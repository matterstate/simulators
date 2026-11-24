//! Deterministic 10x10 Sokoban kernel matching `matterstate_envs.sokoban`.

use std::fmt;

pub const WALL: u8 = 0;
pub const FLOOR: u8 = 1;
pub const TARGET: u8 = 2;
pub const BOX: u8 = 3;
pub const BOX_T: u8 = 4;
pub const PLR: u8 = 5;
pub const PLR_T: u8 = 6;

pub const HEIGHT: usize = 10;
pub const WIDTH: usize = 10;
pub const CELLS: usize = HEIGHT * WIDTH;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Action {
    pub const ALL: [Action; 4] = [Action::Up, Action::Down, Action::Left, Action::Right];

    #[inline]
    pub const fn delta(self) -> (isize, isize) {
        match self {
            Action::Up => (-1, 0),
            Action::Down => (1, 0),
            Action::Left => (0, -1),
            Action::Right => (0, 1),
        }
    }

    #[inline]
    pub const fn from_u8(value: u8) -> Result<Self, SokobanError> {
        match value {
            0 => Ok(Action::Up),
            1 => Ok(Action::Down),
            2 => Ok(Action::Left),
            3 => Ok(Action::Right),
            _ => Err(SokobanError::InvalidAction(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SokobanError {
    InvalidAction(u8),
    InvalidCell(u8),
    MissingPlayer,
    MultiplePlayers,
    NoFreeFloor,
}

impl fmt::Display for SokobanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SokobanError::InvalidAction(a) => write!(f, "invalid action: {a}"),
            SokobanError::InvalidCell(c) => write!(f, "invalid cell value: {c}"),
            SokobanError::MissingPlayer => write!(f, "board has no player cell"),
            SokobanError::MultiplePlayers => write!(f, "board has multiple player cells"),
            SokobanError::NoFreeFloor => write!(f, "board has no free floor cell"),
        }
    }
}

impl std::error::Error for SokobanError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    cells: [u8; CELLS],
    player: u8,
}

impl Board {
    /// Construct a validated board from a flat 10x10 row-major grid.
    pub fn from_cells(cells: [u8; CELLS]) -> Result<Self, SokobanError> {
        let mut player: Option<u8> = None;
        for (idx, &cell) in cells.iter().enumerate() {
            if cell > PLR_T {
                return Err(SokobanError::InvalidCell(cell));
            }
            if cell == PLR || cell == PLR_T {
                if player.is_some() {
                    return Err(SokobanError::MultiplePlayers);
                }
                player = Some(idx as u8);
            }
        }
        let Some(player) = player else {
            return Err(SokobanError::MissingPlayer);
        };
        Ok(Self { cells, player })
    }

    #[inline]
    pub const fn cells(&self) -> &[u8; CELLS] {
        &self.cells
    }

    #[inline]
    pub const fn player_index(&self) -> usize {
        self.player as usize
    }

    /// Apply one move in place. Illegal moves are no-ops.
    #[inline]
    pub fn step_in_place(&mut self, action: Action) {
        let p_idx = self.player as usize;
        let pr = (p_idx / WIDTH) as isize;
        let pc = (p_idx % WIDTH) as isize;
        let (dr, dc) = action.delta();
        let nr = pr + dr;
        let nc = pc + dc;
        if nr < 0 || nr >= HEIGHT as isize || nc < 0 || nc >= WIDTH as isize {
            return;
        }
        let n_idx = (nr as usize) * WIDTH + (nc as usize);
        let next_cell = self.cells[n_idx];
        if next_cell == WALL {
            return;
        }
        let on_target = self.cells[p_idx] == PLR_T;
        if next_cell == BOX || next_cell == BOX_T {
            let br = nr + dr;
            let bc = nc + dc;
            if br < 0 || br >= HEIGHT as isize || bc < 0 || bc >= WIDTH as isize {
                return;
            }
            let b_idx = (br as usize) * WIDTH + (bc as usize);
            let beyond = self.cells[b_idx];
            if beyond == WALL || beyond == BOX || beyond == BOX_T {
                return;
            }
            self.cells[b_idx] = if beyond == TARGET { BOX_T } else { BOX };
            self.cells[n_idx] = if next_cell == BOX_T { PLR_T } else { PLR };
        } else {
            self.cells[n_idx] = if next_cell == TARGET { PLR_T } else { PLR };
        }
        self.cells[p_idx] = if on_target { TARGET } else { FLOOR };
        self.player = n_idx as u8;
    }

    /// Functional copy-and-step matching `matterstate_envs.sokoban.step`.
    #[inline]
    pub fn step(&self, action: Action) -> Self {
        let mut next = *self;
        next.step_in_place(action);
        next
    }

    /// True iff no bare `BOX` cells remain.
    #[inline]
    pub fn solved(&self) -> bool {
        !self.cells.contains(&BOX)
    }

    /// Number of boxes currently on targets.
    #[inline]
    pub fn n_on_target(&self) -> u8 {
        self.cells.iter().filter(|&&c| c == BOX_T).count() as u8
    }

    /// Build a solved board sharing this board's walls and targets.
    pub fn make_solved(&self) -> Result<Self, SokobanError> {
        let mut cells = self.cells;
        for cell in &mut cells {
            match *cell {
                BOX | PLR => *cell = FLOOR,
                BOX_T | PLR_T | TARGET => *cell = BOX_T,
                _ => {}
            }
        }
        let mut free_indices = [0u8; CELLS];
        let mut free_count = 0usize;
        for (idx, &cell) in cells.iter().enumerate() {
            if cell == FLOOR {
                free_indices[free_count] = idx as u8;
                free_count += 1;
            }
        }
        if free_count == 0 {
            return Err(SokobanError::NoFreeFloor);
        }
        let player = free_indices[free_count / 2];
        cells[player as usize] = PLR;
        Ok(Self { cells, player })
    }

    /// All `(predecessor, action)` pairs such that `predecessor.step(action) == *self`.
    pub fn reverse_predecessors(&self) -> Vec<(Self, Action)> {
        let p_idx = self.player as usize;
        let pr = (p_idx / WIDTH) as isize;
        let pc = (p_idx % WIDTH) as isize;
        let p_under_t = self.cells[p_idx] == PLR_T;
        let mut out = Vec::with_capacity(8);

        for action in Action::ALL {
            let (dr, dc) = action.delta();
            let qr = pr - dr;
            let qc = pc - dc;
            if qr < 0 || qr >= HEIGHT as isize || qc < 0 || qc >= WIDTH as isize {
                continue;
            }
            let q_idx = (qr as usize) * WIDTH + (qc as usize);
            let q_cell = self.cells[q_idx];
            if q_cell == WALL || q_cell == BOX || q_cell == BOX_T {
                continue;
            }

            let mut plain = *self;
            plain.cells[q_idx] = if q_cell == TARGET { PLR_T } else { PLR };
            plain.cells[p_idx] = if p_under_t { TARGET } else { FLOOR };
            plain.player = q_idx as u8;
            out.push((plain, action));

            let fr = pr + dr;
            let fc = pc + dc;
            if fr >= 0 && fr < HEIGHT as isize && fc >= 0 && fc < WIDTH as isize {
                let f_idx = (fr as usize) * WIDTH + (fc as usize);
                let f_cell = self.cells[f_idx];
                if f_cell == BOX || f_cell == BOX_T {
                    let mut pull = *self;
                    pull.cells[q_idx] = if q_cell == TARGET { PLR_T } else { PLR };
                    pull.cells[p_idx] = if p_under_t { BOX_T } else { BOX };
                    pull.cells[f_idx] = if f_cell == BOX_T { TARGET } else { FLOOR };
                    pull.player = q_idx as u8;
                    out.push((pull, action));
                }
            }
        }
        out
    }
}

/// Step a batch of boards in place.
#[inline]
pub fn step_batch_in_place(boards: &mut [Board], actions: &[Action]) {
    assert_eq!(boards.len(), actions.len());
    for (board, &action) in boards.iter_mut().zip(actions.iter()) {
        board.step_in_place(action);
    }
}

/// Roll out `horizon` steps over a batch of boards in place.
pub fn rollout_batch(
    horizon: usize,
    boards: &mut [Board],
    actions: &[Action],
    out_boards: &mut [Board],
) {
    let n = boards.len();
    assert_eq!(actions.len(), horizon * n);
    assert_eq!(out_boards.len(), horizon * n);
    for t in 0..horizon {
        let off = t * n;
        step_batch_in_place(boards, &actions[off..off + n]);
        out_boards[off..off + n].copy_from_slice(boards);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walled_board(row4_slice: &[u8]) -> Board {
        let mut cells = [FLOOR; CELLS];
        for c in 0..WIDTH {
            cells[c] = WALL;
            cells[(HEIGHT - 1) * WIDTH + c] = WALL;
        }
        for r in 0..HEIGHT {
            cells[r * WIDTH] = WALL;
            cells[r * WIDTH + (WIDTH - 1)] = WALL;
        }
        for (i, &val) in row4_slice.iter().enumerate() {
            cells[4 * WIDTH + 3 + i] = val;
        }
        Board::from_cells(cells).expect("valid test board")
    }

    #[test]
    fn push_onto_target_and_solve() {
        let mut board = walled_board(&[PLR, BOX, TARGET, FLOOR]);
        assert!(!board.solved());
        assert_eq!(board.n_on_target(), 0);
        board.step_in_place(Action::Right);
        assert!(board.solved());
        assert_eq!(board.n_on_target(), 1);
        assert_eq!(board.cells()[4 * WIDTH + 3], FLOOR);
        assert_eq!(board.cells()[4 * WIDTH + 4], PLR);
        assert_eq!(board.cells()[4 * WIDTH + 5], BOX_T);
    }

    #[test]
    fn blocked_push_is_noop() {
        let board = walled_board(&[PLR, BOX, WALL, TARGET]);
        let next = board.step(Action::Right);
        assert_eq!(next, board);
    }

    #[test]
    fn reverse_predecessors_round_trip() {
        let cases = [
            walled_board(&[PLR, BOX, TARGET, FLOOR]),
            walled_board(&[PLR, BOX_T, FLOOR, FLOOR]),
            walled_board(&[PLR_T, BOX, FLOOR, FLOOR]),
            walled_board(&[PLR, BOX, WALL, TARGET]),
        ];
        for board in cases {
            for (pred, action) in board.reverse_predecessors() {
                assert_eq!(pred.step(action), board);
            }
        }
    }

    #[test]
    fn make_solved_fills_all_targets() {
        let board = walled_board(&[PLR, BOX, TARGET, FLOOR]);
        let solved = board.make_solved().expect("solved board");
        assert!(solved.solved());
        assert_eq!(solved.n_on_target(), 1);
    }
}
