use std::hint::black_box;
use std::time::Instant;

use matterstate_envs::sokoban::{
    Action, BOX, Board, CELLS, FLOOR, HEIGHT, PLR, TARGET, WALL, WIDTH, step_batch_in_place,
};

fn make_board() -> Board {
    let mut cells = [FLOOR; CELLS];
    for c in 0..WIDTH {
        cells[c] = WALL;
        cells[(HEIGHT - 1) * WIDTH + c] = WALL;
    }
    for r in 0..HEIGHT {
        cells[r * WIDTH] = WALL;
        cells[r * WIDTH + (WIDTH - 1)] = WALL;
    }
    cells[4 * WIDTH + 3] = PLR;
    cells[4 * WIDTH + 4] = BOX;
    cells[4 * WIDTH + 5] = TARGET;
    Board::from_cells(cells).expect("valid board")
}

fn main() {
    let board = make_board();
    let iters: usize = 5_000_000;

    let start = Instant::now();
    let mut acc = board;
    for i in 0..iters {
        let action = Action::ALL[i & 3];
        acc.step_in_place(action);
    }
    black_box(acc);
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    println!(
        "sokoban.step_in_place (batch=1):  {:>14.0} steps/s  ({:>5.2} ns/step)",
        iters as f64 / secs,
        elapsed.as_nanos() as f64 / iters as f64,
    );

    let batch = 64usize;
    let batch_iters = 200_000usize;
    let mut boards = vec![board; batch];
    let actions: Vec<Action> = (0..batch).map(|i| Action::ALL[i & 3]).collect();

    let start = Instant::now();
    for _ in 0..batch_iters {
        step_batch_in_place(&mut boards, &actions);
    }
    black_box(&boards);
    let elapsed = start.elapsed();
    let total = (batch * batch_iters) as f64;
    let secs = elapsed.as_secs_f64();
    println!(
        "sokoban.step_batch    (batch=64): {:>14.0} steps/s  ({:>5.2} ns/step)",
        total / secs,
        elapsed.as_nanos() as f64 / total,
    );
}
