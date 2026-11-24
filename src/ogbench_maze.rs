//! Fast 2D force-controlled OGBench-style PointMaze kernel (`[x, y, vx, vy]`).

pub const CELL_SIZE: f32 = 4.0;
pub const AGENT_R: f32 = 0.5;
pub const DT: f32 = 0.1;
pub const DAMPING: f32 = 0.9;
pub const MAX_VEL: f32 = 5.0;

/// Standard 8x8 OGBench `pointmaze-medium` layout (`1` = wall, `0` = free).
pub const MEDIUM_ROWS: usize = 8;
pub const MEDIUM_COLS: usize = 8;
pub const MEDIUM_GRID: [[u8; MEDIUM_COLS]; MEDIUM_ROWS] = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
];

/// Standard 9x12 OGBench `pointmaze-large` layout (`1` = wall, `0` = free).
pub const LARGE_ROWS: usize = 9;
pub const LARGE_COLS: usize = 12;
pub const LARGE_GRID: [[u8; LARGE_COLS]; LARGE_ROWS] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

#[inline]
fn resolve_grid<const R: usize, const C: usize>(
    grid: &[[u8; C]; R],
    x: &mut f32,
    y: &mut f32,
    vx: &mut f32,
    vy: &mut f32,
) {
    let min_c = ((*x - AGENT_R) / CELL_SIZE).floor().max(0.0) as usize;
    let max_c = (((*x + AGENT_R) / CELL_SIZE).floor() as usize).min(C - 1);
    let min_r = ((*y - AGENT_R) / CELL_SIZE).floor().max(0.0) as usize;
    let max_r = (((*y + AGENT_R) / CELL_SIZE).floor() as usize).min(R - 1);

    for (r, row) in grid.iter().enumerate().take(max_r + 1).skip(min_r) {
        for (c, &cell) in row.iter().enumerate().take(max_c + 1).skip(min_c) {
            if cell == 0 {
                continue;
            }
            let x0 = c as f32 * CELL_SIZE;
            let x1 = x0 + CELL_SIZE;
            let y0 = r as f32 * CELL_SIZE;
            let y1 = y0 + CELL_SIZE;
            let cx = x.clamp(x0, x1);
            let cy = y.clamp(y0, y1);
            let dx = *x - cx;
            let dy = *y - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < AGENT_R * AGENT_R && d2 > 1e-10 {
                let d = d2.sqrt();
                let nx = dx / d;
                let ny = dy / d;
                let pen = AGENT_R - d;
                *x += nx * pen;
                *y += ny * pen;
                let vn = *vx * nx + *vy * ny;
                if vn < 0.0 {
                    *vx -= vn * nx;
                    *vy -= vn * ny;
                }
            }
        }
    }
}

/// Step a Struct-of-Arrays batch of 4D states `(x, y, vx, vy)` on the `medium` grid.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_medium_batch_in_place(
    x: &mut [f32],
    y: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    fx: &[f32],
    fy: &[f32],
) {
    step_grid_batch_in_place(&MEDIUM_GRID, x, y, vx, vy, fx, fy);
}

/// Step a Struct-of-Arrays batch of 4D states `(x, y, vx, vy)` on the `large` grid.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_large_batch_in_place(
    x: &mut [f32],
    y: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    fx: &[f32],
    fy: &[f32],
) {
    step_grid_batch_in_place(&LARGE_GRID, x, y, vx, vy, fx, fy);
}

#[inline]
fn step_grid_batch_in_place<const R: usize, const C: usize>(
    grid: &[[u8; C]; R],
    x: &mut [f32],
    y: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    fx: &[f32],
    fy: &[f32],
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(vx.len(), n);
    assert_eq!(vy.len(), n);
    assert_eq!(fx.len(), n);
    assert_eq!(fy.len(), n);

    for i in 0..n {
        vx[i] = (vx[i] * DAMPING + fx[i].clamp(-1.0, 1.0) * 5.0 * DT).clamp(-MAX_VEL, MAX_VEL);
        vy[i] = (vy[i] * DAMPING + fy[i].clamp(-1.0, 1.0) * 5.0 * DT).clamp(-MAX_VEL, MAX_VEL);
        x[i] += vx[i] * DT;
        y[i] += vy[i] * DT;
        resolve_grid(grid, &mut x[i], &mut y[i], &mut vx[i], &mut vy[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_collision_stops_agent() {
        let mut x = [4.6_f32];
        let mut y = [6.0_f32];
        let mut vx = [-2.0_f32];
        let mut vy = [0.0_f32];
        for _ in 0..20 {
            step_medium_batch_in_place(&mut x, &mut y, &mut vx, &mut vy, &[-1.0], &[0.0]);
        }
        assert!(x[0] >= CELL_SIZE + AGENT_R - 1e-3);
    }
}
