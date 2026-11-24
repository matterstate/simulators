//! Vectorized Procedural PointMaze kernel matching `matterstate_envs.pointmaze_proc`.

use crate::render::IMG;

pub const GRID: usize = 6;
pub const CELL: usize = IMG / GRID;
pub const DT: f32 = 0.06;
pub const GOAL_THRESH: f32 = 0.06;
pub const WALL_CELLS: usize = GRID * GRID;

#[inline]
pub fn in_wall(x: f32, y: f32, walls: &[bool; WALL_CELLS]) -> bool {
    let col = ((x * GRID as f32).floor() as isize).clamp(0, (GRID - 1) as isize) as usize;
    let row = ((y * GRID as f32).floor() as isize).clamp(0, (GRID - 1) as isize) as usize;
    walls[row * GRID + col]
}

#[inline]
pub fn step_batch_in_place(
    x: &mut [f32],
    y: &mut [f32],
    ax: &[f32],
    ay: &[f32],
    walls: &[[bool; WALL_CELLS]],
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(ax.len(), n);
    assert_eq!(ay.len(), n);
    assert_eq!(walls.len(), n);

    for i in 0..n {
        let nx = (x[i] + ax[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        let ny = (y[i] + ay[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        if !in_wall(nx, ny, &walls[i]) {
            x[i] = nx;
            y[i] = ny;
        }
    }
}

pub fn render_batch(
    x: &[f32],
    y: &[f32],
    walls: &[[bool; WALL_CELLS]],
    r_px: f32,
    frames: &mut [f32],
) {
    let n = x.len();
    let frame_len = IMG * IMG;
    assert_eq!(y.len(), n);
    assert_eq!(walls.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    let r2 = r_px * r_px;
    let img_f = IMG as f32;

    for i in 0..n {
        let frame = &mut frames[i * frame_len..(i + 1) * frame_len];
        let w = &walls[i];
        for gr in 0..GRID {
            for gc in 0..GRID {
                let val = if w[gr * GRID + gc] { 0.4_f32 } else { 0.0_f32 };
                for dy in 0..CELL {
                    let row_start = (gr * CELL + dy) * IMG + gc * CELL;
                    frame[row_start..row_start + CELL].fill(val);
                }
            }
        }

        let px = x[i] * img_f;
        let py = y[i] * img_f;
        let x0 = (px - r_px).floor().max(0.0) as usize;
        let x1 = ((px + r_px).ceil() as isize + 1).clamp(0, IMG as isize) as usize;
        let y0 = (py - r_px).floor().max(0.0) as usize;
        let y1 = ((py + r_px).ceil() as isize + 1).clamp(0, IMG as isize) as usize;

        for py_i in y0..y1 {
            let dy = py_i as f32 - py;
            let dy2 = dy * dy;
            let row = &mut frame[py_i * IMG..(py_i + 1) * IMG];
            for (px_i, pixel) in row.iter_mut().enumerate().take(x1).skip(x0) {
                let dx = px_i as f32 - px;
                if dx * dx + dy2 <= r2 {
                    *pixel = 1.0;
                }
            }
        }
    }
}
