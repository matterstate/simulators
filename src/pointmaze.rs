//! Vectorized 2D PointMaze kernel matching `matterstate_envs.pointmaze`.

pub const DT: f32 = 0.06;
pub const GOAL_THRESH: f32 = 0.06;
pub const WALL_X0: f32 = 0.45;
pub const WALL_Y0: f32 = 0.00;
pub const WALL_X1: f32 = 0.55;
pub const WALL_Y1: f32 = 0.62;

#[inline]
pub fn in_wall(x: f32, y: f32) -> bool {
    (WALL_X0..=WALL_X1).contains(&x) && (WALL_Y0..=WALL_Y1).contains(&y)
}

/// Step a Struct-of-Arrays batch of positions in place.
#[inline]
pub fn step_batch_in_place(x: &mut [f32], y: &mut [f32], ax: &[f32], ay: &[f32]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(ax.len(), n);
    assert_eq!(ay.len(), n);

    for i in 0..n {
        let nx = (x[i] + ax[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        let ny = (y[i] + ay[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        if !in_wall(nx, ny) {
            x[i] = nx;
            y[i] = ny;
        }
    }
}

#[inline]
pub fn solved_batch(x: &[f32], y: &[f32], gx: &[f32], gy: &[f32], out: &mut [bool]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    assert_eq!(out.len(), n);
    let thresh2 = GOAL_THRESH * GOAL_THRESH;

    for i in 0..n {
        let dx = x[i] - gx[i];
        let dy = y[i] - gy[i];
        out[i] = dx * dx + dy * dy < thresh2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_blocks_move() {
        let mut x = [0.44_f32];
        let mut y = [0.30_f32];
        step_batch_in_place(&mut x, &mut y, &[1.0], &[0.0]);
        assert_eq!(x[0], 0.44);
        assert_eq!(y[0], 0.30);
    }
}
