//! Vectorized 2D PointMaze kernel matching `matterstate_envs.pointmaze`.

use crate::rng::Rng;

pub const DT: f32 = 0.06;
pub const GOAL_THRESH: f32 = 0.06;
pub const WALL_X0: f32 = 0.45;
pub const WALL_Y0: f32 = 0.00;
pub const WALL_X1: f32 = 0.55;
pub const WALL_Y1: f32 = 0.62;
pub const MAX_SAMPLE_TRIES: usize = 10_000;

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

#[inline]
fn sample_free_one(rng: &mut Rng) -> (f32, f32) {
    for _ in 0..MAX_SAMPLE_TRIES {
        let x = rng.next_f32();
        let y = rng.next_f32();
        if !in_wall(x, y) {
            return (x, y);
        }
    }
    panic!("pointmaze::sample_free_one exceeded MAX_SAMPLE_TRIES");
}

pub fn reset_batch(rng: &mut Rng, x: &mut [f32], y: &mut [f32], gx: &mut [f32], gy: &mut [f32]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    let min_dist2 = 0.4_f32 * 0.4_f32;

    for i in 0..n {
        let (sx, sy) = sample_free_one(rng);
        x[i] = sx;
        y[i] = sy;
        let mut found = false;
        for _ in 0..MAX_SAMPLE_TRIES {
            let (cand_x, cand_y) = sample_free_one(rng);
            let dx = sx - cand_x;
            let dy = sy - cand_y;
            if dx * dx + dy * dy >= min_dist2 {
                gx[i] = cand_x;
                gy[i] = cand_y;
                found = true;
                break;
            }
        }
        assert!(found, "pointmaze::reset_batch exceeded MAX_SAMPLE_TRIES");
    }
}

/// Roll out `horizon` steps in place, recording all `horizon * batch` states.
pub fn rollout_batch(
    horizon: usize,
    x: &mut [f32],
    y: &mut [f32],
    ax: &[f32],
    ay: &[f32],
    traj_x: &mut [f32],
    traj_y: &mut [f32],
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(ax.len(), horizon * n);
    assert_eq!(ay.len(), horizon * n);
    assert_eq!(traj_x.len(), horizon * n);
    assert_eq!(traj_y.len(), horizon * n);

    for t in 0..horizon {
        let off = t * n;
        step_batch_in_place(x, y, &ax[off..off + n], &ay[off..off + n]);
        traj_x[off..off + n].copy_from_slice(x);
        traj_y[off..off + n].copy_from_slice(y);
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
