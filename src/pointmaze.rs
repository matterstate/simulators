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
/// Uses explicit AVX2+FMA 8-wide SIMD on x86_64 when available.
#[inline]
pub fn step_batch_in_place(x: &mut [f32], y: &mut [f32], ax: &[f32], ay: &[f32]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(ax.len(), n);
    assert_eq!(ay.len(), n);

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: CPU support for AVX2+FMA verified above; slices have equal length `n`.
            unsafe { step_batch_avx2_fma(x, y, ax, ay) };
            return;
        }
    }

    step_batch_scalar(x, y, ax, ay);
}

#[inline]
pub fn step_batch_scalar(x: &mut [f32], y: &mut [f32], ax: &[f32], ay: &[f32]) {
    let n = x.len();
    for i in 0..n {
        let nx = (x[i] + ax[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        let ny = (y[i] + ay[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        if !in_wall(nx, ny) {
            x[i] = nx;
            y[i] = ny;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn step_batch_avx2_fma(x: &mut [f32], y: &mut [f32], ax: &[f32], ay: &[f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;

    let neg_one = _mm256_set1_ps(-1.0);
    let zero = _mm256_set1_ps(0.0);
    let one = _mm256_set1_ps(1.0);
    let dt = _mm256_set1_ps(DT);
    let wx0 = _mm256_set1_ps(WALL_X0);
    let wy0 = _mm256_set1_ps(WALL_Y0);
    let wx1 = _mm256_set1_ps(WALL_X1);
    let wy1 = _mm256_set1_ps(WALL_Y1);

    let xp = x.as_mut_ptr();
    let yp = y.as_mut_ptr();
    let axp = ax.as_ptr();
    let ayp = ay.as_ptr();

    for c in 0..chunks {
        let off = c * 8;
        unsafe {
            let vx = _mm256_loadu_ps(xp.add(off));
            let vy = _mm256_loadu_ps(yp.add(off));
            let vax = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(axp.add(off)), neg_one), one);
            let vay = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(ayp.add(off)), neg_one), one);

            let nx = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(vax, dt, vx), zero), one);
            let ny = _mm256_min_ps(_mm256_max_ps(_mm256_fmadd_ps(vay, dt, vy), zero), one);

            let in_x = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_GE_OQ>(nx, wx0),
                _mm256_cmp_ps::<_CMP_LE_OQ>(nx, wx1),
            );
            let in_y = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_GE_OQ>(ny, wy0),
                _mm256_cmp_ps::<_CMP_LE_OQ>(ny, wy1),
            );
            let blocked = _mm256_and_ps(in_x, in_y);

            let out_x = _mm256_blendv_ps(nx, vx, blocked);
            let out_y = _mm256_blendv_ps(ny, vy, blocked);

            _mm256_storeu_ps(xp.add(off), out_x);
            _mm256_storeu_ps(yp.add(off), out_y);
        }
    }

    let rem = chunks * 8;
    if rem < n {
        step_batch_scalar(&mut x[rem..], &mut y[rem..], &ax[rem..], &ay[rem..]);
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

    #[test]
    fn simd_matches_scalar() {
        let n = 131;
        let mut rng = Rng::seed_from_u64(77);
        let mut x_simd = vec![0.0_f32; n];
        let mut y_simd = vec![0.0_f32; n];
        let mut gx = vec![0.0_f32; n];
        let mut gy = vec![0.0_f32; n];
        reset_batch(&mut rng, &mut x_simd, &mut y_simd, &mut gx, &mut gy);
        let mut x_ref = x_simd.clone();
        let mut y_ref = y_simd.clone();
        let ax: Vec<f32> = (0..n).map(|_| rng.uniform_f32(-1.5, 1.5)).collect();
        let ay: Vec<f32> = (0..n).map(|_| rng.uniform_f32(-1.5, 1.5)).collect();

        step_batch_in_place(&mut x_simd, &mut y_simd, &ax, &ay);
        step_batch_scalar(&mut x_ref, &mut y_ref, &ax, &ay);
        for i in 0..n {
            assert!((x_simd[i] - x_ref[i]).abs() <= 1e-6);
            assert!((y_simd[i] - y_ref[i]).abs() <= 1e-6);
        }
    }
}
