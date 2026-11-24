//! Vectorized 2-link Reacher kernel matching `matterstate_envs.reacher`.

use std::f32::consts::{PI, TAU};

use crate::render::{paint_blob, paint_seg};
use crate::rng::Rng;

pub const L1: f32 = 0.22;
pub const L2: f32 = 0.20;
pub const BASE_X: f32 = 0.5;
pub const BASE_Y: f32 = 0.5;
pub const STEP: f32 = 0.15;
pub const GOAL_THRESH: f32 = 0.05;
pub const LINK_W: f32 = 0.018;
pub const TIP_R: f32 = 0.03;
pub const MAX_SAMPLE_TRIES: usize = 10_000;

#[inline]
pub fn fk(t1: f32, t2: f32) -> (f32, f32, f32, f32) {
    let t12 = t1 + t2;
    let ex = BASE_X + L1 * t1.cos();
    let ey = BASE_Y + L1 * t1.sin();
    let tx = ex + L2 * t12.cos();
    let ty = ey + L2 * t12.sin();
    (ex, ey, tx, ty)
}

#[inline]
pub fn wrap_angle(theta: f32) -> f32 {
    (theta + PI).rem_euclid(TAU) - PI
}

/// Step a Struct-of-Arrays batch of joint angles in place.
#[inline]
pub fn step_batch_in_place(t1: &mut [f32], t2: &mut [f32], a1: &[f32], a2: &[f32]) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(a1.len(), n);
    assert_eq!(a2.len(), n);

    for i in 0..n {
        t1[i] = wrap_angle(t1[i] + a1[i].clamp(-1.0, 1.0) * STEP);
        t2[i] = wrap_angle(t2[i] + a2[i].clamp(-1.0, 1.0) * STEP);
    }
}

#[inline]
pub fn solved_batch(t1: &[f32], t2: &[f32], gx: &[f32], gy: &[f32], out: &mut [bool]) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    assert_eq!(out.len(), n);
    let thresh2 = GOAL_THRESH * GOAL_THRESH;

    for i in 0..n {
        let (_, _, tx, ty) = fk(t1[i], t2[i]);
        let dx = tx - gx[i];
        let dy = ty - gy[i];
        out[i] = dx * dx + dy * dy < thresh2;
    }
}

/// Render a batch of arms into `frames` of length `batch * img * img`.
pub fn render_batch(t1: &[f32], t2: &[f32], img: usize, frames: &mut [f32]) {
    let n = t1.len();
    let frame_len = img * img;
    assert_eq!(t2.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    frames.fill(0.0);
    for i in 0..n {
        let (ex, ey, tx, ty) = fk(t1[i], t2[i]);
        let frame = &mut frames[i * frame_len..(i + 1) * frame_len];
        paint_seg(frame, img, BASE_X, BASE_Y, ex, ey, LINK_W, 0.5);
        paint_seg(frame, img, ex, ey, tx, ty, LINK_W, 0.8);
        paint_blob(frame, img, tx, ty, TIP_R, 1.0);
    }
}

pub fn reset_batch(rng: &mut Rng, t1: &mut [f32], t2: &mut [f32], gx: &mut [f32], gy: &mut [f32]) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    let r_lo = (L1 - L2).abs() + 0.06;
    let r_hi = L1 + L2 - 0.04;
    let min_dist2 = 0.2_f32 * 0.2_f32;

    for i in 0..n {
        let ang = rng.uniform_f32(-PI, PI);
        let r = rng.uniform_f32(r_lo, r_hi);
        let goal_x = BASE_X + ang.cos() * r;
        let goal_y = BASE_Y + ang.sin() * r;
        gx[i] = goal_x;
        gy[i] = goal_y;

        let mut found = false;
        for _ in 0..MAX_SAMPLE_TRIES {
            let c1 = rng.uniform_f32(-PI, PI);
            let c2 = rng.uniform_f32(-PI, PI);
            let (_, _, tx, ty) = fk(c1, c2);
            let dx = tx - goal_x;
            let dy = ty - goal_y;
            if dx * dx + dy * dy > min_dist2 {
                t1[i] = c1;
                t2[i] = c2;
                found = true;
                break;
            }
        }
        assert!(found, "reacher::reset_batch exceeded MAX_SAMPLE_TRIES");
    }
}

pub fn rollout_batch(
    horizon: usize,
    t1: &mut [f32],
    t2: &mut [f32],
    a1: &[f32],
    a2: &[f32],
    traj_t1: &mut [f32],
    traj_t2: &mut [f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(a1.len(), horizon * n);
    assert_eq!(a2.len(), horizon * n);
    assert_eq!(traj_t1.len(), horizon * n);
    assert_eq!(traj_t2.len(), horizon * n);

    for t in 0..horizon {
        let off = t * n;
        step_batch_in_place(t1, t2, &a1[off..off + n], &a2[off..off + n]);
        traj_t1[off..off + n].copy_from_slice(t1);
        traj_t2[off..off + n].copy_from_slice(t2);
    }
}
