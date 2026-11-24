//! Vectorized 2-link Reacher kernel matching `matterstate_envs.reacher`.

use std::f32::consts::{PI, TAU};

use crate::render::{paint_blob, paint_seg};

pub const L1: f32 = 0.22;
pub const L2: f32 = 0.20;
pub const BASE_X: f32 = 0.5;
pub const BASE_Y: f32 = 0.5;
pub const STEP: f32 = 0.15;
pub const GOAL_THRESH: f32 = 0.05;
pub const LINK_W: f32 = 0.018;
pub const TIP_R: f32 = 0.03;

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
