//! Vectorized variable-morphology 2-link Reacher matching `matterstate_envs.reacher_morph`.

use crate::reacher::{BASE_X, BASE_Y, GOAL_THRESH, LINK_W, STEP, TIP_R, wrap_angle};
use crate::render::{paint_blob, paint_seg};

#[inline]
pub fn fk(t1: f32, t2: f32, l1: f32, l2: f32) -> (f32, f32, f32, f32) {
    let t12 = t1 + t2;
    let ex = BASE_X + l1 * t1.cos();
    let ey = BASE_Y + l1 * t1.sin();
    let tx = ex + l2 * t12.cos();
    let ty = ey + l2 * t12.sin();
    (ex, ey, tx, ty)
}

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
#[allow(clippy::too_many_arguments)]
pub fn solved_batch(
    t1: &[f32],
    t2: &[f32],
    l1: &[f32],
    l2: &[f32],
    gx: &[f32],
    gy: &[f32],
    out: &mut [bool],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(l1.len(), n);
    assert_eq!(l2.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    assert_eq!(out.len(), n);
    let thresh2 = GOAL_THRESH * GOAL_THRESH;

    for i in 0..n {
        let (_, _, tx, ty) = fk(t1[i], t2[i], l1[i], l2[i]);
        let dx = tx - gx[i];
        let dy = ty - gy[i];
        out[i] = dx * dx + dy * dy < thresh2;
    }
}

pub fn render_batch(
    t1: &[f32],
    t2: &[f32],
    l1: &[f32],
    l2: &[f32],
    img: usize,
    frames: &mut [f32],
) {
    let n = t1.len();
    let frame_len = img * img;
    assert_eq!(t2.len(), n);
    assert_eq!(l1.len(), n);
    assert_eq!(l2.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    frames.fill(0.0);
    for i in 0..n {
        let (ex, ey, tx, ty) = fk(t1[i], t2[i], l1[i], l2[i]);
        let frame = &mut frames[i * frame_len..(i + 1) * frame_len];
        paint_seg(frame, img, BASE_X, BASE_Y, ex, ey, LINK_W, 0.5);
        paint_seg(frame, img, ex, ey, tx, ty, LINK_W, 0.8);
        paint_blob(frame, img, tx, ty, TIP_R, 1.0);
    }
}
