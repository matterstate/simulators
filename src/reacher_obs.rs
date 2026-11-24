//! Vectorized Reacher-with-Obstacles kernel matching `matterstate_envs.reacher_obs`.

use crate::reacher::{BASE_X, BASE_Y, LINK_W, STEP, TIP_R, fk, wrap_angle};
use crate::render::{paint_blob, paint_seg};

pub const OBS_R: f32 = 0.09;

#[inline]
pub fn seg_point_dist2(x0: f32, y0: f32, x1: f32, y1: f32, cx: f32, cy: f32) -> f32 {
    let sx = x1 - x0;
    let sy = y1 - y0;
    let t = (((cx - x0) * sx + (cy - y0) * sy) / (sx * sx + sy * sy + 1e-9)).clamp(0.0, 1.0);
    let dx = cx - (x0 + t * sx);
    let dy = cy - (y0 + t * sy);
    dx * dx + dy * dy
}

#[inline]
pub fn collides(t1: f32, t2: f32, cx: f32, cy: f32, r: f32) -> bool {
    let (ex, ey, tx, ty) = fk(t1, t2);
    let d2 = seg_point_dist2(BASE_X, BASE_Y, ex, ey, cx, cy)
        .min(seg_point_dist2(ex, ey, tx, ty, cx, cy));
    d2 < r * r
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_batch_in_place(
    t1: &mut [f32],
    t2: &mut [f32],
    a1: &[f32],
    a2: &[f32],
    ox: &[f32],
    oy: &[f32],
    or: &[f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(a1.len(), n);
    assert_eq!(a2.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(or.len(), n);

    for i in 0..n {
        let p1 = wrap_angle(t1[i] + a1[i].clamp(-1.0, 1.0) * STEP);
        let p2 = wrap_angle(t2[i] + a2[i].clamp(-1.0, 1.0) * STEP);
        if !collides(p1, p2, ox[i], oy[i], or[i]) {
            t1[i] = p1;
            t2[i] = p2;
        }
    }
}

pub fn render_batch(
    t1: &[f32],
    t2: &[f32],
    ox: &[f32],
    oy: &[f32],
    or: &[f32],
    img: usize,
    frames: &mut [f32],
) {
    let n = t1.len();
    let frame_len = img * img;
    assert_eq!(t2.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(or.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    frames.fill(0.0);
    for i in 0..n {
        let (ex, ey, tx, ty) = fk(t1[i], t2[i]);
        let frame = &mut frames[i * frame_len..(i + 1) * frame_len];
        paint_blob(frame, img, ox[i], oy[i], or[i], 0.3);
        paint_seg(frame, img, BASE_X, BASE_Y, ex, ey, LINK_W, 0.5);
        paint_seg(frame, img, ex, ey, tx, ty, LINK_W, 0.8);
        paint_blob(frame, img, tx, ty, TIP_R, 1.0);
    }
}
