//! Vectorized 2-link ArmPush kernel matching `matterstate_envs.armpush`.

use crate::reacher::{BASE_X, BASE_Y, LINK_W, STEP, fk, wrap_angle};
use crate::render::{paint_blob, paint_seg};

pub const R_TIP: f32 = 0.03;
pub const R_OBJ: f32 = 0.06;
pub const R_CONTACT: f32 = R_TIP + R_OBJ;
pub const GOAL_THRESH: f32 = 0.07;

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_batch_in_place(
    t1: &mut [f32],
    t2: &mut [f32],
    ox: &mut [f32],
    oy: &mut [f32],
    a1: &[f32],
    a2: &[f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(a1.len(), n);
    assert_eq!(a2.len(), n);

    for i in 0..n {
        let nt1 = wrap_angle(t1[i] + a1[i].clamp(-1.0, 1.0) * STEP);
        let nt2 = wrap_angle(t2[i] + a2[i].clamp(-1.0, 1.0) * STEP);
        t1[i] = nt1;
        t2[i] = nt2;
        let (_, _, tx, ty) = fk(nt1, nt2);
        let dx = tx - ox[i];
        let dy = ty - oy[i];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < R_CONTACT {
            let scale = R_CONTACT / (dist + 1e-9);
            ox[i] = (tx - dx * scale).clamp(0.0, 1.0);
            oy[i] = (ty - dy * scale).clamp(0.0, 1.0);
        }
    }
}

pub fn render_batch(
    t1: &[f32],
    t2: &[f32],
    ox: &[f32],
    oy: &[f32],
    img: usize,
    frames: &mut [f32],
) {
    let n = t1.len();
    let frame_len = img * img;
    assert_eq!(t2.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    frames.fill(0.0);
    for i in 0..n {
        let (ex, ey, tx, ty) = fk(t1[i], t2[i]);
        let frame = &mut frames[i * frame_len..(i + 1) * frame_len];
        paint_blob(frame, img, ox[i], oy[i], R_OBJ, 0.55);
        paint_seg(frame, img, BASE_X, BASE_Y, ex, ey, LINK_W, 0.5);
        paint_seg(frame, img, ex, ey, tx, ty, LINK_W, 0.8);
        paint_blob(frame, img, tx, ty, R_TIP, 1.0);
    }
}
