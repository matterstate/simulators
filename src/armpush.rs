//! Vectorized 2-link ArmPush kernel matching `matterstate_envs.armpush`.

use std::f32::consts::PI;

use crate::reacher::{BASE_X, BASE_Y, L1, L2, LINK_W, STEP, fk, wrap_angle};
use crate::render::{paint_blob, paint_seg};
use crate::rng::Rng;

pub const R_TIP: f32 = 0.03;
pub const R_OBJ: f32 = 0.06;
pub const R_CONTACT: f32 = R_TIP + R_OBJ;
pub const GOAL_THRESH: f32 = 0.07;
pub const MAX_SAMPLE_TRIES: usize = 10_000;

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

#[inline]
fn sample_annulus(rng: &mut Rng) -> (f32, f32) {
    let r_lo = (L1 - L2).abs() + 0.04;
    let r_hi = L1 + L2 - 0.04;
    let ang = rng.uniform_f32(-PI, PI);
    let r = rng.uniform_f32(r_lo, r_hi);
    (BASE_X + ang.cos() * r, BASE_Y + ang.sin() * r)
}

pub fn reset_batch(
    rng: &mut Rng,
    t1: &mut [f32],
    t2: &mut [f32],
    ox: &mut [f32],
    oy: &mut [f32],
    gx: &mut [f32],
    gy: &mut [f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    let min_dist2 = 0.25_f32 * 0.25_f32;

    for i in 0..n {
        t1[i] = rng.uniform_f32(-PI, PI);
        t2[i] = rng.uniform_f32(-PI, PI);
        let (obj_x, obj_y) = sample_annulus(rng);
        ox[i] = obj_x;
        oy[i] = obj_y;
        let mut found = false;
        for _ in 0..MAX_SAMPLE_TRIES {
            let (cand_x, cand_y) = sample_annulus(rng);
            let dx = obj_x - cand_x;
            let dy = obj_y - cand_y;
            if dx * dx + dy * dy >= min_dist2 {
                gx[i] = cand_x;
                gy[i] = cand_y;
                found = true;
                break;
            }
        }
        assert!(found, "armpush::reset_batch exceeded MAX_SAMPLE_TRIES");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rollout_batch(
    horizon: usize,
    t1: &mut [f32],
    t2: &mut [f32],
    ox: &mut [f32],
    oy: &mut [f32],
    a1: &[f32],
    a2: &[f32],
    traj_t1: &mut [f32],
    traj_t2: &mut [f32],
    traj_ox: &mut [f32],
    traj_oy: &mut [f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(a1.len(), horizon * n);
    assert_eq!(a2.len(), horizon * n);
    assert_eq!(traj_t1.len(), horizon * n);
    assert_eq!(traj_t2.len(), horizon * n);
    assert_eq!(traj_ox.len(), horizon * n);
    assert_eq!(traj_oy.len(), horizon * n);

    for t in 0..horizon {
        let off = t * n;
        step_batch_in_place(t1, t2, ox, oy, &a1[off..off + n], &a2[off..off + n]);
        traj_t1[off..off + n].copy_from_slice(t1);
        traj_t2[off..off + n].copy_from_slice(t2);
        traj_ox[off..off + n].copy_from_slice(ox);
        traj_oy[off..off + n].copy_from_slice(oy);
    }
}
