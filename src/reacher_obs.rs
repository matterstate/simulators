//! Vectorized Reacher-with-Obstacles kernel matching `matterstate_envs.reacher_obs`.

use std::f32::consts::PI;

use crate::reacher::{BASE_X, BASE_Y, L1, L2, LINK_W, STEP, TIP_R, fk, wrap_angle};
use crate::render::{paint_blob, paint_seg};
use crate::rng::Rng;

pub const OBS_R: f32 = 0.09;
pub const MAX_SAMPLE_TRIES: usize = 10_000;

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

#[allow(clippy::too_many_arguments)]
pub fn reset_batch(
    rng: &mut Rng,
    t1: &mut [f32],
    t2: &mut [f32],
    gx: &mut [f32],
    gy: &mut [f32],
    ox: &mut [f32],
    oy: &mut [f32],
    or: &mut [f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(gx.len(), n);
    assert_eq!(gy.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(or.len(), n);
    let r_lo = (L1 - L2).abs() + 0.06;
    let r_hi = L1 + L2 - 0.04;
    let min_tip_goal2 = 0.22_f32 * 0.22_f32;
    let min_goal_obs2 = (OBS_R + 0.03) * (OBS_R + 0.03);

    for i in 0..n {
        let mut found = false;
        for _ in 0..MAX_SAMPLE_TRIES {
            let c1 = rng.uniform_f32(-PI, PI);
            let c2 = rng.uniform_f32(-PI, PI);
            let ang = rng.uniform_f32(-PI, PI);
            let r = rng.uniform_f32(r_lo, r_hi);
            let goal_x = BASE_X + ang.cos() * r;
            let goal_y = BASE_Y + ang.sin() * r;
            let (_, _, tx, ty) = fk(c1, c2);
            let cx = (0.5 * (tx + goal_x) + rng.uniform_f32(-0.05, 0.05)).clamp(0.12, 0.88);
            let cy = (0.5 * (ty + goal_y) + rng.uniform_f32(-0.05, 0.05)).clamp(0.12, 0.88);
            let d_tg2 = (tx - goal_x).powi(2) + (ty - goal_y).powi(2);
            let d_go2 = (goal_x - cx).powi(2) + (goal_y - cy).powi(2);
            if d_tg2 > min_tip_goal2 && d_go2 > min_goal_obs2 && !collides(c1, c2, cx, cy, OBS_R) {
                t1[i] = c1;
                t2[i] = c2;
                gx[i] = goal_x;
                gy[i] = goal_y;
                ox[i] = cx;
                oy[i] = cy;
                or[i] = OBS_R;
                found = true;
                break;
            }
        }
        assert!(found, "reacher_obs::reset_batch exceeded MAX_SAMPLE_TRIES");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rollout_batch(
    horizon: usize,
    t1: &mut [f32],
    t2: &mut [f32],
    a1: &[f32],
    a2: &[f32],
    ox: &[f32],
    oy: &[f32],
    or: &[f32],
    traj_t1: &mut [f32],
    traj_t2: &mut [f32],
) {
    let n = t1.len();
    assert_eq!(t2.len(), n);
    assert_eq!(a1.len(), horizon * n);
    assert_eq!(a2.len(), horizon * n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(or.len(), n);
    assert_eq!(traj_t1.len(), horizon * n);
    assert_eq!(traj_t2.len(), horizon * n);

    for t in 0..horizon {
        let off = t * n;
        step_batch_in_place(t1, t2, &a1[off..off + n], &a2[off..off + n], ox, oy, or);
        traj_t1[off..off + n].copy_from_slice(t1);
        traj_t2[off..off + n].copy_from_slice(t2);
    }
}
