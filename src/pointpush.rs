//! Vectorized 2D PointPush kernel matching `matterstate_envs.pointpush`.

use crate::render::paint_blob;

pub const DT: f32 = 0.05;
pub const R_AGENT: f32 = 0.035;
pub const R_OBJ: f32 = 0.06;
pub const R_CONTACT: f32 = R_AGENT + R_OBJ;
pub const GOAL_THRESH: f32 = 0.07;

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_batch_in_place(
    ax_pos: &mut [f32],
    ay_pos: &mut [f32],
    ox: &mut [f32],
    oy: &mut [f32],
    act_x: &[f32],
    act_y: &[f32],
) {
    let n = ax_pos.len();
    assert_eq!(ay_pos.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(act_x.len(), n);
    assert_eq!(act_y.len(), n);

    for i in 0..n {
        let mut ax = (ax_pos[i] + act_x[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        let mut ay = (ay_pos[i] + act_y[i].clamp(-1.0, 1.0) * DT).clamp(0.0, 1.0);
        let dx = ax - ox[i];
        let dy = ay - oy[i];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < R_CONTACT {
            let scale = R_CONTACT / (dist + 1e-9);
            let onx = (ax - dx * scale).clamp(0.0, 1.0);
            let ony = (ay - dy * scale).clamp(0.0, 1.0);
            let dx2 = ax - onx;
            let dy2 = ay - ony;
            let dist2 = (dx2 * dx2 + dy2 * dy2).sqrt();
            if dist2 < R_CONTACT {
                let scale2 = R_CONTACT / (dist2 + 1e-9);
                ax = (onx + dx2 * scale2).clamp(0.0, 1.0);
                ay = (ony + dy2 * scale2).clamp(0.0, 1.0);
            }
            ox[i] = onx;
            oy[i] = ony;
        }
        ax_pos[i] = ax;
        ay_pos[i] = ay;
    }
}

pub fn render_batch(
    ax_pos: &[f32],
    ay_pos: &[f32],
    ox: &[f32],
    oy: &[f32],
    img: usize,
    frames: &mut [f32],
) {
    let n = ax_pos.len();
    let frame_len = img * img;
    assert_eq!(ay_pos.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    frames.fill(0.0);
    for i in 0..n {
        let frame = &mut frames[i * frame_len..(i + 1) * frame_len];
        paint_blob(frame, img, ox[i], oy[i], R_OBJ, 0.55);
        paint_blob(frame, img, ax_pos[i], ay_pos[i], R_AGENT, 1.0);
    }
}
