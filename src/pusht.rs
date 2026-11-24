//! 2D rigid-body PushT kernel modeled after `gym-pusht` (Chipmunk2D/Pymunk).
//!
//! Geometry and control constants match `gym_pusht/envs/pusht.py`:
//! - 512x512 arena with walls at [7, 504]
//! - Kinematic circular pusher (radius 15) under PD control (k_p=100, k_v=20)
//! - T-block composed of two boxes (120x30 bar + 30x90 stem), mass=1, I=3000
//! - 10 substeps at dt=0.01 (10 Hz control) with zero damping (quasi-static).

use std::f32::consts::{FRAC_PI_4, PI, TAU};

use crate::rng::Rng;

pub const ARENA: f32 = 512.0;
pub const WALL_MIN: f32 = 7.0;
pub const WALL_MAX: f32 = 504.0;
pub const AGENT_R: f32 = 15.0;
pub const DT: f32 = 0.01;
pub const SUBSTEPS: usize = 10;
pub const K_P: f32 = 100.0;
pub const K_V: f32 = 20.0;
pub const MASS: f32 = 1.0;
pub const INV_MASS: f32 = 1.0 / MASS;
pub const INERTIA: f32 = 3000.0;
pub const INV_INERTIA: f32 = 1.0 / INERTIA;

pub const GOAL_X: f32 = 256.0;
pub const GOAL_Y: f32 = 256.0;
pub const GOAL_ANGLE: f32 = FRAC_PI_4;

/// Local-frame AABBs of the two T-block rectangles relative to CoM `(0, 45)`:
/// Bar:  x in [-60, 60], y in [-45, -15]
/// Stem: x in [-15, 15], y in [-15,  75]
const BOXES: [(f32, f32, f32, f32); 2] = [(-60.0, -45.0, 60.0, -15.0), (-15.0, -15.0, 15.0, 75.0)];

const LOCAL_VERTS: [(f32, f32); 8] = [
    (-60.0, -45.0),
    (60.0, -45.0),
    (60.0, -15.0),
    (-60.0, -15.0),
    (-15.0, -15.0),
    (15.0, -15.0),
    (15.0, 75.0),
    (-15.0, 75.0),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub agent_x: f32,
    pub agent_y: f32,
    pub agent_vx: f32,
    pub agent_vy: f32,
    pub block_x: f32,
    pub block_y: f32,
    pub block_angle: f32,
    pub block_vx: f32,
    pub block_vy: f32,
    pub block_omega: f32,
}

impl State {
    pub const fn new(
        agent_x: f32,
        agent_y: f32,
        block_x: f32,
        block_y: f32,
        block_angle: f32,
    ) -> Self {
        Self {
            agent_x,
            agent_y,
            agent_vx: 0.0,
            agent_vy: 0.0,
            block_x,
            block_y,
            block_angle,
            block_vx: 0.0,
            block_vy: 0.0,
            block_omega: 0.0,
        }
    }

    #[inline]
    pub fn step_in_place(&mut self, action_x: f32, action_y: f32) {
        for _ in 0..SUBSTEPS {
            // 1. PD controller for kinematic agent
            let acc_x = K_P * (action_x - self.agent_x) - K_V * self.agent_vx;
            let acc_y = K_P * (action_y - self.agent_y) - K_V * self.agent_vy;
            self.agent_vx += acc_x * DT;
            self.agent_vy += acc_y * DT;
            self.agent_x += self.agent_vx * DT;
            self.agent_y += self.agent_vy * DT;

            // 2. Circle-vs-T-block contact resolution (2 iterations)
            for _ in 0..2 {
                self.resolve_agent_contacts();
                self.resolve_wall_contacts();
            }

            // 3. Integrate block pose and apply space.damping = 0.0
            self.block_x += self.block_vx * DT;
            self.block_y += self.block_vy * DT;
            self.block_angle = (self.block_angle + self.block_omega * DT).rem_euclid(TAU);
            self.block_vx = 0.0;
            self.block_vy = 0.0;
            self.block_omega = 0.0;
        }
    }

    #[inline]
    fn resolve_agent_contacts(&mut self) {
        let cos_a = self.block_angle.cos();
        let sin_a = self.block_angle.sin();
        let dx = self.agent_x - self.block_x;
        let dy = self.agent_y - self.block_y;
        // Agent center in block-local coordinates
        let lx = dx * cos_a + dy * sin_a;
        let ly = -dx * sin_a + dy * cos_a;

        for &(min_x, min_y, max_x, max_y) in &BOXES {
            let cx = lx.clamp(min_x, max_x);
            let cy = ly.clamp(min_y, max_y);
            let diff_x = lx - cx;
            let diff_y = ly - cy;
            let dist2 = diff_x * diff_x + diff_y * diff_y;
            if dist2 >= AGENT_R * AGENT_R {
                continue;
            }
            let (lnx, lny, pen) = if dist2 > 1e-8 {
                let dist = dist2.sqrt();
                // Normal pointing from agent into block
                (-diff_x / dist, -diff_y / dist, AGENT_R - dist)
            } else {
                // Agent center inside box: push out along nearest face
                let dl = (lx - min_x).abs();
                let dr = (max_x - lx).abs();
                let db = (ly - min_y).abs();
                let dt_ = (max_y - ly).abs();
                let m = dl.min(dr).min(db).min(dt_);
                if m == dl {
                    (1.0, 0.0, AGENT_R + dl)
                } else if m == dr {
                    (-1.0, 0.0, AGENT_R + dr)
                } else if m == db {
                    (0.0, 1.0, AGENT_R + db)
                } else {
                    (0.0, -1.0, AGENT_R + dt_)
                }
            };

            // Rotate normal and contact offset back to world frame
            let nx = lnx * cos_a - lny * sin_a;
            let ny = lnx * sin_a + lny * cos_a;
            let rx = cx * cos_a - cy * sin_a;
            let ry = cx * sin_a + cy * cos_a;

            // Relative velocity at contact point (block minus agent)
            let vbx = self.block_vx - self.block_omega * ry;
            let vby = self.block_vy + self.block_omega * rx;
            let vn = (vbx - self.agent_vx) * nx + (vby - self.agent_vy) * ny;
            let rn = rx * ny - ry * nx;
            let inv_k = INV_MASS + rn * rn * INV_INERTIA;

            // Impulse with Baumgarte positional bias
            let bias = 0.2 * pen / DT;
            let j = ((-vn + bias) / inv_k).max(0.0);
            let jx = j * nx;
            let jy = j * ny;
            self.block_vx += jx * INV_MASS;
            self.block_vy += jy * INV_MASS;
            self.block_omega += (rx * jy - ry * jx) * INV_INERTIA;
        }
    }

    #[inline]
    fn resolve_wall_contacts(&mut self) {
        let cos_a = self.block_angle.cos();
        let sin_a = self.block_angle.sin();
        for &(vx, vy) in &LOCAL_VERTS {
            let rx = vx * cos_a - vy * sin_a;
            let ry = vx * sin_a + vy * cos_a;
            let wx = self.block_x + rx;
            let wy = self.block_y + ry;
            if wx < WALL_MIN {
                self.block_x += WALL_MIN - wx;
                self.block_vx = self.block_vx.max(0.0);
            } else if wx > WALL_MAX {
                self.block_x -= wx - WALL_MAX;
                self.block_vx = self.block_vx.min(0.0);
            }
            if wy < WALL_MIN {
                self.block_y += WALL_MIN - wy;
                self.block_vy = self.block_vy.max(0.0);
            } else if wy > WALL_MAX {
                self.block_y -= wy - WALL_MAX;
                self.block_vy = self.block_vy.min(0.0);
            }
        }
    }

    /// World-space coordinates of the 8 T-block keypoints.
    pub fn keypoints(&self) -> [(f32, f32); 8] {
        let cos_a = self.block_angle.cos();
        let sin_a = self.block_angle.sin();
        let mut out = [(0.0, 0.0); 8];
        for (dst, &(vx, vy)) in out.iter_mut().zip(LOCAL_VERTS.iter()) {
            *dst = (
                self.block_x + vx * cos_a - vy * sin_a,
                self.block_y + vx * sin_a + vy * cos_a,
            );
        }
        out
    }

    /// True if local-frame point `(lx, ly)` lies inside either T-block rectangle.
    #[inline]
    pub fn point_in_local_tee(lx: f32, ly: f32) -> bool {
        BOXES
            .iter()
            .any(|&(x0, y0, x1, y1)| (x0..=x1).contains(&lx) && (y0..=y1).contains(&ly))
    }

    /// Approximate fraction of the goal T-block covered by the current T-block.
    /// Uses a deterministic 16x16 grid over each of the two goal rectangles (512 samples).
    pub fn coverage(&self) -> f32 {
        let g_cos = GOAL_ANGLE.cos();
        let g_sin = GOAL_ANGLE.sin();
        let b_cos = self.block_angle.cos();
        let b_sin = self.block_angle.sin();

        let mut inside = 0u32;
        let mut total = 0u32;
        // Weight each box by its area (bar: 120*30=3600, stem: 30*90=2700 -> 4:3)
        for &(x0, y0, x1, y1) in &BOXES {
            let nx = ((x1 - x0) / 5.0) as usize;
            let ny = ((y1 - y0) / 5.0) as usize;
            let dx = (x1 - x0) / nx as f32;
            let dy = (y1 - y0) / ny as f32;
            for ix in 0..nx {
                let lx = x0 + (ix as f32 + 0.5) * dx;
                for iy in 0..ny {
                    let ly = y0 + (iy as f32 + 0.5) * dy;
                    let wx = GOAL_X + lx * g_cos - ly * g_sin;
                    let wy = GOAL_Y + lx * g_sin + ly * g_cos;
                    let rel_x = wx - self.block_x;
                    let rel_y = wy - self.block_y;
                    let blx = rel_x * b_cos + rel_y * b_sin;
                    let bly = -rel_x * b_sin + rel_y * b_cos;
                    if Self::point_in_local_tee(blx, bly) {
                        inside += 1;
                    }
                    total += 1;
                }
            }
        }
        inside as f32 / total as f32
    }

    /// Render a `size x size x 3` RGB `u8` frame (default 96x96) into `rgb`.
    /// Colors match `gym_pusht`: White bg, LightGreen goal, LightSlateGray block, RoyalBlue agent.
    pub fn render_rgb(&self, size: usize, rgb: &mut [u8]) {
        assert_eq!(rgb.len(), size * size * 3);
        rgb.fill(255);

        let scale = ARENA / size as f32;
        let g_cos = GOAL_ANGLE.cos();
        let g_sin = GOAL_ANGLE.sin();
        let b_cos = self.block_angle.cos();
        let b_sin = self.block_angle.sin();
        let r2 = AGENT_R * AGENT_R;

        for py in 0..size {
            let wy = (py as f32 + 0.5) * scale;
            let row_off = py * size * 3;
            for px in 0..size {
                let wx = (px as f32 + 0.5) * scale;
                let off = row_off + px * 3;

                // 1. Goal T-block (LightGreen: 144, 238, 144)
                let gx = wx - GOAL_X;
                let gy = wy - GOAL_Y;
                let glx = gx * g_cos + gy * g_sin;
                let gly = -gx * g_sin + gy * g_cos;
                if Self::point_in_local_tee(glx, gly) {
                    rgb[off] = 144;
                    rgb[off + 1] = 238;
                    rgb[off + 2] = 144;
                }

                // 2. Active T-block (LightSlateGray: 119, 136, 153)
                let bx = wx - self.block_x;
                let by = wy - self.block_y;
                let blx = bx * b_cos + by * b_sin;
                let bly = -bx * b_sin + by * b_cos;
                if Self::point_in_local_tee(blx, bly) {
                    rgb[off] = 119;
                    rgb[off + 1] = 136;
                    rgb[off + 2] = 153;
                }

                // 3. Agent circle (RoyalBlue: 65, 105, 225)
                let ax = wx - self.agent_x;
                let ay = wy - self.agent_y;
                if ax * ax + ay * ay <= r2 {
                    rgb[off] = 65;
                    rgb[off + 1] = 105;
                    rgb[off + 2] = 225;
                }
            }
        }
    }
}

pub fn reset_batch(rng: &mut Rng, states: &mut [State]) {
    for s in states.iter_mut() {
        *s = State::new(
            rng.uniform_f32(50.0, 450.0),
            rng.uniform_f32(50.0, 450.0),
            rng.uniform_f32(100.0, 400.0),
            rng.uniform_f32(100.0, 400.0),
            rng.uniform_f32(-PI, PI).rem_euclid(TAU),
        );
    }
}

pub fn rollout_batch(
    horizon: usize,
    states: &mut [State],
    ax: &[f32],
    ay: &[f32],
    out_states: &mut [State],
) {
    let n = states.len();
    assert_eq!(ax.len(), horizon * n);
    assert_eq!(ay.len(), horizon * n);
    assert_eq!(out_states.len(), horizon * n);
    for t in 0..horizon {
        let off = t * n;
        step_batch_in_place(states, &ax[off..off + n], &ay[off..off + n]);
        out_states[off..off + n].copy_from_slice(states);
    }
}

#[inline]
pub fn step_batch_in_place(states: &mut [State], ax: &[f32], ay: &[f32]) {
    assert_eq!(states.len(), ax.len());
    assert_eq!(states.len(), ay.len());
    for ((s, &x), &y) in states.iter_mut().zip(ax.iter()).zip(ay.iter()) {
        s.step_in_place(x, y);
    }
}

pub fn render_rgb_batch(states: &[State], size: usize, rgb: &mut [u8]) {
    let frame_len = size * size * 3;
    assert_eq!(rgb.len(), states.len() * frame_len);
    for (s, frame) in states.iter().zip(rgb.chunks_exact_mut(frame_len)) {
        s.render_rgb(size, frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_pushes_t_block_and_rotates() {
        // Agent below the right side of the T-bar pushing upward
        let mut s = State::new(290.0, 210.0, 256.0, 256.0, 0.0);
        for _ in 0..20 {
            s.step_in_place(290.0, 300.0);
        }
        assert!(s.block_y > 256.0, "block should be pushed upward");
        assert!(
            s.block_angle != 0.0,
            "off-center push should rotate the T-block"
        );
    }

    #[test]
    fn walls_contain_block() {
        let mut s = State::new(100.0, 256.0, 40.0, 256.0, 0.0);
        for _ in 0..30 {
            s.step_in_place(0.0, 256.0);
        }
        for (kx, ky) in s.keypoints() {
            assert!((WALL_MIN - 1.0..=WALL_MAX + 1.0).contains(&kx));
            assert!((WALL_MIN - 1.0..=WALL_MAX + 1.0).contains(&ky));
        }
    }
}
