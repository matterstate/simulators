//! Fast 2D rasterization primitives matching `matterstate_envs.reacher`.

pub const IMG: usize = 48;

/// Paint a filled disc of radius `r` (normalized `[0, 1]`) centered at `(cx, cy)`.
#[inline]
pub fn paint_blob(frame: &mut [f32], img: usize, cx: f32, cy: f32, r: f32, val: f32) {
    debug_assert_eq!(frame.len(), img * img);
    let img_f = img as f32;
    let px = cx * img_f;
    let py = cy * img_f;
    let r_px = r * img_f;
    let r2 = r_px * r_px;

    let x0 = (px - r_px).floor().max(0.0) as usize;
    let x1 = ((px + r_px).ceil() as isize + 1).clamp(0, img as isize) as usize;
    let y0 = (py - r_px).floor().max(0.0) as usize;
    let y1 = ((py + r_px).ceil() as isize + 1).clamp(0, img as isize) as usize;

    for y in y0..y1 {
        let dy = y as f32 - py;
        let dy2 = dy * dy;
        let row = &mut frame[y * img..(y + 1) * img];
        for (x, pixel) in row.iter_mut().enumerate().take(x1).skip(x0) {
            let dx = x as f32 - px;
            if dx * dx + dy2 <= r2 {
                *pixel = val;
            }
        }
    }
}

/// Paint pixels within normalized distance `w` of segment `(x0, y0) -> (x1, y1)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn paint_seg(
    frame: &mut [f32],
    img: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    w: f32,
    val: f32,
) {
    debug_assert_eq!(frame.len(), img * img);
    let img_f = img as f32;
    let inv_img = 1.0 / img_f;
    let sx = x1 - x0;
    let sy = y1 - y0;
    let inv_seg2 = 1.0 / (sx * sx + sy * sy + 1e-9);
    let w2 = w * w;

    let min_x = ((x0.min(x1) - w) * img_f).floor().max(0.0) as usize;
    let max_x = (((x0.max(x1) + w) * img_f).ceil() as isize + 1).clamp(0, img as isize) as usize;
    let min_y = ((y0.min(y1) - w) * img_f).floor().max(0.0) as usize;
    let max_y = (((y0.max(y1) + w) * img_f).ceil() as isize + 1).clamp(0, img as isize) as usize;

    for py in min_y..max_y {
        let fy = py as f32 * inv_img;
        let dy0 = fy - y0;
        let row = &mut frame[py * img..(py + 1) * img];
        for (px, pixel) in row.iter_mut().enumerate().take(max_x).skip(min_x) {
            let fx = px as f32 * inv_img;
            let dx0 = fx - x0;
            let t = ((dx0 * sx + dy0 * sy) * inv_seg2).clamp(0.0, 1.0);
            let rx = fx - (x0 + t * sx);
            let ry = fy - (y0 + t * sy);
            if rx * rx + ry * ry <= w2 {
                *pixel = val;
            }
        }
    }
}
