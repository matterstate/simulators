//! Zero-dependency multi-threaded batch execution using `std::thread::scope`.

use std::thread;

use crate::{pointmaze, reacher, sokoban};

pub fn step_pointmaze_parallel(
    num_threads: usize,
    x: &mut [f32],
    y: &mut [f32],
    ax: &[f32],
    ay: &[f32],
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(ax.len(), n);
    assert_eq!(ay.len(), n);

    let threads = num_threads.max(1).min(n.max(1));
    if threads == 1 || n < 256 {
        pointmaze::step_batch_in_place(x, y, ax, ay);
        return;
    }

    let chunk = n.div_ceil(threads);
    thread::scope(|s| {
        for (((xc, yc), axc), ayc) in x
            .chunks_mut(chunk)
            .zip(y.chunks_mut(chunk))
            .zip(ax.chunks(chunk))
            .zip(ay.chunks(chunk))
        {
            s.spawn(move || pointmaze::step_batch_in_place(xc, yc, axc, ayc));
        }
    });
}

pub fn render_reacher_parallel(
    num_threads: usize,
    t1: &[f32],
    t2: &[f32],
    img: usize,
    frames: &mut [f32],
) {
    let n = t1.len();
    let frame_len = img * img;
    assert_eq!(t2.len(), n);
    assert_eq!(frames.len(), n * frame_len);

    let threads = num_threads.max(1).min(n.max(1));
    if threads == 1 || n < 16 {
        reacher::render_batch(t1, t2, img, frames);
        return;
    }

    let chunk = n.div_ceil(threads);
    thread::scope(|s| {
        for ((t1c, t2c), fc) in t1
            .chunks(chunk)
            .zip(t2.chunks(chunk))
            .zip(frames.chunks_mut(chunk * frame_len))
        {
            s.spawn(move || reacher::render_batch(t1c, t2c, img, fc));
        }
    });
}

pub fn step_sokoban_parallel(
    num_threads: usize,
    boards: &mut [sokoban::Board],
    actions: &[sokoban::Action],
) {
    let n = boards.len();
    assert_eq!(actions.len(), n);

    let threads = num_threads.max(1).min(n.max(1));
    if threads == 1 || n < 256 {
        sokoban::step_batch_in_place(boards, actions);
        return;
    }

    let chunk = n.div_ceil(threads);
    thread::scope(|s| {
        for (bc, ac) in boards.chunks_mut(chunk).zip(actions.chunks(chunk)) {
            s.spawn(move || sokoban::step_batch_in_place(bc, ac));
        }
    });
}
