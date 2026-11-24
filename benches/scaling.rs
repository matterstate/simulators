use std::hint::black_box;
use std::thread;
use std::time::Instant;

use matterstate_envs::parallel::{
    render_reacher_parallel, step_pointmaze_parallel, step_sokoban_parallel,
};
use matterstate_envs::render::IMG;
use matterstate_envs::sokoban::{
    Action, BOX, Board, CELLS, FLOOR, HEIGHT, PLR, TARGET, WALL, WIDTH,
};

fn make_sokoban_board() -> Board {
    let mut cells = [FLOOR; CELLS];
    for c in 0..WIDTH {
        cells[c] = WALL;
        cells[(HEIGHT - 1) * WIDTH + c] = WALL;
    }
    for r in 0..HEIGHT {
        cells[r * WIDTH] = WALL;
        cells[r * WIDTH + (WIDTH - 1)] = WALL;
    }
    cells[4 * WIDTH + 3] = PLR;
    cells[4 * WIDTH + 4] = BOX;
    cells[4 * WIDTH + 5] = TARGET;
    Board::from_cells(cells).expect("valid board")
}

fn bench<F: FnMut()>(name: &str, batch: usize, threads: usize, iters: usize, mut f: F) {
    for _ in 0..10 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let total = (batch * iters) as f64;
    let secs = elapsed.as_secs_f64();
    println!(
        "{name:<20} {batch:>6} {threads:>7} {:>15.0} {:>10.2}",
        total / secs,
        elapsed.as_nanos() as f64 / total,
    );
}

fn main() {
    let max_threads = thread::available_parallelism().map_or(1, |p| p.get());
    let thread_counts: Vec<usize> = [1, 4, 8, max_threads]
        .into_iter()
        .filter(|&t| t <= max_threads)
        .collect();

    println!(
        "{:<20} {:>6} {:>7} {:>15} {:>10}",
        "kernel", "batch", "threads", "items/s", "ns/item"
    );
    println!("{}", "-".repeat(62));

    for &batch in &[1024usize, 4096, 16384] {
        let mut x = vec![0.2_f32; batch];
        let mut y = vec![0.2_f32; batch];
        let ax: Vec<f32> = (0..batch).map(|i| ((i as f32) * 0.13).sin()).collect();
        let ay: Vec<f32> = (0..batch).map(|i| ((i as f32) * 0.17).cos()).collect();
        let step_iters = (2_000_000 / batch).max(50);

        for &threads in &thread_counts {
            bench("pointmaze.step", batch, threads, step_iters, || {
                step_pointmaze_parallel(threads, &mut x, &mut y, &ax, &ay);
                black_box((&x, &y));
            });
        }

        let t1 = vec![0.4_f32; batch];
        let t2 = vec![-0.6_f32; batch];
        let mut frames = vec![0.0_f32; batch * IMG * IMG];
        let render_iters = (200_000 / batch).max(10);
        for &threads in &thread_counts {
            bench("reacher.render", batch, threads, render_iters, || {
                render_reacher_parallel(threads, &t1, &t2, IMG, &mut frames);
                black_box(&frames);
            });
        }

        let board = make_sokoban_board();
        let mut boards = vec![board; batch];
        let actions: Vec<Action> = (0..batch).map(|i| Action::ALL[i & 3]).collect();
        for &threads in &thread_counts {
            bench("sokoban.step", batch, threads, step_iters, || {
                step_sokoban_parallel(threads, &mut boards, &actions);
                black_box(&boards);
            });
        }
    }
}
