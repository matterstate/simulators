use std::hint::black_box;
use std::time::Instant;

use matterstate_envs::{
    armpush, pointmaze, pointmaze_proc, pointpush, reacher, reacher_morph, reacher_obs, render,
};

fn bench<F: FnMut()>(name: &str, op: &str, batch: usize, iters: usize, mut f: F) {
    for _ in 0..50 {
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
        "{name:<15} {op:<7} {batch:>5} {:>14.0} {:>12.2}",
        total / secs,
        elapsed.as_nanos() as f64 / total,
    );
}

fn main() {
    let batch = 64usize;
    let step_iters = 200_000usize;
    let render_iters = 10_000usize;
    let img = render::IMG;

    let mut frames = vec![0.0_f32; batch * img * img];
    let ax: Vec<f32> = (0..batch).map(|i| ((i as f32) * 0.13).sin()).collect();
    let ay: Vec<f32> = (0..batch).map(|i| ((i as f32) * 0.17).cos()).collect();

    println!(
        "{:<15} {:<7} {:>5} {:>14} {:>12}",
        "kernel", "op", "batch", "items/s", "ns/item"
    );
    println!("{}", "-".repeat(57));

    // pointmaze
    let mut pm_x = vec![0.2_f32; batch];
    let mut pm_y = vec![0.2_f32; batch];
    bench("pointmaze", "step", batch, step_iters, || {
        pointmaze::step_batch_in_place(&mut pm_x, &mut pm_y, &ax, &ay);
        black_box((&pm_x, &pm_y));
    });

    // pointmaze_proc
    let mut pmp_x = vec![0.2_f32; batch];
    let mut pmp_y = vec![0.2_f32; batch];
    let pmp_walls = vec![[false; pointmaze_proc::WALL_CELLS]; batch];
    bench("pointmaze_proc", "step", batch, step_iters, || {
        pointmaze_proc::step_batch_in_place(&mut pmp_x, &mut pmp_y, &ax, &ay, &pmp_walls);
        black_box((&pmp_x, &pmp_y));
    });
    bench("pointmaze_proc", "render", batch, render_iters, || {
        pointmaze_proc::render_batch(&pmp_x, &pmp_y, &pmp_walls, 1.5, &mut frames);
        black_box(&frames);
    });

    // pointpush
    let mut pp_ax = vec![0.3_f32; batch];
    let mut pp_ay = vec![0.3_f32; batch];
    let mut pp_ox = vec![0.5_f32; batch];
    let mut pp_oy = vec![0.5_f32; batch];
    bench("pointpush", "step", batch, step_iters, || {
        pointpush::step_batch_in_place(&mut pp_ax, &mut pp_ay, &mut pp_ox, &mut pp_oy, &ax, &ay);
        black_box((&pp_ax, &pp_ox));
    });
    bench("pointpush", "render", batch, render_iters, || {
        pointpush::render_batch(&pp_ax, &pp_ay, &pp_ox, &pp_oy, img, &mut frames);
        black_box(&frames);
    });

    // reacher
    let mut r_t1 = vec![0.4_f32; batch];
    let mut r_t2 = vec![-0.6_f32; batch];
    bench("reacher", "step", batch, step_iters, || {
        reacher::step_batch_in_place(&mut r_t1, &mut r_t2, &ax, &ay);
        black_box((&r_t1, &r_t2));
    });
    bench("reacher", "render", batch, render_iters, || {
        reacher::render_batch(&r_t1, &r_t2, img, &mut frames);
        black_box(&frames);
    });

    // reacher_morph
    let mut rm_t1 = vec![0.4_f32; batch];
    let mut rm_t2 = vec![-0.6_f32; batch];
    let rm_l1 = vec![0.21_f32; batch];
    let rm_l2 = vec![0.19_f32; batch];
    bench("reacher_morph", "step", batch, step_iters, || {
        reacher_morph::step_batch_in_place(&mut rm_t1, &mut rm_t2, &ax, &ay);
        black_box((&rm_t1, &rm_t2));
    });
    bench("reacher_morph", "render", batch, render_iters, || {
        reacher_morph::render_batch(&rm_t1, &rm_t2, &rm_l1, &rm_l2, img, &mut frames);
        black_box(&frames);
    });

    // reacher_obs
    let mut ro_t1 = vec![0.4_f32; batch];
    let mut ro_t2 = vec![-0.6_f32; batch];
    let ro_ox = vec![0.6_f32; batch];
    let ro_oy = vec![0.6_f32; batch];
    let ro_or = vec![reacher_obs::OBS_R; batch];
    bench("reacher_obs", "step", batch, step_iters, || {
        reacher_obs::step_batch_in_place(&mut ro_t1, &mut ro_t2, &ax, &ay, &ro_ox, &ro_oy, &ro_or);
        black_box((&ro_t1, &ro_t2));
    });
    bench("reacher_obs", "render", batch, render_iters, || {
        reacher_obs::render_batch(&ro_t1, &ro_t2, &ro_ox, &ro_oy, &ro_or, img, &mut frames);
        black_box(&frames);
    });

    // armpush
    let mut ap_t1 = vec![0.4_f32; batch];
    let mut ap_t2 = vec![-0.6_f32; batch];
    let mut ap_ox = vec![0.55_f32; batch];
    let mut ap_oy = vec![0.55_f32; batch];
    bench("armpush", "step", batch, step_iters, || {
        armpush::step_batch_in_place(&mut ap_t1, &mut ap_t2, &mut ap_ox, &mut ap_oy, &ax, &ay);
        black_box((&ap_t1, &ap_ox));
    });
    bench("armpush", "render", batch, render_iters, || {
        armpush::render_batch(&ap_t1, &ap_t2, &ap_ox, &ap_oy, img, &mut frames);
        black_box(&frames);
    });
}
