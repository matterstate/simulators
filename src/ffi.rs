//! C-ABI bridge (`cdylib`) for calling the Rust kernels from Python or C.

use std::slice;

use crate::{
    armpush, pointmaze, pointmaze_proc, pointpush, reacher, reacher_morph, reacher_obs, rng::Rng,
    sokoban,
};

/// # Safety
/// `grid_in` and `grid_out` must point to valid buffers of `sokoban::CELLS` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_sokoban_step(grid_in: *const u8, action: u8, grid_out: *mut u8) -> i32 {
    if grid_in.is_null() || grid_out.is_null() {
        return -1;
    }
    let Ok(act) = sokoban::Action::from_u8(action) else {
        return -1;
    };
    let in_slice = unsafe { slice::from_raw_parts(grid_in, sokoban::CELLS) };
    let mut cells = [0u8; sokoban::CELLS];
    cells.copy_from_slice(in_slice);
    let Ok(mut board) = sokoban::Board::from_cells(cells) else {
        return -1;
    };
    board.step_in_place(act);
    let out_slice = unsafe { slice::from_raw_parts_mut(grid_out, sokoban::CELLS) };
    out_slice.copy_from_slice(board.cells());
    0
}

/// # Safety
/// All pointers must be valid for `n` `f32` elements; `x` and `y` must not alias.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointmaze_step(
    n: usize,
    x: *mut f32,
    y: *mut f32,
    ax: *const f32,
    ay: *const f32,
) {
    let x = unsafe { slice::from_raw_parts_mut(x, n) };
    let y = unsafe { slice::from_raw_parts_mut(y, n) };
    let ax = unsafe { slice::from_raw_parts(ax, n) };
    let ay = unsafe { slice::from_raw_parts(ay, n) };
    pointmaze::step_batch_in_place(x, y, ax, ay);
}

/// # Safety
/// `x`, `y`, `ax`, `ay` must be valid for `n` `f32`s; `walls` for `n * 36` `bool`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointmaze_proc_step(
    n: usize,
    x: *mut f32,
    y: *mut f32,
    ax: *const f32,
    ay: *const f32,
    walls: *const [bool; pointmaze_proc::WALL_CELLS],
) {
    let x = unsafe { slice::from_raw_parts_mut(x, n) };
    let y = unsafe { slice::from_raw_parts_mut(y, n) };
    let ax = unsafe { slice::from_raw_parts(ax, n) };
    let ay = unsafe { slice::from_raw_parts(ay, n) };
    let walls = unsafe { slice::from_raw_parts(walls, n) };
    pointmaze_proc::step_batch_in_place(x, y, ax, ay, walls);
}

/// # Safety
/// `x`, `y` valid for `n` `f32`s; `walls` for `n` grids; `frames` for `n * 48 * 48` `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointmaze_proc_render(
    n: usize,
    x: *const f32,
    y: *const f32,
    walls: *const [bool; pointmaze_proc::WALL_CELLS],
    r_px: f32,
    frames: *mut f32,
) {
    let x = unsafe { slice::from_raw_parts(x, n) };
    let y = unsafe { slice::from_raw_parts(y, n) };
    let walls = unsafe { slice::from_raw_parts(walls, n) };
    let frames =
        unsafe { slice::from_raw_parts_mut(frames, n * crate::render::IMG * crate::render::IMG) };
    pointmaze_proc::render_batch(x, y, walls, r_px, frames);
}

/// # Safety
/// All pointers must be valid for `n` `f32` elements and non-aliasing for mutable slices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointpush_step(
    n: usize,
    ax: *mut f32,
    ay: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    act_x: *const f32,
    act_y: *const f32,
) {
    let ax = unsafe { slice::from_raw_parts_mut(ax, n) };
    let ay = unsafe { slice::from_raw_parts_mut(ay, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let act_x = unsafe { slice::from_raw_parts(act_x, n) };
    let act_y = unsafe { slice::from_raw_parts(act_y, n) };
    pointpush::step_batch_in_place(ax, ay, ox, oy, act_x, act_y);
}

/// # Safety
/// `ax`, `ay`, `ox`, `oy` valid for `n` `f32`s; `frames` for `n * img * img` `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointpush_render(
    n: usize,
    ax: *const f32,
    ay: *const f32,
    ox: *const f32,
    oy: *const f32,
    img: usize,
    frames: *mut f32,
) {
    let ax = unsafe { slice::from_raw_parts(ax, n) };
    let ay = unsafe { slice::from_raw_parts(ay, n) };
    let ox = unsafe { slice::from_raw_parts(ox, n) };
    let oy = unsafe { slice::from_raw_parts(oy, n) };
    let frames = unsafe { slice::from_raw_parts_mut(frames, n * img * img) };
    pointpush::render_batch(ax, ay, ox, oy, img, frames);
}

/// # Safety
/// All pointers must be valid for `n` `f32` elements and non-aliasing for mutable slices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_step(
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    a1: *const f32,
    a2: *const f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, n) };
    let a2 = unsafe { slice::from_raw_parts(a2, n) };
    reacher::step_batch_in_place(t1, t2, a1, a2);
}

/// # Safety
/// `t1`, `t2` valid for `n` `f32`s; `frames` for `n * img * img` `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_render(
    n: usize,
    t1: *const f32,
    t2: *const f32,
    img: usize,
    frames: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts(t1, n) };
    let t2 = unsafe { slice::from_raw_parts(t2, n) };
    let frames = unsafe { slice::from_raw_parts_mut(frames, n * img * img) };
    reacher::render_batch(t1, t2, img, frames);
}

/// # Safety
/// All pointers must be valid for `n` `f32` elements and non-aliasing for mutable slices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_morph_step(
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    a1: *const f32,
    a2: *const f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, n) };
    let a2 = unsafe { slice::from_raw_parts(a2, n) };
    reacher_morph::step_batch_in_place(t1, t2, a1, a2);
}

/// # Safety
/// `t1`, `t2`, `l1`, `l2` valid for `n` `f32`s; `frames` for `n * img * img` `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_morph_render(
    n: usize,
    t1: *const f32,
    t2: *const f32,
    l1: *const f32,
    l2: *const f32,
    img: usize,
    frames: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts(t1, n) };
    let t2 = unsafe { slice::from_raw_parts(t2, n) };
    let l1 = unsafe { slice::from_raw_parts(l1, n) };
    let l2 = unsafe { slice::from_raw_parts(l2, n) };
    let frames = unsafe { slice::from_raw_parts_mut(frames, n * img * img) };
    reacher_morph::render_batch(t1, t2, l1, l2, img, frames);
}

/// # Safety
/// All pointers must be valid for `n` `f32` elements and non-aliasing for mutable slices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_obs_step(
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    a1: *const f32,
    a2: *const f32,
    ox: *const f32,
    oy: *const f32,
    or: *const f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, n) };
    let a2 = unsafe { slice::from_raw_parts(a2, n) };
    let ox = unsafe { slice::from_raw_parts(ox, n) };
    let oy = unsafe { slice::from_raw_parts(oy, n) };
    let or = unsafe { slice::from_raw_parts(or, n) };
    reacher_obs::step_batch_in_place(t1, t2, a1, a2, ox, oy, or);
}

/// # Safety
/// `t1`, `t2`, `ox`, `oy`, `or` valid for `n` `f32`s; `frames` for `n * img * img` `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_obs_render(
    n: usize,
    t1: *const f32,
    t2: *const f32,
    ox: *const f32,
    oy: *const f32,
    or: *const f32,
    img: usize,
    frames: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts(t1, n) };
    let t2 = unsafe { slice::from_raw_parts(t2, n) };
    let ox = unsafe { slice::from_raw_parts(ox, n) };
    let oy = unsafe { slice::from_raw_parts(oy, n) };
    let or = unsafe { slice::from_raw_parts(or, n) };
    let frames = unsafe { slice::from_raw_parts_mut(frames, n * img * img) };
    reacher_obs::render_batch(t1, t2, ox, oy, or, img, frames);
}

/// # Safety
/// All pointers must be valid for `n` `f32` elements and non-aliasing for mutable slices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_armpush_step(
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    a1: *const f32,
    a2: *const f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, n) };
    let a2 = unsafe { slice::from_raw_parts(a2, n) };
    armpush::step_batch_in_place(t1, t2, ox, oy, a1, a2);
}

/// # Safety
/// `t1`, `t2`, `ox`, `oy` valid for `n` `f32`s; `frames` for `n * img * img` `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_armpush_render(
    n: usize,
    t1: *const f32,
    t2: *const f32,
    ox: *const f32,
    oy: *const f32,
    img: usize,
    frames: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts(t1, n) };
    let t2 = unsafe { slice::from_raw_parts(t2, n) };
    let ox = unsafe { slice::from_raw_parts(ox, n) };
    let oy = unsafe { slice::from_raw_parts(oy, n) };
    let frames = unsafe { slice::from_raw_parts_mut(frames, n * img * img) };
    armpush::render_batch(t1, t2, ox, oy, img, frames);
}

/// # Safety
/// All output pointers must be valid for `n` `f32` elements and non-aliasing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointmaze_reset(
    seed: u64,
    n: usize,
    x: *mut f32,
    y: *mut f32,
    gx: *mut f32,
    gy: *mut f32,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let x = unsafe { slice::from_raw_parts_mut(x, n) };
    let y = unsafe { slice::from_raw_parts_mut(y, n) };
    let gx = unsafe { slice::from_raw_parts_mut(gx, n) };
    let gy = unsafe { slice::from_raw_parts_mut(gy, n) };
    pointmaze::reset_batch(&mut rng, x, y, gx, gy);
}

/// # Safety
/// `x`, `y` valid for `n`; `ax`, `ay`, `traj_x`, `traj_y` valid for `horizon * n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointmaze_rollout(
    horizon: usize,
    n: usize,
    x: *mut f32,
    y: *mut f32,
    ax: *const f32,
    ay: *const f32,
    traj_x: *mut f32,
    traj_y: *mut f32,
) {
    let x = unsafe { slice::from_raw_parts_mut(x, n) };
    let y = unsafe { slice::from_raw_parts_mut(y, n) };
    let ax = unsafe { slice::from_raw_parts(ax, horizon * n) };
    let ay = unsafe { slice::from_raw_parts(ay, horizon * n) };
    let traj_x = unsafe { slice::from_raw_parts_mut(traj_x, horizon * n) };
    let traj_y = unsafe { slice::from_raw_parts_mut(traj_y, horizon * n) };
    pointmaze::rollout_batch(horizon, x, y, ax, ay, traj_x, traj_y);
}

/// # Safety
/// All output pointers must be valid for `n` `f32` elements and non-aliasing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_reset(
    seed: u64,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    gx: *mut f32,
    gy: *mut f32,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let gx = unsafe { slice::from_raw_parts_mut(gx, n) };
    let gy = unsafe { slice::from_raw_parts_mut(gy, n) };
    reacher::reset_batch(&mut rng, t1, t2, gx, gy);
}

/// # Safety
/// `t1`, `t2` valid for `n`; `a1`, `a2`, `traj_t1`, `traj_t2` valid for `horizon * n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_rollout(
    horizon: usize,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    a1: *const f32,
    a2: *const f32,
    traj_t1: *mut f32,
    traj_t2: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, horizon * n) };
    let a2 = unsafe { slice::from_raw_parts(a2, horizon * n) };
    let traj_t1 = unsafe { slice::from_raw_parts_mut(traj_t1, horizon * n) };
    let traj_t2 = unsafe { slice::from_raw_parts_mut(traj_t2, horizon * n) };
    reacher::rollout_batch(horizon, t1, t2, a1, a2, traj_t1, traj_t2);
}

/// # Safety
/// All output pointers must be valid for `n` `f32` elements and non-aliasing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_pointpush_reset(
    seed: u64,
    n: usize,
    ax: *mut f32,
    ay: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    gx: *mut f32,
    gy: *mut f32,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let ax = unsafe { slice::from_raw_parts_mut(ax, n) };
    let ay = unsafe { slice::from_raw_parts_mut(ay, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let gx = unsafe { slice::from_raw_parts_mut(gx, n) };
    let gy = unsafe { slice::from_raw_parts_mut(gy, n) };
    pointpush::reset_batch(&mut rng, ax, ay, ox, oy, gx, gy);
}

/// # Safety
/// State pointers valid for `n`; action and trajectory pointers valid for `horizon * n`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ms_pointpush_rollout(
    horizon: usize,
    n: usize,
    ax: *mut f32,
    ay: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    act_x: *const f32,
    act_y: *const f32,
    traj_ax: *mut f32,
    traj_ay: *mut f32,
    traj_ox: *mut f32,
    traj_oy: *mut f32,
) {
    let ax = unsafe { slice::from_raw_parts_mut(ax, n) };
    let ay = unsafe { slice::from_raw_parts_mut(ay, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let act_x = unsafe { slice::from_raw_parts(act_x, horizon * n) };
    let act_y = unsafe { slice::from_raw_parts(act_y, horizon * n) };
    let traj_ax = unsafe { slice::from_raw_parts_mut(traj_ax, horizon * n) };
    let traj_ay = unsafe { slice::from_raw_parts_mut(traj_ay, horizon * n) };
    let traj_ox = unsafe { slice::from_raw_parts_mut(traj_ox, horizon * n) };
    let traj_oy = unsafe { slice::from_raw_parts_mut(traj_oy, horizon * n) };
    pointpush::rollout_batch(
        horizon, ax, ay, ox, oy, act_x, act_y, traj_ax, traj_ay, traj_ox, traj_oy,
    );
}

/// # Safety
/// All output pointers must be valid for `n` `f32` elements and non-aliasing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_armpush_reset(
    seed: u64,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    gx: *mut f32,
    gy: *mut f32,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let gx = unsafe { slice::from_raw_parts_mut(gx, n) };
    let gy = unsafe { slice::from_raw_parts_mut(gy, n) };
    armpush::reset_batch(&mut rng, t1, t2, ox, oy, gx, gy);
}

/// # Safety
/// State pointers valid for `n`; action and trajectory pointers valid for `horizon * n`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ms_armpush_rollout(
    horizon: usize,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    a1: *const f32,
    a2: *const f32,
    traj_t1: *mut f32,
    traj_t2: *mut f32,
    traj_ox: *mut f32,
    traj_oy: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, horizon * n) };
    let a2 = unsafe { slice::from_raw_parts(a2, horizon * n) };
    let traj_t1 = unsafe { slice::from_raw_parts_mut(traj_t1, horizon * n) };
    let traj_t2 = unsafe { slice::from_raw_parts_mut(traj_t2, horizon * n) };
    let traj_ox = unsafe { slice::from_raw_parts_mut(traj_ox, horizon * n) };
    let traj_oy = unsafe { slice::from_raw_parts_mut(traj_oy, horizon * n) };
    armpush::rollout_batch(
        horizon, t1, t2, ox, oy, a1, a2, traj_t1, traj_t2, traj_ox, traj_oy,
    );
}

/// # Safety
/// All output pointers must be valid for `n` `f32` elements and non-aliasing.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ms_reacher_obs_reset(
    seed: u64,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    gx: *mut f32,
    gy: *mut f32,
    ox: *mut f32,
    oy: *mut f32,
    or: *mut f32,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let gx = unsafe { slice::from_raw_parts_mut(gx, n) };
    let gy = unsafe { slice::from_raw_parts_mut(gy, n) };
    let ox = unsafe { slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { slice::from_raw_parts_mut(oy, n) };
    let or = unsafe { slice::from_raw_parts_mut(or, n) };
    reacher_obs::reset_batch(&mut rng, t1, t2, gx, gy, ox, oy, or);
}

/// # Safety
/// State/obs pointers valid for `n`; action and trajectory pointers valid for `horizon * n`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ms_reacher_obs_rollout(
    horizon: usize,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    a1: *const f32,
    a2: *const f32,
    ox: *const f32,
    oy: *const f32,
    or: *const f32,
    traj_t1: *mut f32,
    traj_t2: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, horizon * n) };
    let a2 = unsafe { slice::from_raw_parts(a2, horizon * n) };
    let ox = unsafe { slice::from_raw_parts(ox, n) };
    let oy = unsafe { slice::from_raw_parts(oy, n) };
    let or = unsafe { slice::from_raw_parts(or, n) };
    let traj_t1 = unsafe { slice::from_raw_parts_mut(traj_t1, horizon * n) };
    let traj_t2 = unsafe { slice::from_raw_parts_mut(traj_t2, horizon * n) };
    reacher_obs::rollout_batch(horizon, t1, t2, a1, a2, ox, oy, or, traj_t1, traj_t2);
}

/// # Safety
/// All output pointers must be valid for `n` `f32` elements and non-aliasing.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ms_reacher_morph_reset(
    seed: u64,
    lo: f32,
    hi: f32,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    l1: *mut f32,
    l2: *mut f32,
    gx: *mut f32,
    gy: *mut f32,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let l1 = unsafe { slice::from_raw_parts_mut(l1, n) };
    let l2 = unsafe { slice::from_raw_parts_mut(l2, n) };
    let gx = unsafe { slice::from_raw_parts_mut(gx, n) };
    let gy = unsafe { slice::from_raw_parts_mut(gy, n) };
    reacher_morph::reset_batch(&mut rng, lo, hi, t1, t2, l1, l2, gx, gy);
}

/// # Safety
/// `t1`, `t2` valid for `n`; `a1`, `a2`, `traj_t1`, `traj_t2` valid for `horizon * n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_reacher_morph_rollout(
    horizon: usize,
    n: usize,
    t1: *mut f32,
    t2: *mut f32,
    a1: *const f32,
    a2: *const f32,
    traj_t1: *mut f32,
    traj_t2: *mut f32,
) {
    let t1 = unsafe { slice::from_raw_parts_mut(t1, n) };
    let t2 = unsafe { slice::from_raw_parts_mut(t2, n) };
    let a1 = unsafe { slice::from_raw_parts(a1, horizon * n) };
    let a2 = unsafe { slice::from_raw_parts(a2, horizon * n) };
    let traj_t1 = unsafe { slice::from_raw_parts_mut(traj_t1, horizon * n) };
    let traj_t2 = unsafe { slice::from_raw_parts_mut(traj_t2, horizon * n) };
    reacher_morph::rollout_batch(horizon, t1, t2, a1, a2, traj_t1, traj_t2);
}
