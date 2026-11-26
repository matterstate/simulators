//! Zero-dependency PyO3 bindings (`matterstate_envs._rs`).

#![allow(clippy::type_complexity)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rng::Rng;
use crate::{
    armpush, ogbench_maze, pointmaze, pointmaze_proc, pointpush, pusht, reacher, reacher_morph,
    reacher_obs, render, sokoban, sudoku,
};

fn copy_wall_grids(ws: &[bool]) -> Vec<[bool; pointmaze_proc::WALL_CELLS]> {
    ws.chunks_exact(pointmaze_proc::WALL_CELLS)
        .map(|c| {
            let mut grid = [false; pointmaze_proc::WALL_CELLS];
            grid.copy_from_slice(c);
            grid
        })
        .collect()
}

fn parse_pusht_state(s: &[f32]) -> pusht::State {
    let mut st = pusht::State::new(s[0], s[1], s[4], s[5], s[6]);
    st.agent_vx = s[2];
    st.agent_vy = s[3];
    st
}

fn parse_pusht_states(states: &[f32]) -> PyResult<Vec<pusht::State>> {
    if !states.len().is_multiple_of(7) {
        return Err(PyValueError::new_err("expected flat states of length B*7"));
    }
    Ok(states.chunks_exact(7).map(parse_pusht_state).collect())
}

fn flatten_pusht_states(states: &[pusht::State]) -> Vec<f32> {
    let mut out = Vec::with_capacity(states.len() * 7);
    for s in states {
        out.extend_from_slice(&[
            s.agent_x,
            s.agent_y,
            s.agent_vx,
            s.agent_vy,
            s.block_x,
            s.block_y,
            s.block_angle,
        ]);
    }
    out
}

#[pyfunction]
fn sokoban_step(grid: Vec<i8>, action: u8) -> PyResult<Vec<i8>> {
    if grid.len() != sokoban::CELLS {
        return Err(PyValueError::new_err("expected 100-element Sokoban grid"));
    }
    let act = sokoban::Action::from_u8(action)
        .map_err(|_| PyValueError::new_err("invalid Sokoban action"))?;
    let mut cells = [0u8; sokoban::CELLS];
    for (dst, &src) in cells.iter_mut().zip(grid.iter()) {
        *dst = src as u8;
    }
    let board =
        sokoban::Board::from_cells(cells).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let next = board.step(act);
    Ok(next.cells().iter().map(|&c| c as i8).collect())
}

#[pyfunction]
fn sokoban_solved(grid: Vec<i8>) -> PyResult<bool> {
    if grid.len() != sokoban::CELLS {
        return Err(PyValueError::new_err("expected 100-element Sokoban grid"));
    }
    let mut cells = [0u8; sokoban::CELLS];
    for (dst, &src) in cells.iter_mut().zip(grid.iter()) {
        *dst = src as u8;
    }
    let board =
        sokoban::Board::from_cells(cells).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(board.solved())
}

#[pyfunction]
fn pointmaze_reset(seed: u64, batch: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::seed_from_u64(seed);
    let mut x = vec![0.0_f32; batch];
    let mut y = vec![0.0_f32; batch];
    let mut gx = vec![0.0_f32; batch];
    let mut gy = vec![0.0_f32; batch];
    pointmaze::reset_batch(&mut rng, &mut x, &mut y, &mut gx, &mut gy);
    (x, y, gx, gy)
}

#[pyfunction]
fn pointmaze_in_wall(x: Vec<f32>, y: Vec<f32>) -> PyResult<Vec<bool>> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    Ok(x.iter()
        .zip(y.iter())
        .map(|(&px, &py)| pointmaze::in_wall(px, py))
        .collect())
}

#[pyfunction]
fn pointmaze_step(
    mut x: Vec<f32>,
    mut y: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = x.len();
    if y.len() != n || ax.len() != n || ay.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    pointmaze::step_batch_in_place(&mut x, &mut y, &ax, &ay);
    Ok((x, y))
}

#[pyfunction]
fn pointmaze_rollout(
    horizon: usize,
    mut x: Vec<f32>,
    mut y: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = x.len();
    if y.len() != n || ax.len() != horizon * n || ay.len() != horizon * n {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut tx = vec![0.0_f32; horizon * n];
    let mut ty = vec![0.0_f32; horizon * n];
    pointmaze::rollout_batch(horizon, &mut x, &mut y, &ax, &ay, &mut tx, &mut ty);
    Ok((tx, ty))
}

#[pyfunction]
fn pointmaze_proc_step(
    mut x: Vec<f32>,
    mut y: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
    walls: Vec<bool>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = x.len();
    if y.len() != n
        || ax.len() != n
        || ay.len() != n
        || walls.len() != n * pointmaze_proc::WALL_CELLS
    {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let wall_grids = copy_wall_grids(&walls);
    pointmaze_proc::step_batch_in_place(&mut x, &mut y, &ax, &ay, &wall_grids);
    Ok((x, y))
}

#[pyfunction]
fn pointmaze_proc_render(
    x: Vec<f32>,
    y: Vec<f32>,
    walls: Vec<bool>,
    r_px: f32,
) -> PyResult<Vec<f32>> {
    let n = x.len();
    if y.len() != n || walls.len() != n * pointmaze_proc::WALL_CELLS {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let wall_grids = copy_wall_grids(&walls);
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    pointmaze_proc::render_batch(&x, &y, &wall_grids, r_px, &mut frames);
    Ok(frames)
}

#[pyfunction]
fn pointpush_reset(
    seed: u64,
    batch: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::seed_from_u64(seed);
    let mut ax = vec![0.0_f32; batch];
    let mut ay = vec![0.0_f32; batch];
    let mut ox = vec![0.0_f32; batch];
    let mut oy = vec![0.0_f32; batch];
    let mut gx = vec![0.0_f32; batch];
    let mut gy = vec![0.0_f32; batch];
    pointpush::reset_batch(
        &mut rng, &mut ax, &mut ay, &mut ox, &mut oy, &mut gx, &mut gy,
    );
    (ax, ay, ox, oy, gx, gy)
}

#[pyfunction]
fn pointpush_step(
    mut ax: Vec<f32>,
    mut ay: Vec<f32>,
    mut ox: Vec<f32>,
    mut oy: Vec<f32>,
    ux: Vec<f32>,
    uy: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = ax.len();
    if ay.len() != n || ox.len() != n || oy.len() != n || ux.len() != n || uy.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    pointpush::step_batch_in_place(&mut ax, &mut ay, &mut ox, &mut oy, &ux, &uy);
    Ok((ax, ay, ox, oy))
}

#[pyfunction]
fn pointpush_rollout(
    horizon: usize,
    mut ax: Vec<f32>,
    mut ay: Vec<f32>,
    mut ox: Vec<f32>,
    mut oy: Vec<f32>,
    ux: Vec<f32>,
    uy: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = ax.len();
    if ay.len() != n
        || ox.len() != n
        || oy.len() != n
        || ux.len() != horizon * n
        || uy.len() != horizon * n
    {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut tax = vec![0.0_f32; horizon * n];
    let mut tay = vec![0.0_f32; horizon * n];
    let mut tox = vec![0.0_f32; horizon * n];
    let mut toy = vec![0.0_f32; horizon * n];
    pointpush::rollout_batch(
        horizon, &mut ax, &mut ay, &mut ox, &mut oy, &ux, &uy, &mut tax, &mut tay, &mut tox,
        &mut toy,
    );
    Ok((tax, tay, tox, toy))
}

#[pyfunction]
fn pointpush_render(ax: Vec<f32>, ay: Vec<f32>, ox: Vec<f32>, oy: Vec<f32>) -> PyResult<Vec<f32>> {
    let n = ax.len();
    if ay.len() != n || ox.len() != n || oy.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    pointpush::render_batch(&ax, &ay, &ox, &oy, render::IMG, &mut frames);
    Ok(frames)
}

#[pyfunction]
fn reacher_reset(seed: u64, batch: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::seed_from_u64(seed);
    let mut t1 = vec![0.0_f32; batch];
    let mut t2 = vec![0.0_f32; batch];
    let mut gx = vec![0.0_f32; batch];
    let mut gy = vec![0.0_f32; batch];
    reacher::reset_batch(&mut rng, &mut t1, &mut t2, &mut gx, &mut gy);
    (t1, t2, gx, gy)
}

#[pyfunction]
fn reacher_step(
    mut t1: Vec<f32>,
    mut t2: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = t1.len();
    if t2.len() != n || a1.len() != n || a2.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    reacher::step_batch_in_place(&mut t1, &mut t2, &a1, &a2);
    Ok((t1, t2))
}

#[pyfunction]
fn reacher_rollout(
    horizon: usize,
    mut t1: Vec<f32>,
    mut t2: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = t1.len();
    if t2.len() != n || a1.len() != horizon * n || a2.len() != horizon * n {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut tt1 = vec![0.0_f32; horizon * n];
    let mut tt2 = vec![0.0_f32; horizon * n];
    reacher::rollout_batch(horizon, &mut t1, &mut t2, &a1, &a2, &mut tt1, &mut tt2);
    Ok((tt1, tt2))
}

#[pyfunction]
fn reacher_render(t1: Vec<f32>, t2: Vec<f32>) -> PyResult<Vec<f32>> {
    let n = t1.len();
    if t2.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    reacher::render_batch(&t1, &t2, render::IMG, &mut frames);
    Ok(frames)
}

#[pyfunction]
fn reacher_morph_render(
    t1: Vec<f32>,
    t2: Vec<f32>,
    l1: Vec<f32>,
    l2: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let n = t1.len();
    if t2.len() != n || l1.len() != n || l2.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    reacher_morph::render_batch(&t1, &t2, &l1, &l2, render::IMG, &mut frames);
    Ok(frames)
}

#[pyfunction]
fn reacher_obs_step(
    mut t1: Vec<f32>,
    mut t2: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
    ox: Vec<f32>,
    oy: Vec<f32>,
    or_: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = t1.len();
    if t2.len() != n
        || a1.len() != n
        || a2.len() != n
        || ox.len() != n
        || oy.len() != n
        || or_.len() != n
    {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    reacher_obs::step_batch_in_place(&mut t1, &mut t2, &a1, &a2, &ox, &oy, &or_);
    Ok((t1, t2))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn reacher_obs_rollout(
    horizon: usize,
    mut t1: Vec<f32>,
    mut t2: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
    ox: Vec<f32>,
    oy: Vec<f32>,
    or_: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = t1.len();
    if t2.len() != n
        || ox.len() != n
        || oy.len() != n
        || or_.len() != n
        || a1.len() != horizon * n
        || a2.len() != horizon * n
    {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut tt1 = vec![0.0_f32; horizon * n];
    let mut tt2 = vec![0.0_f32; horizon * n];
    reacher_obs::rollout_batch(
        horizon, &mut t1, &mut t2, &a1, &a2, &ox, &oy, &or_, &mut tt1, &mut tt2,
    );
    Ok((tt1, tt2))
}

#[pyfunction]
fn reacher_obs_render(
    t1: Vec<f32>,
    t2: Vec<f32>,
    ox: Vec<f32>,
    oy: Vec<f32>,
    or_: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let n = t1.len();
    if t2.len() != n || ox.len() != n || oy.len() != n || or_.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    reacher_obs::render_batch(&t1, &t2, &ox, &oy, &or_, render::IMG, &mut frames);
    Ok(frames)
}

#[pyfunction]
fn armpush_step(
    mut t1: Vec<f32>,
    mut t2: Vec<f32>,
    mut ox: Vec<f32>,
    mut oy: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = t1.len();
    if t2.len() != n || ox.len() != n || oy.len() != n || a1.len() != n || a2.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    armpush::step_batch_in_place(&mut t1, &mut t2, &mut ox, &mut oy, &a1, &a2);
    Ok((t1, t2, ox, oy))
}

#[pyfunction]
fn armpush_rollout(
    horizon: usize,
    mut t1: Vec<f32>,
    mut t2: Vec<f32>,
    mut ox: Vec<f32>,
    mut oy: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = t1.len();
    if t2.len() != n
        || ox.len() != n
        || oy.len() != n
        || a1.len() != horizon * n
        || a2.len() != horizon * n
    {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut tt1 = vec![0.0_f32; horizon * n];
    let mut tt2 = vec![0.0_f32; horizon * n];
    let mut tox = vec![0.0_f32; horizon * n];
    let mut toy = vec![0.0_f32; horizon * n];
    armpush::rollout_batch(
        horizon, &mut t1, &mut t2, &mut ox, &mut oy, &a1, &a2, &mut tt1, &mut tt2, &mut tox,
        &mut toy,
    );
    Ok((tt1, tt2, tox, toy))
}

#[pyfunction]
fn armpush_render(t1: Vec<f32>, t2: Vec<f32>, ox: Vec<f32>, oy: Vec<f32>) -> PyResult<Vec<f32>> {
    let n = t1.len();
    if t2.len() != n || ox.len() != n || oy.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    armpush::render_batch(&t1, &t2, &ox, &oy, render::IMG, &mut frames);
    Ok(frames)
}

#[pyfunction]
fn pusht_reset(seed: u64, batch: usize) -> Vec<f32> {
    let mut rng = Rng::seed_from_u64(seed);
    let mut states = vec![pusht::State::new(0.0, 0.0, 0.0, 0.0, 0.0); batch];
    pusht::reset_batch(&mut rng, &mut states);
    flatten_pusht_states(&states)
}

#[pyfunction]
fn pusht_coverage(states: Vec<f32>) -> PyResult<Vec<f32>> {
    let parsed = parse_pusht_states(&states)?;
    Ok(parsed.iter().map(pusht::State::coverage).collect())
}

#[pyfunction]
fn pusht_step(state: Vec<f32>, action_x: f32, action_y: f32) -> PyResult<(Vec<f32>, f32)> {
    if state.len() != 7 {
        return Err(PyValueError::new_err(
            "expected state of length 7 [ax, ay, avx, avy, bx, by, bangle]",
        ));
    }
    let mut st = parse_pusht_state(&state);
    st.step_in_place(action_x, action_y);
    let cov = st.coverage();
    Ok((flatten_pusht_states(&[st]), cov))
}

#[pyfunction]
fn pusht_step_batch(
    states: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let mut parsed = parse_pusht_states(&states)?;
    let n = parsed.len();
    if ax.len() != n || ay.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    pusht::step_batch_in_place(&mut parsed, &ax, &ay);
    let covs = parsed.iter().map(pusht::State::coverage).collect();
    Ok((flatten_pusht_states(&parsed), covs))
}

#[pyfunction]
fn pusht_rollout(
    horizon: usize,
    states: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let mut parsed = parse_pusht_states(&states)?;
    let n = parsed.len();
    if ax.len() != horizon * n || ay.len() != horizon * n {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut traj = vec![0.0_f32; horizon * n * 7];
    for t in 0..horizon {
        pusht::step_batch_in_place(
            &mut parsed,
            &ax[t * n..(t + 1) * n],
            &ay[t * n..(t + 1) * n],
        );
        let dst = &mut traj[t * n * 7..(t + 1) * n * 7];
        dst.copy_from_slice(&flatten_pusht_states(&parsed));
    }
    Ok(traj)
}

#[pyfunction]
fn pusht_render_rgb(state: Vec<f32>, size: usize) -> PyResult<Vec<u8>> {
    if state.len() != 7 {
        return Err(PyValueError::new_err("expected state of length 7"));
    }
    let st = parse_pusht_state(&state);
    let mut rgb = vec![0u8; size * size * 3];
    st.render_rgb(size, &mut rgb);
    Ok(rgb)
}

#[pyfunction]
fn pusht_render_rgb_batch(states: Vec<f32>, size: usize) -> PyResult<Vec<u8>> {
    let parsed = parse_pusht_states(&states)?;
    let mut rgb = vec![0u8; parsed.len() * size * size * 3];
    pusht::render_rgb_batch(&parsed, size, &mut rgb);
    Ok(rgb)
}

#[pyfunction]
fn ogbench_maze_step_medium(
    mut x: Vec<f32>,
    mut y: Vec<f32>,
    mut vx: Vec<f32>,
    mut vy: Vec<f32>,
    fx: Vec<f32>,
    fy: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = x.len();
    if y.len() != n || vx.len() != n || vy.len() != n || fx.len() != n || fy.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    ogbench_maze::step_medium_batch_in_place(&mut x, &mut y, &mut vx, &mut vy, &fx, &fy);
    Ok((x, y, vx, vy))
}

#[pyfunction]
fn ogbench_maze_step_large(
    mut x: Vec<f32>,
    mut y: Vec<f32>,
    mut vx: Vec<f32>,
    mut vy: Vec<f32>,
    fx: Vec<f32>,
    fy: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = x.len();
    if y.len() != n || vx.len() != n || vy.len() != n || fx.len() != n || fy.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    ogbench_maze::step_large_batch_in_place(&mut x, &mut y, &mut vx, &mut vy, &fx, &fy);
    Ok((x, y, vx, vy))
}

#[pyfunction]
#[pyo3(signature = (n, num, clue_frac, seed=0, unique_only=false))]
fn sudoku_generate(
    py: Python<'_>,
    n: usize,
    num: usize,
    clue_frac: f64,
    seed: u64,
    unique_only: bool,
) -> (Vec<i8>, Vec<i8>) {
    py.allow_threads(|| sudoku::generate(n, num, clue_frac, seed, unique_only))
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IMG", render::IMG)?;
    m.add_function(wrap_pyfunction!(sokoban_step, m)?)?;
    m.add_function(wrap_pyfunction!(sokoban_solved, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_reset, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_in_wall, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_step, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_proc_step, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_proc_render, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_reset, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_step, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_render, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_reset, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_step, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_render, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_morph_render, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_obs_step, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_obs_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_obs_render, m)?)?;
    m.add_function(wrap_pyfunction!(armpush_step, m)?)?;
    m.add_function(wrap_pyfunction!(armpush_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(armpush_render, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_reset, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_step, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_step_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_render_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_render_rgb_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ogbench_maze_step_medium, m)?)?;
    m.add_function(wrap_pyfunction!(ogbench_maze_step_large, m)?)?;
    m.add_function(wrap_pyfunction!(sudoku_generate, m)?)?;
    Ok(())
}
