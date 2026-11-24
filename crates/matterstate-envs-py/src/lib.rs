//! PyO3 + NumPy Python bindings (`matterstate_envs_rs`) for `matterstate-envs`.

use matterstate_envs::{
    armpush, ogbench_maze, pointmaze, pointmaze_proc, pointpush, pusht, reacher, reacher_morph,
    reacher_obs, render, rng::Rng, sokoban,
};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn sokoban_step<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, i8>,
    action: u8,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let slice = grid.as_slice()?;
    if slice.len() != sokoban::CELLS {
        return Err(PyValueError::new_err(format!(
            "expected {} cells, got {}",
            sokoban::CELLS,
            slice.len()
        )));
    }
    let act = sokoban::Action::from_u8(action).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut cells = [0u8; sokoban::CELLS];
    for (dst, &src) in cells.iter_mut().zip(slice.iter()) {
        *dst = src as u8;
    }
    let mut board =
        sokoban::Board::from_cells(cells).map_err(|e| PyValueError::new_err(e.to_string()))?;
    board.step_in_place(act);
    let out: Vec<i8> = board.cells().iter().map(|&c| c as i8).collect();
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn pointmaze_reset<'py>(
    py: Python<'py>,
    seed: u64,
    batch: usize,
) -> (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let mut x = vec![0.0_f32; batch];
    let mut y = vec![0.0_f32; batch];
    let mut gx = vec![0.0_f32; batch];
    let mut gy = vec![0.0_f32; batch];
    pointmaze::reset_batch(&mut rng, &mut x, &mut y, &mut gx, &mut gy);
    (
        PyArray1::from_vec(py, x),
        PyArray1::from_vec(py, y),
        PyArray1::from_vec(py, gx),
        PyArray1::from_vec(py, gy),
    )
}

#[pyfunction]
fn pointmaze_step_in_place<'py>(
    mut x: PyReadwriteArray1<'py, f32>,
    mut y: PyReadwriteArray1<'py, f32>,
    ax: PyReadonlyArray1<'py, f32>,
    ay: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let xs = x.as_slice_mut()?;
    let ys = y.as_slice_mut()?;
    let axs = ax.as_slice()?;
    let ays = ay.as_slice()?;
    if xs.len() != ys.len() || xs.len() != axs.len() || xs.len() != ays.len() {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    pointmaze::step_batch_in_place(xs, ys, axs, ays);
    Ok(())
}

#[pyfunction]
fn pointmaze_rollout<'py>(
    py: Python<'py>,
    horizon: usize,
    mut x: PyReadwriteArray1<'py, f32>,
    mut y: PyReadwriteArray1<'py, f32>,
    ax: PyReadonlyArray1<'py, f32>,
    ay: PyReadonlyArray1<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let n = x.len();
    let xs = x.as_slice_mut()?;
    let ys = y.as_slice_mut()?;
    let axs = ax.as_slice()?;
    let ays = ay.as_slice()?;
    if ys.len() != n || axs.len() != horizon * n || ays.len() != horizon * n {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut traj_x = vec![0.0_f32; horizon * n];
    let mut traj_y = vec![0.0_f32; horizon * n];
    pointmaze::rollout_batch(horizon, xs, ys, axs, ays, &mut traj_x, &mut traj_y);
    let out_x = PyArray1::from_vec(py, traj_x).reshape([horizon, n])?;
    let out_y = PyArray1::from_vec(py, traj_y).reshape([horizon, n])?;
    Ok((out_x, out_y))
}

#[pyfunction]
fn pointmaze_proc_step_in_place<'py>(
    mut x: PyReadwriteArray1<'py, f32>,
    mut y: PyReadwriteArray1<'py, f32>,
    ax: PyReadonlyArray1<'py, f32>,
    ay: PyReadonlyArray1<'py, f32>,
    walls: PyReadonlyArray3<'py, bool>,
) -> PyResult<()> {
    let n = x.len();
    let xs = x.as_slice_mut()?;
    let ys = y.as_slice_mut()?;
    let axs = ax.as_slice()?;
    let ays = ay.as_slice()?;
    let ws = walls.as_slice()?;
    if ys.len() != n
        || axs.len() != n
        || ays.len() != n
        || ws.len() != n * pointmaze_proc::WALL_CELLS
    {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let wall_grids: Vec<[bool; pointmaze_proc::WALL_CELLS]> = ws
        .chunks_exact(pointmaze_proc::WALL_CELLS)
        .map(|c| c.try_into().unwrap())
        .collect();
    pointmaze_proc::step_batch_in_place(xs, ys, axs, ays, &wall_grids);
    Ok(())
}

#[pyfunction]
fn pointmaze_proc_render<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
    walls: PyReadonlyArray3<'py, bool>,
    r_px: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let ws = walls.as_slice()?;
    let n = xs.len();
    if ys.len() != n || ws.len() != n * pointmaze_proc::WALL_CELLS {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let wall_grids: Vec<[bool; pointmaze_proc::WALL_CELLS]> = ws
        .chunks_exact(pointmaze_proc::WALL_CELLS)
        .map(|c| c.try_into().unwrap())
        .collect();
    let mut frames = vec![0.0_f32; n * render::IMG * render::IMG];
    pointmaze_proc::render_batch(xs, ys, &wall_grids, r_px, &mut frames);
    Ok(PyArray1::from_vec(py, frames).reshape([n, render::IMG, render::IMG])?)
}

#[pyfunction]
fn pointpush_reset<'py>(
    py: Python<'py>,
    seed: u64,
    batch: usize,
) -> (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
) {
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
    (
        PyArray1::from_vec(py, ax),
        PyArray1::from_vec(py, ay),
        PyArray1::from_vec(py, ox),
        PyArray1::from_vec(py, oy),
        PyArray1::from_vec(py, gx),
        PyArray1::from_vec(py, gy),
    )
}

#[pyfunction]
fn pointpush_step_in_place<'py>(
    mut ax: PyReadwriteArray1<'py, f32>,
    mut ay: PyReadwriteArray1<'py, f32>,
    mut ox: PyReadwriteArray1<'py, f32>,
    mut oy: PyReadwriteArray1<'py, f32>,
    act_x: PyReadonlyArray1<'py, f32>,
    act_y: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let axs = ax.as_slice_mut()?;
    let ays = ay.as_slice_mut()?;
    let oxs = ox.as_slice_mut()?;
    let oys = oy.as_slice_mut()?;
    let uxs = act_x.as_slice()?;
    let uys = act_y.as_slice()?;
    let n = axs.len();
    if ays.len() != n || oxs.len() != n || oys.len() != n || uxs.len() != n || uys.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    pointpush::step_batch_in_place(axs, ays, oxs, oys, uxs, uys);
    Ok(())
}

#[pyfunction]
fn pointpush_render<'py>(
    py: Python<'py>,
    ax: PyReadonlyArray1<'py, f32>,
    ay: PyReadonlyArray1<'py, f32>,
    ox: PyReadonlyArray1<'py, f32>,
    oy: PyReadonlyArray1<'py, f32>,
    img: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let axs = ax.as_slice()?;
    let ays = ay.as_slice()?;
    let oxs = ox.as_slice()?;
    let oys = oy.as_slice()?;
    let n = axs.len();
    if ays.len() != n || oxs.len() != n || oys.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * img * img];
    pointpush::render_batch(axs, ays, oxs, oys, img, &mut frames);
    Ok(PyArray1::from_vec(py, frames).reshape([n, img, img])?)
}

#[pyfunction]
fn reacher_reset<'py>(
    py: Python<'py>,
    seed: u64,
    batch: usize,
) -> (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
) {
    let mut rng = Rng::seed_from_u64(seed);
    let mut t1 = vec![0.0_f32; batch];
    let mut t2 = vec![0.0_f32; batch];
    let mut gx = vec![0.0_f32; batch];
    let mut gy = vec![0.0_f32; batch];
    reacher::reset_batch(&mut rng, &mut t1, &mut t2, &mut gx, &mut gy);
    (
        PyArray1::from_vec(py, t1),
        PyArray1::from_vec(py, t2),
        PyArray1::from_vec(py, gx),
        PyArray1::from_vec(py, gy),
    )
}

#[pyfunction]
fn reacher_step_in_place<'py>(
    mut t1: PyReadwriteArray1<'py, f32>,
    mut t2: PyReadwriteArray1<'py, f32>,
    a1: PyReadonlyArray1<'py, f32>,
    a2: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let t1s = t1.as_slice_mut()?;
    let t2s = t2.as_slice_mut()?;
    let a1s = a1.as_slice()?;
    let a2s = a2.as_slice()?;
    if t1s.len() != t2s.len() || t1s.len() != a1s.len() || t1s.len() != a2s.len() {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    reacher::step_batch_in_place(t1s, t2s, a1s, a2s);
    Ok(())
}

#[pyfunction]
fn reacher_rollout<'py>(
    py: Python<'py>,
    horizon: usize,
    mut t1: PyReadwriteArray1<'py, f32>,
    mut t2: PyReadwriteArray1<'py, f32>,
    a1: PyReadonlyArray1<'py, f32>,
    a2: PyReadonlyArray1<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let n = t1.len();
    let t1s = t1.as_slice_mut()?;
    let t2s = t2.as_slice_mut()?;
    let a1s = a1.as_slice()?;
    let a2s = a2.as_slice()?;
    if t2s.len() != n || a1s.len() != horizon * n || a2s.len() != horizon * n {
        return Err(PyValueError::new_err("mismatched rollout dimensions"));
    }
    let mut traj_t1 = vec![0.0_f32; horizon * n];
    let mut traj_t2 = vec![0.0_f32; horizon * n];
    reacher::rollout_batch(horizon, t1s, t2s, a1s, a2s, &mut traj_t1, &mut traj_t2);
    Ok((
        PyArray1::from_vec(py, traj_t1).reshape([horizon, n])?,
        PyArray1::from_vec(py, traj_t2).reshape([horizon, n])?,
    ))
}

#[pyfunction]
fn reacher_render<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray1<'py, f32>,
    t2: PyReadonlyArray1<'py, f32>,
    img: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let t1s = t1.as_slice()?;
    let t2s = t2.as_slice()?;
    if t1s.len() != t2s.len() {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let n = t1s.len();
    let mut frames = vec![0.0_f32; n * img * img];
    reacher::render_batch(t1s, t2s, img, &mut frames);
    Ok(PyArray1::from_vec(py, frames).reshape([n, img, img])?)
}

#[pyfunction]
fn reacher_obs_step_in_place<'py>(
    mut t1: PyReadwriteArray1<'py, f32>,
    mut t2: PyReadwriteArray1<'py, f32>,
    a1: PyReadonlyArray1<'py, f32>,
    a2: PyReadonlyArray1<'py, f32>,
    ox: PyReadonlyArray1<'py, f32>,
    oy: PyReadonlyArray1<'py, f32>,
    or: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let t1s = t1.as_slice_mut()?;
    let t2s = t2.as_slice_mut()?;
    let a1s = a1.as_slice()?;
    let a2s = a2.as_slice()?;
    let oxs = ox.as_slice()?;
    let oys = oy.as_slice()?;
    let ors = or.as_slice()?;
    let n = t1s.len();
    if t2s.len() != n
        || a1s.len() != n
        || a2s.len() != n
        || oxs.len() != n
        || oys.len() != n
        || ors.len() != n
    {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    reacher_obs::step_batch_in_place(t1s, t2s, a1s, a2s, oxs, oys, ors);
    Ok(())
}

#[pyfunction]
fn reacher_obs_render<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray1<'py, f32>,
    t2: PyReadonlyArray1<'py, f32>,
    ox: PyReadonlyArray1<'py, f32>,
    oy: PyReadonlyArray1<'py, f32>,
    or: PyReadonlyArray1<'py, f32>,
    img: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let t1s = t1.as_slice()?;
    let t2s = t2.as_slice()?;
    let oxs = ox.as_slice()?;
    let oys = oy.as_slice()?;
    let ors = or.as_slice()?;
    let n = t1s.len();
    if t2s.len() != n || oxs.len() != n || oys.len() != n || ors.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * img * img];
    reacher_obs::render_batch(t1s, t2s, oxs, oys, ors, img, &mut frames);
    Ok(PyArray1::from_vec(py, frames).reshape([n, img, img])?)
}

#[pyfunction]
fn reacher_morph_render<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray1<'py, f32>,
    t2: PyReadonlyArray1<'py, f32>,
    l1: PyReadonlyArray1<'py, f32>,
    l2: PyReadonlyArray1<'py, f32>,
    img: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let t1s = t1.as_slice()?;
    let t2s = t2.as_slice()?;
    let l1s = l1.as_slice()?;
    let l2s = l2.as_slice()?;
    let n = t1s.len();
    if t2s.len() != n || l1s.len() != n || l2s.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * img * img];
    reacher_morph::render_batch(t1s, t2s, l1s, l2s, img, &mut frames);
    Ok(PyArray1::from_vec(py, frames).reshape([n, img, img])?)
}

#[pyfunction]
fn armpush_step_in_place<'py>(
    mut t1: PyReadwriteArray1<'py, f32>,
    mut t2: PyReadwriteArray1<'py, f32>,
    mut ox: PyReadwriteArray1<'py, f32>,
    mut oy: PyReadwriteArray1<'py, f32>,
    a1: PyReadonlyArray1<'py, f32>,
    a2: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let t1s = t1.as_slice_mut()?;
    let t2s = t2.as_slice_mut()?;
    let oxs = ox.as_slice_mut()?;
    let oys = oy.as_slice_mut()?;
    let a1s = a1.as_slice()?;
    let a2s = a2.as_slice()?;
    let n = t1s.len();
    if t2s.len() != n || oxs.len() != n || oys.len() != n || a1s.len() != n || a2s.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    armpush::step_batch_in_place(t1s, t2s, oxs, oys, a1s, a2s);
    Ok(())
}

#[pyfunction]
fn armpush_render<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray1<'py, f32>,
    t2: PyReadonlyArray1<'py, f32>,
    ox: PyReadonlyArray1<'py, f32>,
    oy: PyReadonlyArray1<'py, f32>,
    img: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let t1s = t1.as_slice()?;
    let t2s = t2.as_slice()?;
    let oxs = ox.as_slice()?;
    let oys = oy.as_slice()?;
    let n = t1s.len();
    if t2s.len() != n || oxs.len() != n || oys.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let mut frames = vec![0.0_f32; n * img * img];
    armpush::render_batch(t1s, t2s, oxs, oys, img, &mut frames);
    Ok(PyArray1::from_vec(py, frames).reshape([n, img, img])?)
}

#[pyfunction]
fn pusht_step<'py>(
    mut state: PyReadwriteArray1<'py, f32>,
    action_x: f32,
    action_y: f32,
) -> PyResult<f32> {
    let s = state.as_slice_mut()?;
    if s.len() != 7 {
        return Err(PyValueError::new_err(
            "expected state of length 7 [ax, ay, avx, avy, bx, by, bangle]",
        ));
    }
    let mut st = pusht::State {
        agent_x: s[0],
        agent_y: s[1],
        agent_vx: s[2],
        agent_vy: s[3],
        block_x: s[4],
        block_y: s[5],
        block_angle: s[6],
        block_vx: 0.0,
        block_vy: 0.0,
        block_omega: 0.0,
    };
    st.step_in_place(action_x, action_y);
    s[0] = st.agent_x;
    s[1] = st.agent_y;
    s[2] = st.agent_vx;
    s[3] = st.agent_vy;
    s[4] = st.block_x;
    s[5] = st.block_y;
    s[6] = st.block_angle;
    Ok(st.coverage())
}

#[pyfunction]
fn pusht_render_rgb<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f32>,
    size: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let s = state.as_slice()?;
    if s.len() != 7 {
        return Err(PyValueError::new_err("expected state of length 7"));
    }
    let st = pusht::State::new(s[0], s[1], s[4], s[5], s[6]);
    let mut rgb = vec![0u8; size * size * 3];
    st.render_rgb(size, &mut rgb);
    Ok(PyArray1::from_vec(py, rgb).reshape([size, size, 3])?)
}

#[pyfunction]
fn pusht_step_batch<'py>(
    py: Python<'py>,
    mut states: PyReadwriteArray2<'py, f32>,
    ax: PyReadonlyArray1<'py, f32>,
    ay: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = states.shape().to_vec();
    if shape.len() != 2 || shape[1] != 7 {
        return Err(PyValueError::new_err("expected states of shape (B, 7)"));
    }
    let n = shape[0];
    let axs = ax.as_slice()?;
    let ays = ay.as_slice()?;
    if axs.len() != n || ays.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    let flat = states.as_slice_mut()?;
    let mut covs = Vec::with_capacity(n);
    for i in 0..n {
        let s = &mut flat[i * 7..(i + 1) * 7];
        let mut st = pusht::State {
            agent_x: s[0],
            agent_y: s[1],
            agent_vx: s[2],
            agent_vy: s[3],
            block_x: s[4],
            block_y: s[5],
            block_angle: s[6],
            block_vx: 0.0,
            block_vy: 0.0,
            block_omega: 0.0,
        };
        st.step_in_place(axs[i], ays[i]);
        s[0] = st.agent_x;
        s[1] = st.agent_y;
        s[2] = st.agent_vx;
        s[3] = st.agent_vy;
        s[4] = st.block_x;
        s[5] = st.block_y;
        s[6] = st.block_angle;
        covs.push(st.coverage());
    }
    Ok(PyArray1::from_vec(py, covs))
}

#[pyfunction]
fn pusht_render_rgb_batch<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f32>,
    size: usize,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let shape = states.shape();
    if shape.len() != 2 || shape[1] != 7 {
        return Err(PyValueError::new_err("expected states of shape (B, 7)"));
    }
    let n = shape[0];
    let flat = states.as_slice()?;
    let rust_states: Vec<pusht::State> = flat
        .chunks_exact(7)
        .map(|s| pusht::State::new(s[0], s[1], s[4], s[5], s[6]))
        .collect();
    let mut rgb = vec![0u8; n * size * size * 3];
    pusht::render_rgb_batch(&rust_states, size, &mut rgb);
    Ok(PyArray1::from_vec(py, rgb).reshape([n, size, size, 3])?)
}

#[pyfunction]
fn ogbench_maze_large_step_in_place<'py>(
    mut x: PyReadwriteArray1<'py, f32>,
    mut y: PyReadwriteArray1<'py, f32>,
    mut vx: PyReadwriteArray1<'py, f32>,
    mut vy: PyReadwriteArray1<'py, f32>,
    fx: PyReadonlyArray1<'py, f32>,
    fy: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let xs = x.as_slice_mut()?;
    let ys = y.as_slice_mut()?;
    let vxs = vx.as_slice_mut()?;
    let vys = vy.as_slice_mut()?;
    let fxs = fx.as_slice()?;
    let fys = fy.as_slice()?;
    let n = xs.len();
    if ys.len() != n || vxs.len() != n || vys.len() != n || fxs.len() != n || fys.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    ogbench_maze::step_large_batch_in_place(xs, ys, vxs, vys, fxs, fys);
    Ok(())
}

#[pyfunction]
fn ogbench_maze_step_in_place<'py>(
    mut x: PyReadwriteArray1<'py, f32>,
    mut y: PyReadwriteArray1<'py, f32>,
    mut vx: PyReadwriteArray1<'py, f32>,
    mut vy: PyReadwriteArray1<'py, f32>,
    fx: PyReadonlyArray1<'py, f32>,
    fy: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let xs = x.as_slice_mut()?;
    let ys = y.as_slice_mut()?;
    let vxs = vx.as_slice_mut()?;
    let vys = vy.as_slice_mut()?;
    let fxs = fx.as_slice()?;
    let fys = fy.as_slice()?;
    let n = xs.len();
    if ys.len() != n || vxs.len() != n || vys.len() != n || fxs.len() != n || fys.len() != n {
        return Err(PyValueError::new_err("mismatched batch lengths"));
    }
    ogbench_maze::step_medium_batch_in_place(xs, ys, vxs, vys, fxs, fys);
    Ok(())
}

#[pymodule]
fn matterstate_envs_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IMG", render::IMG)?;
    m.add_function(wrap_pyfunction!(sokoban_step, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_reset, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_proc_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_proc_render, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_reset, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(pointpush_render, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_reset, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_render, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_obs_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_obs_render, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_morph_render, m)?)?;
    m.add_function(wrap_pyfunction!(armpush_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(armpush_render, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_step, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_step_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_render_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_render_rgb_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ogbench_maze_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(ogbench_maze_large_step_in_place, m)?)?;
    Ok(())
}
