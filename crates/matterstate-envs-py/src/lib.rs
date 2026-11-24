//! PyO3 + NumPy Python bindings (`matterstate_envs_rs`) for `matterstate-envs`.

use matterstate_envs::{pointmaze, pusht, reacher, render, rng::Rng, sokoban};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadwriteArray1,
    PyUntypedArrayMethods,
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
    let arr = PyArray1::from_vec(py, frames).reshape([n, img, img])?;
    Ok(arr)
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

#[pymodule]
fn matterstate_envs_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IMG", render::IMG)?;
    m.add_function(wrap_pyfunction!(sokoban_step, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_reset, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_step_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(pointmaze_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(reacher_render, m)?)?;
    m.add_function(wrap_pyfunction!(pusht_step, m)?)?;
    Ok(())
}
