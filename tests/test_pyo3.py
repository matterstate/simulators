"""Tests for the PyO3 `matterstate_envs_rs` extension module."""

import numpy as np

from matterstate_envs import _rs as matterstate_envs_rs
from matterstate_envs import pointmaze, reacher, sokoban


def test_pyo3_sokoban_step() -> None:
    grid = np.full((10, 10), sokoban.FLOOR, np.int8)
    grid[[0, -1], :] = sokoban.WALL
    grid[:, [0, -1]] = sokoban.WALL
    grid[4, 3:6] = [sokoban.PLR, sokoban.BOX, sokoban.TARGET]
    flat = grid.ravel()
    expected = sokoban.step(flat, 3)
    actual = matterstate_envs_rs.sokoban_step(flat, 3)
    np.testing.assert_array_equal(actual, expected)


def test_pyo3_pointmaze_reset_and_rollout() -> None:
    batch = 32
    horizon = 16
    x, y, gx, gy = matterstate_envs_rs.pointmaze_reset(42, batch)
    start = np.stack([x, y], -1)
    goal = np.stack([gx, gy], -1)
    assert not pointmaze.in_wall(start).any()
    assert (np.linalg.norm(start - goal, axis=-1) >= 0.4 - 1e-6).all()

    rng = np.random.default_rng(7)
    actions = rng.uniform(-1.0, 1.0, (horizon, batch, 2)).astype(np.float32)
    ax = np.ascontiguousarray(actions[:, :, 0]).ravel()
    ay = np.ascontiguousarray(actions[:, :, 1]).ravel()
    tx, ty = matterstate_envs_rs.pointmaze_rollout(horizon, x, y, ax, ay)

    pos = start.copy()
    exp = []
    for t in range(horizon):
        pos = pointmaze.step(pos, actions[t])
        exp.append(pos)
    np.testing.assert_allclose(np.stack([tx, ty], -1), np.stack(exp), atol=1e-5)


def test_pyo3_reacher_render() -> None:
    rng = np.random.default_rng(11)
    theta, _ = reacher.reset(rng, 16)
    t1 = np.ascontiguousarray(theta[:, 0])
    t2 = np.ascontiguousarray(theta[:, 1])
    frames = matterstate_envs_rs.reacher_render(t1, t2, 48)
    np.testing.assert_array_equal(frames[..., None], reacher.render(theta))


def test_pyo3_pusht_rgb_and_ogbench_maze() -> None:
    state = np.array([290.0, 190.0, 0.0, 0.0, 256.0, 301.0, 0.0], np.float32)
    cov = matterstate_envs_rs.pusht_step(state, 290.0, 320.0)
    assert 0.0 <= cov <= 1.0
    rgb = matterstate_envs_rs.pusht_render_rgb(state, 96)
    assert rgb.shape == (96, 96, 3)
    assert rgb.dtype == np.uint8

    x = np.array([6.0], np.float32)
    y = np.array([6.0], np.float32)
    vx = np.array([0.0], np.float32)
    vy = np.array([0.0], np.float32)
    fx = np.array([1.0], np.float32)
    fy = np.array([0.0], np.float32)
    matterstate_envs_rs.ogbench_maze_step_in_place(x, y, vx, vy, fx, fy)
    assert x[0] > 6.0

    states = np.tile(state, (4, 1))
    covs = matterstate_envs_rs.pusht_step_batch(
        states, np.full(4, 290.0, np.float32), np.full(4, 320.0, np.float32)
    )
    assert covs.shape == (4,)
    rgbs = matterstate_envs_rs.pusht_render_rgb_batch(states, 96)
    assert rgbs.shape == (4, 96, 96, 3)

    matterstate_envs_rs.ogbench_maze_large_step_in_place(x, y, vx, vy, fx, fy)
    assert x[0] > 6.0


def test_pyo3_pusht_reset_coverage_rollout_and_sudoku() -> None:
    states = matterstate_envs_rs.pusht_reset(42, 8)
    assert states.shape == (8, 7)
    covs = matterstate_envs_rs.pusht_coverage(states)
    assert covs.shape == (8,)
    assert ((covs >= 0.0) & (covs <= 1.0)).all()

    ax = np.full(5 * 8, 256.0, np.float32)
    ay = np.full(5 * 8, 256.0, np.float32)
    traj = matterstate_envs_rs.pusht_rollout(5, states, ax, ay)
    assert traj.shape == (5, 8, 7)

    puz, sol = matterstate_envs_rs.sudoku_generate(4, 6, 0.5, 42, True)
    assert puz.shape == (6, 16)
    assert sol.shape == (6, 16)
    assert puz.dtype == np.int8
    assert sol.dtype == np.int8
    assert ((sol >= 1) & (sol <= 4)).all()
