"""Property-based tests (Hegel) for simulator invariants and Rust parity."""

from __future__ import annotations

import hegel
import numpy as np
from hegel import generators as gs

from matterstate_envs import _rs, pointmaze, pointpush, reacher, sokoban


@hegel.test
def test_sokoban_reverse_round_trip_and_rust_parity(
    tc: hegel.TestCase,
) -> None:
    """Generated Sokoban boards obey step parity and reverse round-trip."""
    pr = tc.draw(gs.integers(min_value=1, max_value=8))
    pc = tc.draw(gs.integers(min_value=1, max_value=8))
    br = tc.draw(gs.integers(min_value=1, max_value=8))
    bc = tc.draw(gs.integers(min_value=1, max_value=8))
    action = tc.draw(gs.integers(min_value=0, max_value=3))

    board = np.full((10, 10), sokoban.FLOOR, dtype=np.int8)
    board[0, :] = sokoban.WALL
    board[9, :] = sokoban.WALL
    board[:, 0] = sokoban.WALL
    board[:, 9] = sokoban.WALL
    board[pr, pc] = sokoban.PLR
    if (br, bc) != (pr, pc):
        board[br, bc] = sokoban.BOX

    flat = board.reshape(-1)
    py_next = sokoban.step(flat, action)
    rs_next = _rs.sokoban_step(flat, action)
    np.testing.assert_array_equal(py_next, rs_next)

    for pred, fwd_action in sokoban.reverse_predecessors(flat):
        assert np.array_equal(sokoban.step(pred, fwd_action), flat)


@hegel.test
def test_pointmaze_bounds_wall_and_rust_parity(tc: hegel.TestCase) -> None:
    """PointMaze stays in [0, 1]^2, never enters a wall, and matches Rust."""
    x = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    y = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ax = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))
    ay = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))

    pos = np.array([[x, y]], dtype=np.float32)
    act = np.array([[ax, ay]], dtype=np.float32)
    was_free = not bool(pointmaze.in_wall(pos)[0])

    py_next = pointmaze.step(pos, act)
    assert np.all(np.isfinite(py_next))
    assert np.all((py_next >= 0.0) & (py_next <= 1.0))
    if was_free:
        assert not bool(pointmaze.in_wall(py_next)[0])

    rx = np.array([x], dtype=np.float32)
    ry = np.array([y], dtype=np.float32)
    rax = np.array([ax], dtype=np.float32)
    ray = np.array([ay], dtype=np.float32)
    _rs.pointmaze_step_in_place(rx, ry, rax, ray)
    np.testing.assert_allclose(py_next[0], [rx[0], ry[0]], atol=1e-6)


@hegel.test
def test_pointpush_and_reacher_bounds(tc: hegel.TestCase) -> None:
    """PointPush stays in [0, 1]^2 and Reacher wraps angles to [-pi, pi]."""
    ax = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ay = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ox = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    oy = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ux = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))
    uy = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))
    t1 = tc.draw(gs.floats(min_value=-np.pi, max_value=np.pi))
    t2 = tc.draw(gs.floats(min_value=-np.pi, max_value=np.pi))

    na, no = pointpush.step(
        np.array([[ax, ay]], dtype=np.float32),
        np.array([[ox, oy]], dtype=np.float32),
        np.array([[ux, uy]], dtype=np.float32),
    )
    assert np.all((na >= 0.0) & (na <= 1.0))
    assert np.all((no >= 0.0) & (no <= 1.0))

    nt = reacher.step(
        np.array([[t1, t2]], dtype=np.float32),
        np.array([[ux, uy]], dtype=np.float32),
    )
    assert np.all(np.abs(nt) <= np.pi + 1e-5)
