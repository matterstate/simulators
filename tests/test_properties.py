"""Property-based tests (Hypothesis / Hegel-core) for simulator invariants."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from matterstate_envs import pointmaze, pointpush, reacher, sokoban

_TARGET_DIR = Path(
    os.environ.get(
        "CARGO_TARGET_DIR", Path(__file__).resolve().parents[1] / "target"
    )
)
LIB_PATH = _TARGET_DIR / "release" / "libmatterstate_envs.so"
_F32_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.ms_sokoban_step.argtypes = [_U8_P, ctypes.c_uint8, _U8_P]
    lib.ms_sokoban_step.restype = ctypes.c_int32
    lib.ms_pointmaze_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    return lib


@st.composite
def sokoban_boards(draw: st.DrawFn) -> np.ndarray:
    """Generate a valid 10x10 Sokoban board with wall border and one player."""
    board = np.full((10, 10), sokoban.FLOOR, np.int8)
    board[[0, -1], :] = sokoban.WALL
    board[:, [0, -1]] = sokoban.WALL
    interior = [(r, c) for r in range(1, 9) for c in range(1, 9)]
    n_boxes = draw(st.integers(min_value=1, max_value=4))
    chosen = draw(
        st.lists(
            st.sampled_from(interior),
            min_size=1 + n_boxes,
            max_size=1 + n_boxes,
            unique=True,
        )
    )
    pr, pc = chosen[0]
    board[pr, pc] = draw(st.sampled_from([sokoban.PLR, sokoban.PLR_T]))
    for br, bc in chosen[1:]:
        board[br, bc] = draw(st.sampled_from([sokoban.BOX, sokoban.BOX_T]))
    return board.ravel()


@settings(max_examples=64, deadline=None)
@given(board=sokoban_boards(), action=st.integers(min_value=0, max_value=3))
def test_sokoban_reverse_round_trip_and_rust_parity(
    board: np.ndarray, action: int
) -> None:
    lib = _load_lib()
    for pred, act in sokoban.reverse_predecessors(board):
        pred_copy = pred.copy()
        nxt = sokoban.step(pred, act)
        np.testing.assert_array_equal(nxt, board)
        np.testing.assert_array_equal(pred, pred_copy)

    out = np.empty(100, np.uint8)
    rc = lib.ms_sokoban_step(
        board.view(np.uint8).ctypes.data_as(_U8_P),
        action,
        out.ctypes.data_as(_U8_P),
    )
    assert rc == 0
    np.testing.assert_array_equal(
        out.view(np.int8), sokoban.step(board, action)
    )


@settings(max_examples=100, deadline=None)
@given(
    x=st.floats(min_value=0.0, max_value=1.0, width=32),
    y=st.floats(min_value=0.0, max_value=1.0, width=32),
    ax=st.floats(min_value=-5.0, max_value=5.0, width=32),
    ay=st.floats(min_value=-5.0, max_value=5.0, width=32),
)
def test_pointmaze_bounds_wall_and_rust_parity(
    x: float, y: float, ax: float, ay: float
) -> None:
    pos = np.array([[x, y]], np.float32)
    if pointmaze.in_wall(pos).any():
        return
    act = np.array([[ax, ay]], np.float32)
    nxt = pointmaze.step(pos, act)
    assert (nxt >= 0.0).all() and (nxt <= 1.0).all()
    assert not pointmaze.in_wall(nxt).any()

    lib = _load_lib()
    rx = np.array([x], np.float32)
    ry = np.array([y], np.float32)
    rax = np.array([ax], np.float32)
    ray = np.array([ay], np.float32)
    lib.ms_pointmaze_step(
        1,
        rx.ctypes.data_as(_F32_P),
        ry.ctypes.data_as(_F32_P),
        rax.ctypes.data_as(_F32_P),
        ray.ctypes.data_as(_F32_P),
    )
    np.testing.assert_allclose(np.array([[rx[0], ry[0]]]), nxt, atol=1e-6)


@settings(max_examples=100, deadline=None)
@given(
    t1=st.floats(min_value=-4.0, max_value=4.0, width=32),
    t2=st.floats(min_value=-4.0, max_value=4.0, width=32),
    a1=st.floats(min_value=-5.0, max_value=5.0, width=32),
    a2=st.floats(min_value=-5.0, max_value=5.0, width=32),
)
def test_reacher_angle_wrapping_bounds(
    t1: float, t2: float, a1: float, a2: float
) -> None:
    th = np.array([[t1, t2]], np.float32)
    act = np.array([[a1, a2]], np.float32)
    nxt = reacher.step(th, act)
    assert nxt.dtype == np.float32
    assert (nxt >= -np.pi - 1e-6).all() and (nxt <= np.pi + 1e-6).all()


@settings(max_examples=100, deadline=None)
@given(
    ax=st.floats(min_value=0.0, max_value=1.0, width=32),
    ay=st.floats(min_value=0.0, max_value=1.0, width=32),
    ox=st.floats(min_value=0.0, max_value=1.0, width=32),
    oy=st.floats(min_value=0.0, max_value=1.0, width=32),
    ux=st.floats(min_value=-2.0, max_value=2.0, width=32),
    uy=st.floats(min_value=-2.0, max_value=2.0, width=32),
)
def test_pointpush_stays_in_unit_square(
    ax: float, ay: float, ox: float, oy: float, ux: float, uy: float
) -> None:
    agent = np.array([[ax, ay]], np.float32)
    obj = np.array([[ox, oy]], np.float32)
    act = np.array([[ux, uy]], np.float32)
    na, no = pointpush.step(agent, obj, act)
    assert (na >= 0.0).all() and (na <= 1.0).all()
    assert (no >= 0.0).all() and (no <= 1.0).all()
