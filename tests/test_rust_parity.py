"""Parity tests comparing the Rust C-ABI library against the NumPy kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from matterstate_envs import (
    armpush,
    pointmaze,
    pointmaze_proc,
    pointpush,
    reacher,
    reacher_morph,
    reacher_obs,
    sokoban,
)

LIB_PATH = (
    Path(__file__).resolve().parents[1]
    / "target"
    / "release"
    / "libmatterstate_envs.so"
)

_F32_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)
_BOOL_P = ctypes.POINTER(ctypes.c_bool)


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
    lib.ms_pointmaze_proc_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _BOOL_P,
    ]
    lib.ms_pointmaze_proc_render.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _BOOL_P,
        ctypes.c_float,
        _F32_P,
    ]
    lib.ms_pointpush_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_pointpush_render.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        ctypes.c_size_t,
        _F32_P,
    ]
    lib.ms_reacher_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_reacher_render.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        ctypes.c_size_t,
        _F32_P,
    ]
    lib.ms_reacher_morph_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_reacher_morph_render.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        ctypes.c_size_t,
        _F32_P,
    ]
    lib.ms_reacher_obs_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_reacher_obs_render.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        ctypes.c_size_t,
        _F32_P,
    ]
    lib.ms_armpush_step.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_armpush_render.argtypes = [
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        ctypes.c_size_t,
        _F32_P,
    ]
    lib.ms_pointmaze_reset.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_pointmaze_rollout.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_reacher_reset.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    lib.ms_reacher_rollout.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
        _F32_P,
    ]
    return lib


def _f32_ptr(arr: np.ndarray) -> ctypes.POINTER:
    return arr.ctypes.data_as(_F32_P)


def test_sokoban_byte_exact_parity() -> None:
    lib = _load_lib()
    rng = np.random.default_rng(7)
    for _ in range(32):
        board = np.full((10, 10), sokoban.FLOOR, np.int8)
        board[[0, -1], :] = sokoban.WALL
        board[:, [0, -1]] = sokoban.WALL
        board[1:-1, 1:-1][rng.random((8, 8)) < 0.15] = sokoban.WALL
        avail = np.flatnonzero(board.ravel() == sokoban.FLOOR)
        targets = rng.choice(avail, 4, replace=False)
        board.ravel()[targets] = sokoban.TARGET
        occ = rng.choice(avail, 5, replace=False)
        for idx in occ[:4]:
            board.ravel()[idx] = (
                sokoban.BOX_T if idx in targets else sokoban.BOX
            )
        board.ravel()[occ[4]] = (
            sokoban.PLR_T if occ[4] in targets else sokoban.PLR
        )

        grid = board.ravel().copy()
        out = np.empty(100, np.uint8)
        for act in rng.integers(0, 4, 32):
            expected = sokoban.step(grid, int(act))
            rc = lib.ms_sokoban_step(
                grid.view(np.uint8).ctypes.data_as(_U8_P),
                int(act),
                out.ctypes.data_as(_U8_P),
            )
            assert rc == 0
            np.testing.assert_array_equal(out.view(np.int8), expected)
            grid = expected


def test_continuous_step_and_render_parity() -> None:
    lib = _load_lib()
    rng = np.random.default_rng(123)
    batch = 16
    actions = rng.uniform(-1.5, 1.5, (batch, 2)).astype(np.float32)
    ax = np.ascontiguousarray(actions[:, 0])
    ay = np.ascontiguousarray(actions[:, 1])

    # pointmaze
    pos, _ = pointmaze.reset(rng, batch)
    pos32 = pos.astype(np.float32)
    x = np.ascontiguousarray(pos32[:, 0])
    y = np.ascontiguousarray(pos32[:, 1])
    lib.ms_pointmaze_step(
        batch, _f32_ptr(x), _f32_ptr(y), _f32_ptr(ax), _f32_ptr(ay)
    )
    expected = pointmaze.step(pos32, actions)
    np.testing.assert_allclose(np.stack([x, y], -1), expected, atol=1e-6)

    # pointmaze_proc
    walls = np.ascontiguousarray(pointmaze_proc.sample_mazes(rng, batch))
    pos, _ = pointmaze_proc.reset(rng, walls)
    x = np.ascontiguousarray(pos[:, 0])
    y = np.ascontiguousarray(pos[:, 1])
    walls_ptr = walls.ctypes.data_as(_BOOL_P)
    lib.ms_pointmaze_proc_step(
        batch, _f32_ptr(x), _f32_ptr(y), _f32_ptr(ax), _f32_ptr(ay), walls_ptr
    )
    expected = pointmaze_proc.step(pos, actions, walls)
    np.testing.assert_allclose(np.stack([x, y], -1), expected, atol=1e-6)
    frames = np.empty((batch, 48, 48), np.float32)
    lib.ms_pointmaze_proc_render(
        batch, _f32_ptr(x), _f32_ptr(y), walls_ptr, 1.5, _f32_ptr(frames)
    )
    np.testing.assert_array_equal(
        frames[..., None], pointmaze_proc.render(expected, walls)
    )

    # pointpush
    ag, ob, _ = pointpush.reset(rng, batch)
    agx, agy = np.ascontiguousarray(ag[:, 0]), np.ascontiguousarray(ag[:, 1])
    obx, oby = np.ascontiguousarray(ob[:, 0]), np.ascontiguousarray(ob[:, 1])
    lib.ms_pointpush_step(
        batch,
        _f32_ptr(agx),
        _f32_ptr(agy),
        _f32_ptr(obx),
        _f32_ptr(oby),
        _f32_ptr(ax),
        _f32_ptr(ay),
    )
    exp_ag, exp_ob = pointpush.step(ag, ob, actions)
    np.testing.assert_allclose(np.stack([agx, agy], -1), exp_ag, atol=1e-6)
    np.testing.assert_allclose(np.stack([obx, oby], -1), exp_ob, atol=1e-6)
    lib.ms_pointpush_render(
        batch,
        _f32_ptr(agx),
        _f32_ptr(agy),
        _f32_ptr(obx),
        _f32_ptr(oby),
        48,
        _f32_ptr(frames),
    )
    np.testing.assert_array_equal(
        frames[..., None], pointpush.render(exp_ag, exp_ob)
    )

    # reacher
    th, _ = reacher.reset(rng, batch)
    t1, t2 = np.ascontiguousarray(th[:, 0]), np.ascontiguousarray(th[:, 1])
    lib.ms_reacher_step(
        batch, _f32_ptr(t1), _f32_ptr(t2), _f32_ptr(ax), _f32_ptr(ay)
    )
    exp_th = reacher.step(th, actions)
    np.testing.assert_allclose(np.stack([t1, t2], -1), exp_th, atol=1e-5)
    lib.ms_reacher_render(
        batch, _f32_ptr(t1), _f32_ptr(t2), 48, _f32_ptr(frames)
    )
    np.testing.assert_array_equal(frames[..., None], reacher.render(exp_th))

    # reacher_morph
    th, lengths, _ = reacher_morph.reset(rng, batch, 0.15, 0.27)
    t1, t2 = np.ascontiguousarray(th[:, 0]), np.ascontiguousarray(th[:, 1])
    l1 = np.ascontiguousarray(lengths[:, 0])
    l2 = np.ascontiguousarray(lengths[:, 1])
    lib.ms_reacher_morph_step(
        batch, _f32_ptr(t1), _f32_ptr(t2), _f32_ptr(ax), _f32_ptr(ay)
    )
    exp_th = reacher_morph.step(th, actions)
    np.testing.assert_allclose(np.stack([t1, t2], -1), exp_th, atol=1e-5)
    lib.ms_reacher_morph_render(
        batch,
        _f32_ptr(t1),
        _f32_ptr(t2),
        _f32_ptr(l1),
        _f32_ptr(l2),
        48,
        _f32_ptr(frames),
    )
    np.testing.assert_array_equal(
        frames[..., None], reacher_morph.render(exp_th, lengths)
    )

    # reacher_obs
    th, _, obs = reacher_obs.reset(rng, batch)
    t1, t2 = np.ascontiguousarray(th[:, 0]), np.ascontiguousarray(th[:, 1])
    ox = np.ascontiguousarray(obs[:, 0])
    oy = np.ascontiguousarray(obs[:, 1])
    or_ = np.ascontiguousarray(obs[:, 2])
    lib.ms_reacher_obs_step(
        batch,
        _f32_ptr(t1),
        _f32_ptr(t2),
        _f32_ptr(ax),
        _f32_ptr(ay),
        _f32_ptr(ox),
        _f32_ptr(oy),
        _f32_ptr(or_),
    )
    exp_th = reacher_obs.step_obs(th, actions, obs)
    np.testing.assert_allclose(np.stack([t1, t2], -1), exp_th, atol=1e-5)
    lib.ms_reacher_obs_render(
        batch,
        _f32_ptr(t1),
        _f32_ptr(t2),
        _f32_ptr(ox),
        _f32_ptr(oy),
        _f32_ptr(or_),
        48,
        _f32_ptr(frames),
    )
    np.testing.assert_array_equal(
        frames[..., None], reacher_obs.render(exp_th, obs)
    )

    # armpush
    th, ob, _ = armpush.reset(rng, batch)
    t1, t2 = np.ascontiguousarray(th[:, 0]), np.ascontiguousarray(th[:, 1])
    obx, oby = np.ascontiguousarray(ob[:, 0]), np.ascontiguousarray(ob[:, 1])
    lib.ms_armpush_step(
        batch,
        _f32_ptr(t1),
        _f32_ptr(t2),
        _f32_ptr(obx),
        _f32_ptr(oby),
        _f32_ptr(ax),
        _f32_ptr(ay),
    )
    exp_th, exp_ob = armpush.step(th, ob, actions)
    np.testing.assert_allclose(np.stack([t1, t2], -1), exp_th, atol=1e-5)
    np.testing.assert_allclose(np.stack([obx, oby], -1), exp_ob, atol=1e-5)
    lib.ms_armpush_render(
        batch,
        _f32_ptr(t1),
        _f32_ptr(t2),
        _f32_ptr(obx),
        _f32_ptr(oby),
        48,
        _f32_ptr(frames),
    )
    np.testing.assert_array_equal(
        frames[..., None], armpush.render(exp_th, exp_ob)
    )


def test_reset_and_rollout_parity() -> None:
    lib = _load_lib()
    batch = 32
    horizon = 16
    rng = np.random.default_rng(99)
    actions = rng.uniform(-1.0, 1.0, (horizon, batch, 2)).astype(np.float32)
    ax = np.ascontiguousarray(actions[:, :, 0])
    ay = np.ascontiguousarray(actions[:, :, 1])

    # pointmaze reset invariants + rollout parity
    x = np.empty(batch, np.float32)
    y = np.empty(batch, np.float32)
    gx = np.empty(batch, np.float32)
    gy = np.empty(batch, np.float32)
    lib.ms_pointmaze_reset(
        42, batch, _f32_ptr(x), _f32_ptr(y), _f32_ptr(gx), _f32_ptr(gy)
    )
    start = np.stack([x, y], -1)
    goal = np.stack([gx, gy], -1)
    assert not pointmaze.in_wall(start).any()
    assert not pointmaze.in_wall(goal).any()
    assert (np.linalg.norm(start - goal, axis=-1) >= 0.4 - 1e-6).all()

    traj_x = np.empty((horizon, batch), np.float32)
    traj_y = np.empty((horizon, batch), np.float32)
    lib.ms_pointmaze_rollout(
        horizon,
        batch,
        _f32_ptr(x),
        _f32_ptr(y),
        _f32_ptr(ax),
        _f32_ptr(ay),
        _f32_ptr(traj_x),
        _f32_ptr(traj_y),
    )
    pos = start.copy()
    exp_traj = []
    for t in range(horizon):
        pos = pointmaze.step(pos, actions[t])
        exp_traj.append(pos)
    np.testing.assert_allclose(
        np.stack([traj_x, traj_y], -1), np.stack(exp_traj), atol=1e-5
    )

    # reacher reset invariants + rollout parity
    t1 = np.empty(batch, np.float32)
    t2 = np.empty(batch, np.float32)
    lib.ms_reacher_reset(
        42, batch, _f32_ptr(t1), _f32_ptr(t2), _f32_ptr(gx), _f32_ptr(gy)
    )
    start_th = np.stack([t1, t2], -1)
    goal = np.stack([gx, gy], -1)
    assert (
        np.linalg.norm(reacher.tip(start_th) - goal, axis=-1) > 0.2 - 1e-5
    ).all()

    traj_t1 = np.empty((horizon, batch), np.float32)
    traj_t2 = np.empty((horizon, batch), np.float32)
    lib.ms_reacher_rollout(
        horizon,
        batch,
        _f32_ptr(t1),
        _f32_ptr(t2),
        _f32_ptr(ax),
        _f32_ptr(ay),
        _f32_ptr(traj_t1),
        _f32_ptr(traj_t2),
    )
    th = start_th.copy()
    exp_th = []
    for t in range(horizon):
        th = reacher.step(th, actions[t])
        exp_th.append(th)
    np.testing.assert_allclose(
        np.stack([traj_t1, traj_t2], -1), np.stack(exp_th), atol=1e-4
    )
