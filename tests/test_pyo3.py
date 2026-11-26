"""Tests for the zero-dependency PyO3 Python bindings across all 11 envs."""

import math

from matterstate_envs import (
    armpush,
    ogbench_maze,
    pointmaze,
    pointmaze_proc,
    pointpush,
    pusht,
    reacher,
    reacher_morph,
    reacher_obs,
    sokoban,
    sudoku,
)


def test_pointmaze_and_unchanged_inputs() -> None:
    x, y, gx, gy = pointmaze.reset(8, seed=42)
    assert len(x) == len(y) == len(gx) == len(gy) == 8
    x_orig, y_orig = list(x), list(y)
    ax, ay = [0.5] * 8, [-0.5] * 8
    nx, ny = pointmaze.step(x, y, ax, ay)
    assert x == x_orig and y == y_orig
    assert len(nx) == len(ny) == 8
    assert not any(pointmaze.in_wall(nx, ny))

    tx, ty = pointmaze.rollout(4, x, y, [0.1] * 32, [0.1] * 32)
    assert len(tx) == len(ty) == 32


def test_pointmaze_proc_and_pointpush() -> None:
    walls = [False] * (2 * 36)
    nx, ny = pointmaze_proc.step(
        [0.5, 0.5], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5], walls
    )
    assert len(nx) == len(ny) == 2
    frames = pointmaze_proc.render(nx, ny, walls)
    assert len(frames) == 2 * 48 * 48

    ax, ay, ox, oy, gx, gy = pointpush.reset(4, seed=7)
    assert len(gx) == len(gy) == 4
    nax, nay, nox, noy = pointpush.step(ax, ay, ox, oy, [0.2] * 4, [0.1] * 4)
    assert len(nax) == len(nay) == len(nox) == len(noy) == 4
    pframes = pointpush.render(nax, nay, nox, noy)
    assert len(pframes) == 4 * 48 * 48


def test_reacher_family_and_armpush() -> None:
    t1, t2, gx, gy = reacher.reset(4, seed=3)
    assert len(gx) == len(gy) == 4
    nt1, nt2 = reacher.step(t1, t2, [0.3] * 4, [-0.3] * 4)
    assert all(abs(v) <= math.pi + 1e-5 for v in nt1 + nt2)
    assert len(reacher.render(nt1, nt2)) == 4 * 48 * 48

    assert (
        len(reacher_morph.render(nt1, nt2, [0.25] * 4, [0.2] * 4))
        == 4 * 48 * 48
    )

    ot1, ot2 = reacher_obs.step(
        nt1, nt2, [0.1] * 4, [0.1] * 4, [0.8] * 4, [0.8] * 4, [0.05] * 4
    )
    assert len(ot1) == len(ot2) == 4
    assert (
        len(reacher_obs.render(ot1, ot2, [0.8] * 4, [0.8] * 4, [0.05] * 4))
        == 4 * 48 * 48
    )

    at1, at2, aox, aoy = armpush.step(
        nt1, nt2, [0.6] * 4, [0.6] * 4, [0.1] * 4, [0.1] * 4
    )
    assert len(at1) == len(at2) == len(aox) == len(aoy) == 4
    assert len(armpush.render(at1, at2, aox, aoy)) == 4 * 48 * 48


def test_sokoban_pusht_ogbench_and_sudoku() -> None:
    board = [sokoban.FLOOR] * 100
    for i in range(10):
        board[i] = sokoban.WALL
        board[90 + i] = sokoban.WALL
        board[i * 10] = sokoban.WALL
        board[i * 10 + 9] = sokoban.WALL
    board[55] = sokoban.PLR
    nxt = sokoban.step(board, 0)
    assert nxt[45] == sokoban.PLR
    assert sokoban.solved(nxt)

    states = pusht.reset(4, seed=7)
    assert len(states) == 4 * 7
    covs = pusht.coverage(states)
    assert len(covs) == 4 and all(0.0 <= c <= 1.0 for c in covs)
    s0_next, c0 = pusht.step(states[:7], 256.0, 256.0)
    assert len(s0_next) == 7 and 0.0 <= c0 <= 1.0
    assert len(pusht.render_rgb(s0_next)) == 96 * 96 * 3
    next_batch, bcov = pusht.step_batch(states, [256.0] * 4, [256.0] * 4)
    assert len(next_batch) == 4 * 7 and len(bcov) == 4
    assert len(pusht.render_rgb_batch(next_batch, size=64)) == 4 * 64 * 64 * 3
    traj = pusht.rollout(3, states, [256.0] * 12, [256.0] * 12)
    assert len(traj) == 3 * 4 * 7

    mx, my, mvx, mvy = ogbench_maze.step_medium(
        [6.0], [6.0], [0.0], [0.0], [1.0], [0.0]
    )
    assert mx[0] > 6.0 and len(my) == len(mvx) == len(mvy) == 1
    lx, _, _, _ = ogbench_maze.step_large(
        [6.0], [6.0], [0.0], [0.0], [1.0], [0.0]
    )
    assert lx[0] > 6.0

    puz, sol = sudoku.generate(4, 2, 0.5, seed=7, unique_only=True)
    assert len(puz) == len(sol) == 2 * 16
    assert all(1 <= v <= 4 for v in sol)
