"""Property-based tests (Hegel) for zero-dependency Rust simulator bindings."""

import math

import hegel
from hegel import generators as gs

from matterstate_envs import pointmaze, pointpush, reacher, sokoban


@hegel.test
def test_sokoban_preserves_player_count_and_immutability(
    tc: hegel.TestCase,
) -> None:
    """Sokoban step never mutates input and preserves player count."""
    pr = tc.draw(gs.integers(min_value=1, max_value=8))
    pc = tc.draw(gs.integers(min_value=1, max_value=8))
    br = tc.draw(gs.integers(min_value=1, max_value=8))
    bc = tc.draw(gs.integers(min_value=1, max_value=8))
    action = tc.draw(gs.integers(min_value=0, max_value=3))

    board = [sokoban.FLOOR] * 100
    for i in range(10):
        board[i] = sokoban.WALL
        board[90 + i] = sokoban.WALL
        board[i * 10] = sokoban.WALL
        board[i * 10 + 9] = sokoban.WALL
    board[pr * 10 + pc] = sokoban.PLR
    if (br, bc) != (pr, pc):
        board[br * 10 + bc] = sokoban.BOX

    orig = list(board)
    nxt = sokoban.step(board, action)
    assert board == orig
    players = sum(1 for c in nxt if c in (sokoban.PLR, sokoban.PLR_T))
    assert players == 1


@hegel.test
def test_pointmaze_bounds_walls_and_immutability(tc: hegel.TestCase) -> None:
    """PointMaze stays in [0, 1]^2, never enters a wall, and doesn't mutate."""
    x = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    y = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ax = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))
    ay = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))

    was_free = not pointmaze.in_wall([x], [y])[0]
    xs, ys = [x], [y]
    nx, ny = pointmaze.step(xs, ys, [ax], [ay])
    assert xs == [x] and ys == [y]
    assert 0.0 <= nx[0] <= 1.0 and 0.0 <= ny[0] <= 1.0
    if was_free:
        assert not pointmaze.in_wall(nx, ny)[0]


@hegel.test
def test_pointpush_and_reacher_bounds(tc: hegel.TestCase) -> None:
    """PointPush stays in [0, 1]^2 and Reacher wraps angles to [-pi, pi]."""
    ax = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ay = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ox = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    oy = tc.draw(gs.floats(min_value=0.0, max_value=1.0))
    ux = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))
    uy = tc.draw(gs.floats(min_value=-5.0, max_value=5.0))
    t1 = tc.draw(gs.floats(min_value=-math.pi, max_value=math.pi))
    t2 = tc.draw(gs.floats(min_value=-math.pi, max_value=math.pi))

    nax, nay, nox, noy = pointpush.step([ax], [ay], [ox], [oy], [ux], [uy])
    assert 0.0 <= nax[0] <= 1.0 and 0.0 <= nay[0] <= 1.0
    assert 0.0 <= nox[0] <= 1.0 and 0.0 <= noy[0] <= 1.0

    nt1, nt2 = reacher.step([t1], [t2], [ux], [uy])
    assert abs(nt1[0]) <= math.pi + 1e-5 and abs(nt2[0]) <= math.pi + 1e-5
