"""Procedural PointMaze — random wall layouts per episode, for testing geometry
GENERALIZATION from pixels.

Unlike pointmaze.py (one fixed wall), each maze is a random G×G grid of wall
cells. Solvability is guaranteed by
construction: the free space is reduced to its single largest 4-connected
component, so any two free cells are
mutually reachable. The walls are rendered into the frame, so a pixel
encoder can read the geometry — the whole
point: train on some layouts, plan in UNSEEN ones (a state-only observation
can't, it never sees walls).

Pure vectorized numpy; each batch element carries its OWN maze
(walls:(B,G,G)). step/solved/reset/render act on
batches of distinct mazes at once.
"""

import numpy as np

try:
    import matterstate_envs_rs as _rs
except ImportError:
    _rs = None

G = 6  # maze grid cells per side
IMG = 48  # rendered frame size (CELL = IMG // G px per cell)
CELL = IMG // G
DT = 0.06
GOAL_THRESH = 0.06
MAX_SAMPLE_TRIES = 10_000


# free:(G,G) bool -> mask of the largest 4-connected free component
def _largest_cc(
    free,
):
    seen = np.zeros_like(free)
    best = np.zeros_like(free)
    for i in range(G):
        for j in range(G):
            if free[i, j] and not seen[i, j]:
                stack, comp = [(i, j)], []
                seen[i, j] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < G
                            and 0 <= nx < G
                            and free[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) > int(best.sum()):
                    best = np.zeros_like(free)
                    for y, x in comp:
                        best[y, x] = True
    return best


# -> walls:(n,G,G) bool (True = wall)
def sample_mazes(rng, n, wall_prob=0.22, min_free=16):
    walls = np.ones((n, G, G), bool)
    for k in range(n):
        for _ in range(MAX_SAMPLE_TRIES):
            comp = _largest_cc(rng.random((G, G)) >= wall_prob)
            if comp.sum() >= min_free:
                walls[k] = ~comp
                break
        else:
            raise RuntimeError(
                "pointmaze_proc.sample_mazes exceeded MAX_SAMPLE_TRIES"
            )
    return walls


def _cell(pos):  # pos:(B,2) in [0,1] -> (col,row) int cell
    return np.clip(np.floor(pos * G).astype(int), 0, G - 1)


def in_wall(pos, walls):  # pos:(B,2), walls:(B,G,G) -> bool(B)
    c = _cell(pos)
    return walls[np.arange(len(pos)), c[:, 1], c[:, 0]]


def step(pos, action, walls):  # blocked moves stay put (per-maze walls)
    nxt = np.clip(pos + np.clip(action, -1.0, 1.0) * DT, 0.0, 1.0)
    return np.where(in_wall(nxt, walls)[:, None], pos, nxt)


def solved(pos, goal):
    return np.linalg.norm(pos - goal, axis=-1) < GOAL_THRESH


# start,goal:(n,2) in free cells of each maze, far enough apart to need
# navigation
def reset(
    rng,
    walls,
):
    n = len(walls)
    start, goal = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
    for k in range(n):
        free = np.argwhere(~walls[k])  # (m,2) (row,col)
        if len(free) == 0:
            raise ValueError("maze has no free cells")
        for _ in range(MAX_SAMPLE_TRIES):
            (sr, sc), (gr, gc) = (
                free[rng.integers(len(free))],
                free[rng.integers(len(free))],
            )
            s = (np.array([sc, sr]) + rng.uniform(0.2, 0.8, 2)) / G
            g = (np.array([gc, gr]) + rng.uniform(0.2, 0.8, 2)) / G
            if np.linalg.norm(s - g) > 0.4:
                start[k], goal[k] = s, g
                break
        else:
            raise RuntimeError("pointmaze_proc.reset exceeded MAX_SAMPLE_TRIES")
    return start, goal


def _walls_img(walls):  # (B,G,G) -> (B,IMG,IMG) float, wall cell = 0.4
    return np.repeat(
        np.repeat(walls.astype(np.float32) * 0.4, CELL, 1), CELL, 2
    )


# (B,2),(B,G,G) -> (B,IMG,IMG,1): agent blob over the walls
def render(pos, walls, r=1.5):
    if _rs is not None and pos.dtype == np.float32 and walls.dtype == np.bool_:
        return _rs.pointmaze_proc_render(
            np.ascontiguousarray(pos[:, 0]),
            np.ascontiguousarray(pos[:, 1]),
            np.ascontiguousarray(walls),
            float(r),
        )[..., None]
    wimg = _walls_img(walls)
    yy, xx = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    px, py = pos[:, 0] * IMG, pos[:, 1] * IMG
    d2 = (xx[None] - px[:, None, None]) ** 2 + (
        yy[None] - py[:, None, None]
    ) ** 2
    return np.maximum(wimg, (d2 <= r * r).astype(np.float32))[..., None]


# (B,G,G) -> (B,IMG,IMG,1): walls only (the geometry context, no agent)
def render_walls(
    walls,
):
    return _walls_img(walls)[..., None]
