"""PointMaze — a 2D continuous goal-conditioned navigation task
(DINO-WM-style), CEM-MPC's home turf.

A point agent moves in the unit square under velocity-control actions,
around walls, toward a goal position.
Unlike Sokoban (discrete, irreversible, combinatorial → flat goal-distance
landscape that breaks CEM), this is
continuous with a SMOOTH distance-to-goal — exactly where JEPA world-model +
CEM-MPC planning is proven to work
(V-JEPA 2-AC, DINO-WM beat Dreamer/TD-MPC2). We use it to validate the
proper JEPA recipe end-to-end.

Vectorized numpy: step/solved/in_wall act on (B,2) batches so CEM can roll a
whole population at once.
"""

import numpy as np

# one central wall rising from the floor with a gap at the top → straight line
# is often blocked, must route over.
WALLS = np.array(
    [
        [0.45, 0.00, 0.55, 0.62],  # vertical wall, gap above y=0.62
    ]
)
DT = 0.06
GOAL_THRESH = 0.06


def in_wall(pos):  # pos:(...,2) -> bool(...)
    x, y = pos[..., 0], pos[..., 1]
    inside = np.zeros(pos.shape[:-1], dtype=bool)
    for x0, y0, x1, y1 in WALLS:
        inside |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return inside


# pos:(B,2), action:(B,2) in [-1,1] -> next pos:(B,2); blocked moves stay put
def step(
    pos,
    action,
):
    nxt = np.clip(pos + np.clip(action, -1.0, 1.0) * DT, 0.0, 1.0)
    blocked = in_wall(nxt)
    return np.where(blocked[..., None], pos, nxt)


def solved(pos, goal):  # (B,2),(B,2) -> bool(B)
    return np.linalg.norm(pos - goal, axis=-1) < GOAL_THRESH


def sample_free(rng, n):  # uniform points not inside any wall
    pts = rng.uniform(0, 1, (n, 2))
    bad = in_wall(pts)
    while bad.any():
        pts[bad] = rng.uniform(0, 1, (int(bad.sum()), 2))
        bad = in_wall(pts)
    return pts


# (start, goal) both in free space, and far enough apart to need real
# navigation
def reset(
    rng,
    n,
):
    start = sample_free(rng, n)
    goal = sample_free(rng, n)
    close = np.linalg.norm(start - goal, axis=-1) < 0.4
    while close.any():
        goal[close] = sample_free(rng, int(close.sum()))
        close = np.linalg.norm(start - goal, axis=-1) < 0.4
    return start, goal
