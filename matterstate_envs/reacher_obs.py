"""Reacher with OBSTACLES — a 2-link arm whose end-effector must reach a goal
while routing its links AROUND a circular
obstacle. The step from reaching (near-reactive) to genuine PLANNING: greedy
"move the tip toward the goal" gets the arm
stuck against the obstacle, so reaching the goal requires a non-trivial PATH
(back up, go around) — amortized THINKING
becomes load-bearing. Collision blocks the move (the arm can't pass
through), like a wall. Obstacle is rendered into the
frame so the conv encoder sees it (obstacle-conditioned dynamics, as in the
maze-generalization WM).

Reuses reacher fk/step/joint-velocity control. Pure vectorized numpy.
"""

import numpy as np
from numpy.typing import NDArray

try:
    from matterstate_envs import _rs
except ImportError:
    _rs = None

from matterstate_envs.reacher import (
    BASE,
    GOAL_THRESH,
    L1,
    L2,
    LINK_W,
    _blob,
    _seg,
    fk,
    step,
    tip,
)

# obstacle radius (image-normalized); placed in the arm's workspace so
# straight-line tip paths cross it
OBS_R = 0.09
MAX_SAMPLE_TRIES = 10_000


# (B,2)x3 -> (B,) min distance from point c to segment p0->p1
def _seg_circle_dist(p0, p1, c):
    seg = p1 - p0
    t = np.clip(((c - p0) * seg).sum(-1) / ((seg**2).sum(-1) + 1e-9), 0.0, 1.0)[
        :, None
    ]
    return np.linalg.norm(c - (p0 + t * seg), axis=-1)


# (B,) min distance from either link to the obstacle CENTER (collide if < r)
def link_obstacle_dist(
    theta: NDArray[np.float32], obs: NDArray[np.float32]
) -> NDArray[np.float32]:
    elbow, ee = fk(theta)
    b = np.broadcast_to(BASE, elbow.shape)
    c = obs[:, :2]
    return np.minimum(
        _seg_circle_dist(b, elbow, c), _seg_circle_dist(elbow, ee, c)
    )


# obs:(B,3)=[cx,cy,r] -> (B,) bool: either link enters the obstacle disk
def collides(
    theta: NDArray[np.float32], obs: NDArray[np.float32]
) -> NDArray[np.bool_]:
    return link_obstacle_dist(theta, obs) < obs[:, 2]


# joint-velocity step, but a colliding move is BLOCKED (arm stays put)
def step_obs(
    theta: NDArray[np.float32],
    action: NDArray[np.float32],
    obs: NDArray[np.float32],
) -> NDArray[np.float32]:
    if (
        _rs is not None
        and theta.ndim == 2
        and theta.shape[1] == 2
        and obs.ndim == 2
        and obs.shape[1] == 3
    ):
        t1 = np.ascontiguousarray(theta[:, 0], dtype=np.float32)
        t2 = np.ascontiguousarray(theta[:, 1], dtype=np.float32)
        a1 = np.ascontiguousarray(action[:, 0], dtype=np.float32)
        a2 = np.ascontiguousarray(action[:, 1], dtype=np.float32)
        ox = np.ascontiguousarray(obs[:, 0], dtype=np.float32)
        oy = np.ascontiguousarray(obs[:, 1], dtype=np.float32)
        or_ = np.ascontiguousarray(obs[:, 2], dtype=np.float32)
        _rs.reacher_obs_step_in_place(t1, t2, a1, a2, ox, oy, or_)
        return np.stack([t1, t2], axis=-1)
    proposed = step(theta, action)
    return np.where(collides(proposed, obs)[:, None], theta, proposed).astype(
        np.float32
    )


def solved(
    theta: NDArray[np.float32], goal: NDArray[np.float32]
) -> NDArray[np.bool_]:
    return np.linalg.norm(tip(theta) - goal, axis=-1) < GOAL_THRESH


# theta (non-colliding start), goal (reachable, outside obstacle), obs — set
# up so routing is needed
def reset(
    rng,
    n,
):
    obs = np.zeros((n, 3), np.float32)
    obs[:, 2] = OBS_R
    theta = np.zeros((n, 2), np.float32)
    goal = np.zeros((n, 2), np.float32)
    todo = np.ones(n, bool)
    for _ in range(MAX_SAMPLE_TRIES):
        if not todo.any():
            return theta, goal, obs
        k = int(todo.sum())
        idx = np.where(todo)[0]
        th = rng.uniform(-np.pi, np.pi, (k, 2)).astype(np.float32)
        ang = rng.uniform(-np.pi, np.pi, k)
        r = rng.uniform(abs(L1 - L2) + 0.06, L1 + L2 - 0.04, k)
        g = (
            BASE + np.stack([np.cos(ang), np.sin(ang)], -1) * r[:, None]
        ).astype(np.float32)
        # obstacle near the midpoint of the start-tip -> goal straight path, so
        # greedy reaching runs into it
        mid = 0.5 * (tip(th) + g)
        jit = rng.uniform(-0.05, 0.05, (k, 2))
        c = np.clip(mid + jit, 0.12, 0.88).astype(np.float32)
        o = np.concatenate([c, np.full((k, 1), OBS_R, np.float32)], 1)
        far = np.linalg.norm(tip(th) - g, axis=-1) > 0.22
        startok = ~collides(th, o)
        goalok = (
            np.linalg.norm(g - c, axis=-1) > OBS_R + 0.03
        )  # goal itself outside the obstacle
        ok = far & startok & goalok
        sel = idx[ok]
        theta[sel], goal[sel], obs[sel] = th[ok], g[ok], o[ok]
        todo[sel] = False
    raise RuntimeError("reacher_obs.reset exceeded MAX_SAMPLE_TRIES")


# Jacobian-transpose tip->goal descent (the REACTIVE baseline that gets stuck)
def greedy_action(
    theta,
    goal,
):
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t12), np.cos(t12)
    err = tip(theta) - goal
    g1 = (-L1 * s1 - L2 * s12) * err[:, 0] + (L1 * c1 + L2 * c12) * err[:, 1]
    g2 = (-L2 * s12) * err[:, 0] + (L2 * c12) * err[:, 1]
    g = np.stack([g1, g2], -1)
    return (-g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)).astype(
        np.float32
    )


# joint action that drives the tip along desired velocity dv via
# Jacobian-transpose
def _jt_action(
    theta,
    dv,
):
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t12), np.cos(t12)
    g1 = (-L1 * s1 - L2 * s12) * dv[:, 0] + (L1 * c1 + L2 * c12) * dv[:, 1]
    g2 = (-L2 * s12) * dv[:, 0] + (L2 * c12) * dv[:, 1]
    g = np.stack([g1, g2], -1)
    return (g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)).astype(
        np.float32
    )


# DETERMINISTIC tangent-router: skirt the obstacle on the goal-ward side, else
# go straight
def router_action(
    theta,
    goal,
    obs,
):
    tp = tip(theta)
    c, r = obs[:, :2], obs[:, 2:3]
    to_goal = goal - tp
    blocked = (_seg_circle_dist(tp, goal, c) < (r[:, 0] + 0.04))[:, None]
    from_c = tp - c
    perp = np.stack(
        [-from_c[:, 1], from_c[:, 0]], -1
    )  # +90 deg; flip to whichever skirts toward the goal
    perp = np.where((perp * to_goal).sum(-1, keepdims=True) >= 0, perp, -perp)

    # noqa: E731
    def norm(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-09)

    skirt = norm(perp) + 0.4 * norm(
        from_c
    )  # tangential + slight outward so it clears, not hugs, the obstacle
    return _jt_action(theta, np.where(blocked, skirt, to_goal))


# obstacle (0.3) UNDER the arm (upper 0.5, fore 0.8, tip 1.0); conv reads both
def render(
    theta,
    obs,
    img=48,
):
    if (
        _rs is not None
        and theta.dtype == np.float32
        and obs.dtype == np.float32
    ):
        return _rs.reacher_obs_render(
            np.ascontiguousarray(theta[:, 0]),
            np.ascontiguousarray(theta[:, 1]),
            np.ascontiguousarray(obs[:, 0]),
            np.ascontiguousarray(obs[:, 1]),
            np.ascontiguousarray(obs[:, 2]),
            img,
        )[..., None]
    elbow, ee = fk(theta)
    b = np.broadcast_to(BASE, elbow.shape)
    frame = np.zeros((len(theta), img, img), np.float32)
    frame = _blob(frame, obs[:, :2], obs[:, 2], 0.3)
    frame = _seg(frame, b, elbow, LINK_W, 0.5)
    frame = _seg(frame, elbow, ee, LINK_W, 0.8)
    frame = _blob(frame, ee, 0.03, 1.0)
    return frame[..., None]
