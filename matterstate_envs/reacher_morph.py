"""Variable-MORPHOLOGY 2-link Reacher — link lengths L1,L2 vary PER EPISODE,
rendered into the frame.

The generalist test: instead of one fixed arm, a FAMILY of arms with
different link lengths. The morphology is drawn per
episode and DRAWN into the pixels (the conv encoder reads the arm's geometry
from the image), so one frozen WM + one
amortized recursive planner must control ANY arm in the family — and
zero-shot EXTRAPOLATE to link lengths outside the
training range. Reuses reacher's render primitives; all kinematics take a
per-batch L=(B,2)=[L1,L2].
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
    LINK_W,
    STEP,
    _blob,
    _seg,
)

L_TRAIN = (0.15, 0.27)  # link-length range seen in training
L_OOD = (0.28, 0.33)  # held-out longer arms (extrapolation)
MAX_SAMPLE_TRIES = 10_000


def sample_L(
    rng: np.random.Generator, n: int, lo: float, hi: float
) -> NDArray[np.float32]:
    return rng.uniform(lo, hi, (n, 2)).astype(np.float32)


def fk(
    theta: NDArray[np.float32], L: NDArray[np.float32]
) -> tuple[
    NDArray[np.float32], NDArray[np.float32]
]:  # theta:(B,2), L:(B,2) -> elbow, tip
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    elbow = BASE + L[:, 0:1] * np.stack([np.cos(t1), np.sin(t1)], -1)
    tip = elbow + L[:, 1:2] * np.stack([np.cos(t12), np.sin(t12)], -1)
    return elbow.astype(np.float32), tip.astype(np.float32)


def tip(
    theta: NDArray[np.float32], L: NDArray[np.float32]
) -> NDArray[np.float32]:
    return fk(theta, L)[1]


def step(
    theta: NDArray[np.float32], action: NDArray[np.float32]
) -> NDArray[np.float32]:  # joint-velocity control (morphology-independent)
    th = theta + np.clip(action, -1.0, 1.0) * STEP
    return ((th + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def solved(
    theta: NDArray[np.float32],
    L: NDArray[np.float32],
    goal: NDArray[np.float32],
) -> NDArray[np.bool_]:
    return np.linalg.norm(tip(theta, L) - goal, axis=-1) < GOAL_THRESH


# theta, L, goal — goal reachable for THIS env's L, start tip far enough to
# need motion
def reset(
    rng,
    n,
    lo,
    hi,
):
    L = sample_L(rng, n, lo, hi)
    theta = rng.uniform(-np.pi, np.pi, (n, 2)).astype(np.float32)
    ang = rng.uniform(-np.pi, np.pi, n)
    rmin, rmax = np.abs(L[:, 0] - L[:, 1]) + 0.06, L[:, 0] + L[:, 1] - 0.04
    r = rmin + rng.uniform(0, 1, n) * (rmax - rmin)
    goal = (
        BASE + np.stack([np.cos(ang), np.sin(ang)], -1) * r[:, None]
    ).astype(np.float32)
    far = np.linalg.norm(tip(theta, L) - goal, axis=-1) > 0.18
    for _ in range(MAX_SAMPLE_TRIES):
        if far.all():
            return theta, L, goal
        k = int((~far).sum())
        theta[~far] = rng.uniform(-np.pi, np.pi, (k, 2))
        far = np.linalg.norm(tip(theta, L) - goal, axis=-1) > 0.18
    raise RuntimeError("reacher_morph.reset exceeded MAX_SAMPLE_TRIES")


# (B,2),(B,2) -> (B,img,img,1): upper arm (0.5), forearm (0.8), tip (1.0)
def render(
    theta: NDArray[np.float32], L: NDArray[np.float32], img: int = 48
) -> NDArray[np.float32]:
    if _rs is not None and theta.dtype == np.float32 and L.dtype == np.float32:
        return _rs.reacher_morph_render(
            np.ascontiguousarray(theta[:, 0]),
            np.ascontiguousarray(theta[:, 1]),
            np.ascontiguousarray(L[:, 0]),
            np.ascontiguousarray(L[:, 1]),
            img,
        )[..., None]
    elbow, ee = fk(theta, L)
    b = np.broadcast_to(BASE, elbow.shape)
    frame = np.zeros((len(theta), img, img), np.float32)
    frame = _seg(frame, b, elbow, LINK_W, 0.5)
    frame = _seg(frame, elbow, ee, LINK_W, 0.8)
    frame = _blob(frame, ee, 0.03, 1.0)
    return frame[..., None]


# Jacobian-transpose descent with per-batch link lengths
def oracle_action(
    theta: NDArray[np.float32],
    L: NDArray[np.float32],
    goal: NDArray[np.float32],
) -> NDArray[np.float32]:
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t12), np.cos(t12)
    l1, l2 = L[:, 0], L[:, 1]
    err = tip(theta, L) - goal
    g1 = (-l1 * s1 - l2 * s12) * err[:, 0] + (l1 * c1 + l2 * c12) * err[:, 1]
    g2 = (-l2 * s12) * err[:, 0] + (l2 * c12) * err[:, 1]
    g = np.stack([g1, g2], -1)
    return (-g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)).astype(
        np.float32
    )
