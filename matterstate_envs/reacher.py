"""Reacher — a planar 2-link arm whose END-EFFECTOR must reach a goal,
controlled by JOINT velocities.

The north-star step from point-mass nav/push to real robot control: the
agent commands the two JOINTS (theta1, theta2),
not the end-effector directly, so reaching a goal requires inverting the
kinematic chain through the world model. There
is kinematic redundancy (many joint configs reach the same point) and the
action->effect map is nonlinear (a joint
nudge moves the tip by an amount that depends on the whole arm pose) — the
defining features of articulated control.
Goal is on the END-EFFECTOR position (like PointPush's goal-on-object):
decode the tip from the latent, plan joint
actions to bring it to the goal.

Pure vectorized numpy; render draws the two links + tip + goal so a conv
encoder can read the arm pose from pixels.
"""

import numpy as np

try:
    import matterstate_envs_rs as _rs
except ImportError:
    _rs = None

L1, L2 = (
    0.22,
    0.2,
    # link lengths (image-normalized); reach annulus = [|L1-L2|, L1+L2] = [0.02,
    # 0.42]
)
BASE = np.array([0.5, 0.5], np.float32)  # shoulder at image center
STEP = 0.15  # rad per unit action per env step
GOAL_THRESH = 0.05
IMG = 48
LINK_W = 0.018  # render half-thickness of a link
MAX_SAMPLE_TRIES = 10_000


def fk(theta):  # theta:(B,2) -> elbow:(B,2), tip:(B,2)  forward kinematics
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    elbow = BASE + L1 * np.stack([np.cos(t1), np.sin(t1)], -1)
    tip = elbow + L2 * np.stack([np.cos(t12), np.sin(t12)], -1)
    return elbow.astype(np.float32), tip.astype(np.float32)


def tip(theta):
    return fk(theta)[1]


# theta,action:(B,2) -> next theta (joint-velocity control, angles wrap to
# [-pi,pi])
def step(
    theta,
    action,
):
    th = theta + np.clip(action, -1.0, 1.0) * STEP
    return ((th + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def solved(theta, goal):
    return np.linalg.norm(tip(theta) - goal, axis=-1) < GOAL_THRESH


# theta, goal — goal a reachable point, start tip far enough to need real
# motion
def reset(
    rng,
    n,
):
    theta = rng.uniform(-np.pi, np.pi, (n, 2)).astype(np.float32)
    ang = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(abs(L1 - L2) + 0.06, L1 + L2 - 0.04, n)
    goal = (
        BASE + np.stack([np.cos(ang), np.sin(ang)], -1) * r[:, None]
    ).astype(np.float32)
    far = np.linalg.norm(tip(theta) - goal, axis=-1) > 0.2
    for _ in range(MAX_SAMPLE_TRIES):
        if far.all():
            return theta, goal
        k = int((~far).sum())
        theta[~far] = rng.uniform(-np.pi, np.pi, (k, 2))
        far = np.linalg.norm(tip(theta) - goal, axis=-1) > 0.2
    raise RuntimeError("reacher.reset exceeded MAX_SAMPLE_TRIES")


# paint pixels within w of the segment p0->p1 onto (B,img,img)
def _seg(frame, p0, p1, w, val):
    img = frame.shape[1]
    yy, xx = np.meshgrid(np.arange(img), np.arange(img), indexing="ij")
    P = np.stack([xx, yy], -1)[None] / img  # (1,img,img,2) in [0,1]
    seg = (p1 - p0)[:, None, None, :]
    t = ((P - p0[:, None, None, :]) * seg).sum(-1) / ((seg**2).sum(-1) + 1e-9)
    t = np.clip(t, 0.0, 1.0)[..., None]
    closest = p0[:, None, None, :] + t * seg
    d2 = ((P - closest) ** 2).sum(-1)
    return np.where(d2 <= w**2, val, frame)


def _blob(frame, pos, r, val):
    img = frame.shape[1]
    yy, xx = np.meshgrid(np.arange(img), np.arange(img), indexing="ij")
    px, py = pos[:, 0] * img, pos[:, 1] * img
    d2 = (xx[None] - px[:, None, None]) ** 2 + (
        yy[None] - py[:, None, None]
    ) ** 2
    radius = np.asarray(r)
    if radius.ndim == 1:
        radius = radius[:, None, None]
    return np.where(d2 <= (radius * img) ** 2, val, frame)


# (B,2) -> (B,img,img,1): upper arm (0.5), forearm (0.8), tip (1.0)
def render(theta, img=48):
    if _rs is not None and theta.dtype == np.float32:
        t1 = np.ascontiguousarray(theta[:, 0])
        t2 = np.ascontiguousarray(theta[:, 1])
        return _rs.reacher_render(t1, t2, img)[..., None]
    elbow, ee = fk(theta)
    b = np.broadcast_to(BASE, elbow.shape)
    frame = np.zeros((len(theta), img, img), np.float32)
    frame = _seg(frame, b, elbow, LINK_W, 0.5)
    frame = _seg(frame, elbow, ee, LINK_W, 0.8)
    frame = _blob(frame, ee, 0.03, 1.0)
    return frame[..., None]


# Jacobian-transpose descent on ||tip - goal||^2 (solvability check)
def oracle_action(theta, goal):
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t12), np.cos(t12)
    err = tip(theta) - goal  # (B,2)
    # columns of J^T applied to err: g_i = (d tip / d theta_i) . err
    g1 = (-L1 * s1 - L2 * s12) * err[:, 0] + (L1 * c1 + L2 * c12) * err[:, 1]
    g2 = (-L2 * s12) * err[:, 0] + (L2 * c12) * err[:, 1]
    g = np.stack([g1, g2], -1)
    return (-g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)).astype(
        np.float32
    )
