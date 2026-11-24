"""ArmPush — a 2-link arm that PUSHES an object to a goal, commanding only the
JOINTS.

The capstone robot-control task: combine the Reacher's articulated control
(you command θ1/θ2, the tip moves through the
kinematic chain) with PointPush's indirect contact (the tip only moves the
object on contact, and the goal is on the
OBJECT). So the planner must (a) drive the arm tip behind the object via
joint motion it does not directly control, and
(b) push the object to the goal — articulated manipulation, the real robot
task. Object + goal live inside the arm's
reachable annulus so the task is solvable.

Pure vectorized numpy; render draws the arm links + tip + object so a conv
encoder can read the whole scene. Reuses
reacher FK/step and PointPush's contact model.
"""

import numpy as np

from matterstate_envs import reacher as RE

R_TIP = 0.03  # the arm tip acts as the pusher
R_OBJ = 0.06
R_CONTACT = R_TIP + R_OBJ
GOAL_THRESH = 0.07
IMG = 48
MAX_SAMPLE_TRIES = 10_000
_REACH = (
    abs(RE.L1 - RE.L2) + 0.04,
    RE.L1 + RE.L2 - 0.04,
)  # object/goal annulus from base (kept reachable)


# PointPush contact: tip shoves the object to the far side on overlap
def _contact_push(tip, obj):
    d = tip - obj
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    contact = dist < R_CONTACT
    return np.where(
        contact, np.clip(tip - d / (dist + 1e-9) * R_CONTACT, 0.0, 1.0), obj
    ).astype(np.float32)


# theta,action:(B,2), obj:(B,2) -> next theta, obj
def step(theta, obj, action):
    theta = RE.step(theta, action)
    tip = RE.tip(theta)
    return theta, _contact_push(tip, obj)


def solved(obj, goal):
    return np.linalg.norm(obj - goal, axis=-1) < GOAL_THRESH


def _annulus(rng, n):  # a point in the reachable ring around the base
    ang = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(*_REACH, n)
    return (
        RE.BASE + np.stack([np.cos(ang), np.sin(ang)], -1) * r[:, None]
    ).astype(np.float32)


# theta, object, goal — object & goal reachable, far enough apart to need real
# pushing
def reset(
    rng,
    n,
):
    theta = rng.uniform(-np.pi, np.pi, (n, 2)).astype(np.float32)
    obj = _annulus(rng, n)
    goal = _annulus(rng, n)
    close = np.linalg.norm(obj - goal, axis=-1) < 0.25
    for _ in range(MAX_SAMPLE_TRIES):
        if not close.any():
            return theta, obj, goal
        goal[close] = _annulus(rng, int(close.sum()))
        close = np.linalg.norm(obj - goal, axis=-1) < 0.25
    raise RuntimeError("armpush.reset exceeded MAX_SAMPLE_TRIES")


# (B,2),(B,2) -> (B,img,img,1): object (0.55) under arm (links 0.5/0.8, tip
# 1.0)
def render(
    theta,
    obj,
    img=48,
):
    elbow, tip = RE.fk(theta)
    b = np.broadcast_to(RE.BASE, elbow.shape)
    frame = np.zeros((len(theta), img, img), np.float32)
    frame = RE._blob(frame, obj, R_OBJ, 0.55)
    frame = RE._seg(frame, b, elbow, RE.LINK_W, 0.5)
    frame = RE._seg(frame, elbow, tip, RE.LINK_W, 0.8)
    frame = RE._blob(frame, tip, R_TIP, 1.0)
    return frame[..., None]


# drive the tip toward the push spot via Jacobian-transpose (solvability
# check)
def oracle_action(
    theta,
    obj,
    goal,
):
    tip = RE.tip(theta)
    to_goal = goal - obj
    tg = to_goal / (np.linalg.norm(to_goal, axis=-1, keepdims=True) + 1e-9)
    behind = obj - tg * (
        R_CONTACT + 0.01
    )  # the spot behind the object, opposite the goal
    d_behind = behind - tip
    db = np.linalg.norm(d_behind, axis=-1, keepdims=True)
    aligned = db < 0.05
    desired_tip_vel = np.where(
        aligned, tg, d_behind / (db + 1e-9)
    )  # push toward goal, else go behind the object
    t1, t12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t12), np.cos(t12)
    # joint action ∝ J^T · desired_tip_vel  (move the tip in the desired
    # direction through the joints)
    g1 = (-RE.L1 * s1 - RE.L2 * s12) * desired_tip_vel[:, 0] + (
        RE.L1 * c1 + RE.L2 * c12
    ) * desired_tip_vel[:, 1]
    g2 = (-RE.L2 * s12) * desired_tip_vel[:, 0] + (
        RE.L2 * c12
    ) * desired_tip_vel[:, 1]
    g = np.stack([g1, g2], -1)
    return (g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)).astype(
        np.float32
    )
