"""PointPush — a 2D contact/pushing task: move an OBJECT to a goal by pushing
it with a point agent.

The north-star step toward manipulation: control is INDIRECT (the agent only
moves the object on contact) and the
goal is on the OBJECT, not the agent — so the planner must position the
agent behind the object and push it toward
the goal. Contact model: the agent moves under velocity control; when it
overlaps the object, the object is shoved
to sit exactly at contact distance on the far side (and the agent is
prevented from penetrating a wall-pinned
object). Irreversible-ish (object can be pinned in a corner) → needs
planning.

Pure vectorized numpy; render shows agent (bright) + object (mid) blobs so a
conv encoder can localize both.
"""

import numpy as np

DT = 0.05
R_AGENT = 0.035
R_OBJ = 0.06
R_CONTACT = R_AGENT + R_OBJ
GOAL_THRESH = 0.07
IMG = 48


def step(agent, obj, action):  # agent,obj,action:(B,2) -> next agent,obj
    a = np.clip(agent + np.clip(action, -1.0, 1.0) * DT, 0.0, 1.0)
    d = a - obj
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    contact = dist < R_CONTACT
    obj_new = np.where(
        contact, np.clip(a - d / (dist + 1e-9) * R_CONTACT, 0.0, 1.0), obj
    )  # shoved to far side
    d2 = a - obj_new
    dist2 = np.linalg.norm(d2, axis=-1, keepdims=True)
    penetrate = contact & (
        dist2 < R_CONTACT
    )  # object hit a bound -> push the agent back instead
    a = np.where(penetrate, obj_new + d2 / (dist2 + 1e-9) * R_CONTACT, a)
    return np.clip(a, 0.0, 1.0), obj_new


def solved(obj, goal):
    return np.linalg.norm(obj - goal, axis=-1) < GOAL_THRESH


# agent, object, goal — object/goal far enough apart to need real pushing
def reset(rng, n):
    obj = rng.uniform(0.15, 0.85, (n, 2))
    goal = rng.uniform(0.15, 0.85, (n, 2))
    close = np.linalg.norm(obj - goal, axis=-1) < 0.35
    while close.any():
        goal[close] = rng.uniform(0.15, 0.85, (int(close.sum()), 2))
        close = np.linalg.norm(obj - goal, axis=-1) < 0.35
    agent = rng.uniform(0.1, 0.9, (n, 2))
    return (
        agent.astype(np.float32),
        obj.astype(np.float32),
        goal.astype(np.float32),
    )


# paint a disc of intensity val at pos onto (B,img,img)
def _blob(frame, pos, r, val):
    img = frame.shape[1]
    yy, xx = np.meshgrid(np.arange(img), np.arange(img), indexing="ij")
    px, py = pos[:, 0] * img, pos[:, 1] * img
    d2 = (xx[None] - px[:, None, None]) ** 2 + (
        yy[None] - py[:, None, None]
    ) ** 2
    return np.where(d2 <= (r * img) ** 2, val, frame)


# (B,2),(B,2) -> (B,img,img,1): object (0.55) under agent (1.0)
def render(agent, obj, img=48):
    frame = np.zeros((len(agent), img, img), np.float32)
    frame = _blob(frame, obj, R_OBJ, 0.55)
    frame = _blob(frame, agent, R_AGENT, 1.0)
    return frame[..., None]


# scripted pusher (for solvability check): get behind object, push to goal
def oracle_action(agent, obj, goal):
    to_goal = goal - obj
    tg = to_goal / (np.linalg.norm(to_goal, axis=-1, keepdims=True) + 1e-9)
    behind = obj - tg * (
        R_CONTACT + 0.02
    )  # the spot behind the object, opposite the goal
    d_behind = behind - agent
    db = np.linalg.norm(d_behind, axis=-1, keepdims=True)
    aligned = (
        db < 0.06
    )  # close to the push spot -> push toward goal, else go to the push spot
    return np.where(aligned, tg, d_behind / (db + 1e-9))
