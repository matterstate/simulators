"""PushT: 2D rigid-body T-block pushing with Rust physics and 96x96 RGB."""

import numpy as np
from numpy.typing import NDArray

from matterstate_envs import _rs

IMG: int = 96


def reset(n: int, seed: int = 0) -> NDArray[np.float32]:
    """Sample ``n`` initial PushT states of shape ``(n, 7)`` float32."""
    return _rs.pusht_reset(n, seed)


def coverage(bx: float, by: float, ba: float) -> float:
    """Return goal overlap coverage in ``[0, 1]`` for ``(bx, by, ba)``."""
    return _rs.pusht_coverage(bx, by, ba)


def step(
    state: NDArray[np.float32],
    tx: float,
    ty: float,
) -> tuple[NDArray[np.float32], float]:
    """Advance a single PushT state ``(7,)`` toward target ``(tx, ty)``."""
    return _rs.pusht_step(state, tx, ty)


def step_batch(
    states: NDArray[np.float32],
    tx: NDArray[np.float32],
    ty: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Advance a batch of PushT states ``(n, 7)`` toward ``(tx, ty)``."""
    return _rs.pusht_step_batch(states, tx, ty)


def rollout(
    horizon: int,
    states: NDArray[np.float32],
    tx: NDArray[np.float32],
    ty: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Run a ``horizon``-step PushT rollout for a batch of states."""
    return _rs.pusht_rollout(horizon, states, tx, ty)


def render_rgb(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    ba: float,
) -> NDArray[np.uint8]:
    """Render a single PushT state to a ``(96, 96, 3)`` uint8 RGB image."""
    return _rs.pusht_render_rgb(ax, ay, bx, by, ba)


def render_rgb_batch(states: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Render a batch of PushT states ``(n, 7)`` to ``(n, 96, 96, 3)`` RGB."""
    return _rs.pusht_render_rgb_batch(states)
