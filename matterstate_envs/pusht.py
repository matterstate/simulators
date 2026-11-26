"""PushT: 2D rigid-body T-block pushing with Rust physics and RGB rendering."""

import numpy as np
from numpy.typing import NDArray

from matterstate_envs import _rs

IMG: int = 96


def reset(n: int, seed: int = 0) -> NDArray[np.float32]:
    """Sample ``n`` initial PushT states of shape ``(n, 7)`` float32."""
    return _rs.pusht_reset(seed, n)


def coverage(states: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return goal overlap coverage in ``[0, 1]`` for ``(n, 7)`` states."""
    return _rs.pusht_coverage(states)


def step(
    state: NDArray[np.float32],
    tx: float,
    ty: float,
) -> float:
    """Advance ``state`` ``(7,)`` in place toward ``(tx, ty)``."""
    return _rs.pusht_step(state, tx, ty)


def step_batch(
    states: NDArray[np.float32],
    tx: NDArray[np.float32],
    ty: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Advance ``states`` ``(n, 7)`` in place and return coverage ``(n,)``."""
    return _rs.pusht_step_batch(states, tx, ty)


def rollout(
    horizon: int,
    states: NDArray[np.float32],
    tx: NDArray[np.float32],
    ty: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Run a ``horizon``-step PushT rollout and return ``(horizon, n, 7)``."""
    return _rs.pusht_rollout(horizon, states, tx, ty)


def render_rgb(
    state: NDArray[np.float32],
    size: int = IMG,
) -> NDArray[np.uint8]:
    """Render a single PushT state ``(7,)`` to ``(size, size, 3)`` RGB."""
    return _rs.pusht_render_rgb(state, size)


def render_rgb_batch(
    states: NDArray[np.float32],
    size: int = IMG,
) -> NDArray[np.uint8]:
    """Render ``states`` ``(n, 7)`` to ``(n, size, size, 3)`` RGB."""
    return _rs.pusht_render_rgb_batch(states, size)
