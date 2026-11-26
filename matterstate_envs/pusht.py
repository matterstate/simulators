"""PushT: 2D rigid-body T-block pushing with Rust physics and RGB rendering."""

from matterstate_envs import _rs

IMG: int = 96


def reset(batch: int, seed: int = 0) -> list[float]:
    """Sample ``batch`` initial PushT states (flat length ``batch * 7``)."""
    return _rs.pusht_reset(seed, batch)


def coverage(states: list[float]) -> list[float]:
    """Return goal overlap coverage in ``[0, 1]`` for flat ``(B, 7)`` states."""
    return _rs.pusht_coverage(states)


def step(
    state: list[float],
    tx: float,
    ty: float,
) -> tuple[list[float], float]:
    """Advance ``state`` (length 7) toward ``(tx, ty)`` without mutation."""
    return _rs.pusht_step(state, tx, ty)


def step_batch(
    states: list[float],
    tx: list[float],
    ty: list[float],
) -> tuple[list[float], list[float]]:
    """Advance flat ``(B, 7)`` states and return ``(next_states, coverage)``."""
    return _rs.pusht_step_batch(states, tx, ty)


def rollout(
    horizon: int,
    states: list[float],
    tx: list[float],
    ty: list[float],
) -> list[float]:
    """Run a ``horizon``-step PushT rollout (flat ``horizon * B * 7``)."""
    return _rs.pusht_rollout(horizon, states, tx, ty)


def render_rgb(state: list[float], size: int = IMG) -> list[int]:
    """Render a single PushT state to flat ``size * size * 3`` RGB bytes."""
    return _rs.pusht_render_rgb(state, size)


def render_rgb_batch(states: list[float], size: int = IMG) -> list[int]:
    """Render flat ``(B, 7)`` states to flat ``B * size * size * 3`` RGB."""
    return _rs.pusht_render_rgb_batch(states, size)
