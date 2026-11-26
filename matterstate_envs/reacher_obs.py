"""2-link Reacher with circular obstacle backed by the Rust core."""

from matterstate_envs import _rs

IMG: int = _rs.IMG


def step(
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
    ox: list[float],
    oy: list[float],
    or_: list[float],
) -> tuple[list[float], list[float]]:
    """Advance joint angles with obstacle collision blocking."""
    return _rs.reacher_obs_step(t1, t2, a1, a2, ox, oy, or_)


def rollout(
    horizon: int,
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
    ox: list[float],
    oy: list[float],
    or_: list[float],
) -> tuple[list[float], list[float]]:
    """Run a ``horizon``-step batched rollout."""
    return _rs.reacher_obs_rollout(horizon, t1, t2, a1, a2, ox, oy, or_)


def render(
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
    or_: list[float],
) -> list[float]:
    """Render ``(B, 48, 48)`` flat float32 frames."""
    return _rs.reacher_obs_render(t1, t2, ox, oy, or_)
