"""2-link ArmPush backed by the Rust core."""

from matterstate_envs import _rs

IMG: int = _rs.IMG


def step(
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Advance arm angles and pushed object without mutating inputs."""
    return _rs.armpush_step(t1, t2, ox, oy, a1, a2)


def rollout(
    horizon: int,
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Run a ``horizon``-step batched rollout."""
    return _rs.armpush_rollout(horizon, t1, t2, ox, oy, a1, a2)


def render(
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
) -> list[float]:
    """Render ``(B, 48, 48)`` flat float32 frames."""
    return _rs.armpush_render(t1, t2, ox, oy)
