"""2-link planar Reacher backed by the Rust core."""

from matterstate_envs import _rs

IMG: int = _rs.IMG


def reset(
    batch: int,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Sample ``batch`` ``(t1, t2, gx, gy)`` states."""
    return _rs.reacher_reset(seed, batch)


def step(
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float]]:
    """Advance joint angles ``(t1, t2)`` without mutating inputs."""
    return _rs.reacher_step(t1, t2, a1, a2)


def rollout(
    horizon: int,
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float]]:
    """Run a ``horizon``-step batched rollout."""
    return _rs.reacher_rollout(horizon, t1, t2, a1, a2)


def render(t1: list[float], t2: list[float]) -> list[float]:
    """Render ``(B, 48, 48)`` flat float32 frames."""
    return _rs.reacher_render(t1, t2)
