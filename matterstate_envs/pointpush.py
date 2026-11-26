"""2D PointPush backed by the Rust core."""

from matterstate_envs import _rs

IMG: int = _rs.IMG


def reset(
    batch: int,
    seed: int = 0,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Sample ``batch`` ``(ax, ay, ox, oy, gx, gy)`` states."""
    return _rs.pointpush_reset(seed, batch)


def step(
    ax: list[float],
    ay: list[float],
    ox: list[float],
    oy: list[float],
    ux: list[float],
    uy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Advance agent and object positions without mutating inputs."""
    return _rs.pointpush_step(ax, ay, ox, oy, ux, uy)


def rollout(
    horizon: int,
    ax: list[float],
    ay: list[float],
    ox: list[float],
    oy: list[float],
    ux: list[float],
    uy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Run a ``horizon``-step batched rollout."""
    return _rs.pointpush_rollout(horizon, ax, ay, ox, oy, ux, uy)


def render(
    ax: list[float],
    ay: list[float],
    ox: list[float],
    oy: list[float],
) -> list[float]:
    """Render ``(B, 48, 48)`` flat float32 frames."""
    return _rs.pointpush_render(ax, ay, ox, oy)
