"""2D PointMaze backed by the Rust AVX2+FMA core."""

from matterstate_envs import _rs


def reset(
    batch: int,
    seed: int = 0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Sample ``batch`` valid ``(x, y, gx, gy)`` states."""
    return _rs.pointmaze_reset(seed, batch)


def in_wall(x: list[float], y: list[float]) -> list[bool]:
    """Return wall-membership booleans for ``(x, y)`` coordinates."""
    return _rs.pointmaze_in_wall(x, y)


def step(
    x: list[float],
    y: list[float],
    ax: list[float],
    ay: list[float],
) -> tuple[list[float], list[float]]:
    """Advance ``(x, y)`` by ``(ax, ay)`` without mutating inputs."""
    return _rs.pointmaze_step(x, y, ax, ay)


def rollout(
    horizon: int,
    x: list[float],
    y: list[float],
    ax: list[float],
    ay: list[float],
) -> tuple[list[float], list[float]]:
    """Run a ``horizon``-step batched rollout."""
    return _rs.pointmaze_rollout(horizon, x, y, ax, ay)
