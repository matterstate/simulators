"""Procedural 2D PointMaze backed by the Rust core."""

from matterstate_envs import _rs

IMG: int = _rs.IMG


def step(
    x: list[float],
    y: list[float],
    ax: list[float],
    ay: list[float],
    walls: list[bool],
) -> tuple[list[float], list[float]]:
    """Advance ``(x, y)`` against per-maze ``walls``."""
    return _rs.pointmaze_proc_step(x, y, ax, ay, walls)


def render(
    x: list[float],
    y: list[float],
    walls: list[bool],
    r_px: float = 1.5,
) -> list[float]:
    """Render ``(B, 48, 48)`` flat float32 frames."""
    return _rs.pointmaze_proc_render(x, y, walls, r_px)
