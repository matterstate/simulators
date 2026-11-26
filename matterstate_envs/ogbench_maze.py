"""OGBench 2D point-maze (medium and large grids) backed by the Rust core."""

from matterstate_envs import _rs


def step_medium(
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    fx: list[float],
    fy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Advance a batch of medium-maze states without mutating inputs."""
    return _rs.ogbench_maze_step_medium(x, y, vx, vy, fx, fy)


def step_large(
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    fx: list[float],
    fy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Advance a batch of large-maze states without mutating inputs."""
    return _rs.ogbench_maze_step_large(x, y, vx, vy, fx, fy)
