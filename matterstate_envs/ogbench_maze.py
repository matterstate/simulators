"""OGBench 2D point-maze (medium and large grids) backed by the Rust core."""

import numpy as np
from numpy.typing import NDArray

from matterstate_envs import _rs


def step_medium_in_place(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    vx: NDArray[np.float32],
    vy: NDArray[np.float32],
    fx: NDArray[np.float32],
    fy: NDArray[np.float32],
) -> None:
    """Advance a batch of medium-maze states in place."""
    _rs.ogbench_maze_step_in_place(x, y, vx, vy, fx, fy)


def step_large_in_place(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    vx: NDArray[np.float32],
    vy: NDArray[np.float32],
    fx: NDArray[np.float32],
    fy: NDArray[np.float32],
) -> None:
    """Advance a batch of large-maze states in place."""
    _rs.ogbench_maze_large_step_in_place(x, y, vx, vy, fx, fy)
