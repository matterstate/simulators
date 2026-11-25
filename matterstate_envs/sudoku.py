"""Parallel Sudoku puzzle generator backed by the Rust core."""

import numpy as np
from numpy.typing import NDArray

from matterstate_envs import _rs


def generate(
    n: int,
    num: int,
    clue_frac: float,
    seed: int = 0,
    unique_only: bool = False,
) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
    """Generate ``num`` ``n x n`` Sudoku puzzles and solutions."""
    return _rs.sudoku_generate(n, num, clue_frac, seed, unique_only)
