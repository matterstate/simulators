"""Deterministic 10x10 Sokoban backed by the Rust core."""

from matterstate_envs import _rs

WALL: int = 0
FLOOR: int = 1
TARGET: int = 2
BOX: int = 3
BOX_T: int = 4
PLR: int = 5
PLR_T: int = 6
H: int = 10
W: int = 10
N_ACT: int = 4


def step(grid: list[int], action: int) -> list[int]:
    """Apply one move to a flat 100-cell Sokoban grid."""
    return _rs.sokoban_step(grid, action)


def solved(grid: list[int]) -> bool:
    """Return True when no unplaced boxes remain on the grid."""
    return _rs.sokoban_solved(grid)
