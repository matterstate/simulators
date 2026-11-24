"""Exact transitions and success against gym-sokoban, not RGB/reset/rewards."""

import numpy as np
import pytest
from gym_sokoban.envs import SokobanEnv

from matterstate_envs import sokoban as simulator

pytestmark = pytest.mark.reference


def reference_for(grid: np.ndarray) -> SokobanEnv:
    board = grid.reshape(10, 10)
    boxes = np.isin(board, [simulator.BOX, simulator.BOX_T])
    reference = SokobanEnv(num_boxes=int(boxes.sum()), reset=False)
    targets = np.isin(
        board, [simulator.TARGET, simulator.BOX_T, simulator.PLR_T]
    )
    reference.room_fixed = np.where(
        board == simulator.WALL, 0, np.where(targets, 2, 1)
    )
    # gym-sokoban reverses BOX/BOX_T and stores player-on-target in terrain.
    reference.room_state = np.array([0, 1, 2, 4, 3, 5, 5])[board]
    reference.player_position = np.argwhere(
        np.isin(board, [simulator.PLR, simulator.PLR_T])
    )[0]
    reference.num_env_steps = 0
    reference.boxes_on_target = int((board == simulator.BOX_T).sum())
    return reference


def canonical_state(reference: SokobanEnv) -> np.ndarray:
    grid = np.array([0, 1, 2, 4, 3, 5], dtype=np.int8)[reference.room_state]
    player_target = (reference.room_state == 5) & (reference.room_fixed == 2)
    grid[player_target] = simulator.PLR_T
    return grid.ravel()


def check_trajectory(grid: np.ndarray, actions: np.ndarray) -> None:
    reference = reference_for(grid)
    try:
        np.testing.assert_array_equal(
            canonical_state(reference), grid, strict=True
        )
        for action in actions:
            original = grid.copy()
            expected = simulator.step(grid, int(action))
            # The external engine's public step includes pushing and walking;
            # actions 1..4 correspond to our 0..3. No renderer is substituted.
            _, _, done, _ = reference.step(
                int(action) + 1, observation_mode="tiny_rgb_array"
            )
            np.testing.assert_array_equal(
                canonical_state(reference), expected, strict=True
            )
            np.testing.assert_array_equal(grid, original, strict=True)
            assert bool(done) == simulator.solved(expected)
            grid = expected
            if done:
                break
    finally:
        reference.close()


@pytest.mark.parametrize("seed", range(32))
def test_seeded_transition_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    board = np.full((10, 10), simulator.FLOOR, np.int8)
    board[[0, -1], :] = simulator.WALL
    board[:, [0, -1]] = simulator.WALL
    board[1:-1, 1:-1][rng.random((8, 8)) < 0.15] = simulator.WALL
    available = np.flatnonzero(board.ravel() == simulator.FLOOR)
    targets = rng.choice(available, 4, replace=False)
    board.ravel()[targets] = simulator.TARGET
    occupants = rng.choice(available, 5, replace=False)
    for index in occupants[:4]:
        board.ravel()[index] = (
            simulator.BOX_T if index in targets else simulator.BOX
        )
    board.ravel()[occupants[4]] = (
        simulator.PLR_T if occupants[4] in targets else simulator.PLR
    )
    check_trajectory(board.ravel(), rng.integers(0, 4, 32))


@pytest.mark.parametrize(
    "cells",
    [
        [simulator.PLR, simulator.BOX, simulator.TARGET, simulator.FLOOR],
        [simulator.PLR, simulator.BOX_T, simulator.FLOOR, simulator.FLOOR],
        [simulator.PLR_T, simulator.BOX, simulator.FLOOR, simulator.FLOOR],
        [simulator.PLR, simulator.BOX, simulator.WALL, simulator.TARGET],
    ],
)
def test_contact_and_target_parity(cells: list[int]) -> None:
    board = np.full((10, 10), simulator.FLOOR, np.int8)
    board[[0, -1], :] = simulator.WALL
    board[:, [0, -1]] = simulator.WALL
    board[4, 3:7] = cells
    check_trajectory(board.ravel(), np.array([3, 2, 0, 1, 3, 3, 2]))
