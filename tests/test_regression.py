"""Byte-level snapshots captured from latent-planner before extraction."""

import hashlib

import numpy as np
from syrupy.assertion import SnapshotAssertion

from matterstate_envs import (
    armpush,
    pointmaze,
    pointmaze_proc,
    pointpush,
    reacher,
    reacher_morph,
    reacher_obs,
    sokoban,
)


def capture_trajectories() -> dict[str, np.ndarray]:
    arrays = {}
    rng = np.random.default_rng(2026)
    actions = rng.uniform(-2, 2, (16, 8, 2)).astype(np.float32)

    position, goal = pointmaze.reset(rng, 8)
    arrays["pointmaze.reset"] = np.stack([position, goal])
    trajectory = []
    for action in actions:
        position = pointmaze.step(position, action)
        trajectory.append(position)
    arrays["pointmaze.trajectory"] = np.stack(trajectory)
    arrays["pointmaze.success"] = pointmaze.solved(position, goal)

    theta, goal = reacher.reset(rng, 8)
    arrays["reacher.reset"] = np.stack([theta, goal])
    trajectory = []
    for action in actions:
        theta = reacher.step(theta, action)
        trajectory.append(theta)
    arrays["reacher.trajectory"] = np.stack(trajectory)
    arrays["reacher.render"] = reacher.render(theta)
    arrays["reacher.success"] = reacher.solved(theta, goal)

    theta, obj, goal = armpush.reset(rng, 8)
    arrays["armpush.reset"] = np.stack([theta, obj, goal])
    trajectory = []
    for action in actions:
        theta, obj = armpush.step(theta, obj, action)
        trajectory.append(np.stack([theta, obj]))
    arrays["armpush.trajectory"] = np.stack(trajectory)
    arrays["armpush.render"] = armpush.render(theta, obj)
    arrays["armpush.success"] = armpush.solved(obj, goal)

    agent, obj, goal = pointpush.reset(rng, 8)
    arrays["pointpush.reset"] = np.stack([agent, obj, goal])
    trajectory = []
    for action in actions:
        agent, obj = pointpush.step(agent, obj, action)
        trajectory.append(np.stack([agent, obj]))
    arrays["pointpush.trajectory"] = np.stack(trajectory)
    arrays["pointpush.render"] = pointpush.render(agent, obj)
    arrays["pointpush.success"] = pointpush.solved(obj, goal)

    theta, goal, obstacle = reacher_obs.reset(rng, 8)
    arrays["reacher_obs.reset"] = np.concatenate([theta, goal, obstacle], 1)
    trajectory = []
    for action in actions:
        theta = reacher_obs.step_obs(theta, action, obstacle)
        trajectory.append(theta)
    arrays["reacher_obs.trajectory"] = np.stack(trajectory)
    arrays["reacher_obs.render"] = reacher_obs.render(theta, obstacle)
    arrays["reacher_obs.success"] = reacher_obs.solved(theta, goal)

    theta, lengths, goal = reacher_morph.reset(rng, 8, 0.15, 0.27)
    arrays["reacher_morph.reset"] = np.stack([theta, lengths, goal])
    trajectory = []
    for action in actions:
        theta = reacher_morph.step(theta, action)
        trajectory.append(theta)
    arrays["reacher_morph.trajectory"] = np.stack(trajectory)
    arrays["reacher_morph.render"] = reacher_morph.render(theta, lengths)
    arrays["reacher_morph.success"] = reacher_morph.solved(theta, lengths, goal)

    walls = pointmaze_proc.sample_mazes(rng, 8)
    position, goal = pointmaze_proc.reset(rng, walls)
    arrays["pointmaze_proc.walls"] = walls
    arrays["pointmaze_proc.reset"] = np.stack([position, goal])
    trajectory = []
    for action in actions:
        position = pointmaze_proc.step(position, action, walls)
        trajectory.append(position)
    arrays["pointmaze_proc.trajectory"] = np.stack(trajectory)
    arrays["pointmaze_proc.render"] = pointmaze_proc.render(position, walls)
    arrays["pointmaze_proc.success"] = pointmaze_proc.solved(position, goal)

    grid = np.full((10, 10), sokoban.FLOOR, np.int8)
    grid[[0, -1], :] = sokoban.WALL
    grid[:, [0, -1]] = sokoban.WALL
    grid[4, 3:6] = [sokoban.PLR, sokoban.BOX, sokoban.TARGET]
    grid = grid.ravel()
    trajectory = [grid]
    for action in [3, 2, 0, 3, 1, 2, 3, 3]:
        grid = sokoban.step(grid, action)
        trajectory.append(grid)
    arrays["sokoban.trajectory"] = np.stack(trajectory)
    arrays["sokoban.success"] = np.asarray(sokoban.solved(grid))
    return arrays


def fingerprint(arrays: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }


def test_extraction_snapshot(snapshot: SnapshotAssertion) -> None:
    assert fingerprint(capture_trajectories()) == snapshot
