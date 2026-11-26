"""Type stubs for the compiled Rust extension module matterstate_envs._rs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

IMG: int

def sokoban_step(
    grid: NDArray[np.int8],
    action: int,
) -> NDArray[np.int8]: ...
def pointmaze_reset(
    seed: int,
    batch: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]: ...
def pointmaze_step_in_place(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
) -> None: ...
def pointmaze_rollout(
    horizon: int,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def pointmaze_proc_step_in_place(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
    walls: NDArray[np.bool_],
) -> None: ...
def pointmaze_proc_render(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    walls: NDArray[np.bool_],
    r_px: float,
) -> NDArray[np.float32]: ...
def pointpush_reset(
    seed: int,
    batch: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]: ...
def pointpush_step_in_place(
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    ux: NDArray[np.float32],
    uy: NDArray[np.float32],
) -> None: ...
def pointpush_rollout(
    horizon: int,
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    ux: NDArray[np.float32],
    uy: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]: ...
def pointpush_render(
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def reacher_reset(
    seed: int,
    batch: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]: ...
def reacher_step_in_place(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    a1: NDArray[np.float32],
    a2: NDArray[np.float32],
) -> None: ...
def reacher_rollout(
    horizon: int,
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    a1: NDArray[np.float32],
    a2: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def reacher_render(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def reacher_obs_step_in_place(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    a1: NDArray[np.float32],
    a2: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    or_: NDArray[np.float32],
) -> None: ...
def reacher_obs_rollout(
    horizon: int,
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    a1: NDArray[np.float32],
    a2: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    or_: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def reacher_obs_render(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    or_: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def reacher_morph_render(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    l1: NDArray[np.float32],
    l2: NDArray[np.float32],
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def armpush_step_in_place(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    a1: NDArray[np.float32],
    a2: NDArray[np.float32],
) -> None: ...
def armpush_rollout(
    horizon: int,
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    a1: NDArray[np.float32],
    a2: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]: ...
def armpush_render(
    t1: NDArray[np.float32],
    t2: NDArray[np.float32],
    ox: NDArray[np.float32],
    oy: NDArray[np.float32],
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def pusht_reset(seed: int, batch: int) -> NDArray[np.float32]: ...
def pusht_coverage(states: NDArray[np.float32]) -> NDArray[np.float32]: ...
def pusht_step(
    state: NDArray[np.float32],
    action_x: float,
    action_y: float,
) -> float: ...
def pusht_step_batch(
    states: NDArray[np.float32],
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def pusht_rollout(
    horizon: int,
    states: NDArray[np.float32],
    ax: NDArray[np.float32],
    ay: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def pusht_render_rgb(
    state: NDArray[np.float32],
    size: int,
) -> NDArray[np.uint8]: ...
def pusht_render_rgb_batch(
    states: NDArray[np.float32],
    size: int,
) -> NDArray[np.uint8]: ...
def ogbench_maze_step_in_place(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    vx: NDArray[np.float32],
    vy: NDArray[np.float32],
    fx: NDArray[np.float32],
    fy: NDArray[np.float32],
) -> None: ...
def ogbench_maze_large_step_in_place(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    vx: NDArray[np.float32],
    vy: NDArray[np.float32],
    fx: NDArray[np.float32],
    fy: NDArray[np.float32],
) -> None: ...
def sudoku_generate(
    n: int,
    num: int,
    clue_frac: float,
    seed: int = 0,
    unique_only: bool = False,
) -> tuple[NDArray[np.int8], NDArray[np.int8]]: ...
