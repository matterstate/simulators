"""Type stubs for the zero-dependency Rust extension matterstate_envs._rs."""

IMG: int

def sokoban_step(grid: list[int], action: int) -> list[int]: ...
def sokoban_solved(grid: list[int]) -> bool: ...
def pointmaze_reset(
    seed: int,
    batch: int,
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def pointmaze_in_wall(x: list[float], y: list[float]) -> list[bool]: ...
def pointmaze_step(
    x: list[float],
    y: list[float],
    ax: list[float],
    ay: list[float],
) -> tuple[list[float], list[float]]: ...
def pointmaze_rollout(
    horizon: int,
    x: list[float],
    y: list[float],
    ax: list[float],
    ay: list[float],
) -> tuple[list[float], list[float]]: ...
def pointmaze_proc_step(
    x: list[float],
    y: list[float],
    ax: list[float],
    ay: list[float],
    walls: list[bool],
) -> tuple[list[float], list[float]]: ...
def pointmaze_proc_render(
    x: list[float],
    y: list[float],
    walls: list[bool],
    r_px: float,
) -> list[float]: ...
def pointpush_reset(
    seed: int,
    batch: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]: ...
def pointpush_step(
    ax: list[float],
    ay: list[float],
    ox: list[float],
    oy: list[float],
    ux: list[float],
    uy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def pointpush_rollout(
    horizon: int,
    ax: list[float],
    ay: list[float],
    ox: list[float],
    oy: list[float],
    ux: list[float],
    uy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def pointpush_render(
    ax: list[float],
    ay: list[float],
    ox: list[float],
    oy: list[float],
) -> list[float]: ...
def reacher_reset(
    seed: int,
    batch: int,
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def reacher_step(
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float]]: ...
def reacher_rollout(
    horizon: int,
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float]]: ...
def reacher_render(t1: list[float], t2: list[float]) -> list[float]: ...
def reacher_morph_render(
    t1: list[float],
    t2: list[float],
    l1: list[float],
    l2: list[float],
) -> list[float]: ...
def reacher_obs_step(
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
    ox: list[float],
    oy: list[float],
    or_: list[float],
) -> tuple[list[float], list[float]]: ...
def reacher_obs_rollout(
    horizon: int,
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
    ox: list[float],
    oy: list[float],
    or_: list[float],
) -> tuple[list[float], list[float]]: ...
def reacher_obs_render(
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
    or_: list[float],
) -> list[float]: ...
def armpush_step(
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def armpush_rollout(
    horizon: int,
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def armpush_render(
    t1: list[float],
    t2: list[float],
    ox: list[float],
    oy: list[float],
) -> list[float]: ...
def pusht_reset(seed: int, batch: int) -> list[float]: ...
def pusht_coverage(states: list[float]) -> list[float]: ...
def pusht_step(
    state: list[float],
    action_x: float,
    action_y: float,
) -> tuple[list[float], float]: ...
def pusht_step_batch(
    states: list[float],
    ax: list[float],
    ay: list[float],
) -> tuple[list[float], list[float]]: ...
def pusht_rollout(
    horizon: int,
    states: list[float],
    ax: list[float],
    ay: list[float],
) -> list[float]: ...
def pusht_render_rgb(state: list[float], size: int) -> list[int]: ...
def pusht_render_rgb_batch(states: list[float], size: int) -> list[int]: ...
def ogbench_maze_step_medium(
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    fx: list[float],
    fy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def ogbench_maze_step_large(
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    fx: list[float],
    fy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]: ...
def sudoku_generate(
    n: int,
    num: int,
    clue_frac: float,
    seed: int = 0,
    unique_only: bool = False,
) -> tuple[list[int], list[int]]: ...
