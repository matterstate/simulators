"""Baseline throughput benchmarks for the NumPy simulator kernels."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np

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


def _time_op(
    fn: Callable[[], None],
    *,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    start_ns = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    elapsed_ns = time.perf_counter_ns() - start_ns
    total_items = batch_size * iters
    elapsed_s = elapsed_ns / 1e9
    return {
        "batch_size": batch_size,
        "iters": iters,
        "elapsed_s": elapsed_s,
        "items_per_sec": total_items / elapsed_s,
        "ns_per_item": elapsed_ns / total_items,
    }


def _make_sokoban_board() -> np.ndarray:
    grid = np.full((10, 10), sokoban.FLOOR, np.int8)
    grid[[0, -1], :] = sokoban.WALL
    grid[:, [0, -1]] = sokoban.WALL
    grid[4, 3:6] = [sokoban.PLR, sokoban.BOX, sokoban.TARGET]
    return grid.ravel()


def run_benchmarks(
    batch_size: int,
    step_iters: int,
    render_iters: int,
    warmup: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(2026)
    actions = rng.uniform(-1.0, 1.0, (batch_size, 2)).astype(np.float32)
    results: list[dict[str, object]] = []

    def record(
        kernel: str,
        op: str,
        fn: Callable[[], None],
        *,
        b: int,
        iters: int,
    ) -> None:
        stats = _time_op(fn, batch_size=b, warmup=warmup, iters=iters)
        results.append({"kernel": kernel, "op": op, **stats})

    # pointmaze
    pm_pos, _ = pointmaze.reset(rng, batch_size)
    record(
        "pointmaze",
        "step",
        lambda: pointmaze.step(pm_pos, actions),
        b=batch_size,
        iters=step_iters,
    )

    # pointmaze_proc
    pmp_walls = pointmaze_proc.sample_mazes(rng, batch_size)
    pmp_pos, _ = pointmaze_proc.reset(rng, pmp_walls)
    record(
        "pointmaze_proc",
        "step",
        lambda: pointmaze_proc.step(pmp_pos, actions, pmp_walls),
        b=batch_size,
        iters=step_iters,
    )
    record(
        "pointmaze_proc",
        "render",
        lambda: pointmaze_proc.render(pmp_pos, pmp_walls),
        b=batch_size,
        iters=render_iters,
    )

    # pointpush
    pp_agent, pp_obj, _ = pointpush.reset(rng, batch_size)
    record(
        "pointpush",
        "step",
        lambda: pointpush.step(pp_agent, pp_obj, actions),
        b=batch_size,
        iters=step_iters,
    )
    record(
        "pointpush",
        "render",
        lambda: pointpush.render(pp_agent, pp_obj),
        b=batch_size,
        iters=render_iters,
    )

    # reacher
    r_theta, _ = reacher.reset(rng, batch_size)
    record(
        "reacher",
        "step",
        lambda: reacher.step(r_theta, actions),
        b=batch_size,
        iters=step_iters,
    )
    record(
        "reacher",
        "render",
        lambda: reacher.render(r_theta),
        b=batch_size,
        iters=render_iters,
    )

    # reacher_morph
    rm_theta, rm_lengths, _ = reacher_morph.reset(rng, batch_size, 0.15, 0.27)
    record(
        "reacher_morph",
        "step",
        lambda: reacher_morph.step(rm_theta, actions),
        b=batch_size,
        iters=step_iters,
    )
    record(
        "reacher_morph",
        "render",
        lambda: reacher_morph.render(rm_theta, rm_lengths),
        b=batch_size,
        iters=render_iters,
    )

    # reacher_obs
    ro_theta, _, ro_obs = reacher_obs.reset(rng, batch_size)
    record(
        "reacher_obs",
        "step",
        lambda: reacher_obs.step_obs(ro_theta, actions, ro_obs),
        b=batch_size,
        iters=step_iters,
    )
    record(
        "reacher_obs",
        "render",
        lambda: reacher_obs.render(ro_theta, ro_obs),
        b=batch_size,
        iters=render_iters,
    )

    # armpush
    ap_theta, ap_obj, _ = armpush.reset(rng, batch_size)
    record(
        "armpush",
        "step",
        lambda: armpush.step(ap_theta, ap_obj, actions),
        b=batch_size,
        iters=step_iters,
    )
    record(
        "armpush",
        "render",
        lambda: armpush.render(ap_theta, ap_obj),
        b=batch_size,
        iters=render_iters,
    )

    # sokoban (single-grid API)
    sok_grid = _make_sokoban_board()
    record(
        "sokoban",
        "step",
        lambda: sokoban.step(sok_grid, 3),
        b=1,
        iters=step_iters,
    )

    return results


def _print_table(results: list[dict[str, object]]) -> None:
    header = (
        f"{'kernel':<15} {'op':<7} {'batch':>5} {'items/s':>14} {'ns/item':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['kernel']:<15} {row['op']:<7} {row['batch_size']:>5} "
            f"{row['items_per_sec']:>14,.0f} {row['ns_per_item']:>12,.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--step-iters", type=int, default=2000)
    parser.add_argument("--render-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_benchmarks(
        batch_size=args.batch,
        step_iters=args.step_iters,
        render_iters=args.render_iters,
        warmup=args.warmup,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
