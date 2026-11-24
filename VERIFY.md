# Verification

## Contract map

Last-reviewed upstream source: `matterstate/latent-planner@df2f5b2`.
The initial simulator commit preserves its kernel function bodies; only package
imports and research-only `__main__` demos were removed.

| Property | Code/check | Bounds and assumptions |
|---|---|---|
| Extraction preserves reset, trajectory, render, and success bytes | `tests/test_regression.py:capture_trajectories`, reviewed Syrupy snapshot | Seed 2026, batch 8, 16 continuous actions; 8 Sokoban actions; NumPy 2.3.5, x86_64 Linux |
| Sokoban matches independent engine transitions and success | `tests/reference/test_sokoban.py:check_trajectory` | gym-sokoban 0.0.6, 32 seeds, at most 32 moves each; four directed cases of at most 7 moves; 10x10 boards with wall borders |
| Simulator step does not mutate Sokoban input | `check_trajectory` | Checked after every compared transition |
| No external engine or research imports in runtime | `.ast-grep/rules/runtime-imports.yml` | Python import statements; tested matching/nonmatching fixtures |
| No import-path injection | `.ast-grep/rules/python-root-imports.yml` | Tested explicit `sys.path` mutation forms |
| External OGBench dependency environment remains loadable | `check` import smoke | Python 3.11, independent lockfile; no dataset download or simulation |

Sokoban equality concerns canonical integer grid bytes and success, not RGB,
reward, reset generation, or time-limit wrappers. The reference is initialized
with known board state rather than a random reset. Its constructor, public
`step`, reward computation, termination computation, and tiny RGB renderer are
unmodified. The four directed cases exercise target entry/exit, player-on-target,
and blocked pushes. Seeded boards have equal box and target counts.

## Run and observe

```sh
nix develop
uv sync --frozen --group reference
check
```

Exit zero means all configured contracts passed. Pytest reports snapshot and
external-reference failures; ast-grep reports forbidden source locations.
The external legacy Gym package emits deprecation warnings; it is deliberately
isolated and pinned as a reference, not imported by runtime kernels.
No server, credentials, network dataset, GPU, or training is required for tests
after dependencies have been installed. Separate checkouts can run independently;
do not update snapshots concurrently in the same checkout.

## Failure sensitivity and updates

The initial verification injects a stationary Sokoban step in a separate Python
process and requires the directed external parity tests to fail. Discard that
mutation and rerun the unchanged gate before accepting work. Future transition
or adapter changes must repeat an appropriate behavioral mutation, not merely
check for syntax/import failure.

Before changing a kernel, inspect source history since the reviewed snapshot
revision. Compare changed functions with their reference contract. Use
`uv run --group reference python -m pytest --snapshot-update` only after reviewing
why every byte difference is expected; updating a snapshot is not proof of
external parity. Numerical backend changes require a separately stated precision
contract and performance measurements.
