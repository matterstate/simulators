# Matterstate simulators

NumPy reference simulator kernels extracted from `matterstate/latent-planner`
alongside a zero-dependency Rust core (`matterstate-envs`) with C-ABI (`cdylib`)
and PyO3 (`matterstate_envs_rs`) bindings. External engines are isolated as
test-only references. Python reference imports use `matterstate_envs` (NumPy
only); the Rust core provides SIMD-accelerated batched stepping, deterministic
PRNG resets, multi-step rollouts, and 48×48 rasterization.

## Use

### Python (NumPy reference)

```python
import numpy as np
from matterstate_envs import pointmaze

position, goal = pointmaze.reset(np.random.default_rng(0), 8)
position = pointmaze.step(position, np.zeros_like(position))
```

### Python (Rust PyO3 extension)

Build and install `matterstate_envs_rs` into `.venv`:

```sh
nix develop --command bash -c \
  'unset UV_PYTHON && VIRTUAL_ENV="$PWD/.venv" maturin develop --release --manifest-path crates/matterstate-envs-py/Cargo.toml'
```

```python
import numpy as np
import matterstate_envs_rs as rs

x, y, gx, gy = rs.pointmaze_reset(42, 64)
ax = np.zeros(16 * 64, np.float32)
ay = np.zeros(16 * 64, np.float32)
traj_x, traj_y = rs.pointmaze_rollout(16, x, y, ax, ay)
```

### Rust / C-ABI

The root crate (`Cargo.toml`, `src/`) has zero external dependencies and builds
both `rlib` and `cdylib` (`target/release/libmatterstate_envs.so`).

The package contains `pointmaze`, `pointmaze_proc`, `reacher`, `reacher_obs`,
`reacher_morph`, `pointpush`, `armpush`, and `sokoban`. Research collectors,
learned models, optimizers, and in-graph JAX renderers remain in latent-planner.
There are no import-path shims or dependencies back into the research repo.

## Develop and verify

```sh
nix develop
uv sync --frozen --group reference
check
```

`check` is the same gate used in CI: Nix/Python/Rust formatting, Ruff, Clippy,
Rust unit tests, release `cdylib` build, structural import/loop policies and
their fixtures, regression snapshots, Rust-vs-NumPy C-ABI parity, PyO3 parity,
external Sokoban parity, and the isolated OGBench import check. Neovim Conform
uses the same Ruff/nixfmt executables when launched from this dev shell.

The default development group contains pytest and Syrupy. Runtime installation
of the package does not install either development or reference groups.
To run only extraction regression tests without external engines:

```sh
uv run --frozen python -m pytest tests/test_regression.py
```

## CI runner access

CI runs on the Matterstate `atlas-trusted` group with labels
`self-hosted, linux, x64, atlas, matterstate, nixos`. Nix is already present.
Only `main` pushes and manual dispatches of `main` run on this shared host;
PRs require an isolated runner. Superseded checks are cancelled.

The org runner group must permit this repository and selected workflow
`matterstate/simulators/.github/workflows/check.yml@refs/heads/main`, in addition
to its existing allowlist. This requires org-admin or runner-group permissions;
do not replace existing entries or authorize arbitrary PR code. Queued jobs
are not evidence of passing parity checks.

## Parity status

| Surface | Reference | Evidence |
|---|---|---|
| All eight extracted NumPy kernels | latent-planner `df2f5b2` | Exact dtype, shape, and byte-hash snapshots of 30 seeded reset/trajectory/render/success arrays |
| All eight Rust kernels (C-ABI + PyO3) | NumPy reference kernels | Byte-exact Sokoban (32×32 moves), pixel-exact 48×48 renders, `atol=1e-6..1e-4` continuous steps & rollouts |
| Sokoban grid transitions and success | gym-sokoban 0.0.6 | Exact equality over 32 seeded bounded trajectories and four directed contact/target cases |
| Custom navigation, arm, and pushing tasks | None equivalent yet | Custom tasks, not benchmark replacements |
| Gymnasium, robotics, and PushT | Locked `reference` dependency group | Reference engines available; no replacement parity claim |
| OGBench | Isolated Python 3.11 reference project | Locked dependencies/import check only; no replacement parity claim |

Sokoban tests explicitly translate tile and action encodings, construct equal
initial states, and execute the external engine's real public `step`. They do
not substitute its movement or renderer. Reward, RGB equality, reset/level
distributions, time-limit behavior, and performance are outside that test's
contract. See `VERIFY.md` for bounds and failure-sensitivity checks.

Floating-point extraction snapshots target the locked NumPy 2.3.5 environment
on x86_64 Linux. They do not establish cross-platform bit equality. Future
benchmark replacements must define exact or tolerance-based contracts before
being used instead of the reference; tests must not silently relax equality.

OGBench uses Python 3.11 because its `labmaze` dependency has no usable wheel
for this project's Python 3.13 environment. Its dependencies and lockfile live
under `tests/reference/ogbench/`; they are never runtime package dependencies.

## Measured performance (x86_64 Linux, 24-core AVX2+FMA)

Run `uv run --frozen python benchmarks/bench_numpy.py`, `cargo bench --bench sokoban`,
`cargo bench --bench continuous`, and
`RUSTFLAGS="-C target-cpu=native" cargo bench --bench scaling`.

| Kernel | Op | Batch | NumPy items/s | Rust items/s | Speedup |
|---|---|---:|---:|---:|---:|
| `sokoban` | `step` | 1 | 24,435 | 287,014,016 | ~11,700× |
| `sokoban` | `step` | 64 | — | 530,216,961 | ~21,700× vs B=1 |
| `pointmaze` | `step` (AVX2+FMA) | 1,024 | 2,912,704 (B=64) | 5,339,556,094 | ~1,830× |
| `pointmaze_proc` | `step` | 64 | 2,796,923 | 176,524,554 | ~63× |
| `pointmaze_proc` | `render` | 64 | 29,228 | 6,427,694 | ~220× |
| `pointpush` | `step` | 64 | 1,523,733 | 438,563,686 | ~288× |
| `pointpush` | `render` | 64 | 21,178 | 6,016,974 | ~284× |
| `reacher` | `step` | 64 | 5,727,996 | 197,708,268 | ~35× |
| `reacher` | `render` (24t, B=4096) | 4,096 | 2,409 (B=64) | 5,236,743 | ~2,170× |
| `reacher_morph` | `step` | 64 | 5,654,653 | 197,386,407 | ~35× |
| `reacher_morph` | `render` | 64 | 1,482 | 2,064,099 | ~1,393× |
| `reacher_obs` | `step` | 64 | 336,222 | 47,439,953 | ~141× |
| `reacher_obs` | `render` | 64 | 1,534 | 1,888,573 | ~1,231× |
| `armpush` | `step` | 64 | 1,060,090 | 45,257,876 | ~43× |
| `armpush` | `render` | 64 | 1,677 | 1,803,951 | ~1,076× |
