# matterstate-envs

Fast, zero-dependency Rust simulator kernels with C-ABI (`cdylib`) and
Python/NumPy (`PyO3`) bindings for reinforcement learning and world-model
planning.

## Environments

| Environment | Type | Rust (`src/`) | Python (`matterstate_envs`) |
|---|---|---|---|
| `sokoban` | 10×10 discrete pushing | `sokoban.rs` | `sokoban`, `_rs.sokoban_step` |
| `pointmaze` | 2D navigation (AVX2+FMA SIMD) | `pointmaze.rs` | `pointmaze`, `_rs.pointmaze_*` |
| `pointmaze_proc` | Procedural 6×6 mazes + 48×48 render | `pointmaze_proc.rs` | `pointmaze_proc`, `_rs.pointmaze_proc_*` |
| `pointpush` | 2D contact pushing + 48×48 render | `pointpush.rs` | `pointpush`, `_rs.pointpush_*` |
| `reacher` | 2-link planar arm + 48×48 render | `reacher.rs` | `reacher`, `_rs.reacher_*` |
| `reacher_obs` | 2-link arm with circular obstacle | `reacher_obs.rs` | `reacher_obs`, `_rs.reacher_obs_*` |
| `reacher_morph` | Variable-length 2-link arm | `reacher_morph.rs` | `reacher_morph`, `_rs.reacher_morph_*` |
| `armpush` | 2-link arm pushing an object | `armpush.rs` | `armpush`, `_rs.armpush_*` |
| `pusht` | 2D rigid-body T-block + 96×96 RGB | `pusht.rs` | `_rs.pusht_*` |
| `ogbench_maze` | 4D force-controlled PointMaze (medium/large) | `ogbench_maze.rs` | `_rs.ogbench_maze_*` |
| `sudoku` | Parallel puzzle generator | `sudoku.rs` | `_rs.sudoku_generate` |

## Quickstart

### Python

`matterstate-envs` builds both the Python package and the native `_rs`
extension in a single wheel via `maturin`:

```python
import numpy as np
from matterstate_envs import _rs, pointmaze

# High-level API (automatically delegates hot paths to Rust)
pos, goal = pointmaze.reset(np.random.default_rng(0), 64)
pos = pointmaze.step(pos, np.zeros_like(pos, np.float32))

# Direct Rust batched rollout API (zero Python loop overhead)
x, y, gx, gy = _rs.pointmaze_reset(42, 64)
ax = np.zeros(16 * 64, np.float32)
ay = np.zeros(16 * 64, np.float32)
traj_x, traj_y = _rs.pointmaze_rollout(16, x, y, ax, ay)
```

### Rust

The root crate (`Cargo.toml`) has **zero external dependencies** by default
(PyO3/NumPy bindings are gated behind `--features python`):

```sh
cargo test
cargo bench --bench sokoban
cargo bench --bench continuous
RUSTFLAGS="-C target-cpu=native" cargo bench --bench scaling
```

## Benchmarks (x86_64 Linux, 24-core AVX2+FMA)

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
| `reacher` | `render` (24t) | 4,096 | 2,409 (B=64) | 5,236,743 | ~2,170× |
| `reacher_morph` | `step` | 64 | 5,654,653 | 197,386,407 | ~35× |
| `reacher_morph` | `render` | 64 | 1,482 | 2,064,099 | ~1,393× |
| `reacher_obs` | `step` | 64 | 336,222 | 47,439,953 | ~141× |
| `reacher_obs` | `render` | 64 | 1,534 | 1,888,573 | ~1,231× |
| `armpush` | `step` | 64 | 1,060,090 | 45,257,876 | ~43× |
| `armpush` | `render` | 64 | 1,677 | 1,803,951 | ~1,076× |
| `pusht` | `step` (10 substeps) | 64 | — | 1,272,912 | — |

## Verification & Development

```sh
nix develop
uv sync --frozen --group reference
check
```

`check` runs formatting (`nixfmt`, `ruff`, `rustfmt`), lints (`actionlint`,
`ruff`, `clippy` on both default and `--features python`, `ast-grep`), Rust
unit/SIMD tests, property-based tests (`tests/test_properties.py`), Rust-vs-NumPy
C-ABI & PyO3 parity tests (`tests/test_rust_parity.py`, `tests/test_pyo3.py`),
and external reference parity against `gym-sokoban` and `pymunk`
(`tests/reference/`). See `VERIFY.md` for details.

## License

MIT — see [LICENSE](LICENSE).
