# Changelog

All notable changes to `matterstate-envs` will be documented in this file.

## [Unreleased]

### Added
- Zero-dependency Rust core (`src/`) for 11 environments: `sokoban`,
  `pointmaze` (explicit AVX2+FMA SIMD), `pointmaze_proc`, `pointpush`,
  `reacher`, `reacher_obs`, `reacher_morph`, `armpush`, `pusht` (2D rigid-body
  T-block + 96×96 RGB), `ogbench_maze` (`medium` and `large` grids), and
  `sudoku` (parallel generator).
- Unified `maturin` Python wheel (`matterstate_envs` + `matterstate_envs._rs`)
  with automatic Rust delegation for hot `sokoban.step` and 48×48 `render`
  paths, plus batched `reset` and multi-step `rollout` APIs.
- C-ABI (`cdylib`) exports (`src/ffi.rs`) and full parity/property test suites
  (`tests/test_rust_parity.py`, `tests/test_pyo3.py`, `tests/test_properties.py`,
  `tests/reference/test_sokoban.py`, `tests/reference/test_pusht.py`).
- Throughput and multi-core scaling benchmarks (`benchmarks/bench_numpy.py`,
  `benches/*.rs`).
- MIT `LICENSE`.
