# Agent instructions

This is a simulator library, not a planner or training repository.

- Keep runtime kernels under `matterstate_envs/` and dependent only on NumPy.
  External engines belong exclusively in test-only reference environments.
- The Python kernels were extracted without numerical changes. Preserve their
  existing APIs until an explicit breaking change is requested. No compatibility
  imports, `sys.path` injection, or research-repository dependencies.
- Do not call a custom task benchmark-equivalent. Define and verify dynamics,
  observations, actions, reset, success, and episode contracts independently.
  Exact Sokoban transition parity is not pixel/reward/reset parity.
- Separate byte-exact checks from explicit floating-point tolerance contracts.
  Never weaken a parity check simply to make a replacement pass.
- Keep kernel optimization separate from extraction. Measure performance before
  claiming a speedup. New compute implementations should follow the global
  language rules; this initial extraction retains the existing NumPy code.
- Use `nix develop`, `uv sync --frozen --group reference`, then `check`.
  Format changed Python with the pinned Ruff and Nix with nixfmt first.
  CI runs the same non-mutating source checks and tests.
- Review snapshot changes. `VERIFY.md` documents the contract map and bounds.
- Keep datasets, binaries, native build products, and environments Git-ignored.
  Root `.gitignore` is default-deny; narrowly allow intentional new source.
