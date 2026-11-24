# Matterstate simulators

NumPy simulator kernels extracted from `matterstate/latent-planner`, with
external engines isolated as test-only references. Python imports use
`matterstate_envs`; the installed runtime requires only NumPy. This extraction
preserves existing behavior; it is not a speed optimization or a claim that
our custom tasks implement MuJoCo, PushT, or OGBench physics.

## Use

Install a pinned Git revision of this repository, then import directly:

```python
import numpy as np
from matterstate_envs import pointmaze

position, goal = pointmaze.reset(np.random.default_rng(0), 8)
position = pointmaze.step(position, np.zeros_like(position))
```

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

`check` is the same gate used in CI: Nix/Python formatting, Ruff, structural
import policies and their fixtures, regression snapshots, external Sokoban
parity, and the isolated OGBench import check. Neovim Conform uses the same
Ruff/nixfmt executables when launched from this dev shell.

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
| All eight extracted kernels | latent-planner `df2f5b2` | Exact dtype, shape, and byte-hash snapshots of 30 seeded reset/trajectory/render/success arrays |
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

## Known inherited limitation

`reacher_obs.render` cannot broadcast per-item obstacle radii for batches
larger than one. The extraction snapshot renders those frames individually;
it does not hide or fix the existing batched-rendering defect. Rejection
samplers also retain their existing unbounded retry behavior. Neither issue
was changed during behavior-preserving extraction.
