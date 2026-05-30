# Compatibility-Preserving Internal Cleanup

Umbrella issue: `7237df6`.
Branch: `feature/compat-internal-cleanup`.
Base: `develop` at `349ef42`.

## Implemented

- Preserved the Mandible-facing provider manifest and mapping contract added in
  `78af270`; provider schemas, manifest kinds, HTTP routes, CLI entry points,
  env vars, and storage layouts were not changed.
- Restored `feedbax.xabdeef.losses.delayed_reach_loss()` by replacing the
  undefined `EffectorFixationLoss` reference with an explicit hold-signal-masked
  loss term.
- Added focused tests for delayed-reach fixation masking and delayed loss
  construction.
- Removed documented dead analysis files:
  `feedbax/analysis/fps_tmp2.py` and `feedbax/analysis/nn_utils.py`.
- Fixed `feedbax.analysis.fp_finder` undefined-name lint on jaxtyping shape
  annotations without hiding real undefined names elsewhere in the module.
- Fixed `feedbax.training.rl.ppo.train_ppo_batched()` using undefined
  `batch_size` and `minibatch_size` inside the JIT update loop.
- Consolidated Studio frontend package-manager workflow on npm, matching
  repo-local instructions and `web/package-lock.json`; removed the stale
  `web/pnpm-lock.yaml`, updated helper scripts, and corrected README frontend
  port guidance to `3008`.

## Cloc

Commands used:

- `cloc feedbax --exclude-dir=__pycache__ --out=.unfrack/baseline-feedbax.txt`
- `cloc feedbax --exclude-dir=__pycache__ --out=.unfrack/final-feedbax.txt`
- `cloc web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/baseline-web.txt`
- `cloc web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/final-web.txt`
- `cloc feedbax web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/baseline-total.txt`
- `cloc feedbax web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/final-total.txt`

Summary:

- `feedbax/`: 219 files / 50,382 code lines -> 218 files / 50,022 code lines
  (`-1` file, `-360` code lines).
- `web/` maintained source set: 153 files / 44,441 code lines -> unchanged.
- Combined maintained source set: 372 files / 94,823 code lines -> 371 files /
  94,463 code lines (`-1` file, `-360` code lines).

## Verification

- `uv run ruff check feedbax/xabdeef/losses.py feedbax/analysis/fp_finder.py tests/test_xabdeef_losses.py --select F821,F822,F823,F401`: passed.
- `uv run ruff check feedbax/training/rl/ppo.py --select F821,F822,F823`: passed.
- `uv run ruff check feedbax/analysis --select F821,F822,F823`: passed.
- `uv run python -m compileall -q feedbax/training/rl/ppo.py feedbax/analysis/fp_finder.py feedbax/xabdeef/losses.py tests/test_xabdeef_losses.py`: passed.
- `uv run pytest tests/test_xabdeef_losses.py tests/test_cde_controller.py tests/test_provider_contract.py -q`: passed (`66` passed, `2` xfailed).
- `uv run pytest tests/test_xabdeef_losses.py tests/test_rl_ppo.py -q`: passed (`12` passed).
- `npm ci`: passed; reported `11` audit vulnerabilities (`6` moderate, `5` high).
- `npm test -- --run`: passed (`24` files, `185` tests).
- `npm run build`: passed with the existing large Plotly chunk warning.
- `bash -n scripts/dev.sh` and `bash -n scripts/build.sh`: passed.

## Blocked/Deferred

- `uv run pytest tests/test_batched_ppo.py -q` could not run in this worktree
  because the optional `mujoco` dependency is not installed. The touched PPO
  module compiles and its non-MJX tests pass.
- A repo-wide `uv run ruff check feedbax tests --select F821,F822,F823` still
  reports historical jaxtyping annotation false positives in untouched modules.
  The scan no longer reports the real `train_ppo_batched` undefined-name bug.
- `npm audit` reports existing frontend dependency vulnerabilities; this pass
  did not run `npm audit fix` because that can upgrade dependencies and change
  frontend behavior.
