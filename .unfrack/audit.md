# Residual Compatibility-Preserving Internal Cleanup

Umbrella issue: `ce814d1`.
Branch: `feature/compat-internal-cleanup-residuals`.
Base: `develop` at `78e87b9`.

## Implemented

- Preserved Mandible-facing provider manifest mappings, provider/execution
  schemas, HTTP route paths, CLI entry points, env var names, and storage
  layouts.
- Fixed Studio local execution to use the current interpreter for local backend
  runs instead of shelling out to bare `python`; remote/cloud execution specs
  still use `python`.
- Aligned the FastAPI development CORS origin with the actual Vite dev port
  `3008` and added a regression test for that origin.
- Removed the residual pnpm package-manager drift by ignoring regenerated
  `pnpm-lock.yaml` files now that `web/package-lock.json` is canonical.
- Updated the web UI spec to match npm and Vite port `3008`.
- Removed unused internal frontend compatibility aliases:
  `RunSelector`, `fetchAnalysisGraph`, and `saveAnalysisGraph`.
- Made `uv run ruff check feedbax tests --select F821,F822,F823` pass
  repo-wide by importing the type-only `AbstractTask` reference and adding
  narrow `F821` suppressions only where Ruff misreads jaxtyping shape tokens as
  undefined names.
- Removed unused imports in the touched annotation-heavy modules.

## Cloc

Commands used:

- `cloc feedbax --exclude-dir=__pycache__ --out=.unfrack/baseline-feedbax.txt`
- `cloc feedbax --exclude-dir=__pycache__ --out=.unfrack/final-feedbax.txt`
- `cloc web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/baseline-web.txt`
- `cloc web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/final-web.txt`
- `cloc feedbax web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/baseline-total.txt`
- `cloc feedbax web/src web/package.json web/vite.config.ts web/tsconfig.json web/tsconfig.node.json --exclude-dir=__pycache__,dist,node_modules --out=.unfrack/final-total.txt`

Summary:

- `feedbax/`: 218 files / 50,022 code lines -> 218 files / 50,010 code lines
  (`-12` code lines).
- `web/` maintained source set: 153 files / 44,441 code lines -> 153 files /
  44,420 code lines (`-21` code lines).
- Combined maintained source set: 371 files / 94,463 code lines -> 371 files /
  94,430 code lines (`-33` code lines).

## Verification

- `uv run ruff check feedbax tests --select F821,F822,F823 --output-format concise`: passed.
- `uv run ruff check feedbax/intervene/schedule.py feedbax/mechanics/geometry.py feedbax/mechanics/musculoskeletal.py feedbax/mechanics/plant.py feedbax/nn.py feedbax/nn_cde.py feedbax/train.py tests/test_web_app.py --select F821,F822,F823,F401 --output-format concise`: passed.
- `uv run ruff check feedbax/studio_execution.py feedbax/web/app.py tests/test_web_app.py --select F821,F822,F823,F401 --output-format concise`: passed.
- `uv run python -m compileall -q ...`: passed for touched Python modules.
- `uv run pytest tests/test_web_app.py -q`: passed.
- `uv run pytest tests/test_studio_execution.py::test_run_studio_training_local_execution_materializes_snapshot_and_refs tests/test_studio_execution.py::test_studio_training_run_local_endpoint_returns_execution_result -q`: passed.
- `uv run pytest tests/test_web_app.py tests/test_provider_contract.py tests/test_studio_execution.py tests/test_execution_contract.py tests/test_cde_controller.py tests/test_backend.py -q`: passed (`98` passed, `13` skipped, `2` xfailed).
- `npm ci`: passed; reported existing `11` audit vulnerabilities (`6` moderate,
  `5` high).
- `npm test -- --run`: passed (`24` files, `185` tests).
- `npm run build`: passed with the existing large Plotly chunk warning.

## Blocked/Deferred

- Full `uv run pytest -q` is still not green in this environment:
  `650` passed, `15` skipped, `2` xfailed, `16` failed, `56` errors. The errors
  are dominated by optional MuJoCo/MJX suites failing with
  `ModuleNotFoundError: No module named 'mujoco'`; additional historical
  failures remain in older mechanics, intervention, observation-normalization,
  artifact-materialization, and worker-execution tests. The focused provider,
  Studio execution, backend compatibility, CDE, frontend test, and frontend
  build checks for this patch passed.
- Existing frontend dependency audit findings were not mutated with
  `npm audit fix` because that can change dependency versions and behavior.
