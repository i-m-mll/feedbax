# Unfrack Audit

Umbrella issue: `6128292`.
Branch: `unfrack/develop-cleanup`.
Base: `develop` at `4da5df0`.

## Implemented

- Fixed `feedbax._tree.anyf` and `allf`, which closed over an undefined `funcs` name and failed when their returned predicates were called.
- Added focused predicate tests for `anyf`, `allf`, and `notf`.
- Preserved root `feedbax` re-export behavior by adding explicit `__all__` instead of deleting public imports.
- Restored `nan_safe_mse` so public/importable analysis modules `feedbax.analysis.network` and `feedbax.analysis.regression` import successfully.
- Moved notebook widget imports in `feedbax.setup_utils` behind the notebook-only file chooser path, so importing analysis/training helpers no longer requires notebook extras.
- Updated the stale CDE zero-dX test to match current semantics: hidden state is unchanged only when vector-field input change, decay, and Anti-NF feedback are all disabled/zero.
- Fixed generated Equinox wrapper imports so generated jaxtyping annotations for `Array` and `Float` resolve, and updated the generator to preserve the fix.
- Centralized the frontend `feedbax:lastProjectId` storage access behind the existing safe storage helper.
- Fixed local Studio tab restore behavior so a restored unsaved local tab is not overwritten by stale last-project autoload.
- Added a `web` test script and a regression test for restored local tab state.

## Remaining Findings

- `CDENetwork`, `Channel`, and related mechanics paths still emit static-JAX-array warnings. This is a real Equinox/JAX architecture issue and should be fixed by storing dynamic array initial state as PyTree data or by rebuilding it from static shape/config.
- `feedbax/analysis/fps_tmp2.py` remains temporary/dead-looking and contains undefined names. It should be deleted or repaired after checking notebooks/scripts for direct imports.
- Broad Ruff and Pyright remain noisy across the repo. This pass fixed correctness-level issues in touched surfaces, not the whole historical lint backlog.
- Frontend settings still expose some controls that are not fully wired to graph/canvas behavior, and some param editor/value type duplication remains.
- Frontend package manager state remains mixed because both npm and pnpm lockfiles exist. This pass used pnpm, matching current local tooling.

## Verification

- `uv run ruff check ... --select F821,F822,F823,F401` on touched Python files: passed.
- `uv run pytest tests/test_tree_predicates.py tests/test_cde_controller.py -q`: passed, with known static-array warnings.
- Import check for `feedbax.analysis.network`, `feedbax.analysis.regression`, and `feedbax.setup_utils`: passed.
- `pnpm test -- src/stores/projectsStore.test.ts`: passed all current frontend tests (`19` files, `145` tests).
- `pnpm run build`: passed with the existing large Plotly chunk warning.

