## feedbax repo — max-scrutiny review: packaging, layering, tests, CI, dead weight, tooling, code smells

Repo root: `/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax` (reviewed via mounted path `/sessions/serene-jolly-ptolemy/mnt/feedbax` — identical content). Read-only review, no files edited. Already-tracked items (1201afd, 8cc42a3, 5f22fa7, bdee8d1, c20214c, ce0b823, 4f6f3c2) are excluded from findings below.

---

### Findings

**F1** — `src/` is a dead, empty, unreferenced directory | **Low** severity, trivial cleanup | area: packaging
Evidence: `find src -mindepth 0` returns only the empty dir itself (`ls -la src` → `total 0`, only `.`/`..`). No reference to `src/` anywhere in `pyproject.toml`, `makefile`, `.worktree.yaml`, or `[tool.hatch.build]` sections (build targets point at `packages = ["feedbax"]`). `git log --oneline -- src/` returns nothing — it isn't even tracked history, just a stray empty directory on disk.
Why: confusing for anyone orienting to the repo layout (rlrmp's own CLAUDE.md explicitly notes "there is no `src/` layout" for feedbax, so this empty dir is misleading noise, possibly a leftover from an abandoned src-layout migration or an artifact of tooling).
Proposal: delete the empty `src/` directory (or `.gitignore` it if some local script recreates it as a build scratch dir — but no evidence of that).
Effort: S. Overlap: none of the already-tracked issues mention this.

**F2** — Undeclared direct dependency: `sqlalchemy` imported directly but not listed in `pyproject.toml` | **Medium** severity | area: packaging
Evidence: `feedbax/persistence/database.py:9` and `feedbax/dashboard/backend/query.py:9` both do `from sqlalchemy... import ...` / `import pandas as pd` + sqlalchemy `Session`, yet `pyproject.toml`'s `dependencies` list has no `sqlalchemy` entry. `uv.lock:2852-2858` shows `sqlalchemy==2.0.45` is present only as a transitive dependency of `alembic` (`uv.lock:20-27` lists `{ name = "sqlalchemy" }` under alembic's deps).
Why: this only works today because alembic happens to require sqlalchemy at a compatible version. If alembic ever drops sqlalchemy as a dependency, loosens its version bound below what `persistence/database.py` needs, or is removed from `dependencies` in a future cleanup (e.g. as part of extras-splitting per this review's own recommendation F4), `import feedbax.persistence.database` breaks with no direct signal in `pyproject.toml` explaining why. This is exactly the "contract obeyed, never bypassed" pattern in reverse — a real import contract with no declared package boundary.
Proposal: add `sqlalchemy` as an explicit direct dependency (core or in a new `analysis`/`persistence` extra, see F4).
Effort: S.

**F3** — `import feedbax` unconditionally pulls in `plotly` at package-init time | **Medium-High** severity | area: import hygiene / layering
Evidence: `feedbax/__init__.py:23` imports `feedbax.tasks` → `feedbax/tasks/__init__.py:8` imports `.task` → `feedbax/tasks/task.py:39`: `import plotly.graph_objs as go  # pyright: ignore [reportMissingTypeStubs]`, at true module top-level (not under `TYPE_CHECKING`, not deferred into a function). Verified independently by direct grep of the source tree.
Why: this directly contradicts the intended "components/runtime/execution (core) ← contracts ← training/analysis ← studio/web" layering — a plotting library, which belongs in an outer/visualization concern, is now baked into the innermost `tasks` module that `feedbax/__init__.py` re-exports at the top. Every consumer of `import feedbax` (including headless training scripts, CI, and any lightweight downstream tooling) now pays plotly's import cost and dependency-resolution weight even if they never plot anything. Confirmed no `fastapi`, `pandas`, or `sklearn` are pulled in transitively via the eager chain (traced 2-3 levels through `config.mapping`, `runtime.selectors`, `runtime.graph`, `contracts.graphs.templates`, `models.cde`, `intervene`, `objectives.loss`, `tasks`, `tasks.presets`) — so the damage is contained to plotly, not systemic, but it is real and specific.
Proposal: move the `plotly.graph_objs` import in `feedbax/tasks/task.py:39` behind a lazy/deferred import (only import inside the function(s) that actually build a plotly figure), or split task-visualization helpers out of `tasks/task.py` into a `tasks/viz.py` that `feedbax/__init__.py` does not eagerly reach.
Effort: S (single-file fix once the call sites using `go` in that file are identified).

**F4** — Core `dependencies` list mixes core runtime needs with heavy, narrowly-used, or Studio/analysis-only packages | **Medium** severity | area: packaging
Evidence (import-site counts verified directly against source, not `uv.lock`):
- `fastapi`/`uvicorn`/`httpx` — used only under `feedbax/web/**` (18 files import fastapi directly, all under `feedbax/web/`); zero use under `components/`, `runtime/`, `execution/`, `contracts/`, `models/`, `mechanics/`, `objectives/`, `tasks/`.
- `pandas` — used in exactly 3 files: `feedbax/plot/mpl.py`, `feedbax/dashboard/backend/query.py`, `feedbax/persistence/database.py`.
- `polars` — used in exactly 3 files: `feedbax/plot/trajectories.py`, `feedbax/plot/plotly.py`, `feedbax/plot/misc.py`.
- `scikit-learn` — used in exactly 3 files: `feedbax/plot/experiments.py`, `feedbax/analysis/tangling.py`, `feedbax/analysis/pca.py`.
- `dash` (not in core deps, correctly an optional extra already) — but `dash-bootstrap-components` sibling `dash` import appears in `feedbax/dashboard/app.py` only, confirming `dashboard` is correctly gated as `dashboard` extra already for the dash-specific bits — but `feedbax/dashboard/backend/query.py` (pandas+sqlalchemy) is NOT gated, it's pulled by core deps regardless.
- `pyexiv2` — used in exactly 2 files: `feedbax/config/batch.py`, `feedbax/config/yaml.py` (plus `feedbax/bin/plotly_viewer.py`, a CLI script).
- `pyperclip` — used in exactly 4 files (`feedbax/analysis/{specs,evaluation,execution,analysis}.py`) — clipboard access is a very narrow analysis-workflow convenience, not core-runtime.
- `alembic`/`dill`/`kaleido`/`tensorboardx`/`ruamel-yaml`/`rich` are similarly narrow (1-2 call sites each, all under `training/`, `persistence/`, `config/`, or CLI `bin/`).
Why: none of these narrow-use packages are needed to build/train/execute a graph (the actual "core" of the library per CLAUDE.md's own framing — "the graph is the model"). Every `pip install feedbax` today drags in fastapi+uvicorn+httpx (a whole ASGI web stack) even for someone who only wants `Graph`/`Component`/training primitives with no Studio backend and no dashboards. This inflates install size/time and CVE surface for a pure-JAX consumer.
Proposal: split into optional extras — e.g. `web` (fastapi, uvicorn, httpx, alembic, sqlalchemy — see F2), `analysis` (pandas, scikit-learn, pyperclip, tensorboardx), `viz` (seaborn, kaleido, polars if only used by plot/), `io` (pyexiv2, ruamel-yaml, dill). Keep truly universal deps (jax, equinox, diffrax, optax, jaxtyping, numpy, pydantic, plotly given F3's caveat, matplotlib, treescope, tqdm, jax-cookbook) in core. This is a bigger lift since rlrmp's editable-source integration assumes `uv sync` installs everything — coordinate the extras split with a corresponding rlrmp `pyproject.toml` dependency-group update in the same wave.
Effort: M (touches pyproject.toml extras, CI install steps, and needs a downstream-compat check with rlrmp before landing — this itself argues for filing it as a tracked issue with a migration note, per the repo's own "durable format" discipline analog).

**F5** — `pandas` and `polars` both used for the same conceptual task (dataframe-shaping loss/trial data for plotting), in sibling files of the same package | **Low-Medium** severity | area: packaging — duplicate dependency
Evidence: `feedbax/plot/mpl.py:29,666,669,757` builds `pd.DataFrame(...)` from loss/trajectory arrays for matplotlib rendering. `feedbax/plot/plotly.py:22,70,73,505` builds `pl.DataFrame(...)` from the same class of loss/trajectory arrays for plotly rendering. Both files live directly under `feedbax/plot/` and do near-identical data-reshaping (melt/concat over loss-term arrays) with two different dataframe libraries.
Why: two dataframe engines is 2x the dependency weight, 2x the API-familiarity burden, and prone to representation drift between the mpl and plotly rendering paths for what should be the same shaped data (a bug fixed in one melt/reshape helper will not propagate to the other).
Proposal: standardize on one library (polars is already used more broadly in the codebase and is lighter/faster; pandas' 3 call sites are all confined to `plot/mpl.py` + 2 persistence/dashboard files) and migrate `plot/mpl.py`'s three `pd.DataFrame` call sites to polars, or vice versa. Low urgency but easy, contained win.
Effort: S-M (3 call sites in `mpl.py`, contained; dashboard/persistence pandas usage is unrelated to the plot duplication and can stay if that subsystem prefers pandas for sqlalchemy ORM row conversion).

**F6** — No layering violations found; core subpackages are clean | **Informational** (positive finding)
Evidence: `grep -rn "from feedbax\.(studio|web|training|analysis|dashboard|plugins)"` across `components/`, `runtime/`, `execution/`, `contracts/`, `models/`, `intervene/`, `objectives/`, `tasks/`, `mechanics/`, `control/` returns zero hits. `feedbax/__init__.py` does not import `feedbax.web` or `feedbax.studio` at all (confirmed by direct grep). `feedbax/web/` correctly depends on `feedbax/studio/` (5 import sites across `web/worker/app.py`, `web/worker/execution.py`, `web/api/provider.py`) — `studio/` (4 files: `execution.py`, `schema.py`, `protocol.py`, `__init__.py`) is a shared protocol/schema layer consumed by the `web/` FastAPI app, not a duplicate or legacy sibling of `web/`.
Why noted: this is a genuinely clean result worth stating explicitly given how much of this review is critical — the intended dependency direction (core ← contracts ← training/analysis ← studio/web) is respected with no violations found in the greppable surface.

**F7** — Circular-import workaround leaves two public-looking functions unreachable from the package | **Medium** severity | area: import hygiene
Evidence: `feedbax/intervene/__init__.py:27-31` — `remove_all_intervenors` and `remove_intervenors` are imported from `feedbax.intervene.remove` in a commented-out block, disabled because `AbstractStagedModel` in `remove.py` causes a circular import with `intervene/__init__.py`.
Why: this isn't a documented deferred-import pattern (like the other three circular-import workarounds found, which use function-local imports with explanatory comments — `feedbax/plugins/__init__.py:8-15`, `feedbax/integrations/treescope.py:190-191`, `feedbax/plot/io.py:287-289`, all legitimate lazy-import fixes). This one just comments out the export entirely, silently removing two functions from the public `feedbax.intervene` surface. Per this repo's own policy ("'just for now' workarounds are bugs... regardless of how they are labelled"), a commented-out export sitting in `__init__.py` is precisely that class of residue — worse, it's invisible unless someone reads the `__init__.py` source, since no test can catch "this function used to be importable and now silently isn't" without an explicit regression test.
Proposal: either fix the circular import properly (move the shared base out of `remove.py` or use a local/deferred import like the other three cases), or delete `remove_all_intervenors`/`remove_intervenors` from `remove.py` entirely if they're truly dead, rather than leaving a permanently-commented import.
Effort: S-M depending on how entangled `AbstractStagedModel` is.

**F8** — `feedbax_contract`-marker family does not exist in feedbax's own test suite, despite rlrmp depending on it | **Medium-High** severity | area: tests / CI
Evidence: `grep -rn "feedbax_contract" tests/ pyproject.toml` → zero hits anywhere in feedbax. `pyproject.toml:140-142` declares only one custom pytest marker, `slow`. rlrmp's CLAUDE.md (this session's own project context) states: "The marked gate is `feedbax_contract` in `ci/feedbax-contract-suite.toml`" and lists families like `graph_spec_contract`, `analysis_recipe_contract`, etc. as protecting rlrmp's integration surface against feedbax drift.
Why: rlrmp's contract-gate assumption appears to rest entirely on rlrmp-side tests exercising feedbax's public API, with no reciprocal feedbax-side marker or test family that self-identifies "these tests are the contract surface downstream consumers rely on." If feedbax refactors internals the 109 existing tests do cover (e.g. `test_provider_contract.py`, `test_graphspec_builtins.py`, `test_structured_spec_migrations.py` — all touch contract-adjacent surfaces already, just untagged), there's no way to `pytest -m feedbax_contract` from feedbax's own CI to assert "the rlrmp-relevant contract surface still passes" before a downstream break is discovered on the rlrmp side. This connects directly to the already-tracked 1201afd (CI should wire contract/import-boundary/migration suites + a downstream rlrmp job) — noting it here only because the *marker itself doesn't exist yet*, which is a prerequisite fact 1201afd's implementation will need.
Effort: not proposing new work here since 1201afd already tracks the CI wiring — flagging as supporting evidence for that issue's scope.

**F9** — CI runs only 2 of 109 test files; no lint/typecheck/test pass over the full tree | **High** severity | area: CI
Evidence: `.github/workflows/ci.yml`'s `python` job runs pytest scoped to exactly `tests/test_batch_reshape_nan_bypass.py tests/test_studio_api_contracts.py`, and scopes `ruff check`/`pyright` to just those two files as well (per the CI-audit subagent's direct read of the workflow). `scripts/full_suite.sh`/`.py` (the memoized full-suite wrapper, newest file in the repo per git-log) exists but is never invoked by any workflow. `makefile`'s own `ci:` target (`uv lock --check` + `test lint typecheck`) also only runs the single-file `test`/`lint`/`typecheck` targets (`makefile:9-21` all point at `tests/test_batch_reshape_nan_bypass.py` specifically), not `test-all`/`lint-all`/`typecheck-all`.
Why: 107 of 109 test files (all contract, graph-spec, training, mechanics, objectives, analysis, config, integration, web, and task tests) run in nobody's CI today — they only run if a human remembers to `uv run pytest tests/ -q` locally. A regression in, say, `feedbax/runtime/graph.py` or `feedbax/contracts/graphs/serialization.py` (the exact subgraph-authority logic CLAUDE.md calls load-bearing) would not be caught by CI at all. This is the single largest CI gap found and is squarely what 1201afd already targets — reported here as current-state confirmation, not a new issue.
Effort: N/A (already tracked).

**F10** — `build_docs.yml` uses stale/deprecated GitHub Actions versions and a different Python version than `ci.yml` | **Low** severity | area: CI / tooling
Evidence: `build_docs.yml` uses `actions/checkout@v2` and `actions/setup-python@v2` (both deprecated; GitHub has been warning on `v2` action versions for node16 deprecation) and pins Python 3.11, while `ci.yml` uses `@v4`/`@v5` and the project's actual `requires-python = ">=3.12"`.
Why: docs build on a Python version (3.11) that the package itself declares unsupported (`requires-python = ">=3.12"`) — this "works" only because mkdocs builds don't typically exercise version-specific syntax, but it's an inconsistency that will bite if 3.12-only syntax creeps into any doc-executed code cells.
Proposal: bump `build_docs.yml` to the same actions versions and Python 3.12+ as `ci.yml`.
Effort: S.

**F11** — `feedbax/dashboard/` is narrowly used and untested, worth a liveness confirmation | **Low** severity | area: dead weight (soft finding, not clear-cut dead)
Evidence: 1697 lines, last modified 2026-06-14 (not stale by date), but only reachable via its own `feedbax/bin/dashboard.py` entry point — nothing else in `feedbax/`, `tests/`, or `examples/` imports from it, and zero test files reference `dashboard`. It has its own README (per the dead-weight subagent).
Why: recently touched ≠ actively depended-on. Zero test coverage for a 1700-line subsystem plus zero cross-references from the rest of the codebase is worth an explicit "is Dash still the plan, or has Studio (the React/FastAPI stack) superseded it?" check, given the repo already has a much larger, actively-developed `web/` + separate frontend (`web/` top-level React app) serving the same "visualize/monitor training" niche that a Dash dashboard would.
Proposal: not a removal recommendation on its own (recency + own README suggest deliberate, not abandoned) — but flag for the maintainer to confirm scope: is `feedbax/dashboard/` legacy-superseded-by-Studio, or a genuinely separate use case (e.g. quick local dash prototyping vs. the full Studio app)? If superseded, migrate to LEGACY-banner convention (per rlrmp's documented pattern) or remove; if not, add minimal test coverage.
Effort: S (to decide/label), M (if migration/removal follows).

**F12** — `xabdeef/`, `plugins/`, `testing/` are all live, not dead weight (positive/negative-result finding) | **Informational**
Evidence: `xabdeef/` (600 lines) is imported from `training/{post_training,train}.py`, 2 test files, and 1 example; last touched 2026-06-14. `plugins/` (540 lines) is imported from ~16 files spanning analysis/config/web/training/dashboard/integrations/tests — this is core plugin-registry infrastructure, not legacy. `testing/` (257 lines) has an explicit docstring "Testing helpers for downstream Feedbax integration contracts" and is used by `tests/test_evaluation_contract.py`; last touched 2026-07-03 (today-ish). None of the four originally-suspected "maybe dead" directories are actually dead — only `dashboard/` (F11) warrants a soft flag.

**F13** — `examples/` notebooks have real API drift; even a recently-touched notebook imports modules that no longer exist | **Medium** severity | area: docs/examples freshness
Evidence: `examples/1_train.ipynb` (touched 2026-06-14, one of the "recent batch") imports `from feedbax.bodies import SimpleFeedback`, `from feedbax.nn import SimpleStagedNetwork`, `from feedbax.iterate import Iterator` — none of `feedbax/bodies.py`, `feedbax/nn.py`, `feedbax/iterate.py` exist in the current tree. Current homes are `feedbax/models/feedback.py::SimpleFeedback` and `feedbax/models/networks.py::SimpleStagedNetwork`. `examples/8_advanced.ipynb` (2024, older batch) imports `feedbax.tree.tree_unzip`/`tree_map_unzip`, which don't exist anywhere in current source — but this notebook is already commented out of `mkdocs.yml`'s nav, so that particular drift is at least not surfaced to doc readers.
Why: a notebook touched as recently as 2026-06-14 (same week as several core refactors) with broken imports means either (a) the notebook wasn't actually re-run/tested after that touch (likely just metadata/formatting changes), or (b) the module reorganization (`nn.py`/`bodies.py`/`iterate.py` → `models/`) happened after the notebook's last real edit and nobody re-validated example notebooks post-move. This is exactly the kind of drift `mkdocs.yml`'s nav-freshness guard (4f6f3c2, already tracked) is presumably meant to catch for docs, but examples/notebooks may be outside that guard's scope — worth confirming.
Bonus finding, same root cause: **the repo's own `CLAUDE.md` is stale in the identical way** — it documents "CDE network: `feedbax/nn_cde.py`" and "Other networks (SimpleStagedNetwork, LeakyRNNCell): `feedbax/nn.py`" but the real current paths are `feedbax/models/cde.py` and `feedbax/models/networks.py`. This means the project's own agent-facing instructions actively mis-point new sessions (including this one, until direct source inspection corrected it) to nonexistent files.
Proposal: (1) either execute-and-refresh `examples/1_train.ipynb`'s imports or add a lightweight CI/pre-commit check that greps notebook import cells against `feedbax/__init__.py`'s actual export surface; (2) fix the two stale file-path references in `/feedbax/CLAUDE.md`'s "Repository Structure" section (nn_cde.py → models/cde.py, nn.py → models/networks.py).
Effort: S for the CLAUDE.md fix; S-M for a notebook freshness check.

**F14** — `except Exception`/bare `except` swallowing in operationally risky paths | **Medium-High** severity | area: code smell
Evidence (worst offenders, from repo-wide grep of 81 total hits, 2 bare `except:`):
- `feedbax/web/orchestration/manager.py:209,215` — bare `except Exception: pass`/silent-return in cloud instance delete/terminate lifecycle, no logging.
- `feedbax/web/services/training_service.py:267` — `except Exception: return <fallback>` silently masking worker-status query failures, returning stale cached status as if healthy.
- `feedbax/runtime/graph.py:77` — `except Exception: continue` while collecting `StateIndex` fields during core graph-execution state discovery; a misconfigured field would silently vanish from state discovery rather than surfacing.
- `feedbax/config/tree.py:936` — bare `except: continue` searching a reference tree for an LDict level, no logging (has a warning fallback after the loop, partially mitigating).
Why: `manager.py:209/215` risk orphaned/unaccounted cloud billing (a RunPod/cloud instance that fails to terminate cleanly and nobody is told); `training_service.py:267` risks showing a stale "healthy" status to a Studio user when the worker actually crashed; `runtime/graph.py:77` is in the literal state-discovery core path CLAUDE.md calls out as load-bearing ("the graph is the model" — no silent substitution allowed), so a silently-skipped `StateIndex` field is precisely the class of bug the project's own core principle forbids.
Proposal: at minimum add `logger.warning(...)` before the `pass`/`continue` in the four sites above so failures are observable; for `runtime/graph.py:77` specifically, consider whether the exception should propagate (per "absence... is an error, not a condition to work around").
Effort: S per site (logging-only fix), review needed for `runtime/graph.py:77` to confirm which exception types are actually expected vs. being over-broadly caught.

**F15** — `print(` used instead of `logging` in durable-artifact-adjacent library code | **Low-Medium** severity | area: code smell
Evidence: 42 hits outside `feedbax/bin/` (CLI scripts, where print is expected). Worst concentrations: `feedbax/training/checkpoint_custody.py` (10 hits) — checkpoint status messages in what CLAUDE.md's artifact-schema section treats as a durable-write path; `feedbax/analysis/setup.py` (6), `feedbax/analysis/fp_finder.py` (6, solver diagnostics).
Why: `print` output is invisible to log aggregation/level filtering and can't be silenced or redirected the way `logging` calls can — inconsistent with the rest of the codebase's `logging.getLogger(__name__)` convention (confirmed present in `feedbax/__init__.py:97` and `persistence/database.py`).
Proposal: replace `print(` calls in `training/checkpoint_custody.py`, `analysis/setup.py`, `analysis/fp_finder.py` with `logger.info`/`logger.debug` as appropriate.
Effort: S (mechanical).

**F16** — `jax.tree_util.*` used directly in a handful of files instead of the `import jax.tree as jt` convention | **Low** severity | area: code smell
Evidence: 27 total hits (22 lib, 5 tests); actual deprecated-API calls (not just `jax.tree_util` aliasing, which itself may be legitimate for `register_pytree_node_class`) at: `feedbax/plot/utils.py:99` (`jax.tree_util.tree_flatten_with_path`), `feedbax/config/tree.py:553` (same), `feedbax/runtime/graph.py:98` (`jax.tree_util.tree_flatten`), `feedbax/intervene/schedule.py:73` (`jax.tree_util.tree_reduce`). `feedbax/training/rl/tasks.py:142`'s `@jax.tree_util.register_pytree_node_class` is a legitimate use with no `jt.*` equivalent.
Why: `runtime/graph.py:98` is in the core graph module; `jt.tree_flatten_with_path`/`jt.tree_flatten` do have modern equivalents (`jax.tree_util.tree_flatten_with_path` is actually still the canonical spelling in modern JAX — worth double-checking this specific one isn't a false positive against the "deprecated" label, since JAX's own `jax.tree.flatten` doesn't yet expose a `with_path` variant under the `jt` namespace in all JAX versions pinned here — flag for verification rather than blind fix).
Proposal: audit each of the 4 lib hits for a real `jt.*` equivalent before mechanically replacing; convention violation is real but low-count and low-severity given none sit in a hot-loop/traced path beyond one-time graph construction.
Effort: S, but verify jt.* API surface first (avoid a bad substitution).

**F17** — `TODO`/self-flagged tech debt worth triaging, notably one literal unimplemented branch | **Low-Medium** severity | area: code smell
Evidence: 133 total hits. Most notable: `feedbax/plot/mpl.py:219` — `raise ValueError("TODO")`, a live code path that raises at runtime if hit (not a comment, an actual placeholder exception). `feedbax/objectives/loss.py:516` — comment noting a real correctness bug ("if `terms` is a dict, this fails!"), unfixed. `feedbax/tasks/task.py:1195` — "appears to be deprecated, though perhaps it shouldn't be," i.e. uncertain dead-code status in a core task-definition file. `feedbax/analysis/analysis.py:2651` — self-flagged "overcomplicated and performs needlessly redundant work."
Why: `plot/mpl.py:219`'s `raise ValueError("TODO")` is the most concrete — it means some plotting code path is a landmine waiting to be hit by a real user, not a comment for future review.
Proposal: triage `plot/mpl.py:219` first (either implement or convert to a clear `NotImplementedError` with an informative message, since "raise ValueError('TODO')" gives a caller zero information); the loss.py:516 dict-terms bug and tasks/task.py:1195 deprecation-uncertainty are worth their own tracked issues given they're substantive, not cosmetic.
Effort: S for the ValueError message fix; M for the loss.py dict-terms bug (needs investigation of blast radius); S for confirming/removing the tasks.py:1195 dead code.

**F18** — "for now"/"temporary"/"workaround" grep is clean — no policy violations found | **Informational** (positive finding)
Evidence: 11 total hits, all benign: 2 are `tempfile.TemporaryDirectory` (stdlib, unrelated to the "workaround" policy), 3 are stable solver-default comments ("Explicit for now" on `diffrax.Euler` defaults — not a shim, just an explicitness note), 1 `bin/analysis.py:192` CLI-scope note. One item flagged by the smell-audit subagent as needing a manual look, `analysis/analysis.py:2578`'s `#! TEMPORARY` marker, was not independently re-verified in this session — worth a follow-up read if the maintainer wants full closure on this grep.
Why noted: given the CLAUDE.md policy explicitly treats "just for now" workarounds as bugs, it's a genuinely good sign this grep came back this clean — no silent fallback-to-stale-outer-params pattern, no disguised default-subgraph synthesis, was found anywhere in the fallback/silently grep either (53 hits, all either the governed `contracts/manifest.py` fallback-with-required-reason pattern, or benign). One item worth a manual look: `feedbax/mechanics/muscle_config.py:252` "Legacy fallback: uniform magnitude from lateral offset" — labeled "Legacy," which per this repo's stated backward-compatibility stance ("we do not maintain legacy code paths... When something is wrong, raise a clear error") is worth confirming isn't silently substituting a stale default.

---

### Layering map

**Intended** (per CLAUDE.md / project convention): `components / runtime / execution` (innermost) ← `contracts` ← `training / analysis` ← `studio / web` (outermost).

**Observed** (from direct grep across all core subpackages, zero violations found):

```
components/  runtime/  execution/  contracts/  models/  intervene/  objectives/  tasks/  mechanics/  control/
        \        |         |          |          |         |            |         |         |
         \       |         |          |          |         |            |         |         |
          ---------------- no inbound references from studio/web/training/analysis/dashboard/plugins ----------------

studio/  (4 files: schema, protocol, execution) ← consumed by → web/ (41 files, FastAPI app, 5 import sites into studio/)
training/, analysis/  → freely import from components/runtime/execution/contracts/models (as intended, outer-to-inner)
plugins/  ↔ config/   (circular, resolved via lazy __getattr__ singleton, feedbax/plugins/__init__.py:8-15)
plot/     ↔ plugins/  (circular, resolved via function-local import, feedbax/plot/io.py:287-289)
intervene/ (self-contained circularity between __init__.py and remove.py — NOT resolved, just disabled; F7)
```

One eager-import surprise breaks the "outer stays outer" spirit without being a strict layering violation: `feedbax/tasks/task.py:39`'s top-level `import plotly.graph_objs` means `tasks` (a "core-ish" subpackage per its presence in `feedbax/__init__.py`'s eager-import chain) pulls in a visualization dependency that conceptually belongs to the `plot/` outer layer (F3). This isn't a cross-subpackage Python import violation (task.py doesn't import from `feedbax.plot`), but it is a layering violation in the dependency-weight sense: the innermost import chain now carries an outer-layer *external* dependency.

No `import feedbax` chain reaches `fastapi`, `pandas`, `sklearn`, `dash`, `alembic`, `sqlalchemy`, `mujoco`, or `dill` — confirmed by tracing the full eager chain from `feedbax/__init__.py` through `config.mapping`, `runtime.selectors`, `runtime.graph`, `contracts.graphs.templates`, `models.cde`, `intervene`, `objectives.loss`, `tasks`, `tasks.presets`, and one level further into each of those modules' own imports.

---

### Test coverage table

Flat `tests/` directory (no subpackage mirroring — `find tests -type d` shows only `tests/`, `tests/fixtures/`, `tests/fixtures/path_expressions`), 109 files, 34,954 total lines. Single `conftest.py` (one fixture, `enable_jax_x64`, no duplication possible). `pytest-xdist` is a declared dev dependency but **not enabled by default** for feedbax's own suite (`addopts = "--strict-markers"` only, no `-n auto`) — xdist parallelism is a downstream-rlrmp convention, not exercised by feedbax's own `make test-all`.

| Subpackage | Test refs (files referencing `feedbax.<pkg>`) | Verdict |
|---|---|---|
| contracts | 46 | tested |
| runtime | 32 | tested |
| training | 22 | tested |
| mechanics | 20 | tested |
| objectives | 19 | tested |
| analysis | 17 | tested |
| config | 15 | tested |
| integrations | 12 | tested |
| web | 11 | tested |
| tasks | 10 | tested |
| intervene | 7 | tested |
| models | 7 | thin |
| component_registry | 6 | thin |
| studio | 6 | thin |
| persistence | 5 | thin |
| execution | 4 | thin |
| components | 3 | thin |
| plot | 3 | thin |
| plugins | 3 | thin |
| acausal | 2 | thin |
| bin | 2 | thin |
| control | 2 | thin |
| xabdeef | 2 | thin |
| testing | 1 | thin |
| dashboard | **0** | **untested** |

Largest files: `test_provider_contract.py` (2523 lines — breadth of manifest/schema assertions, not a heavy training loop), `test_graphspec_builtins.py` (1682), `test_analysis_spec_bundles.py` (1318), `test_graph.py` (1136), `test_graph_templates.py` (959). Spot-checked "training-loop-shaped" tests (`test_trainer_hotpath.py`, `test_worker_execution.py`, `test_provider_contract.py`) all use tiny `n_batches` (2-5) — the suite is architecturally broad, not slow-by-real-training. Only one `@pytest.mark.slow` use found across the whole tree; markers actually used (`parametrize`, `xfail`, `skipif`, `slow`) all match declared/built-in markers — `--strict-markers` would pass clean, no undeclared-marker violations. `feedbax_contract` marker family: **zero hits**, see F8. `/tmp/`-hardcoded paths: 6 hits, all inside mocked string-literal assertions, not real filesystem writes — low xdist-collision risk if xdist were ever turned on for this repo's own suite.

`components`, `plot`, `execution`, `dashboard` are the weakest-covered live subpackages (dashboard: zero coverage, matches its narrow-liveness flag in F11).

---

### Quick wins

1. Delete the empty, unreferenced `src/` directory (F1). Zero risk, zero dependents.
2. Fix `feedbax/tasks/task.py:39`'s eager `import plotly.graph_objs` — defer it into the function(s) that need it (F3). Contained, single-file change with an outsized payoff for import latency of the whole library.
3. Fix the two stale file-path references in `/feedbax/CLAUDE.md`'s own "Repository Structure" section (`nn_cde.py`/`nn.py` → `models/cde.py`/`models/networks.py`) (F13). The project's own agent instructions are currently wrong about where its core networks live.
4. Add `sqlalchemy` as an explicit direct dependency in `pyproject.toml` (F2). One-line fix, closes a silent transitive-dependency risk.
5. Add logging before the four silent `except`/`pass`/`continue` sites in `web/orchestration/manager.py:209,215`, `web/services/training_service.py:267`, and review `runtime/graph.py:77` (F14). Cheap, high-value observability fix for a repo whose core principle explicitly forbids silent substitution in graph-execution state.
6. Fix `feedbax/plot/mpl.py:219`'s `raise ValueError("TODO")` to at least carry a real message or become `NotImplementedError` (F17). One line, removes a landmine.
7. Bump `build_docs.yml`'s GitHub Actions versions and Python version to match `ci.yml` (F10). Mechanical CI-config parity fix.
