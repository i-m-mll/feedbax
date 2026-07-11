# MAX-SCRUTINY Code Review: feedbax Studio Backend (Python)

**Scope reviewed:** `feedbax/web/` (app.py, api/*, models/*, services/*, ws/*, worker/*, orchestration/*, config.py, decorators.py), `feedbax/studio/schema.py`, `feedbax/studio/execution.py`, `feedbax/integrations/provider.py`, `feedbax/dashboard/app.py`, `feedbax/bin/*`.

Repo root: `/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax`. React/TypeScript frontend under `web/` was explicitly out of scope and not reviewed. No repo files were edited (read-only review); no `.git` state was touched.

The top finding (C1) was independently re-verified by direct file inspection: `feedbax/web/worker/__init__.py` contains `main()` gated behind `if __name__ == "__main__":` (lines 37-38), and no `__main__.py` exists anywhere under `feedbax/web/worker/` (confirmed via glob). Python's `-m <package>` execution protocol requires a `__main__.py` submodule — an `__init__.py`'s `__main__` guard never fires that way. The invocation `python -m feedbax.web.worker` is used at exactly two call sites (`feedbax/web/services/training_service.py`, `feedbax/web/orchestration/startup_script.py`) plus a spec doc — confirmed via grep. This finding is real, not a false positive.

## Findings

### CRITICAL

**C1 — `python -m feedbax.web.worker` cannot execute; the worker subprocess launch path is broken as written, with the failure hidden by `DEVNULL` redirection**
Severity: critical | Area: worker/async
Evidence: `feedbax/web/worker/__init__.py:1-4` documents invocation as `python -m feedbax.web.worker ...` and gates `main()` behind `if __name__ == "__main__":` (lines 37-38) — but package-execution via `-m` requires a `__main__.py` submodule, which doesn't exist (verified: no `__main__.py` under `feedbax/web/worker/`). Both call sites use this broken invocation: `feedbax/web/services/training_service.py:128-131` (`subprocess.Popen([sys.executable, "-m", "feedbax.web.worker", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`) and `feedbax/web/orchestration/startup_script.py` (GCP VM bootstrap via `nohup`).
Why it matters: this is the mechanism by which Studio spawns local training and by which a GCP worker VM starts its worker process. The subprocess would exit immediately with `ModuleNotFoundError`, but `stdout/stderr=subprocess.DEVNULL` discards that traceback entirely — the caller only sees `wait_for_health` time out after 5s (`training_service.py:134`) and raise a generic `RuntimeError`, with zero indication of root cause. On GCP this is worse: the VM boots, bills, and never becomes usable, with the failure buried in a VM-local log nothing on the Studio side inspects.
Proposal: add `feedbax/web/worker/__main__.py` (`from feedbax.web.worker import main; main()`); stop discarding subprocess stderr (capture via `subprocess.PIPE`); add a CI smoke test invoking `python -m feedbax.web.worker --help`.
Effort: S (fix) / M (regression test + investigating how this shipped)
Overlap: none of the tracked items cover this; it's upstream of and would block d18cf9c/9aa8ff2/d90b3e5 work that exercises the actual worker subprocess.

**C2 — `/api/orchestration/launch` accepts an unvalidated free-form shell-command string executed as root on a freshly billed GCP VM**
Severity: critical | Area: orchestration
Evidence: `feedbax/web/api/orchestration.py:42` (`feedbax_install_cmd: Optional[str] = None`) flows unvalidated into `InstanceConfig` (lines 105-108), then `feedbax/web/orchestration/gcp.py:183` (`make_startup_script(config.feedbax_install_cmd)`) f-string-embeds it directly into the instance's `startup-script` metadata (`startup_script.py:18-20`), executed as root by GCP's guest agent on boot. No allowlist, regex check, or escaping anywhere in the path.
Why it matters: any caller of this endpoint is a full RCE primitive against billing cloud infrastructure the moment this API is reachable beyond localhost (including CSRF-style requests from a malicious webpage while Studio runs locally).
Proposal: drop the parameter from the public request schema (server-config-only, or a small allowlisted set of named install profiles), or validate strictly (e.g., `pip install (feedbax|feedbax==\S+)` only).
Effort: S | Overlap: none.

### HIGH

**H1 — Worker health polled only once at VM creation; nothing re-checks liveness afterward**
Severity: high | Area: orchestration
Evidence: `feedbax/web/orchestration/manager.py:221-256` (`refresh_status`) only re-polls GCP VM-level status. Once `status="running"` is set (`manager.py:167-168`), `wait_for_health` is never called again.
Why it matters: combined with C1, a VM can sit `RUNNING` at the GCP layer indefinitely while its worker never started or has since crashed, invisible to Studio until a training request times out with no diagnostic.
Proposal: periodically re-poll worker `/health` while `status == "running"`; transition to `"error"` after repeated failures.
Effort: M | Overlap: none.

**H2 — No process supervision for the worker on the GCP VM; a crash is silent and unrecovered**
Severity: high | Area: orchestration
Evidence: `feedbax/web/orchestration/startup_script.py:29-34` launches the worker via bare `nohup ... &` — no systemd unit, no restart-on-failure, no log forwarding.
Why it matters: combined with H1, any worker crash permanently strands the VM in an apparently-healthy state with no recovery or signal.
Proposal: wrap the worker launch in a systemd unit with `Restart=on-failure`, or a minimal supervisor; surface degraded/crash state via `/health`.
Effort: M | Overlap: none.

**H3 — No confirmation gate, spend cap, or concurrent-launch guard before a billable GCP instance is created**
Severity: high | Area: orchestration
Evidence: `feedbax/web/api/orchestration.py:80-116` (`launch_instance`) creates a billable instance via `background_tasks.add_task` on any POST to `/launch` — no confirmation step, no dry-run, no rejection when an instance is already `running`/`creating`.
Why it matters: rlrmp's own RunPod runbook (spec-lock + explicit user confirmation before billable launches) establishes exactly this discipline for a sibling cloud-training path; the GCP path here has no analogous gate, and a double-click/retry could provision two concurrent VMs.
Proposal: require an explicit confirm flag or two-step validate-then-confirm flow; reject `/launch` when current state isn't `("idle", "error")`.
Effort: M | Overlap: none — parallel gap to the hardened RunPod/Modal convention elsewhere, worth harmonizing but not duplicating any tracked item.

**H4 — Hardcoded trainable-node type list silently substitutes for anything the canvas actually expresses**
Severity: high | Area: worker
Evidence: `feedbax/web/worker/execution.py:42-49` defines `_DEFAULT_TRAINABLE_COMPONENT_TYPES = {"Linear", "MLP", "GRU", "LSTM", "Recurrent Controller", "Simple Feedback Loop"}`; `_derive_trainable_nodes` (lines 535-543): `if raw is True or node.type in _DEFAULT_TRAINABLE_COMPONENT_TYPES: trainable.append(node_id)`. The frontend never emits an explicit `trainable` param on any node, so in practice every graph's trainable-node set is determined entirely by this hardcoded type allowlist.
Why it matters: this is precisely the "inferring architectural choices the canvas doesn't express" pattern the project's core policy calls a bug. A user wanting to freeze a `GRU` backbone while training a `Linear` readout has no way to express that — the worker silently decides for them.
Proposal: require every trainable-eligible node type to carry an explicit `trainable` param emitted by Studio's own node-creation defaults (making the graph spec authoritative even in the common case), or if the type-based default is an intentional authoring convenience, surface it visibly in the Studio UI (per-node trainable toggle) so what's rendered matches what's built.
Effort: M | Overlap: none.

**H5 — Studio's schema/validation layer never checks composite/Network nodes for a missing subgraph, unlike the actual build path**
Severity: high | Area: schema
Evidence: `feedbax/studio/schema.py:795-853` (`_enumerate_graph_ports`) computes `subgraph = graph.subgraphs.get(node_id) if graph.subgraphs else None` (line 805) but proceeds to `_port_schema` regardless; `_component_port_dimension` derives shape from `node.params["out_size"]`/`["hidden_size"]` directly (lines 989-993), consulting `subgraph` only as secondary fallback. No `missing_subgraph`-class issue type exists anywhere in `schema.py` (grepped, zero hits) — contrast with the correct build-time enforcement in `feedbax/contracts/graphs/serialization.py:1035-1043` ("Network node {node_name!r} has no subgraph...").
Why it matters: this is the outer/stale-params-over-subgraph pattern the CLAUDE.md policy explicitly forbids, occurring in exactly the surface (live Studio validation) meant to catch it before training is attempted. A Network node whose subgraph was cleared still enumerates a plausible `PortSchema` from stale params with zero validation warning, then hard-fails later at build time through a disconnected error path. (The build-time enforcement itself is correct and verified solid — this is a validation-surface gap, not evidence the build guarantee is broken.)
Proposal: in `_enumerate_graph_ports`, emit a `SchemaValidationIssue` (type `missing_subgraph`, severity `error`) for any subgraph-bearing node type when `subgraph is None`, mirroring the build-time message, before falling back to param-based shape inference.
Effort: M | Overlap: none.

### MEDIUM

**M1 — `TrainingConfig.batch_size` is parsed, stored, and completely unused; only `TrainingSpec.batch_size` (fixed at 1) is enforced**
Evidence: `feedbax/contracts/training.py` declares both `TrainingSpec.batch_size` (~line 87-96) and `TrainingConfig.batch_size: int = 128` (~line 107-136). `execution.py:116-120` validates `training_model.batch_size != 1 → raise`, but the parsed `_TrainingCfg.batch_size` from `worker/app.py:203` never appears in `run_training_graph`'s body.
Why it matters: a Studio panel wired to `TrainingConfig.batch_size` could let a user set "64" and see it silently accepted even though the worker only ever runs `batch_size=1` — a silent-stale-value trap on a training hyperparameter.
Proposal: remove `batch_size` from `TrainingConfig`, or wire it through and drop the `== 1` restriction.
Effort: S | Overlap: none.

**M2 — `_write_checkpoint` leaves permanent, unmanaged `tempfile.mkdtemp` directories with no cleanup on job eviction**
Evidence: `execution.py:680-685` writes every checkpoint under a fresh `tempfile.mkdtemp(prefix="feedbax_ckpt_")`, never removed. `_evict_terminal_jobs_locked` (bounded at `_TERMINAL_JOB_RETENTION_MAX = 32`) evicts the in-memory job record but never touches `job.checkpoint_path` on disk.
Why it matters: unbounded `/tmp` growth over long Studio sessions.
Proposal: `shutil.rmtree` on eviction, or write under a single managed, periodically-swept root.
Effort: S | Overlap: none.

**M3 — `training_service` module-level singleton relies on `__del__` to terminate its subprocess, which has no reliable timing guarantee**
Evidence: `services/training_service.py:411` (`training_service = TrainingService()` at import time); `__del__` (~407-408) calls `_terminate_worker()`. Consumed via direct module import (not `Depends`) from `api/training.py:24`, `api/orchestration.py:24`, `ws/training.py:6`.
Why it matters: `__del__` timing is unreliable under CPython GC/interpreter shutdown ordering; an ungraceful exit can orphan the worker subprocess. Lack of DI also blocks isolated testing.
Proposal: terminate the worker subprocess from an explicit FastAPI shutdown/lifespan hook; convert to `Depends`-injected singleton.
Effort: M | Overlap: none.

**M4 — `ws/training.py`'s WebSocket handler has no `WebSocketDisconnect`-specific handling; disconnect produces a noisy secondary exception**
Evidence: `feedbax/web/ws/training.py:11-26` — `except Exception` catches what `websocket.send_json` raises on a closed socket, then the `except` block itself attempts another `send_json` to the same closed socket, uncaught.
Proposal: catch `WebSocketDisconnect` specifically and return without further sends; wrap `finally`'s `close()` in its own try/except.
Effort: S | Overlap: none.

**M5 — `api/runs.py`'s `create_eval_run` swallows DB persistence failures and still returns success with an untracked `run_id`**
Evidence: `api/runs.py:263-291` — `except Exception as exc: logger.warning(...)` followed by an unconditional success `EvalRunInfo` response.
Why it matters: caller receives identical success whether or not the record was written; a subsequent lookup would never find it, with no signal to the user.
Proposal: return a `persisted: bool` field, or fail the request with 500 if persistence is required.
Effort: S | Overlap: none.

**M6 — `terminate()` resets orchestration state to `"idle"` even when delete/disconnect calls fail**
Evidence: `manager.py:206-219` — `delete_instance` and `_terminate_worker()` wrapped in bare `try/except Exception: pass`, then unconditional `OrchestrationState(status="idle")`. Contrast: `launch()`'s failure path does record `orphaned_instance`.
Why it matters: a failed `DELETE /instance` call leaves the VM running/billing while Studio's state claims nothing is running.
Proposal: on delete failure, surface an error/orphaned state rather than silently resetting to idle.
Effort: S | Overlap: none.

**M7 — In-memory-only orchestration state; a backend restart permanently orphans any running GCP instance, and `list_instances` (which could reconcile this) is never called**
Evidence: `OrchestrationManager.__init__` (`manager.py:65-67`) is purely in-process. `gcp.py:254-280` (`list_instances`) is never imported/called from `manager.py` or `orchestration.py`.
Proposal: persist minimal instance state on each transition; reconcile via `list_instances` on backend startup.
Effort: M | Overlap: none.

**M8 — `provider.py` is a genuine god-module mixing five distinct responsibilities across 2224 lines**
Evidence: mixes (a) Pydantic schema definitions (~102-227), (b) a static capability/manifest registry with ~300 lines of inline literal mapping data (`provider_manifest()` 526-836, `_mandible_manifest_mappings()` 311-523), (c) five bespoke `*_registry_snapshot()` functions with no shared abstraction, (d) a large `validate_*_spec` family (1038-2192), (e) an analysis-data-product mismatch-reconciliation subsystem (1805-1987).
Proposal: split into `provider/manifest.py`, `provider/registries.py`, `provider/validation.py`, `provider/data_products.py` — pure decomposition, no behavior change.
Effort: L | Overlap: none.

**M9 — Mux/Demux port-count normalization silently rewrites node arity from incidental wiring, with no validation issue raised**
Evidence: `schema.py:742-792` (`_normalize_dynamic_graph_ports`) mutates `Demux.output_ports` and `Mux.input_ports`/`n_inputs` from observed wiring, used for all downstream validation, with no `SchemaValidationIssue` when the node's declared shape disagrees with wiring.
Why it matters: borders on "background construction" — inferring/overriding structure the canvas-authored spec didn't declare, silently, and undocumented as intentional.
Proposal: document explicitly as wiring-derived-by-design, or emit a `mux_arity_mismatch` issue instead of silent override.
Effort: S (doc) / M (new issue type) | Overlap: none.

**M10 — Derived-dimension issue reporting and the returned schema graph can disagree**
Evidence: `schema.py:464-477` catches `DerivedDimensionError` and returns the original, unresolved graph silently; `_derived_dimension_issues` (479-517) independently re-derives and correctly reports the same failure as an issue.
Why it matters: internally inconsistent output — issues say "conflict" while ports/selectors in the same response are computed against an unresolved graph.
Proposal: compute normalization once; feed both graph and error into the same issue-reporting path; tag ports from a failed normalization as provisional.
Effort: M | Overlap: none.

**M11 — `evaluation_type`/`analysis_type`/`report_type` are unconstrained `str`, validated only by truthiness, not against a registered set**
Evidence: `contracts/manifest.py:402,561,581`; `execution.py:889-895` reads `analysis_type` from a free-form dict and raises only if falsy, not if unregistered.
Why it matters: a typo'd/drifted value fails deep in the analysis dispatcher with a less specific error than schema-level rejection would give.
Proposal: validate against the provider's registry before constructing the spec, raising a clear error listing known types.
Effort: M | Overlap: none.

**M12 — `dashboard/app.py` shares one module-level SQLAlchemy `Session` across Dash's callback execution**
Evidence: `feedbax/dashboard/app.py:38,394` — a single `_db_session` read from multiple `@app.callback` functions; SQLAlchemy `Session` is not thread-safe.
Proposal: use `scoped_session`, or open/close per callback.
Effort: S-M | Overlap: none (relevance depends on dashboard's keep/retire status — see dead-code verdict).

**M13 — `db_merge.py` builds `ALTER TABLE`/`ATTACH DATABASE` SQL via raw f-string interpolation of identifiers sourced from arbitrary input `.db` files**
Evidence: `feedbax/bin/db_merge.py:35,90,148-153,196-203`.
Why it matters: `sqlite3` can't parameterize identifiers, so a crafted `.db` file with adversarial table/column names achieves SQL injection via schema metadata.
Proposal: validate identifiers against `^[A-Za-z_][A-Za-z0-9_]*$` before interpolation.
Effort: S-M | Overlap: none.

**M14 — `studio_pipeline.py` fabricates a synthetic training-loss curve, self-labeled only in prose `notes`, not a machine-checkable schema flag**
Evidence: `feedbax/bin/studio_pipeline.py:48-71` — `final_loss = round(1.0/(1.0+total_batches), 8)`, with notes stating this is a stub.
Why it matters: any downstream consumer parsing `final_loss`/`history` programmatically (rather than reading `notes`) treats fabricated numbers as real telemetry.
Proposal: add a structured `"materialization_kind": "stub"` field so downstream tooling can fail closed instead of relying on prose.
Effort: S | Overlap: adjacent to but not duplicating tracked worker items (d18cf9c, d90b3e5) — none of those explicitly cover this CLI's fabricated-metrics shape.

### LOW

- **L1** — `execution.py:560-565`'s `_dry_run` preflight discards JAX traceback structure at the HTTP boundary. Log full traceback server-side before re-raising cleaned message. (S, none)
- **L2** — `worker/client.py:281-283`'s SSE consumer treats any unrecognized exception as silent clean termination (`except Exception: return`). Log at warning level first. (S, none)
- **L3** — `execution.py:627-649`'s `_jsonable_value` fallback silently `repr()`s unknown pytree leaf types into trajectory JSON with no warning. Log/warn on fallback. (S, none)
- **L4** — No payload size cap on `_retained_observables_payload` (`execution.py:613-624`), unlike `StatisticsService`'s explicit 200-point downsample (`services/statistics_service.py:57,100-107`). Apply same convention. (M, adjacent to d18cf9c but distinct — retention semantics vs raw payload size)
- **L5** — `services/graph_service.py:206-237`'s cycle-detection DFS is unbounded recursion, no depth guard — large graphs could hit `RecursionError` → 500. Convert to iterative DFS. (S, none)
- **L6** — GCP startup script transmits Tailscale auth key and worker auth token via plain instance metadata (`gcp.py:184-189,198`), readable by any VM process or anyone with `compute.instances.get`. Use Secret Manager references. (M, none)
- **L7** — GCP instance creation sets no explicit `--service-account`/`--scopes` (`gcp.py:167-209`), inheriting default (likely broad) scope. Add a pinned minimal-scope service account field. (S/M, none)
- **L8** — `bin/analysis.py:192` silently overrides a user-supplied `--no-pickle` flag in batched mode (`no_pickle=... if args.single else True,  #! For now, ...`) — exactly the "just for now" pattern CLAUDE.md calls a bug regardless of comment framing. Warn or error instead of silent override. (S, none)
- **L9** — `db_merge.py:91-96` matches expected SQLite errors by message substring (`"duplicate column name" not in str(e)`) with no logging on the swallowed path. Check via `PRAGMA table_info` instead; log swallow at debug. (S, none)
- **L10** — `bin/run.py:55-60`'s `_dispatch` has a dead error-handling branch: `importlib.import_module` never returns `None`, so the friendly "could not locate module" message is unreachable and real failures surface as raw `ModuleNotFoundError`. Wrap import in try/except. (S, none)
- **L11** — `bin/dashboard.py:29-31,53-57` permits `--host 0.0.0.0 --debug` simultaneously — a known Werkzeug/Dash RCE footgun. Refuse to start (or require explicit unsafe-override) when combined. (S, none)
- **L12** — `schema.py:1510-1579`'s `_static_task_parameter_labels` reads only outer `node.params` for composite nodes, never the subgraph; only `AffineValueComposer` is special-cased by literal type-string match. Extend to consult subgraphs when present; move to a registry-level `ComponentMeta` hook. (M, none)
- **L13** — `StudioExecutionPreparationError` (`execution.py:138`) is a single flat exception class for every failure mode, unlike the disciplined `MigrationError` hierarchy in `contracts/migrations.py`. Split into subclasses or add `.error_code`. (S, none)
- **L14** — `execution.py:600-603` computes repo root via magic-number `Path(__file__).resolve().parents[2]` with no assertion it's actually the repo root. Add assertion or derive from `feedbax.__file__`. (S, none)
- **L15** — `provider.py` performs no direct file/manifest I/O — the atomic/partial-write risk named in scope applies to other modules (`worker/execution.py`, `contracts/manifest.py`), not this one. Recorded so follow-up work isn't mis-scoped. Not an action item.
- **L16** — `api/analysis.py:30-33,105`'s single-worker `ThreadPoolExecutor(max_workers=1)` silently serializes concurrent analysis requests with no queued-state visibility — a second request looks identical to a hung one. Expose a distinct `queued` status. (S, none)

## Dead-code verdict: `feedbax/dashboard/`

**Not orphaned — live but isolated.** `pyproject.toml` declares a `dashboard` optional-dependency extra; `feedbax/bin/dashboard.py` imports `feedbax.dashboard.app.create_app` with a working CLI documented in `feedbax/dashboard/README.md`. However, a repo-wide search found zero references to `feedbax.dashboard`/`dashboard.app` outside those two locations — not imported by `feedbax/__init__.py`, not referenced anywhere in `feedbax/web/`, absent from `[project.scripts]` (unlike `feedbax-run`/`-train`/`-analysis`/`-provider`), zero test coverage. It operates on a separate SQLite-backed `FigureRecord`/`EvaluationRecord` substrate entirely distinct from the GraphSpec/manifest artifacts Studio operates on — it never touches a GraphSpec, so "the graph is the model" policy doesn't directly bind it. It's a narrow, single-entry-point, untested, non-Studio figure-browsing tool; whether it's deliberately kept distinct or quietly stale is a maintainer call the evidence doesn't resolve on its own.

## Architecture Assessment

The Studio backend is a fundamentally sound design let down by two categories of gap: operational hardening around the worker/orchestration boundary, and a validation surface that lags the (correctly implemented) build-time enforcement of "the graph is the model."

On the positive side, the policy's actual hard constraint — a composite node's subgraph, when present, is authoritative, and its absence is a build-time error — is implemented correctly and verified directly in `feedbax/contracts/graphs/serialization.py`, which hard-errors on missing `Network`/`Subgraph` subgraphs rather than falling back to outer params. The schema-versioning/migration infrastructure in `feedbax/contracts/migrations.py` is unusually rigorous for a single-developer project: explicit per-family migrate-vs-reject stance, registry-declared required test files, and correct conversion of migration failures into structured issues rather than crashes or silent acceptance of stale formats. No hardcoded secrets, `shell=True`, or `os.system` calls were found anywhere in the CLI layer, and the codebase is nearly free of "TODO/HACK/for now" smells — the two instances found (L8, M14) are both honestly documented in comments/notes, just not enforced at a type-checkable level.

The most consequential problem is **C1** (verified independently): the worker subprocess launch invocation cannot succeed because the package lacks a `__main__.py`, and the `DEVNULL`-redirected subprocess output means this failure is currently invisible. Both the local dev path and the GCP-provisioned path route through this exact same broken invocation — either the worker subprocess path is currently non-functional in the reviewed tree, or there's an out-of-tree fix not reflected here. This should be triaged first: it's upstream of essentially every other worker/training/orchestration finding — if the worker can't start, H1/H2/M2/M3 about its lifecycle are moot until this is fixed.

The second systemic risk is the orchestration layer's **optimistic-success posture**: in-memory-only state (M7), silent success-on-partial-failure in `terminate()` (M6), no post-launch health re-polling (H1), no process supervision on the VM (H2), no cost-safety gate before billable launch (H3, C2) compose into real risk of orphaned, silently-billing GCP instances Studio has no way to rediscover — notably, `list_instances` already exists to reconcile this and is simply never called. This is architecturally the same "silent stale-value substitution" failure mode the project's core policy calls out for models, just manifesting in infrastructure state; the fix pattern (fail loud, make absence-of-truth an error) is one the codebase already applies well elsewhere.

The third theme is a **validation/build asymmetry in Studio's schema layer**: the build path enforces subgraph-authority correctly, but the live validation surface a Studio user sees while editing doesn't mirror that enforcement (H5), and a related normalization function can silently diverge from what its own issue list reports (M10). Neither breaks the core guarantee — a bad graph still can't be trained — but both mean a user gets no useful signal until "train" is clicked, rather than fast in-canvas feedback.

`provider.py` (M8) and the worker's hardcoded trainable-type inference (H4) are the clearest instances of "policy says X, code quietly does something adjacent to X" outside infrastructure: the former is a maintainability risk rather than a correctness violation; the latter is the review's best candidate for an actual policy violation in the sense CLAUDE.md cares most about — an architectural choice (which nodes train) made by the worker rather than expressed by the canvas.

`feedbax/dashboard/` is best understood as accumulated-but-contained scope: real, packaged, launchable, but with zero cross-references and zero tests — worth an explicit keep-or-retire decision rather than continued silent investment or assumed deadness.

## Quick Wins

- **C1**: add `feedbax/web/worker/__main__.py`; stop `DEVNULL`-discarding worker subprocess stderr. Highest value/lowest effort fix in the review — likely unblocks local training entirely.
- **C2**: drop or strictly allowlist `feedbax_install_cmd` in the orchestration launch request.
- **M1**: delete the unused `TrainingConfig.batch_size` field (or wire it through).
- **M6**: don't silently reset orchestration state to `idle` on failed `terminate()` — mirror the existing `orphaned_instance` pattern already used in `launch()`.
- **M7**: wire the already-existing but unused `list_instances` into a startup reconciliation pass.
- **L8**: stop silently overriding `--no-pickle` in batched mode.
- **L10**: fix the dead/unreachable error branch in `bin/run.py`'s `_dispatch`.
- **L11**: refuse `--host 0.0.0.0 --debug` together in `bin/dashboard.py`.
- **M2**: clean up orphaned `_write_checkpoint` temp directories on job eviction.
- **H5/M9**: add the missing `missing_subgraph` validation issue type to mirror the build-time check — cheapest fix in the schema layer.

---

Note: this review did not create a ledger issue. Given the mandible-cowork global policy (implementation work requires a ledger issue before starting), if any of these findings — especially C1 and C2 — should be turned into actual fixes, that follow-on work should go through a tracked feedbax issue first.
</content>
