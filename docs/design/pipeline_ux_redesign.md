# Pipeline Pane Redesign — Staging, Sweeps, Lineage, and Launch

> **Reconstruction notice:** This document was reconstructed post-hoc. The code is
> authoritative wherever it disagrees with this document.

Status: PROPOSAL (uncommitted working notes, 2026-07-05). Companion: `pipeline_ux_ledger_map.md`
(maps every proposal here onto existing ledger issues or proposed new children of `4007bf2`).

This document redesigns the bottom pane of Feedbax Studio — the pipeline/provenance workspace —
around a small set of unifying concepts. It is grounded in code archaeology (frontend, backend,
ledger) done 2026-07-05; §2 records the diagnosis so claims are checkable.

---

## 1. The one-sentence design

**The bottom pane is a workspace over manifests: every intention the user stages becomes a
persisted, content-addressed, typed spec (a pending manifest); sweeps are declared as axes and
explicitly expanded through a previewed matrix; downstream stages bind to upstream work through
selections-that-are-queries; and one selection context drives bidirectional scoping between the
run tables and the top-pane canvas.**

Everything else in this document is an elaboration of those four moves. They deliberately echo
the project's core principle: *the graph is the model* → *the pipeline is the provenance graph*.
Nothing in the bottom pane is decorative; every visible row lowers to a typed feedbax spec
(0c46f69 decision #5) and every executed row is a manifest.

---

## 2. Diagnosis (current state, verified 2026-07-05)

Condensed; each claim was verified against code or ledger this session.

**D1 — The launch surface is orphaned.** (Corrected 2026-07-06 after direct inspection.)
`TrainingPanel.tsx` — loss tree, optimizer config, the Prepare/Run-local/Materialize/Start
buttons, WebSocket loss chart, checkpoint download — is defined but **mounted nowhere**; nothing
imports it. The right sidebar renders `ScenarioInspectorPanel`/`TreescopePanel`/
`ValidationPanel`, and objective editing has already migrated to the Objectives projection. So
today: the bottom-shelf `TrainCollectionPanel` has config fields and a run table but no launch,
the staged-execution flow (`POST /api/provider/studio/training/plan` → `run-local`) is
unreachable from any mounted UI, and the Eval tab's "Run selected" button is unconditionally
`disabled` ("Backend execution wiring is pending"; `RunCollectionStagePanel.tsx:356`). There is
no reachable way to launch training. The design question is not whether to dissolve
`TrainingPanel` (it's de facto dissolved) but where its organs land: scenario-describing config →
top-pane projections/sidebar; run-creating actions → run table + queue; loss chart + checkpoint
download → focused-run detail.

**D2 — Two parallel selection models.** `workspaceStore` stage `selection_spec` arrays
(multi-select run tables) vs `runStore`'s single `selectedTrainingRunId`/`selectedEvalRunId`
(dropdown pickers, only half-mounted). They sync into each other; neither is authoritative.

**D3 — Runs are untyped metadata reshaping.** Run tables are derived ad hoc from
`StudioManifestRef.metadata` (`pipelineCollections.ts`), seeded from a static example. The
backend has a full manifest system (`TrainingRunManifest` etc. under `FEEDBAX_RUNS_DIR`) that the
tables do not read. `createTrainingRun` is a client-side stub (`runAPI.ts:146` TODO); eval
creation fakes completion with a 1.5 s `setTimeout`.

**D4 — Sweeps are a dead end.** `StudioValueSpec` supports `sampling_scope: "sweep"`, and the
sidebar `ValueSpecField` will happily author one — but *nothing consumes it*. There is no
expansion code anywhere (`grep sweep` across `feedbax/` finds only: the `ExecutionCell` docstring
in `execution/models.py:92`, the `authored_sweep` `AxisRole` in `contracts/worker.py:35`, and
legacy zipped/broadcast YAML batch sweeps in `config/batch.py`). `TrainingRunSetManifest` has
`run_ids` but no axes metadata, and nothing populates it from a ValueSpec.

**D5 — ValueSpec lowering is untyped.** `literalFromValueSpec` unwraps `constant` specs to plain
literals; any *non*-constant spec is stored as a raw dict inside `ComponentSpec.params`
(`ParamValue = Union[..., Dict[str, Any]]`). A sweep-marked component param is a schemaless dict
parked in the graph spec with no contract and no consumer. (Task-timeline signals are the one
place ValueSpec has a real backend path, via `StudioTaskBindingSpec.exposed_data[*].value_spec`.)

**D6 — The dependency machinery landed but has no UI.** The 2026-07-03/04 wave (196bd09 path
expressions, 3300ced `run_condition` + `depends_on_roles` + materialization stages, b0e28c2
extraction, 7af0adf eval-recipe contract, 1c89aa7 durable eval custody, 132f98c report registry,
589535c import/export packets) gives the backend runtime-conditional stages, artifact-role
dependencies, `skipped`/`not_applicable`/`missing` statuses, and content-addressed eval identity
— none of which is surfaced anywhere in Studio.

**D7 — No delete/duplicate/compare/group affordances for runs; no report UI; `compare`/`import`/
`export`/`protocol` stage kinds fall through to a JSON dump panel.**

Ledger context: live umbrella `4007bf2`; `0c46f69` owns the pipeline-workspace spec ("stages
operate on collections, not one globally active run"); `96ac771` (fresh, 0 comments) owns run
authoring/sweep/run-matrix UX; `bd5aefb` closed having landed only a narrow ValueSpec substrate —
the generalized system is explicitly not done; `d90b3e5` root-cause fixed, UI pending.

---

## 3. Design principles

Distilled from a precedent survey (W&B sweeps, MLflow, Optuna, Ray Tune, Guild, ClearML, Neptune,
Simulink Test Manager, COMSOL, Ansys, JMP DOE, Dagster, Prefect, Airflow, GitHub Actions matrix;
digest in §11):

- **P1 Preview before commit.** Every expansion (sweep matrix, query match set, conditional-stage
  dry run) is materialized visibly *before* compute is spent. (Simulink "Show Iterations", JMP
  design table, Ansys DOE preview. COMSOL's missing count-preview is the anti-example.)
- **P2 One launch entry point per semantic.** Two buttons that look like "run this" must share
  one data model and one code path. (Dagster's job-tab vs asset-graph backfill divergence,
  dagster#17665, is the cautionary tale — and D1 is our local instance of it.)
- **P3 Intention is typed and persistent.** Staged-but-not-launched work survives save/load
  (No-volatility rule). Nothing user-visible lives only in a Zustand store.
- **P4 Pinned vs varied is first-class.** A parameter that varies across runs is visually and
  schematically distinct from one that doesn't, everywhere it appears. (W&B `value` vs `values`;
  Figma variant properties.)
- **P5 Cross-view sync is total or explicitly off.** Hover-preview + click-commit, bidirectional,
  with a pin/decouple escape hatch. Partial sync trains distrust (wandb#6335).
- **P6 Every status is attributable.** Skipped, stale, filtered, evicted, superseded — each badge
  carries a reason string traceable to what set it. (wandb#3917 is the anti-example.)

---

## 4. Core model

### 4.1 Staged work = pending manifests

Kill the notion of frontend-only "staged runs". **Staging writes a real manifest with
`status: "pending"`** (the vocabulary already exists: `ManifestStatus = pending | running |
completed | failed | cancelled`). A pending `TrainingRunManifest` carries full snapshot
`SpecPayload`s (graph/training/task/binding specs) exactly like a completed one — it just hasn't
run.

Consequences, all free:

- **No-volatility satisfied by construction.** Plans persist in `FEEDBAX_RUNS_DIR` and reload
  with the project; the run table is uniformly manifest-backed and D3's ad-hoc metadata
  reshaping is replaced by one typed run-summary/index API (this is `e33f487`'s scope).
- **Idempotent staging.** Give planned runs deterministic content-addressed ids —
  `training_run_manifest_id(spec)` hashing (graph_spec, training_spec, task_spec, seed, axis
  coordinates), exactly as `evaluation_run_manifest_id` already does. Re-staging an unchanged
  matrix is a no-op; the preview can say "3 of 15 already staged".
- **One lifecycle.** `draft → pending → running → completed | failed | cancelled`, plus the
  bundle-stage vocabulary `skipped | not_applicable | missing` for conditional work. One status
  chip component, one color language, everywhere.
- **Launch = execute a pending manifest.** The existing plan/run-local/orchestration machinery
  becomes the *executor* of pending manifests rather than a parallel path (resolves D1/P2).
  `TrainingPanel`'s config editing migrates into the top pane's scenario editing (where loss
  tree/optimizer conceptually belong — they are scenario properties); its launch buttons migrate
  into the run table and queue.

Deletion is allowed for `pending` only; completed manifests are immutable provenance (a
`superseded` marker, not deletion, hides them from default views).

### 4.2 Axes, matrices, expansion (the sweep model)

Vocabulary (aligned with the worker contract's existing `AxisRole "authored_sweep"`):

- **Axis** — a parameter path bound to an *enumerable* variation: an explicit value list, a
  linspace/logspace range (min, max, n), or a sampler (distribution + n_samples + seed). Declared
  by giving the parameter's ValueSpec variation-scope `sweep` (§5).
- **Matrix** — the set of axes currently declared on a stage's scenario, plus a **combination
  mode**: `cross` (Cartesian), `zip` (lockstep; validated equal lengths, cf. `config/batch.py`'s
  existing zipped/broadcast semantics), or `manual` (start from cross/zip, then hand-prune rows).
  Mixed designs compose: axes are grouped, groups zip internally and cross with each other —
  COMSOL's "specified vs all combinations" made compositional.
- **Expansion** — Matrix → N pending `TrainingRunManifest`s + one `TrainingRunSetManifest`
  extended with an `axes` block (parameter paths, per-axis values, combination spec, and each
  run's coordinates). Schema change → versioned migration rule per repo policy.

UX in the Train tab (this is `96ac771`'s acceptance slice, generalized):

1. User marks `hidden_activity_loss.weight` as a sweep axis with values `[0, 1e-5]` in the
   sidebar (§5). The Train tab badge increments: `⇉ 1 axis · 2 planned`.
2. The Train tab's **Matrix builder** strip shows each axis as a chip (param path, value count,
   remove/edit), the combination-mode control, and a live count: `2 × 3 = 6 runs`.
3. Below it, the **expansion preview**: ghost rows in the run table itself (not a separate
   widget), one per would-be run, with **dynamic columns for varied axes only** and
   auto-generated labels from varied axes only (`weight=0`, `weight=1e-5` — the rlrmp run-label
   convention, enforced by construction: constants live in the set spec, not in labels).
4. **Stage N runs** writes the pending manifests + set manifest. Ghost rows solidify into
   pending rows. Launch is a separate act (§6, Queue).

Sweeping is stage-generic: `Matrix = base × axes`. For Train, base = the scenario. For Eval,
base = the training-run selection, and axes come from eval params (perturbation amplitude grid,
noise levels...) plus a checkpoint policy dimension (§6). Same builder, same preview, same
combination modes.

**Multi-param and after-the-fact sweeps** (the "sweep 2 params over 5 runs" family) are handled
by *bulk-edit verbs* on selected pending rows, not by more modal machinery (§5.4).

### 4.3 Selections are queries

Today `selection_spec.training_run_ids` is a frozen id list. Generalize `SelectionSpec` to three
forms:

- **Explicit** — the id list (checkbox selection), as now.
- **Query** — a `ManifestPredicate`/path-expression (196bd09 — already the binding mechanism of
  `AnalysisBundleSpec.predicate`). Authored via chips, not raw AST: *from set X* · *status
  completed* · *has checkpoint* · *lowest `final_validation_loss` per set* · *tag Y*. A `top-k
  by metric per group` term is the one genuinely new predicate primitive needed.
- **Frozen** — a query plus the snapshot of what it matched when downstream work was staged
  (recorded as `ParentRef`s in the consuming manifest — reproducibility is never query-relative).

UI: a query selection shows both the chips and the live match count/preview (P1). A **refresh
matches** action diffs old vs new materialization (`+2 new, 1 no longer matches`) and offers the
Airflow-style reprocess tri-state: run for *missing only* / *missing + failed* / *all*. This —
predicate binding + dry-run + content-addressed caching — is precisely the "exchangeable
eval↔analysis mapping": retargeting a bundle at a different eval set is editing its predicate,
and unchanged eval specs are cache hits by manifest identity, so re-pointing is nearly free.

### 4.4 One selection context; provenance mode

Replace the D2 duality with a single `SelectionContext`: `{stage, collection, selectedIds:
Set, focusedId}` (selected set ≠ focused item, per Figma/Blender/Houdini convention). Hover =
preview highlight, click = commit, bidirectional between run tables, lineage view, and the top
pane; an explicit pin toggle decouples when wanted (P5).

**Provenance mode** is the run↔canvas integration: focusing a run and hitting *View snapshot*
puts the top pane into a read-only projection of that run's snapshotted specs — banner
("Viewing `weight=1e-5` · pending — frozen snapshot"), param fields showing the run's values
with **diff badges vs the current draft** (spec-hash comparison per subtree), one-click *Back to
draft*. Two actions close the loop:

- **Promote to draft** — copy the run's params into the current scenario (the
  duplicate-and-iterate gesture; replaces a "duplicate run" button with something better).
- **Restage from here** — new pending manifest from this snapshot with a param override diff.

This also answers "how do eval runs scope the canvas": an eval run's provenance mode scopes the
*task* projection to the eval's task/condition spec, and is exactly the state in which the
workspace view (sibling design lane) can animate that run's trajectories. The two designs meet
at this seam: `SelectionContext.focusedId` + provenance mode is the contract the workspace view
should consume.

---

## 5. ValueSpec: diagnosis and redesign

(Yes — `StudioValueSpec` is the right name: `feedbax/contracts/graph.py:470`, mirrored in
`web/src/types/workspace.ts`; editor in `web/src/components/values/ValueSpecField.tsx` with
vocabulary in `features/scenario/valueSpecs.ts`.)

### 5.1 Why the modal gives you a headache

Four compounding causes:

1. **`mode` conflates two orthogonal questions.** *What is this value?* (literal, expression,
   reference, function-of-time, schedule, random variable) and *where does it vary?* (fixed,
   across runs, across replicates, across trials, over time). The current modal presents `mode`
   as primary and `sampling_scope` as an afterthought dropdown, so "uniform distribution over
   trials" (task randomization) and "uniform sampler as a random-search axis" (sweep) look like
   the same thing with a different dropdown, while "grid axis of 5 values" has no honest
   representation at all (`categorical` is a semantic abuse).
2. **No enumerable-axis forms.** No list, no linspace/logspace. The two most common sweep shapes
   in practice are unexpressible.
3. **No consequences.** Nothing downstream consumes a sweep spec (D4), so the modal edits feel
   inert — there is no feedback loop telling you what you just did.
4. **Mechanical debt.** Portal positioned by bottom-shelf-height arithmetic instead of anchored
   popover; JSON textareas for schedules/vectors; heuristic chip labels.

### 5.2 The two-part editor

Redesign the editor around **Value × Variation**:

- **Value** (form picker + form-specific structured editors): Literal · Expression · Reference ·
  Function · Schedule · Distribution. Real editors: list editor with add/remove/paste-CSV; range
  editor (min/max/n, lin/log toggle) with rendered tick preview; distribution editor with a tiny
  density sketch; schedule editor with a mini curve preview (replacing the JSON textarea).
- **Variation** (segmented control, options filtered by the field descriptor's eligibility
  matrix — the bd5aefb scope matrix, enforced rather than documented): Fixed · **Across runs
  (sweep axis)** · Across replicates · Across trials · Over time.

Legality is the cross product filtered per field: a sweep axis requires an enumerable Value form
(list, range, or sampler+n); `distribution × trials` is within-run stochasticity;
`distribution × replicates` is per-replicate init; `schedule × timestep` is a time-varying
signal. The descriptor system (`allowedModes`/`allowedScopes`) already exists — it becomes the
eligibility matrix carrier.

When Variation = *Across runs*, the modal footer becomes consequential (fixes cause 3): "Adds
axis to **Train matrix** · 5 values · matrix would be 5 × 3 = 15 runs", with a jump-link to the
matrix builder.

### 5.3 Which matrix does an axis join? (the scoping question)

Proposal: **the axis joins the matrix of the scenario being edited** — and since each bottom-pane
stage owns a scenario, that means the active stage. Editing the training scenario's params with
the Train tab active → train axis; editing eval condition params → eval axis. No extra "this is
a training sweep" declaration (it would be redundant 95% of the time), but three guards against
hidden magic (P6):

- The chip on the param field names its matrix: `⇉ 5 · train`.
- The stage tab badge counts axes and planned rows, so a declared axis is always visible from
  the shelf even when the sidebar is closed.
- Expansion never happens implicitly — axes accumulate; only *Stage N runs* creates rows.

### 5.4 Bulk-edit verbs (the second-sweep problem)

Multi-selecting M pending rows turns the sidebar into a bulk editor: one row per parameter, each
showing either the common value or a "varies (M values)" chip. Editing a param in bulk mode adds
a **verb** (Jira's bulk-change vocabulary, adapted):

- **Keep** — leave per-row values untouched (default; makes clobbering impossible).
- **Set** — broadcast one value to all M.
- **Distribute** — zip an M-length list across the selection *in table order* (this is exactly
  "override one param of M staged runs with a sweep of M"). Validation: list length must equal M.
- **Cross** — multiply: selection × K values → M×K rows (stage-time expansion of a second axis
  over an existing selection).

With ghost-row preview before commit, as always (P1). This one mechanism subsumes the
"dual sweep" scenarios: stage axis A, select the M rows, Distribute or Cross axis B — or just
declare both axes up front and pick zip/cross in the matrix builder. All three routes the user
sketched exist, and they are the same primitive underneath.

---

## 6. Stage-by-stage UX

The shelf keeps stage tabs (Train / Eval / Analysis / Report), and gains three cross-cutting
**projections** of the same underlying manifest collections — mirroring the top pane's
projection concept: **Table** (per-stage, default) · **Lineage** (provenance DAG) · **Queue**
(execution). Compare is a mode of Table (selection ≥ 2), not a separate surface — resisting the
Simulink failure mode of three disconnected tools.

**Train.** Left: matrix builder + run table (pending + running + completed in one table; ghost →
pending → live progress via the existing training WebSocket, now keyed by manifest id).
Dynamic axis columns; set-grouped rows (collapsible; a set header shows a Dagster-style status
strip — N green / M gray cells — legible at 100+ runs where flat lists fail). Right: focused-run
detail (replaces `RunDetailOverlay`), or bulk editor when multi-selected. Row actions: launch,
cancel, view snapshot, restage, supersede.

**Eval.** Base selection (query chips or explicit, §4.3) × condition axes × **checkpoint
policy** (last / best-by-metric / every-k — lowering to `CheckpointSelectionManifest`; today
eval is silently run-granular). Preview: `4 runs × 3 conditions × 1 checkpoint = 12 evals — 5
already materialized (cache hit)`. Content-addressed eval identity makes the cache-hit preview
honest and free. "Run selected" finally wires to `execute_evaluation_run_spec`.

**Analysis.** The bundle becomes the first-class object: a bundle card shows its predicate
binding (chips + match preview), its stage list, and per-stage **dry-run status** — evaluate
`run_condition`s and `depends_on_roles` against currently-matched inputs *before launch*:
`would run` / `would skip (has_reach_epoch = false on 2/6 inputs)` / `missing required role
'gain_matrices' from stage fit_gains`. The existing analysis-DAG canvas remains the authoring
view inside a bundle; the card view is the pipeline-facing face. Skipped/not_applicable statuses
render as first-class chips with reason tooltips (P6), never hidden.

**Report.** Minimal honest slice until 4d1558c (template layer) lands: recipe picker + inputs
binding (same selection machinery) + rendered `report_render` artifact viewer (md/html) +
regeneration button. No bespoke report builder yet.

**Queue projection.** All pending manifests across stages, with: execution target per run/set
(local worker / GCP / RunPod / manual export — RunPod exists in `feedbax/execution` but is not
wired into web orchestration; that's a gap to close), concurrency limit, drag-reorder,
pause/cancel, explicit eviction/supersession events (never silent — GH Actions
`cancel-in-progress` anti-pattern). **Billable-launch gate:** launching to a paid target renders
a spec-lock table (varied axes, counts, target, est. duration) requiring explicit confirmation —
the rlrmp RunPod protocol, encoded in the product instead of in agent instructions.

**Lineage projection.** Manifests as nodes (grouped by set), `ParentRef`s as edges; click-through
focuses the row in its stage tab; hover cross-highlights (P5). Staleness and skip statuses
render on edges/nodes. This is a projection, not a second source of truth (P2).

---

## 7. Dependency health and staleness

- **Staleness is spec-hash comparison** — cheap because everything is content-addressed. A
  pending run whose snapshot diverges from the current draft scenario gets `stale (draft
  changed)` with a diff view; a downstream eval whose upstream was re-run gets `stale (upstream
  superseded)`. Staleness never auto-cancels or auto-reruns anything; it feeds the reprocess
  tri-state (*missing / missing+failed / all*, plus *stale* as a filter) on each stage's launch
  control.
- **Superseded chains stay navigable.** Restaging creates a `superseded_by` link; default views
  filter superseded rows, a toggle reveals history.
- **Dry-run is the universal pre-launch surface** (§6 Analysis) — one endpoint that takes any
  pending manifest/bundle and reports per-stage would-run/would-skip/missing against current
  inputs, rendered identically wherever launches happen.

---

## 8. The fifth things (inventory of what the prompt didn't name)

1. **Queue & execution-target management** as a first-class projection, incl. the billable
   spec-lock gate and wiring the RunPod execution backend into web orchestration (§6).
2. **Comparison** — the `compare` stage kind exists as a stub; make it a Table mode: N-selection
   → param+metric diff, identical fields collapsed by default (MLflow/ClearML convention),
   optional per-axis small-multiples once figures are in reach.
3. **Staleness/invalidation semantics** (§7) — nobody asked, everybody needs it the first time
   they edit the graph after staging 20 runs.
4. **Import/ingest UI** — 589535c landed manifest import/export packets; the empty run table's
   real fix is an ingest surface: drop a packet (or point at a `FEEDBAX_RUNS_DIR`) to import
   runs — including rlrmp-trained runs into Studio. The `import` stage stub becomes real.
5. **Auto-labeling from varied axes** (§4.2) — encode the rlrmp label convention by construction.
6. **Per-run logs/failure triage** — Console tab becomes per-run: focused run → its log stream
   (nohup/worker), scanned for the standard error signatures; failed rows carry the first error
   line in a tooltip.
7. **Checkpoint granularity for evals** (§6 Eval).
8. **Column/metric configuration with restraint** — dynamic axis columns are automatic; metric
   columns user-pickable, persisted in workspace ui_state (No-volatility), never auto-fetched
   unboundedly (Neptune's 51-second-table lesson).
9. **Keyboard flow** — stage/launch/annotate without leaving the table (range-select,
   Enter=focus, Space=select, L=launch). Cheap now, impossible to retrofit culturally later.

---

## 9. Backend gaps this design requires

Ordered roughly by how much the UI story depends on them:

| # | Gap | Notes |
|---|---|---|
| B1 | Pending-manifest staging API + typed run-summary/index endpoints | Subsumes `e33f487` DB indexing; replaces `runAPI.ts` stubs; deterministic training-run ids |
| B2 | Axis/matrix expansion: ValueSpec sweep axes → pending manifests + `TrainingRunSetManifest.axes` | Schema-versioned change + migration rule; align with worker `AxisRole "authored_sweep"`; reuse `config/batch.py` zip/broadcast semantics as the reference implementation |
| B3 | ValueSpec v2: Value×Variation split, enumerable axis forms (list/range/sampler+n), eligibility matrix enforcement; typed lowering for non-constant component params (fixing D5) | Successor to `bd5aefb`; new schema version + migration |
| B4 | SelectionSpec v2: explicit/query/frozen, `top-k per group` predicate primitive | Builds on 196bd09/ManifestPredicate |
| B5 | Dry-run endpoint (conditions + role deps vs current inputs, no side effects) | Thin wrapper over existing `_run_condition_skip_reason`/`_resolve_stage_inputs` logic |
| B6 | Eval execution wiring from Studio (+ checkpoint policy → `CheckpointSelectionManifest`) | "Run selected" goes live |
| B7 | Queue/target unification; RunPod backend into web orchestration; spec-lock gate | |
| B8 | Staleness service (spec-hash diff; superseded links) | |

Per repo policy: B2/B3/B4 are durable-schema changes and must land with versioned migration
rules and old-version accept/migrate/reject tests.

---

## 10. Open questions — status after 2026-07-06 review

1. **`TrainingPanel` dissolution** — mooted by the D1 correction: the panel is already
   unmounted. Decision reduces to organ placement (config → projections/sidebar; launch →
   table/queue; loss chart + checkpoint download → focused-run detail). Pending Matt's
   confirmation after plain-language walkthrough.
2. **Pending manifests as staging substrate** — **RESOLVED: yes** (Matt agrees). GC story for
   cancelled/stale pending manifests still needs a design note in the P1 spec.
3. **Query liveness** — proposal restated plainly: staged downstream work always freezes a
   snapshot of what the query matched; new matches only surface as a badge ("2 new runs now
   match") with a manual *refresh* action (missing / missing+failed / all). No auto-restaging.
   Pending confirmation.
4. **Replicates** — **RESOLVED: within-run ensemble dimension** (statistical variability over
   random inits etc.). Matt's residual unease ("is the init distribution resampled per
   replicate or constant?") is answered by the Value×Variation split itself: an init
   distribution with variation scope `replicate` resamples per replicate; scope `run` samples
   once and shares. Selective replication = per-ValueSpec scope choice.
5. **Seeds vs replicates** — unified under the same mechanism: a *seed sweep* is variation
   scope `sweep` on the run's master PRNG key (separate runs; all streams fork — batch order,
   init, noise); *replicates* are scope `replicate` on selected streams within one vmapped run.
   One mechanism (declaring which PRNG streams fork at which level), two conventional presets.
   Seed becomes a first-class axis at zero extra cost.
6. **Naming** — **RESOLVED**: "matrix" for the axis-composition object; "pending" for the
   manifest status (reuses the existing `ManifestStatus` literal — no schema change).

---

## 11. Precedent digest (what we're stealing, from whom)

Steal: materialize-before-run preview (Simulink Test Manager "Show Iterations"; JMP design
tables); explicit combination modes (COMSOL specified/all/switch); `value` vs `values` as the
pinned/swept schema split (W&B sweep config); verb-explicit bulk edit (Jira); schema-derived
launch forms (Prefect/Airflow params); collapse-identical diff defaults (MLflow/ClearML); status
grids over flat lists for large sweeps (Dagster partitions bar); reprocess tri-state
(Airflow backfill: missing / missing+errored / all); hover-preview + click-commit bidirectional
selection sync with pin (Figma/Blender/Houdini/TouchDesigner); queue overrides as auditable
diffs (W&B Launch).

Avoid: silent unattributable state (wandb#3917); divergent semantics behind similar launch
buttons (dagster#17665); partial cross-view sync (wandb#6335); unbounded comparison fetching
(neptune#730); fragmenting run/results/compare into separate tools (Simulink's Test
Manager / Simulation Manager / Data Inspector split).
