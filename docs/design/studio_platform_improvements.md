# Studio Platform Improvements: performance, contract integrity, UX foundations

Status: filed 2026-07-06 as umbrella `e59ed00` with 17 children (ids in §8).
Provenance: synthesized from four read-only archaeology passes (frontend
perf/architecture, backend↔frontend seam, cross-surface UX, spec/ledger gaps)
run in a Cowork session on `develop`.
Restoration note (2026-07-08): this file was lost from the working tree by a
host-side tree operation before it was ever committed (see the 2026-07-07
post-merge review, finding A1 — all three umbrella design docs were lost the
same way). Restored verbatim from the authoring session; content reflects the
2026-07-06 pre-implementation state. Where this doc and the merged code
disagree, the code is authoritative.

Relationship to existing surfaces:

- `efa2093` (pipeline pane redesign) and `f3159c7` (Workspace view) own their
  respective surfaces. Nothing here proposes work inside them; §7 records
  integration judgments about them.
- `d3f9af7` (autosave-robustness umbrella) owns the in-flight save race
  (`eb2ee06`), beforeunload (`e107e37`), autosave-failure toast (`5bdcd40`),
  and dimensions spurious-save (`6fa4176`). Save-related items below are
  scoped to what that umbrella does *not* cover (payload cost, OCC/multi-tab,
  server-side read-modify-write).
- `6116155` (typed subgraph taxonomy + pluggable canvas) owns per-subgraph-type
  canvas editors. The shared-primitive work in §5 is upstream of it, not
  duplicative.
- `93f79f2` (Studio design coordination) is the decision-log surface; if this
  doc becomes an umbrella, comment there.

The doc is organized as: P0 trust/principle bugs (§2), then three improvement
themes — rendering/state performance (§3), contract integrity (§4), UX
foundations (§5) — then testing/spec hygiene (§6), umbrella judgments (§7),
and a candidate child-issue breakdown (§8). Appendix A holds the factual
inventories the findings cite.

---

## 1. Scope and method

Scope: everything in Feedbax Studio that is *not* the bottom pipeline pane's
redesigned staging/sweep/selection semantics and *not* the Workspace view's
representation/playback contract. Concretely: canvas rendering and state
architecture, the REST/WS/SSE seam, save/load and persistence machinery,
cross-surface navigation, editing ergonomics, feedback/error surfacing, shared
UI primitives, tests, and spec freshness. Backend-internal improvements are
excluded (owned by a parallel session); the *seam* is in scope.

Method: four parallel subagent passes with file:line evidence, cross-checked
against the ledger (open Studio issues, umbrella bodies) and the design docs
(`pipeline_ux_redesign.md`, `workspace_view/DESIGN.md`, `WEB_UI_SPEC.md`,
`COLLIMATOR_COMPARISON.md`). Confidence and evidence per finding live in the
sections below; effort grades are S/M/L.

---

## 2. P0: core-principle and trust bugs

These are bugs under existing repo policy, not enhancements. They should be
filed as `error` issues regardless of whether the rest of this doc becomes an
umbrella.

### 2.1 Frontend subgraph synthesis violates "graph is the model"

`web/src/stores/graphStore.ts:2041-2051` falls through
`SUBGRAPH_FACTORIES[type]?.() ?? componentDef?.template_graph ??
createEmptySubgraph(nodeId)` on first expansion of a composite node —
synthesizing a subgraph client-side instead of raising. Two hardcoded factory
functions (`createArm6MuscleSubgraph` at `:432-555`,
`createPointMass8MuscleSubgraph` at `:557-634`) bake node types and params
(e.g. `tau_rise: 0.015`) into frontend TypeScript, and the synthesized result
persists into save state. The comment at `:2042-2047` cites Bug `5e8895e`, but
that fix only addressed a registry-load race; the unconditional synthesis
still fires. CLAUDE.md is explicit: "Absence of a subgraph is an error, not a
condition to work around" and "No background construction." The backend build
path (`feedbax/web/worker/execution.py` → `compile_graph`) honors the policy;
the frontend does not.

Related: the frontend renders subgraph presence from `params._subgraph`
truthiness while the build path reads `graph.subgraphs[nodeId]` — two
representations with no regeneration-on-save keeping them synced. This is the
exact latent-bug class the policy names. Intersects `6116155` (subgraph
taxonomy) and `c77b227` (composite-cleanup-3); the fix should be coordinated
there rather than duplicated, but the synthesis fallback itself is
independent and immediately fileable. Effort: M.

### 2.2 Silent fabricated data when the backend is unreachable

`web/src/api/runAPI.ts:52-71,144-155,185-193`: `STUB_TRAINING_RUNS` /
`STUB_EVAL_RUNS` are silently substituted when the backend is unreachable —
fake run history (`tr-001`, `tr-002`) indistinguishable from real data. Worse,
`createEvalRun` fabricates a phantom *successful write*: on failure it
synthesizes a client-side ID with `status: 'running'`. The same
silent-stub pattern exists in `analysisAPI.ts`, including on schema-validation
failure — so a genuine contract-drift bug degrades invisibly into fake data
instead of surfacing. Meanwhile `client.ts`/`figureAPI.ts` throw bare errors
with no fallback: two opposed error philosophies in one directory.

Fix: delete the stub fallbacks; fail loudly with a user-visible error state
(consuming the toast layer of §5.3). A write that failed must never report
success. Effort: S–M, high confidence, high severity — this is the most
trust-corrosive finding in the report.

### 2.3 `param_schema_version` silently dropped on every normalize pass

`normalize_graph_for_studio_authoring` strips `param_schema_version` on every
create/update/load, even though `compile_graph` gates migration behavior on
it. The migration infrastructure itself is well-built (explicit
`GRAPH_SPEC_SCHEMA_VERSION` v3, migration registry, hard error on unsupported
versions) — this one pass undermines it. Looks like an oversight, not a
decision. Effort: S. (Seam bug; coordinate with the backend session if it is
already touching normalize.)

---

## 3. Theme A: rendering and state performance

The findings here compound: whole-store subscriptions (A1) make every edit a
broadcast; full array rebuilds (A2) make every broadcast expensive; O(n²)
validation (A4) re-runs on every rebuild; and the autosave loop (A5)
serializes the whole project on the same trigger. Fixing any one helps; the
ordering below is the dependency order.

### A1. Whole-store Zustand subscriptions (S per site, M systematic)

`Canvas.tsx:590-614` destructures 22 fields from a selector-less
`useGraphStore()`; same pattern in `useShortcuts.ts:22-23`, `Header.tsx:41-47`,
`AnalysisPanel.tsx:142-162`, `TrainingPanel.tsx:41`, `RunSelector.tsx:36-43`.
`useWorkspaceStore((s) => s.workspace)` (~15 sites) selects one large nested
object, behaving identically. `useShallow`/`subscribeWithSelector` are used
zero times in `src/`. Any edit to a 62-member store re-renders Canvas, Header,
and the global shortcut hook on the hottest interaction paths. Stopgap:
`useShallow` at these sites. Durable fix: A6.

### A2. Full node/edge array rebuild on every single-node edit (L)

`graphStore.ts:940` (`buildNodes`) and `:1021` (`buildEdges`) reconstruct the
entire arrays from `graph`+`uiState` on every param edit and every
`onNodesChange`. New references for unaffected nodes defeat React
Flow/React memoization wholesale; cost scales with graph size, not edit size.
This is the main scaling risk for 100+-node graphs and multiplies everything
else. Fix: diff-based reconciliation preserving object identity for untouched
nodes/edges.

### A3. `React.memo` on node/edge renderers (S, after A2)

`CustomNode.tsx:38`, `SubgraphNode.tsx:48`, `TapNode.tsx:6`,
`RoutedEdge.tsx:62`, `StateFlowEdge.tsx:3` — none memoized. Useless until A2
lands (props get new references every render regardless), cheap after.
(Positive: `nodeTypes`/`edgeTypes` are correctly hoisted at
`Canvas.tsx:333,340`.)

### A4. O(n²) validation/schema passes on every edit (M point fixes, L incremental)

`features/graph/validation.ts:63-98` scans all wires per port;
`features/schema/project.ts:939-956` linear `findPort` called twice per wire;
`features/graph/dynamicPorts.ts:36-99` is O(nodes × wires). All keyed on
`graph` identity, which changes every edit per A2 — so this pure CPU work
re-runs constantly and is likely the dominant source of perceptible lag on
large graphs. Fix: Map/Set indices built once per registry construction.

### A5. Autosave payload and trigger breadth (M)

`projectsStore.ts:655-671` wires whole-store `.subscribe()` on five stores
into a 250 ms-debounced `persistLocalProjectTabs()` (`:351-372`) that
`JSON.stringify`s the full tabs array — including undo/redo history — to
localStorage. Nearly any keystroke/drag restarts the cycle. Fix: narrow the
subscriptions to persistence-relevant fields; exclude undo history or persist
it on a longer cadence. Scope note: this is the *cost* side of autosave;
`d3f9af7` owns the *correctness* side (races, beforeunload, failure toast) —
file this as a distinct child or offer it to that umbrella, don't fork it.

### A6. Split the god-object `graphStore` (L)

`graphStore.ts:1467-1556`: 19 state fields, 43 actions, 62 members mixing
graph topology, React Flow adapter state, subgraph navigation, selection,
undo/redo history, and registry/persistence/merge workflow. A pure selection
change costs the same as a topology mutation for any subscriber. Slicing
along those six concerns is the prerequisite for A1's durable form, and is
also what `efa2093`'s unified SelectionContext (P5) will want to land on top
of — see §7.

### A7. Code splitting (M)

No `manualChunks` in `vite.config.ts`, zero `React.lazy`; `BottomShelf.tsx`
and `TopShelf.tsx` statically import all panels though one renders at a time;
recharts eagerly imported. Only Plotly is dynamically imported. Fix:
`React.lazy`+Suspense per shelf panel; vendor chunks for recharts and
`@xyflow/react`. Latent landmine: the dead three.js viewer stack
(`components/viewer/`, only imported by the unmounted `TrajectoryPanel.tsx`)
would silently add ~600 KB if reconnected without a lazy guard — either prune
it or guard it now (S). Sequencing note: `f3159c7`'s Workspace renderer and
playback lanes will add real weight here; landing lazy boundaries first is
cheap insurance.

### A8. Small hot-path fixes (S each, high confidence)

`Canvas.tsx:630` `useViewport()` re-renders the whole 1839-line component per
pan/zoom frame to feed one handle size — hoist into a scoped child.
`RoutedEdge.tsx:186-191` updates edge points on raw `pointermove` with no rAF
gating. `Canvas.tsx:545-549` runs a rAF loop *and* a parallel 100 ms
`setInterval` calling the same connector-update function. Figure-generation
status polling is triplicated (`FigureOutputPin.tsx`, `AnalysisNode.tsx:59-93`,
`AnalysisPanel.tsx:597-632`) — extract one shared hook.

---

## 4. Theme B: contract integrity across the seam

The seam has good bones — codegen exists (`scripts/generate_studio_contracts.py`
→ `web/src/generated/studioContracts.ts`), graph-spec migration infrastructure
is genuinely solid, WS reconnect logic is well-engineered. The problem is
coverage and discipline around those bones.

### B1. One type system, not three (L)

Hand-written `types/graph.ts`/`types/training.ts` disagree with generated
contracts in both directions: `ComponentSpec` required-vs-optional fields and
missing `param_schema_version`; `GraphSpec.additive_channel_adapters` /
`.parameter_constraints` missing from hand files; `TrainingConfig` 9 Pydantic
fields vs 5 in TS; `NodeUIState.position` typed `Record<string,number>` vs
concrete `{x,y}`. `client.ts` validates inbound with generated `parseContract`
but types outbound bodies from the hand-written files. Fix: re-export hand
types from generated contracts (or delete them), and bring `runAPI.ts` (zero
`parseContract` calls) into the validation net.

### B2. Extend codegen coverage (M)

`trajectory.py`, `statistics.py`, `inspection.py` and their TS counterparts
are entirely hand-written with no drift detection. Add them to the contract
generator.

### B3. Component-registry parity (M)

Backend registry: 65 components. Frontend fallback `web/src/data/components.ts`
(860 lines): 53 entries — 13 missing including
`AnalyticalMusculoskeletalPlant` (directly relevant to active CDE work), plus
one dead frontend-only `Mechanics` entry. Decide: generate this file from the
backend registry, or delete the fallback and make offline mode an explicit
error state (consistent with §2.2). A field-by-field parity test belongs in
CI either way.

### B4. CI freshness gates (S)

Suspected-broken generated-file gate: `LossTermSpec.matrix`/`matrix_kind`
postdate the last regen commit (~30 commits since). Verify with
`uv run python scripts/generate_studio_contracts.py && git diff --exit-code`
and wire exactly that into CI. Note the existing `.github/workflows/ci.yml`
only runs two test files; see §6.

### B5. WS/SSE protocol hardening (M)

`training_error` has two divergent shapes (WS-handler-originated carries
`job_id`; worker-relayed carries `batch`). No resync strategy after
reconnect — events during disconnect are permanently lost with no signal
(reconnect itself is solid: exponential backoff, bounded retries). No
batching/throttling on training events — every worker event is a synchronous
Zustand update (flood risk, unmeasured). Confirmed race: `_last_status_by_job`
is written by both the WS relay loop and the REST `get_status()` poll with no
monotonicity check, so a stale REST response can regress the displayed batch
number (S fix, high confidence). `ws/simulation.py` remains a 17-line stub —
fine, but it should be labeled as such wherever the spec claims otherwise
(§6.2).

### B6. Save concurrency beyond d3f9af7 (M–L)

Four independent save call sites (autosave, manual button, Cmd+S, pagehide
beacon); backend does unconditional read-modify-write per call;
`GraphMetadata.version` is a hardcoded string never used as a concurrency
token; zero multi-tab coordination (two tabs silently last-write-wins).
`d3f9af7` owns the in-flight-race child (`eb2ee06`); what it does not cover:
optimistic-concurrency (ETag/If-Match or version token), a BroadcastChannel
multi-tab warning (M for the warning, L for full OCC + conflict UI), and
rollback/reconciliation-against-server-truth on failure.

### B7. Route hygiene (S–M)

`analysis.py` (singular) and `analyses.py` (plural) collide on the
`/api/analyses` URL space; `trajectories.py` and `statistics.py` share the
`/api/trajectories` prefix; one collection route requires a trailing slash
(`figures.py:159`) which the frontend special-cases. Thirteen backend routes
have no frontend caller (all of `provider.py`, `execution.py`,
`inspection.py`, `DELETE /api/graphs/{graph_id}`, `GET
/api/training/{job_id}/manifest`, …) — triage each: intended-for-agents, dead,
or future. Seven endpoints + 2 request fields use untyped `dict`/`Any`
payloads. Coordinate with the backend session — the fix is backend-side, but
the triage input (who calls what) is seam knowledge from this pass.

---

## 5. Theme C: UX foundations

Cross-cutting machinery every surface consumes — including both umbrellas'
new panes once they land. Priority order per daily-use friction.

### C1. Undo/redo integrity (M)

History is silently wiped on every subgraph enter/exit
(`graphStore.ts:2089-2090`, `:2271-2272` reset `past: [], future: []`) — the
biggest silent data-loss-adjacent surprise in daily editing. Fix: per-layer
history stacks keyed on `GraphLayer`. Separately, `toggleNodeCollapse`
(`:3114`), `toggleNodeReversed` (`:3150`), `setAllNodesCollapsed` (`:3173`)
set `isDirty` but never push history — collapse can't be Cmd+Z'd while every
other mutation can (S fix: route through the existing snapshot pattern at
`:2968`).

### C2. Copy/paste/duplicate (M)

Does not exist at all — no clipboard actions in the store, no Cmd+C/V/D
bindings. Cloning a configured composite node currently means manually
rebuilding its internal wiring. Constraint from repo policy: duplicate of a
composite MUST deep-copy `graph.subgraphs[nodeId]` +
`uiState.subgraph_states[nodeId]` and raise if the source subgraph is absent
(never silently drop — that would recreate §2.1).

### C3. Feedback layer (S per site, L systemic)

Sonner is mounted globally but used at exactly two call sites (both errors).
Concrete gaps, each a one-line-catch-class fix: `Header.tsx:191-206` export
has `try/finally` with no `catch` (failed export looks identical to success);
`useTraining.ts:299-309` `stop()` has no error handling and can strand the
Stop button (while `start()` handles errors properly); template-load/open
fall back to a local-only unsaved tab with only a `console.error`
(believed-but-false persistence); manual save success gives no feedback;
delete gives no feedback (a "deleted — Cmd+Z to undo" toast would also make
undo discoverable). Systemic: zero `catch` blocks across seven of nine
stores. Standardize an error wrapper for store actions that call APIs, and a
toast policy (errors always; success for save/export/training transitions).
§2.2's loud-failure requirement consumes this layer.

### C4. No-volatility sweep (S per site)

Direct violations of the project's own no-exceptions rule, all
component-local `useState` with a ready-made persisted pattern sitting in
`layoutStore` (which correctly persists divider/collapse state):
`SubgraphNode.tsx:52` inline-preview expanded; `BottomShelf.tsx:53`
stage-vs-console mode (while `activeStage` itself IS persisted);
`ComponentLibrary.tsx:109` + `AnalysisLibrary.tsx:205` category expansion;
`StatisticsPanel.tsx:51-52` sub-tabs; `trajectoryStore`/`statisticsStore`
have no persistence layer at all. Related viewport bug:
`Canvas.tsx:1102-1144` auto-`fitView` on every layer transition overrides any
saved per-subgraph viewport in `uiState.subgraph_states` — skip auto-fit when
a cached viewport exists (S).

### C5. Shared UI primitives (M–L)

Six node renderers independently reimplement card/header/selection/ports: four
selection-ring variants, port dots from `w-1.5` to `w-4` for the same
conceptual handle, divergent header radii. Two hardcoded, disagreeing edge
hex palettes (`RoutedEdge.tsx:144-149` vs `StateFlowEdge.tsx:22-24`). No
shared Dialog primitive — `SettingsOverlay` and `FigureViewer` each hand-roll
Escape handling; `AddLossTermModal` has neither Escape nor backdrop-close.
Panel headers drift across 5 letter-spacing values and 6–7 padding values;
chart tooltip styles copy-pasted three times with a palette-length mismatch
(8 vs 12 entries — latent bug past 8 groups). Semantic color drift:
`ValidationPanel.tsx:107` renders *errors in amber*; "selected" is emerald in
one panel, brand elsewhere. Proposal: `NodeShell`/`PortHandle`, `Dialog`,
`PanelSectionHeader`/`CloseButton`, `edgeStyle.ts`, `chartTheme.ts`, and
error/selected color tokens. This is the root cause that mints new
inconsistencies with every added node type or panel; it is also upstream
groundwork `6116155`'s pluggable canvas will benefit from.

### C6. Shortcuts and discoverability (S each)

Full current binding set: save, undo, redo, delete. Missing:
Escape-to-deselect, zoom-to-fit, select-all, any shortcut cheat-sheet, any
label pairing a button with its key. Invisible gestures: right-click port
context menu (objectives/probes) has zero on-screen affordance; edge
waypoint-insertion and double-click-to-insert-probe (`Canvas.tsx:1605-1609`)
are undocumented anywhere. No empty-canvas onboarding overlay (blank dotted
canvas, no pointer to the library). Libraries are drag-only — no click-to-add
(accessibility). Sidebar returns `null` for the `'task'` projection
(`Sidebar.tsx:33-35`) — the left column vanishes as a side effect; render an
explicit empty state or task panel instead.

---

## 6. Theme D: testing and spec hygiene

### D1. Test coverage shape (M–L)

Frontend: 25 Vitest files, all `.test.ts` — zero `.test.tsx`, so no
component/DOM test exists anywhere; entire `canvas/` dir, all node types, all
24 panels, and 5 of 11 stores untested; no Playwright/Cypress. Backend web:
`analyses.py`, `components.py`, `execution.py`, `figures.py`, `graphs.py`,
`inspection.py`, `runs.py`, `statistics.py`, `trajectories.py` HTTP layers and
BOTH WS handlers have zero tests (no test opens a WebSocket). CI runs only
two test files (`test_batch_reshape_nan_bypass.py`,
`test_studio_api_contracts.py`) — other existing backend web tests never run
in CI. Minimum viable slice: wire existing tests into CI (S), add contract
regen gate (§B4), one WS handler test, one Playwright smoke (load project →
edit param → save → reload → assert No-volatility). The No-volatility
convention is currently enforced by nobody; a save/reload smoke test is its
natural structural guard.

### D2. WEB_UI_SPEC.md staleness (M, mostly editorial)

The 2542-line spec (v0.1.0, 2026-01-26) is wrong about: `RightPanel` tab
vocabulary (replaced by TopShelf/BottomShelf/stage vocabulary), Radix UI and
React DnD as dependencies (absent), SQLite storage (filesystem), in-process
training threading (subprocess + SSE, and CLAUDE.md's own "(stub)" label for
`training_service.py` is stale — it's a 411-line subprocess/SSE/checkpoint
client), §11 simulation preview (a literal 501), §13.2 simulation WS
(hardcoded one-message stub). Entire subsystems are undocumented
(`components/analysis/`, the three.js viewer, ~2x the documented endpoint
surface). Options: (a) demote the spec to historical with a banner and move
authority to per-surface design docs (the pipeline/workspace pattern), or
(b) a section-by-section refresh. Given two active design docs already own
the two biggest surfaces, (a) plus a slim current-architecture doc is the
cheaper honest state. Also fix the CLAUDE.md "(stub)" label.

### D3. Unowned Collimator-gap decisions (decision items, not proposals)

`COLLIMATOR_COMPARISON.md` capabilities with no owning design doc among the
three UX docs: general block library (math ops, sources, filters, the "lost
in merge" restoration list); per-block observability "barnacles" beyond
playback; acausal/physical-domain modeling (partially intersects `6116155`'s
AcausalSystem lane — check before filing); linearization/PID/codegen/MPC
tooling; state-machine support for task phases; SCC-based subsystem grouping.
These need explicit scope-in/scope-out rulings on `93f79f2`, not silent
non-ownership.

Update (2026-07-08): umbrella `6116155` now owns the acausal/physical-domain
modeling scope; the remaining D3 items still need explicit rulings on `93f79f2`.

---

## 7. Judgments on the existing umbrellas (integration only)

**efa2093 (pipeline pane).** Its P5 child (unified SelectionContext) will be
built on top of whatever store architecture exists — landing A6 (graphStore
split) first, or co-designing the slice boundaries with P5, avoids building
the new selection model against the god object and re-splitting later. Its
run-table children (P4, P9) will be the first consumers of §2.2's
loud-failure semantics — the current silent `STUB_TRAINING_RUNS` path would
make the new run table lie about backend state on day one; recommend §2.2
land before or with P4. Its "TrainingPanel is orphaned" diagnosis is
independently corroborated; the panel's recharts imports are also a bundle
consideration for A7 when its organs redistribute.

**f3159c7 (Workspace view).** Its playback lane will push frames through the
store→canvas render path; under the current buildNodes-rebuild-everything
regime (A2) live overlays will pay full-graph cost per frame. A2/A1 are
effectively performance prerequisites for pleasant playback. Its renderer
also adds bundle weight on top of an app with zero code splitting — A7's lazy
boundaries should precede C9/C10 of that umbrella. Neither point changes its
design; both affect sequencing.

**Cross-umbrella observation.** Both umbrellas add new persisted view state
(stage scoping, WorkspaceViewState, pending manifests). The No-volatility
sweep (C4) plus the Playwright save/reload smoke (D1) would give them a
structural guard to land against, rather than each hand-verifying the
convention.

---

## 8. Umbrella structure (filed 2026-07-06)

Umbrella: `e59ed00`. P0 items carry the `error` label; H14 `maintenance`; the
rest `feature`. D3 decision items were posted as a comment on `93f79f2`.

| # | Issue | Slice | Contents | Effort | Depends on |
|---|---|---|---|---|---|
| E0a | `7da2278` | Subgraph synthesis violation | §2.1 (coordinate with 6116155/c77b227) | M | — |
| E0b | `1153f60` | Silent stub data + phantom writes | §2.2 | S–M | H3 (toast layer) soft |
| E0c | `258f8aa` | param_schema_version drop | §2.3 | S | backend session |
| H1 | `5f3957e` | Hot-path quick wins | A1 stopgap, A8, C4 viewport fix | S×n | — |
| H2 | `3def564` | Store split + identity-preserving builds | A6, A2, A3 | L | — (before efa2093 P5 ideally) |
| H3 | `e973e34` | Feedback layer | C3, toast policy, store error wrapper | M | — |
| H4 | `33135b5` | Validation indexing | A4 | M | H2 helps |
| H5 | `782b73a` | Autosave payload narrowing | A5 | M | cross-linked to d3f9af7 |
| H6 | `7bd2d94` | Code splitting + dead viewer decision | A7 | M | before f3159c7 C9 |
| H7 | `5977ece` | Contract unification | B1, B2, B3, B4 | L | — |
| H8 | `fcb7f94` | WS hardening | B5 | M | backend session coordination |
| H9 | `7239ae7` | Save OCC/multi-tab | B6 | M–L | after d3f9af7 children |
| H10 | `1877173` | Undo integrity + duplicate | C1, C2 | M | H2 helps |
| H11 | `b0c7c8f` | Shared primitives | C5 | M–L | feeds 6116155 |
| H12 | `d64b5a6` | Shortcuts/discoverability/No-volatility sweep | C6, C4 remainder | S×n | — |
| H13 | `2ab6af2` | Test floor + CI wiring | D1, B4 | M | — |
| H14 | `b0b1256` | Spec retirement/refresh | D2 | M | — |
| — | — | Collimator scope rulings | D3 | decision | posted on 93f79f2 |

Suggested first wave: E0a–E0c + H1 + H3 + H13 (small, high-trust-yield,
unblock nothing). Second wave: H2 (the big one), then H4/H10 which it
cheapens. H7/H8/H9 pace with the backend session and d3f9af7.

---

## Appendix A: factual inventories

**Stores** (`web/src/stores/`, lines): graphStore 3475 · analysisStore 1210 ·
workspaceStore 1067 · projectsStore 683 · runStore 424 · trainingStore 387 ·
layoutStore 236 · statisticsStore 207 · trajectoryStore 196 · demandStore 124
· settingsStore 33. Zero uses of `useShallow`/`subscribeWithSelector`.

**Largest components**: Canvas 1839 · PropertiesPanel 1414 · TrainingPanel
1379 (unmounted) · ScenarioInspectorPanel 1179 · RunCollectionStagePanel 1057
· AnalysisPanel 1043 · ValueSpecField 794 · ScenarioProjectionWorkspace 752 ·
Header 614 · CustomNode 605.

**Node renderers** live in `web/src/components/canvas/` (there is no
`components/nodes/` despite CLAUDE.md's claim — fix that line too):
CustomNode, SubgraphNode, TapNode; plus `components/analysis/` AnalysisNode,
DataSourceNode, TransformNode.

**REST surface**: 56 endpoints across graphs, components, provider, training,
execution (501 stub), inspection, trajectories, statistics, orchestration,
figures, analysis, analyses, runs. 13 with no frontend caller. No frontend
calls to nonexistent endpoints (verified healthy).

**WS message types**: training_progress, training_complete, training_error
(two shapes), training_log, training_trajectory, simulation_state (stub).

**TS type locations**: generated — `src/generated/studioContracts.ts` (2054
lines); hand-written duplicates — `src/types/{graph,training,components}.ts`;
hand-written no-codegen — `src/types/{trajectory,statistics}.ts`; registry
fallback — `src/data/components.ts` (53/65 components).

**Verified healthy** (do not re-investigate): React Flow single source of
truth (no `useNodesState` dual-state); WS reconnect backoff; orchestration
polling cleanup; nodeTypes hoisting; undo snapshot gating to drag-end; canvas
selection surviving tab switches; subgraph breadcrumbs; multi-select
move/delete; modal Escape consistency except AddLossTermModal; icon-button
tooltip coverage outside the three named gaps; backend `compile_graph`
raising on missing subgraphs; graph-spec migration registry (v3, versioned,
fail-closed).
