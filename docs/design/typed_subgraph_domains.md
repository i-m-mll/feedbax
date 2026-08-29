# Typed subgraph domains: acausal/DAE, mechanics, and network-internals editing in Studio

Status: PROPOSAL — uncommitted working notes, 2026-07-08 (Cowork session).
Not yet a spec or auth request. No ledger issues filed yet; a candidate
issue mapping is at the end so an umbrella can be filed later without
re-deriving context.

Relation to existing surfaces: this is the design body of work that ledger
umbrella `6116155` ("typed subgraph taxonomy + pluggable canvas architecture")
names but does not yet contain. It subsumes/advances `a6efafc` (DAE/acausal
subgraphs with domain palettes, ex-GitHub #49), `a8efe7e` (compatibility-matrix
bug), and touches `2f8dd61`, `56dfd97`, `c1ad986`/`7c0c4b2`, `f22dbf4`,
`7da2278`. Spec ancestors: `docs/DAE_BIOMECHANICS_SPEC.md` (Phase 3),
`docs/ode_graph_ui_analysis.md`, `5378a5d` structural-authoring spec,
`docs/design/workspace_view/DESIGN.md`.

## 0. Grounding: what exists today (verified 2026-07-08)

Backend:

- `GraphSpec` v3 (`feedbax/contracts/graph.py`) is recursive: composite
  interiors live in a sibling `subgraphs: Dict[str, GraphSpec]` keyed by node
  id — never a field on `ComponentSpec`. `WireSpec` is directed and carries no
  dtype; port typing lives on `ComponentMeta.port_types`.
- The "requires a subgraph" rule is general and enforced at four layers
  (`compile_graph`, `studio/schema.py::_missing_subgraph_issues`,
  `prototypes.py`, `integrations/provider.py`): `is_composite and builder is
  None` ⇒ hard error when no interior exists. But the rule cannot distinguish
  "needs a causal GraphSpec interior" from "needs a different formalism
  entirely" — `Subgraph`/`Network` are special-cased by literal type string.
- `feedbax/acausal/` is a complete, tested Modelica-style equation-graph
  subsystem: `AcausalElement`/`AcausalPort` (across/through vars, domains
  TRANSLATIONAL/ROTATIONAL), genuinely undirected `AcausalConnection`,
  `assemble_system()` (union-find node merging, through-eq topo-sort, mass
  balance → pure-jnp vector field; index-0 only, no CAS), and
  `AcausalSystem(DAEComponent)` which behaves as an ordinary causal
  `Component` at runtime. `DAEComponent` (`mechanics/dae.py`) is a generic
  diffrax+optimistix stepping harness.
- **The crux gap**: `AcausalSystem` is registered in the component registry
  (palette-visible, `is_composite=True`) but has no builder and no
  `compile_graph` path; the acausal element classes (`Mass`, `LinearSpring`,
  `Ground`, …) are not registered at all. Ditto
  `AnalyticalMusculoskeletalPlant` (registered, no builder;
  `_DISPLAY_ONLY_MESSAGES`). CDE templates are display-only because their
  primitive node types (`MatMul`, `Scale`, `Sigmoid`, …) have no builders
  (issue `2f8dd61` lineage). `PenzaiSubgraph` works as an opaque leaf but its
  interior builder registry (`_PENZAI_MODEL_BUILDERS`) is invisible to Studio
  and stateful Penzai models raise `NotImplementedError`.
- Schema governance (`SpecSchemaRegistry`, `schema_namespace.py`,
  migration-edge BFS with fail-closed unknown versions) is the mandatory home
  for any new durable spec family. `AcausalElement`/`AcausalConnection`/
  `StateLayout` currently have **no schema identity** (review finding AC7).
- Error transport to the frontend collapses everything into one string
  (`training_error` events; validation issues pre-joined with "; ").

Frontend:

- One React Flow canvas; node renderer chosen by cache presence
  (`Boolean(graph.subgraphs[id])` → `SubgraphNode` else `CustomNode`), fixed
  `nodeTypes` maps everywhere (incl. nested preview `NESTED_NODE_TYPES`).
- Subgraph navigation via `graphStack` in `graphStore`; `currentContext` comes
  from a hardcoded switch `getSubgraphContext(type)` duplicated in two places;
  palette filtering via hardcoded `CONTEXT_SUGGESTED_CATEGORIES` /
  `CONTEXT_EXCLUSIVE_FILTER` in `ComponentLibrary.tsx` (contexts `'muscle'`,
  `'acausal'` live; `'network'`, `'penzai'` dead). Composite detection is
  triple-sourced (`_compositeTypes`, `DEFAULT_COMPOSITE_TYPES`,
  `SUBGRAPH_TYPES`).
- Known bug `a8efe7e`: a causal `Subgraph` can still be dropped inside an
  `AcausalSystem` context.
- No compile-status concept exists anywhere in the UI. Closest precedents:
  per-node async status in `demandStore` + `FigureOutputPin`; the WS
  training-error channel; client-side `validateGraph` + `ValidationPanel`.
- UI persistence is already recursive and adequate: `GraphUIState`
  (`node_states`/`edge_states`/`subgraph_states`/`viewport`) +
  `graph_stack_path` on `StudioWorkspaceSpec` — new domain editors must nest
  into this, not invent parallel storage (No-volatility).
- The analysis pipeline is a full copy-paste second canvas/store/palette —
  the anti-pattern this design must avoid repeating.

Ledger decisions that bind this design: graph-is-the-model + no background
construction (`07c0ec9`, CLAUDE.md); outer params never authoritative, Option
B is default (`2f8dd61`); template import is non-destructive (`07c0ec9` #11);
one `StateIndex` cell per subgraph holding a structured Module, generalizing
`MechanicsState` (`a6efafc` #4); state cells separated from named observables
(`a6efafc` #5–7); recurrent-edge UX conventions (`7f02d30`); causal Subgraph
inside AcausalSystem = no (`6116155`); Penzai core not adopted for NN state
(`c1ad986`); "never a shadow model" for derived views (workspace DESIGN.md).

## 1. Core concept: graph domains as a first-class registry concept

Introduce a **DomainRegistry** in the backend, parallel to (and referenced by)
the component registry. A domain is the unit that answers: what components may
be placed here, what an edge means, how the interior is validated, how it
compiles to a runnable `Component`, and how the editor should look.

```python
class DomainMeta(BaseModel):
    id: str                      # "feedbax.domain.causal", ".acausal", ".mechanics", ".penzai"
    display_name: str
    interior_schema_id: str      # spec family of the interior payload
    edge_semantics: Literal["directed", "undirected"]
    allows_multi_edge_per_port: bool
    nestable_domains: list[str]  # compatibility matrix, declarative
    editor: EditorCapability     # kind: canvas|tree|inspector|none; editable: bool
    theme: DomainTheme           # canvas tint, node tone, edge style, icon, legend entry
    compiler_id: str             # entry in the domain-compiler registry
```

`ComponentMeta` gains two fields:

- `domain: str` — where this component may be *placed* (replaces the
  hardcoded frontend `CONTEXT_*` tables). Acausal elements declare
  `feedbax.domain.acausal`; causal primitives declare `feedbax.domain.causal`.
- `interior_domain: Optional[str]` — set on composite types; declares what
  formalism lives *inside* (replaces `SUBGRAPH_TYPES`, `getSubgraphContext`,
  `DEFAULT_COMPOSITE_TYPES` string switches). `Subgraph` → causal;
  `AcausalSystem` → acausal; mechanics templates → mechanics; `PenzaiAdapter`
  → penzai.

This is the "data-driven vs code-driven" question `6116155` left open:
**data-driven metadata, code-backed behavior.** The registry carries the
declarative facts (palette membership, nestability, theme, schema ids);
compilers and editors are registered code keyed by those ids. Everything the
frontend needs arrives through the existing `GET /api/components` payload plus
a small `GET /api/domains`, so no frontend hardcoding survives.

The four-layer "missing subgraph is an error" rule generalizes: a node whose
type has `interior_domain=d` and no interior payload of schema `d` is a hard
error with a domain-specific message. No fallbacks, per the core principle.

## 2. Interior representation: discriminated subgraph payloads

Keep the load-bearing pattern (interior keyed by node id in a sibling dict),
but make the value a **discriminated union on `schema_id`**:

```python
subgraphs: Optional[Dict[str, CausalGraphSpec | AcausalGraphSpec | ...]]
```

(GraphSpec v4; migration rule: v3 payloads' subgraph values are all causal.)

`AcausalGraphSpec` (`feedbax.spec.acausal_graph.v1`):

- `nodes: Dict[str, ComponentSpec]` — **reuse `ComponentSpec` unchanged.**
  Node types resolve to registered acausal element components. This is what
  lets the sidebar, param schemas, properties panel, and drag-drop machinery
  work identically in both domains.
- `connections: List[AcausalConnectionSpec]` — undirected:
  `{a: (node, port), b: (node, port)}`. Multiple connections per port are
  legal (conservation at junctions). This is a new edge kind, deliberately
  *not* a `WireSpec`.
- `boundary: List[BoundaryAdapterSpec]` — see §4.
- `physical_domain` (translational/rotational/…), `solver: SolverConfigSpec`
  (solver type, dt, root-finder — currently `AcausalSystem.__init__` kwargs,
  now persisted spec).
- `subgraphs: Optional[Dict[str, AcausalGraphSpec]]` — acausal composites
  nest **acausal-only** (compatibility matrix enforces this). An actuator
  (e.g., a Hill-muscle element assembly) is an acausal composite; a mechanics
  model wraps it as one node in a larger acausal graph. Compilation flattens
  nested acausal graphs into one element/connection set before
  `assemble_system` (boundary ports of an acausal composite are conserving
  ports, merged by the same union-find).

This closes AC7 by construction: `AcausalElement`/`AcausalConnection` get
schema identity because the spec layer now owns them; `feedbax/acausal/base.py`
objects become the *compiled* form, built from specs by the domain compiler.

Registry work: register every acausal element (`Mass`, `LinearSpring`,
`LinearDamper`, `Ground`, `ForceSource`, `PrescribedMotion`, sensors,
`Inertia`, `TorsionalSpring`, `GearRatio`, …) as `ComponentMeta` with
`domain="feedbax.domain.acausal"`, param schemas, and **conserving-port**
declarations (port kind: `conserving` vs `signal`, with across/through
variable names and physical domain — an extension of `PortTypeSpec`).

## 3. Compilation as a first-class, visible step

Each domain registers a compiler: `interior spec → Component` (acausal:
specs → elements/connections → `assemble_system` → `AcausalSystem`). Two
integration points:

1. **Build time (worker)**: `compile_graph` dispatches on `interior_domain`
   through the domain-compiler registry instead of literal type strings. The
   worker always compiles from the spec — the graph is the model; no cached
   artifact is ever trusted.
2. **Authoring time (Studio)**: a new endpoint
   `POST /api/graphs/{id}/nodes/{node_path}/compile` runs the same compiler's
   *structural* phase (no JAX, no numerics — union-find, balance counting,
   topology checks are pure Python and fast) and returns a structured report:

```python
class DomainCompileReport(BaseModel):
    status: Literal["ok", "ok_with_warnings", "error"]
    interior_content_hash: str
    diagnostics: List[DomainDiagnostic]   # severity, message, node_ids, port/variable names, counts
    derived_interface: PortTypeSpec        # the causal ports the node will expose
    summary: dict                          # e.g. n_equations, n_unknowns, n_states, n_networks
```

Diagnostics follow the best precedents rather than the worst:

- **Name the offenders, not just counts** (ModelingToolkit's "here are the
  problematic variables: y(t)"), and give per-network equation/unknown counts
  (OpenModelica) — but never report "success" alongside a mismatch
  (OpenModelica's `checkModel` pitfall).
- Check **each topologically distinct network** separately (grounding/domain
  reference required per network — Simscape's Solver Configuration
  discipline), and recurse into acausal composites so local imbalance can't
  hide behind global balance (OpenModelica ticket #3977 pitfall).
- Detect the named illegal topologies: parallel across-sources, series
  through-sources (Simscape).

Compile-status is an explicit state machine per composite node:
`never_compiled | stale | compiling | ok | ok_with_warnings | error`.
Staleness = `interior_content_hash` mismatch with the current interior spec.
The last report (status + hash + diagnostics) is persisted with the project as
a **cache keyed by content hash** — that satisfies No-volatility (the badge
the user saw survives reload) without creating a shadow model (the worker
recompiles from spec unconditionally; a stale hash visibly downgrades the
badge). Auto-compile runs debounced on interior edits, since the structural
phase is cheap; a full numeric dry-run remains a worker-preflight concern.

**Structured error transport, end to end.** Replace the "; "-joined
single-string error path with a `DomainDiagnostic[]`-bearing payload used
uniformly by save-validate, node compile, and worker preflight/`training_error`
events. This is a general Studio improvement the domain work forces.

## 4. The causal boundary: explicit adapters, derived ports

Follow the Simulink↔Simscape precedent: the causal/acausal boundary is
crossed only by **explicit adapter nodes inside the acausal interior**:

- `ActuationInput` — causal signal in → through/across source (force, torque,
  prescribed motion) on a conserving port.
- `SensorOutput` — across/through measurement → causal signal out.

The parent `AcausalSystem` node's causal ports are **derived** from its
adapters (name, order, dtype) — read-only in the parent canvas, edited only
from inside, with a provenance tooltip ("derived from ActuationInput 'u' in
this node's interior"). This replaces the current
frozen-at-`__init__` port derivation with a spec-level rule, keeps the
outer node honest (no stale outer interface), and mirrors how causal
subgraph boundary ports already work. Adapters live in a pinned
"Boundary/Utilities" palette bucket (Simscape Utilities precedent). Adapters
carry unit/dimension annotations; unit mismatch is a compile diagnostic
(adapters are the classic silent-failure point).

Runtime state contract: one `StateIndex` cell per acausal node holding the
structured solver state (generalizing `MechanicsState`, per `a6efafc` #4),
with named observables exposed through `SensorOutput` adapters and the
declared-view contract rather than literal state paths (`a6efafc` #5–7).

## 5. Frontend: one canvas, pluggable domain contexts

Do **not** copy the canvas a third time. Extract the pattern the analysis
stack duplicated into a parameterized `DomainContext` consumed by the single
model canvas:

```ts
interface DomainContext {
  id: string;
  theme: { canvasTint; nodeTone; edgeStyle; icon; legendEntries };
  nodeTypes: Record<string, ReactFlowNodeComponent>;  // per-domain renderer map
  edgeTypes: Record<string, ReactFlowEdgeComponent>;
  paletteFilter: (c: ComponentDefinition) => boolean; // c.domain === this.id
  validateDrop: (c, ctx) => DropVerdict;              // compatibility matrix
  connectionSemantics: 'directed' | 'undirected';
}
```

Contexts are constructed from the backend `/api/domains` + `ComponentDefinition
.domain`/`interior_domain` — deleting `getSubgraphContext`,
`CONTEXT_SUGGESTED_CATEGORIES`, `CONTEXT_EXCLUSIVE_FILTER`,
`SUBGRAPH_TYPES`, `DEFAULT_COMPOSITE_TYPES` (five hardcoded tables, one of
which is duplicated). Entering any composite pushes a `GraphLayer` exactly as
today; `contextType` becomes the registry-declared `interior_domain`. Fixes
`a8efe7e` properly: the palette filters by domain *and* `validateDrop`
rejects at drop time with a toast naming the rule.

Visual grammar (the "colour of the node prior to entry, and of the canvas
after entry" requirement, generalized):

- **Before entry**: composite nodes get a domain tone + icon chip (extend
  `NodeTone` from a closed union to registry-driven tones): e.g. causal
  subgraph violet (as now), acausal copper/teal, mechanics green, penzai
  amber. Same tone appears on the palette card and the breadcrumb segment.
- **After entry**: canvas background tint + a persistent domain chip next to
  the breadcrumb ("Acausal — translational"); undirected connections render
  with no arrowheads, thicker stroke, domain color, round conserving-port
  handles on the node boundary (Modelica: physical connectors on the icon
  boundary; Simscape: per-domain line styles). Distinct enter/exit actions
  (avoid Blender's overloaded-Tab confusion); breadcrumbs already exist.
- **Legend**: a small toggleable legend overlay listing domain colors/edge
  styles (Simscape Legend precedent).
- A **balance meter** in the status bar while inside an acausal context:
  live per-network "equations vs unknowns" from client-side structural
  validation, always visible, honest about mismatch.

**Resolution: the `'network'` context is not a domain.** The dead
`'network'` entry in `CONTEXT_SUGGESTED_CATEGORIES`/`CONTEXT_EXCLUSIVE_FILTER`
was an aspirational palette filter for the interior of the causal `Network`
composite (Neural Networks / Math / Signal Processing categories), never wired
up (`getSubgraphContext` never returns it). A `Network` interior is an
ordinary causal graph, so an *exclusive* filter is semantically wrong (it
would forbid nesting a `Subgraph`, `Constant`, etc.). Decision: delete it
with the other hardcoded tables. If soft guidance proves wanted later, add an
optional `suggested_categories: list[str]` hint to `ComponentMeta` for
composite types that floats categories to the top of the palette without
hiding anything — registry-driven, never exclusive.

Compile-status UX is dual-channel + upward-propagating (Unreal/LabVIEW/VS
Code precedents):

- **On the parent node**: a status pip (ok/warn/error/stale/never-compiled)
  on every domain-composite node; error/stale states bubble to ancestors
  (a parent composite shows a rolled-up "contains errors" pip) so broken
  interiors are discoverable without diving.
- **Inside the editor**: offending elements highlighted on-canvas (Unreal
  Material "ERROR!" precedent) plus the aggregate `ValidationPanel` extended
  with per-domain sections, click-to-jump to the offending node across
  nesting levels (Simulink Diagnostic Viewer).
- **Persistence**: badge states derive from the cached report + hash; the
  interior editor's UI state (positions, viewport, collapse) nests into the
  existing recursive `GraphUIState.subgraph_states` — no parallel storage.

Store architecture: extend `graphStore`'s existing slices with
domain-parameterized behavior rather than adding a store per domain; longer
term the same `DomainContext` machinery is the path to unifying the analysis
canvas (`f22dbf4`, `e8ec78e`) instead of a third fork.

## 6. Mechanics: an acausal domain flavor with an assembly projection

Mechanics nodes are **acausal subgraphs in mechanical physical domains** —
Matt's instinct is right, and the code agrees (`AcausalSystem` already spans
translational/rotational; a Hill-muscle actuator is naturally an acausal
composite that a larger arm model wraps as a node). What mechanics adds is a
better *editing metaphor*, not a different formalism:

- **Graph view**: the ordinary acausal canvas (elements + conserving
  connections). Always available; source of truth is the `AcausalGraphSpec`.
- **Assembly view**: a tree-with-sockets projection (bodies → joints →
  attached actuators/muscles → sensors) with a synchronized 2D workspace
  preview, following OpenSim Creator/Simscape Multibody precedent that
  articulated assembly fits a tree+sockets UI better than free-form
  node placement. This is a *projection/editor over the same spec* —
  derived, cached, never persisted ("never a shadow model", workspace
  DESIGN.md §3.5) — and plugs into the workspace representation contract
  (`f3159c7`: `planar_chain`, `muscle_path`, reserved `kinematic_tree`).
  Structural edits in either view mutate the one spec.
- **Preformed templates first** (DAE_BIOMECHANICS_SPEC's ordering stands):
  ship `PointMass`, `TwoLinkArm`, `Arm + N Hill muscles` as mechanics-domain
  *templates* — real `AcausalGraphSpec` payloads materialized on first entry
  (non-destructive, `07c0ec9` #11), not display-only stubs. This is the path
  that makes `AnalyticalMusculoskeletalPlant` honest: either it becomes a
  template whose interior is a real equation graph, or it stays a preformed
  leaf with a working builder — never registered-but-broken.
- Assembly-specific diagnostics ride the same compile report: DOF counts,
  unassembled-joint errors, muscle path-point references to missing bodies
  (Simscape Multibody's Variable/Statistics Viewer precedent).

Multibody beyond planar arms (3D kinematic trees, MJX coupling) stays out of
scope here; the domain/compiler seams are where it would later attach
(`mjcf_scene` archetype reserved in AUX_3D.md).

## 7. Penzai / network internals: an inspector domain first

Per `c1ad986`, Penzai's core is not the NN state paradigm; and per repo
policy, an opaque-but-lying node is worse than an honest one. So:

- Define `feedbax.domain.penzai` with `editor: {kind: "inspector",
  editable: false}`. Entering a `PenzaiAdapter` node opens a **Treescope
  panel** (structural pretty-print, named-axis array views) inside the same
  navigation chrome (breadcrumb, domain chip, exit) instead of a canvas.
  The palette shows an explanatory empty state (the dead `'penzai'` context
  code in `ComponentLibrary.tsx` already anticipated exactly this — wire it
  up rather than delete it).
- Surface the hidden interior registry: `_PENZAI_MODEL_BUILDERS` entries
  become registry-visible variants (the `builder_name` enum param already
  exists — give it a real schema + descriptions so the properties panel is
  informative).
- Fix the stateful gap (`PenzaiStateManager.from_model` NotImplementedError)
  or have the compile report state it plainly as an `error` diagnostic on
  nodes wrapping stateful models — no silent wrongness.
- The `editor.kind` capability generalizes: `canvas` (causal, acausal),
  `tree` (mechanics assembly view), `inspector` (penzai), `none` (pure
  leaves). An editable penzai canvas, if ever wanted, becomes a new
  `EditorCapability` without schema churn. The same inspector pattern later
  serves any opaque-wrapper family (`components/equinox.py` wrappers).

Related but causal-domain: make the CDE primitive node types (`MatMul`,
`Scale`, `Sigmoid`, `Subtract`, `Reshape`, …) executable so the four CDE
templates stop being display-only (`2f8dd61` lineage). "Display-only" should
then disappear as a category — replaced by honest compile-status on ordinary
templates.

## 8. Schema and migration plan (Artifact Schema policy compliance)

New/changed spec families, all registered with `SpecSchemaRegistry` under
`feedbax.spec.*` with explicit migrate/reject stances:

| Change | Identity | Migration stance |
|---|---|---|
| `GraphSpec` v3 → v4 (discriminated subgraph union, `interior_domain` refs) | `feedbax.spec.graph.v4` | migrate: v3 subgraph values tagged causal; recursive, parent-before-child (existing machinery) |
| Acausal interior | `feedbax.spec.acausal_graph.v1` | new family; reject unknown |
| Boundary adapters, solver config | nested in acausal family | — |
| Domain registry payload | `feedbax.spec.domain.v1` | new family |
| Compile report (cache) | `feedbax.spec.domain_compile_report.v1` | new family; cache-invalidate rather than migrate is acceptable (derived data) |
| Conserving-port extension of `PortTypeSpec` | bump component-def schema | migrate: default `signal` |

Acceptance evidence per policy: old-v3 project loads and migrates; unknown
acausal versions reject with clear errors; a saved project with a stale
compile report shows `stale`, not `ok`.

## 9. Performance notes

- Structural compile is pure Python (union-find + counting + topo checks);
  keep it JAX-free and debounce it (~same cost class as current client
  validation). Numeric dry-run stays in worker preflight.
- Cache compile reports by interior content hash server-side; hash computed
  from canonicalized spec JSON.
- Frontend: one React Flow instance; per-domain `nodeTypes`/`edgeTypes` maps
  memoized at module scope (React Flow re-renders everything when these maps
  change identity); domain switch on subgraph entry already forces remount
  via `graphViewKey`, so no extra invalidation machinery needed.
- Registry payload growth (dozens of acausal elements) is negligible against
  the existing 65-component payload; keep the 5-min TanStack staleTime.

## 10. Testing / verification bar

- Contract tests per domain compiler: golden diagnostics for balance
  mismatch (named variables + counts), ungrounded network, parallel
  across-sources, series through-sources, unit mismatch at adapters, nested
  acausal composite flattening, adapter-derived interface stability.
- Schema tests: v3→v4 accept/migrate, unknown-version reject (policy).
- e2e (Playwright, alongside `no-volatility.spec.ts`): enter acausal node →
  palette switches + canvas tint; drop causal component → rejected with
  named rule; edit interior → badge goes `stale`; compile → `ok`; reload →
  badge/positions/viewport survive; parent pip reflects child error.
- Worker: `compile_graph` on a graph containing an acausal node produces a
  running `AcausalSystem`; missing interior → domain-specific hard error.

## 11. Phasing

- **A — Domain plumbing** (unblocks everything): `DomainMeta` registry +
  `/api/domains`; `domain`/`interior_domain` on `ComponentMeta`; frontend
  `DomainContext` consuming registry (delete the five hardcoded tables);
  compatibility-matrix enforcement (closes `a8efe7e`); generalized
  missing-interior errors keyed on `interior_domain`.
- **B — Acausal end-to-end**: `AcausalGraphSpec` family + GraphSpec v4;
  register acausal elements + adapters; domain compiler
  (spec → assemble_system → AcausalSystem) wired into `compile_graph`;
  compile endpoint + report cache; undirected edge renderer + conserving
  ports; status pips + diagnostics panel + balance meter; structured
  diagnostic transport.
- **C — Mechanics flavor**: mechanics palette (bodies/joints/muscles as
  acausal composites); preformed templates (PointMass, TwoLinkArm, Hill-arm)
  as real spec payloads; assembly tree view + workspace-preview coupling
  (with `f3159c7` C1–C3); `AnalyticalMusculoskeletalPlant` made honest.
- **D — Inspector domains**: penzai Treescope inspector + registry-visible
  builders + stateful fix-or-honest-error; CDE primitives executable,
  display-only category retired.
- **E — Polish/unification**: legend overlay, diagnostics click-to-jump
  across nesting, warning-suppression-with-justification (Simulink
  precedent), evaluate migrating the analysis canvas onto `DomainContext`.

## 12. Candidate ledger mapping (when an umbrella is requested)

- Umbrella: extend `6116155` (it already owns this taxonomy) rather than a
  new one; adopt `a6efafc`, `a8efe7e` as children.
- New children per phase: A (registry/domain plumbing, frontend context), B
  (acausal spec family + compiler + builder; compile service + report;
  undirected canvas; structured diagnostics), C (mechanics templates +
  assembly view — coordinate with `f3159c7` to avoid double-owning the
  workspace projection), D (penzai inspector; CDE primitives — relates
  `2f8dd61`, `c1ad986`, `7c0c4b2`), plus a docs child updating
  `DAE_BIOMECHANICS_SPEC.md` Phase 3 → this design.
- Out-of-scope pointers: inner-iteration/multirate subgraphs stay on
  `56dfd97`; analysis-canvas unification on `f22dbf4`/`e8ec78e`; 3D/MJX on
  AUX_3D reservations.

## 13. Precedent index (research citations)

Simscape domain line styles/legend and converter blocks
(mathworks.com/help/simscape/ug/domain-specific-line-styles.html,
…/connecting-simscape-diagrams-to-simulink-sources-and-scopes.html);
structural-singularity diagnostics and illegal topologies
(…/troubleshooting-simulation-errors.html); Simulink Diagnostic Viewer stages,
severity escalation, fix suggestions
(mathworks.com/help/simulink/slref/diagnosticviewer.html). Modelica connector
semantics (specification.modelica.org/master/connectors-and-connections.html),
per-domain colors and icon-boundary connectors (mbe.modelica.university),
balance checking and its pitfalls (openmodelica.org; ticket #3977).
ModelingToolkit named-variable singularity errors
(sciml.github.io/ModelingToolkitCourse/dev/lectures/lecture7/). Collimator
groups-vs-submodels and acausal block taxonomy (docs.collimator.ai). Houdini
context-sensitive Tab menu and subnet dive/breadcrumbs (sidefx.com docs).
Blender shared node-editor with per-editor-type palettes (docs.blender.org).
Unreal shared SGraphEditor, Blueprint compile-button states, Material
inline ERROR! + Compiler Results panel (dev.epicgames.com). LabVIEW broken
run arrow + Error List (ni.com). VS Code error-propagation-to-container
proposal (github.com/microsoft/vscode/issues/119023). OpenSim Creator
tree+sockets and muscle path points (docs.opensimcreator.com). MuJoCo MJCF
fail-fast compiler (mujoco.readthedocs.io). Penzai/Treescope WYSIWYG
structure-is-the-computation (arxiv.org/html/2408.00211v1,
treescope.readthedocs.io).
