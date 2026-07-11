# Workspace View: Representation Contract, Scoping, and Playback

> **Reconstruction notice:** This document was reconstructed post-hoc. The code is
> authoritative wherever it disagrees with this document.

**Status:** working draft (design session 2026-07-05, not yet a spec or auth request).
**Relation to ledger:** elaborates and partially supersedes the sketches in `946c859`
(spatial representation + objective authoring contract) and `6a094c9` (runtime playback +
validation overlays). Both remain the right work split; this doc proposes the unifying
data model they should share, plus concrete archetypes, scoping semantics, and a build
plan (see `LEDGER_PLAN.md`). 3D/MuJoCo strategy is in `AUX_3D.md`.

---

## 1. Problem and current state

The Workspace projection today (`web/src/components/scenario/ScenarioProjectionWorkspace.tsx`,
`WorkspaceProjection`) is a hardcoded SVG: fixed viewBox, a synthesized ring of targets driven
only by `n_targets`/`target_radius`, and a two-segment "arm" at fixed pixel coordinates. Its
selection plumbing (entity registry → `selected_entity_id` → Properties panel) is real and
reusable; its geometry is decorative.

Separately, a **dormant, orphaned viewer chain** exists (`web/src/components/viewer/`:
`Scene.tsx`, `BodyRig.tsx`, `TargetMarker.tsx`, `PlaybackControls.tsx`, sole importer
`TrajectoryPanel.tsx` which is itself unimported). It has real forward kinematics, task-type
colored target markers with motion traces, and a complete rAF playback transport with
scrubber/speed/keyboard control — but it is 3D-perspective (r3f/three.js), wired to the offline
NPZ trajectory-browser API rather than the live scenario pipeline, and none of its state is
persisted (violating no-volatility).

Contracts that already anticipate this feature:

- The frontend selector type (`web/src/types/workspace.ts`) already enumerates
  `mechanics_object` and `biomechanics_object` namespaces; the Python `StudioSelectorRef.namespace`
  is a free string (tightening it is C1 work).
- `StudioScenarioSpec.biomechanics_spec` is an explicitly reserved, untyped slot.
- `RetainedObservableSpec`/`RetainedObservableTargetSpec` is the versioned, general "retain
  this value during rollout" envelope (kinds: port/edge/graph_output/recurrent_carry/state_path/
  task_data; retention modes stream/window/trajectory).
- `ComponentDefinition` (served by `/api/components`) carries param schemas, port types,
  identity/provenance, and migrations — but no visual metadata.
- Every skeleton implements `forward_kinematics` returning per-joint Cartesian state
  (`AbstractSkeleton` contract); `BodyPreset`/`MuscleTopology`/`MuscleConfig` carry exactly the
  geometry a muscle schematic needs (origin/insertion bodies and positions, signed moment arms).
- `TrialTimeline` (epochs, named events like go cue) is the temporal model a scrubber needs.
- `TargetSpec` (value, `time_idxs`, `time_mask`, `discount`) is the loss-target model an
  overlay needs — a loss term is *where* (selector), *versus what* (target), *when* (mask/discount).
- The frontend's `isMechanicsNode()` heuristic (substring match on
  `['mechanic','plant','arm','muscle',...]` in `web/src/features/scenario/entities.ts`) is the
  clearest sign that a first-class representation contract is missing.

**The design problem is therefore not "draw an arm."** It is: define the contract by which any
component/task/objective — including ones that don't exist yet — declares its geometric
presence, its interaction affordances, and its data needs, such that the Workspace view is a
pure projection of specs + artifacts (never a shadow model), across authoring, preview, and
playback modes.

---

## 2. Core abstraction: entities, anchors, pose sources

Three ideas carry the whole design.

### 2.1 Scene entities with typed anchors

The workspace scene is a set of **spatial entities**, each derived from a spec-level object
(a mechanics node, a task, an objective term, an intervention) and identified by the *same
entity id* used everywhere else in Studio (the existing scenario entity registry). Every
entity exposes named **anchors**: points (or small frames) with semantic roles.

Anchors are the common language the user asked for. A point mass exposes
`{body, effector}` (coincident). A planar N-link arm exposes
`{base, joint[0..N-1], effector, link_midpoint[i], ...}`. A task exposes
`{goal, start, target[i], center}`. A muscle exposes `{origin, insertion}` — resolved into
*another entity's* frames. "The purple line for the goal-position loss" is then simply an
`objective_link` entity whose two endpoints are anchor references resolved from the objective
term's `source_selector`/`target_selector`. Nothing about the loss overlay knows what an arm
is; it knows how to resolve a selector to an anchor.

Anchor references are expressed as `StudioSelectorRef`s in the `mechanics_object` /
`task_object` namespaces with a sub-path, e.g. `mechanics:<node-id>.effector`,
`task:<scenario>.goal`. This makes anchors addressable by objectives, bindings, interventions,
analyses — the same addressing vocabulary as everything else, per the 946c859 comment's
constraint that a rendered object is selectable for authoring only if it maps to a canonical
selector.

### 2.2 Pose sources: one scene, four ways to pose it

Every anchor's position is supplied by a **pose source**. The scene's *structure* (which
entities, which anchors, which archetypes) never changes across modes; only the pose source
and overlay set change:

| Pose source | Definition | Mode it generates |
|---|---|---|
| `rest` | Default configuration declared/derived from params only (e.g. arm at preset rest angles; task targets at canonical schematic positions) | **Authoring schematic** |
| `sampled(trial_spec, seed)` | Positions from a task-sampled trial spec (inits, targets), no controller | **Sampled preview** |
| `recorded(artifact, trial, t)` | Frame `t` of a stored rollout track | **Validation/eval playback** |
| `live(stream)` | Latest frame from a streaming event during training | **Live overlay** |

This single distinction cleanly generates the mode taxonomy in `6a094c9` (authoring /
sampled preview / validation playback / evaluation playback / live overlay / imported
artifact) without any mode-specific scene code. "Imported" is `recorded` with degraded
metadata and an explicit provenance banner.

It also resolves the "schematic vs to-scale" tension in the motivating brief: the view is
**always to-scale in world units** (meters, y-up, single scenario workspace frame, pan/zoom +
scale bar). What is *schematic* is the pose source (`rest`, or a distribution glyph standing
in for a family of trials) — not the geometry. Purely symbolic glyphs (markers, distribution
icons, annotations) additionally declare `scale_invariant: true`, meaning their *position* is
world-frame but their *size* is screen-space. That is how a target can be a crisp 8-px dot at
any zoom while the arm segments remain physically proportioned.

### 2.3 Scene spec compiles to a data request

The third idea makes playback parsimonious: given a scene, the set of anchor selectors it
needs *is* a retention request. The scene resolver emits the required selector set, and the
backend materializes it as `RetainedObservableSpec`s with `retention.mode = "trajectory"`.
There is no separate, hand-maintained "what to record for the workspace view" list — the
representation contract compiles to it. Eval/validation artifacts then carry per-trial,
per-anchor Cartesian tracks (see §6), and the renderer never runs model code.

---

## 3. The representation contract (`WorkspaceRepresentationSpec`)

### 3.1 Who owns it: provider-declared data, not component methods, not frontend heuristics

The mechanics classes should **not** implement `draw()` — that couples simulation code to a
rendering vocabulary and makes representations unavailable without instantiating JAX objects.
The frontend should **not** infer representations — that is the `isMechanicsNode` name-sniffing
dead end. Instead:

- The **component registry entry** (`ComponentMeta`/`ComponentDefinition`) gains an optional
  `representation: RepresentationSpec` block — declarative data, versioned with its own schema
  id, served through the existing `/api/components` catalog next to `param_schema`. This is the
  same pattern as MJCF (visual metadata rides on the model spec, one authoring surface) and
  Simscape/Modelica (geometry params on the component feed both physics and rendering), and it
  respects the registry's existing ownership/provenance/migration machinery for free.
- **Tasks** get the analogous block on their registry/spec records.
- A registered **renderer id** escape hatch exists for genuinely bespoke cases
  (`renderer: "feedbax.workspace.<name>"`), but it names a *registered capability* with declared
  required fields and fallback behavior — never an ad hoc React branch keyed on type strings
  (the 946c859 comment's constraint). The expectation is that ≥90% of components never need it.

Honesty rule (mirrors "absence of a subgraph is an error"): a component with no
`representation` block renders as an explicit **unrepresented placeholder** (labeled chip at
the workspace margin, not fake geometry), and the validation panel reports it. Executable but
not representable is a visible state, never silently invented geometry.

### 3.2 Archetype vocabulary

`RepresentationSpec` declares one or more entities built from a small, closed set of
**archetypes** (rerun.io's archetype model is the precedent worth copying: typed, versioned,
composable "what to draw" records over an entity path). Proposed initial vocabulary:

| Archetype | Semantics | Bindings (examples) | Anchors exposed |
|---|---|---|---|
| `point_body` | Point mass / particle | mass→param `mass` (affects marker size channel, optional) | `body`, `effector` |
| `planar_chain` | N-link serial linkage | `link_lengths`→param `l`; `rest_angles`→literal or param; `joint_limits`→param `bounds` | `base`, `joint[i]`, `effector` |
| `muscle_path` | Line of action between frames | `origin`/`insertion`→anchor refs (possibly via frame provider, §3.4); `activation`→state selector (style channel) | `origin`, `insertion`, `midpoint` |
| `marker` | Task point (start/goal/via/cue) | `position`→task param or trial-spec path; `role`; `temporality: static\|scheduled\|moving` | `point` |
| `region` | Workspace bounds, obstacle, distribution support | `bounds`→param (e.g. task `workspace`) | `center`, `corner[i]` |
| `distribution_glyph` | Schematic of a sampled family (center-out ring, uniform rect) | `distribution`→ValueSpec of the task param; `n_preview` | `representative` |
| `vector` | Force/velocity/perturbation arrow at an anchor | `at`→anchor ref; `value`→state selector | — |
| `trace` | Trajectory polyline for an anchor over time | `anchor`→anchor ref; `window` | — |
| `objective_link` | Loss/objective edge between two anchors | derived from `StudioObjectiveTermSpec` selectors + `TargetSpec` timing | — |
| `annotation` | Label/callout | free | — |

Notes:

- `planar_chain` covers PointMass-on-a-stick, TwoLinkArm, and future N-link arms with one
  archetype — "an N-link arm mechanics object with different lengths" is just data. The
  `AnalyticalMusculoskeletalPlant` composes `planar_chain` + N × `muscle_path`.
- `marker.temporality` drives the static-goal vs tracking-target glyph distinction from the
  brief (plain dot vs dot-with-motion-arcs). It should be *derived from task semantics*
  (does the target's `TargetSpec` value vary in time? does the timeline schedule target onset?)
  rather than hand-set, with the provider able to override.
- Style is separated from geometry via **style channels** (color, width, opacity, dash) that
  bindings can drive: muscle activation → width/heat; loss `discount` over time → objective_link
  opacity during playback; per-condition trial colors → trace color (reuse
  `feedbax/plot/colors.py` conventions so workspace and analysis figures agree).
- Cross-view color semantics get promoted to named design tokens: **objective violet**
  (currently ad hoc `violet-500` on graph port markers) is the color of every objective
  projection in every view; emerald stays task-binding; amber stays live-training highlight.

### 3.3 Bindings

Each archetype field binds to one of: a **param path** on the owning component spec (live —
editing `l[0]` in the sidebar re-renders the arm immediately, and vice versa §7.2); a
**state/anchor selector** (pose- or overlay-driving); a **trial-spec path** (task data); or a
**literal**. Bindings are validated against the existing `ParamSchema`/`ValueSchema`/
`SelectorTargetSchema` machinery at catalog load, not draw time. A `ValueSpec`-valued param
(distribution/expression rather than constant) is legal where the archetype declares it —
that is what makes `distribution_glyph` and sampled previews fall out naturally.

### 3.4 Composition, subgraphs, and frame providers

Per the Core Principle, the subgraph is the model. Representation follows the same rule:

- A **composite node's representation is the composition of its subgraph children's
  representations**, resolved recursively. No outer/stale representation may shadow the
  subgraph. A composite may declare a *collapsed* fallback (single glyph) used only in the
  graph view, never in the workspace.
- **Frame providers:** a muscle component's `origin`/`insertion` anchors live in frames owned
  by a skeleton entity. The muscle's representation declares
  `frame_provider: from_input_port("angles")` (or similar): the resolver follows the graph
  wiring from that port to find the host entity and resolve `link0`/`link1` frames against it.
  This keeps attachment topology in data (`MuscleConfig.origin_body/insertion_body/origin_pos/
  insertion_pos` is exactly the needed shape) while the *binding* between muscle and skeleton
  is the graph topology itself — no duplicate declaration, and rewiring the graph re-binds the
  visual automatically. An unresolvable frame provider is a visible validation error.

Prerequisite cleanup: there are at least three divergent "canonical 6-muscle 2-link"
parameterizations in the Python package (`MuscledArm` constructor defaults,
`TwoLinkArmMuscleGeometry.default_six_muscle`, `default_6muscle_2link_topology` +
`default_muscle_config`). Consolidate to one source before binding representations to them
(child issue in `LEDGER_PLAN.md`).

### 3.5 Resolution pipeline

```
ComponentDefinition.representation (catalog, static)
        × GraphSpec node params (scenario)
        × graph wiring (frame providers)
        × task spec/registry representation
        × objective_spec terms
        × stage/run/artifact selection (scoping, §5)
   → ResolvedScene { entities, anchors, required_selectors, validation }
        × PoseSource (rest | sampled | recorded | live)
   → posed frame(s) → renderer
```

`ResolvedScene` is **derived, cached, never persisted** — persisting it would create a shadow
model. What persists is only view state (§8). `StudioScenarioSpec.biomechanics_spec` should
*not* become a stored scene; if used at all, it holds scenario-level representation
*overrides* (e.g. a chosen workspace frame/extent, per-scenario visibility defaults), typed
and versioned, defaulting to empty.

---

## 4. Loss and objective overlays

An objective term is already `(source_selector, target_selector, weight, timing)`. The overlay
projection is:

- Resolve both selectors to anchors (or to a task target value). Render an `objective_link`
  in objective violet between them — e.g. effector anchor ↔ goal marker for the canonical
  goal-position loss.
- Terms whose selectors do not resolve to spatial anchors (hidden-state penalties, control
  effort) do **not** invent geometry — they remain graph-view port markers (existing violet
  crosshairs) and appear in the workspace only in a compact legend ("2 non-spatial terms"),
  keeping the two views complementary rather than duplicative.
- `TargetSpec.time_mask`/`discount` project onto **time**, not space: the scrubber (§6.3)
  shows violet activity bands per spatial term, and during playback the link's opacity follows
  the discount weight at `t`. A final-state-only loss reads as a link that "arms" near the end
  of the trial — the temporal structure of the objective becomes directly visible, which no
  static figure currently shows.
- Weights map to a width/intensity channel with a legend; editing stays in the sidebar.

---

## 5. Scoping: stage, run, trial, time

The scene is scoped by exactly four coordinates, in strict order, each with a persisted
selection and an explicit "nothing selected" rendering:

1. **Stage** (existing bottom-shelf tab → scenario): determines *which specs* project. The
   training stage projects the training scenario's task + objective terms; an eval stage
   projects *its* task variant and *its* metrics/objectives — so loss terms relevant to
   training don't leak into eval views, per the brief. This linkage (bottom tab scopes top
   pane) is currently unwired in the frontend (top and bottom shelves scope independently
   today); making it real is part of this work and consistent with the open `0c46f69`
   pipeline spec.
2. **Run/artifact** (existing `selection_spec`/collection machinery): determines which
   recorded pose sources are *available*. No run selected → playback modes disabled with an
   explanatory affordance, authoring/preview still fully functional.
3. **Trial(s)**: within a selected artifact. Single trial armed → animation enabled.
   Multiple trials/conditions selected → static overlay of traces (condition-colored, reusing
   plot color conventions) with optional mean + dispersion band; animation disabled rather
   than ambiguous. Aggregate views are first-class, not an afterthought — comparing an
   authored task against 32 rollout traces is a primary inspection act.
4. **Time**: the scrubber. Shared cursor semantics (rerun-style): the workspace, any open
   profile plots, and the loss-window bands all follow one time cursor.

**Mode is derived, not chosen from a free menu:** available modes = f(stage kind, data
present). The mode banner (always visible, part of no-jitter chrome) states what is shown and
its provenance: "Authoring — rest pose", "Preview — 8 sampled trials, seed 17",
"Eval playback — run `baseline__12k`, trial 3/32, checkpoint 8000". Authored intent vs
observed behavior must be unmistakable at a glance (6a094c9's core UX requirement); ghost
rendering (§7.4) reinforces it.

---

## 6. Playback data path

### 6.1 Replay product

Extend the eval/validation artifact contract with a `workspace_replay` product: per trial,
per anchor-selector, a Cartesian track `(T, 2)` (plus per-frame overlay channels like muscle
activation where bound), together with the trial spec snapshot (targets, `TrialTimeline`),
and manifest references to spec snapshot / checkpoint / task variant / seed. The required
selector set comes from the scene compile (§2.3). Feedbax owns kinematic truth: anchors are
resolved server-side via the same selector machinery used everywhere, so the frontend never
re-implements FK for playback correctness. (Client-side planar-chain FK exists only as an
interaction nicety in authoring mode, §7.2.)

This deliberately supersedes the offline NPZ trajectory-browser path as the workspace's data
source (that subsystem can remain as an importer: NPZ → imported-artifact mode with explicit
missing-metadata warnings).

### 6.2 Live overlay

Generalize the existing `training_trajectory` WS event from hardcoded `{effector, target, t}`
to tracks keyed by anchor selector — same shape as replay frames, lower fidelity, explicitly
badged as periodic snapshots. One frame vocabulary across live and recorded is what keeps the
renderer mode-free. This is a versioned change to a durable wire shape → migration rule per
repo policy.

### 6.3 Transport UI

Revive the dormant transport (`PlaybackControls` logic: rAF loop, speed options, keyboard
map, scrubber) as a 2D-agnostic component. Additions: epoch bands and named event ticks from
`TrialTimeline` (hold/target_on/movement shading, go-cue tick), violet loss-window bands
(§4), loop toggle, and frame-step honoring `dt`. Placement: overlay strip at the bottom of
the workspace canvas, only in playback-capable modes, constant height (no-jitter).

### 6.4 Sampled preview

Needs a small server endpoint: "sample N trial specs for this task spec + seed" (task
sampling without any controller; `POST /api/execution/simulate` is currently a 501 stub and
full simulation is *not* required for this). Preview shows N ghosted trial instances (start/
goal pairs, schedule), with a visible seed and a reseed control. This is the cheapest
high-value slice: it makes task authoring feel physical before any training exists.

---

## 7. Interaction and authoring UX

### 7.1 Selection and cross-view linking

Workspace entities participate in the existing entity-selection system (same ids →
`selected_entity_id` → Properties panel), which already works in the mockup. Add symmetric
hover-linking: hovering the arm node in the graph view highlights the arm in the workspace
and vice versa; hovering an objective term (sidebar, graph port marker, workspace link)
highlights all three projections. Cheap, high-payoff, and it teaches the mental model that
all views project one spec.

### 7.2 Direct manipulation (bidirectional editing)

Desmos-style: geometry is a *second editor* for the same params the sidebar edits.

- Drag a fixed goal marker → writes the task param (with snapping and numeric readout while
  dragging, PhET-style).
- Drag on a distribution-backed marker cannot silently pin a sample: it opens a small choice —
  *edit distribution bounds* vs *fix endpoint* (writes `fixed_endpoints`) — mirroring the
  ValueSpec modes rather than bypassing them.
- Drag workspace-region corner → task `workspace` bounds.
- Drag effector/joints in authoring mode to inspect configurations (client planar-chain FK/IK
  for responsiveness). Whether a posed rest configuration is *persisted* as a spec-level
  rest-pose param or is transient inspection is an open question (§10); no-volatility says if
  it stays visible, it must be saved.
- Every drag respects no-jitter (fixed-size handles appearing on hover within reserved
  hit-areas) and writes through the normal dirty/save path.

### 7.3 Objective authoring with canonicalization

Per `946c859`: drawing from an effector anchor to a goal marker creates a real
`StudioObjectiveTermSpec` with canonical selectors. Anchors carry an `interaction_role`
(objective-source / objective-target / canonical-for:`<entity>` / illustrative). Starting a
drag from a decorative center-out target snaps — visibly, with a brief explanation — to the
task's canonical goal entity. Illustrative glyphs can *initiate* but never *terminate as
themselves*.

### 7.4 Honest-state affordances

- **Ghosting/onion-skin:** in playback, the authored rest pose renders as a faint ghost under
  the animated body; in preview, unsampled schematic markers ghost under sampled instances.
- **Reachability validation:** `TwoLinkArm.bounds` + `workspace_test` already exist — render
  the reachable envelope as a subtle region in authoring mode, and style goals outside it as
  validation warnings (wired into the existing ValidationPanel). This turns a silent
  train-time pathology into an authoring-time visual.
- **Comparison overlay:** two selected runs/checkpoints on the same trial → paired traces/
  bodies, condition-colored, shared time cursor. (Data-cheap once §6.1 exists; huge for the
  "did robustness training change the reach" question.)

---

## 8. Persistence and migrations

Typed `WorkspaceViewState` under the scenario/workspace `ui_state` (versioned, migratable):
active mode *inputs* (selected artifact/trial refs — mode itself is derived), camera
(pan/zoom), visibility toggles per overlay class, playback position/speed, comparison
selection. All of it survives save/load per no-volatility (the dormant viewer's unpersisted
camera/playback is the counterexample to fix). Validation at snapshot-build time mirrors
`assertGraphUiStateConsistency` (a persisted trial index must exist in the referenced
artifact; if not, fail visibly to a defined fallback).

New durable schemas introduced (each with schema id + version + migration or explicit
rejection): `RepresentationSpec` (registry-level), `workspace_replay` artifact product,
generalized live-trajectory event, `WorkspaceViewState`, optional scenario representation
overrides. The GraphSpec itself should not need changes — that is a feature of the design.

---

## 9. Rendering technology

**SVG first, renderer behind an interface.** Rationale: current scene sizes are tens of
elements; SVG gives free hit-testing, CSS-token styling, crisp export, accessibility, and
no-jitter is easiest to audit in the DOM. The animation hot path (traces + body pose at
60fps, possibly 32+ trials) is the only risk; keep the *frame application* layer (pose →
transforms) separate from the *scene construction* layer so a Canvas2D/PixiJS backend can
replace it if profiling demands (precedent research: React Flow itself is DOM/SVG; PixiJS is
the standard escape hatch for 100s–1000s of animated elements). Do not adopt the dormant
three.js scene for the 2D workspace: a perspective camera on a planar task is worse UX than a
true 2D orthographic viewport, and the 3D question is deliberately separate (`AUX_3D.md`).
Cannibalize from the dormant chain: transport logic, FK utils, marker/trace semantics.

---

## 10. Open questions (for user decision or first-slice discovery)

1. **Rest pose ownership.** Is a mechanics rest configuration a real spec param (persisted,
   schema'd) or derived-only (zeros/preset)? Affects §7.2 drag-to-pose semantics.
2. **Trial identity in artifacts.** Stable trial ids vs indices — needed for persisted trial
   selection to survive re-evaluation; interacts with manifest/lineage design (`c6c6da0`,
   `e33f487`).
3. **Aggregate animation.** Is animating a mean trajectory (with dispersion band evolving in
   time) worth the statistical honesty problems, or is aggregate = static-only the right
   permanent rule?
4. **Where task representations live.** Tasks aren't graph nodes; their registry surface is
   less developed than components'. Does task representation metadata ride on task presets/
   registry entries, or on the task spec schema itself?
5. **Muscle overlay default.** Activation → width vs color-heat as default channel (width
   reads better at small scale; color collides with condition coloring).
6. **`biomechanics_spec` fate.** Keep reserved-and-empty, type it for overrides (§3.5), or
   deprecate in favor of representation specs + view state only?
7. **Interventions/perturbations.** Force-field or bump perturbations have natural `vector`/
   `region` projections scoped by intervention specs — in scope for the first playback slice
   or a follow-up?

---

## 11. Precedents (condensed; from research pass 2026-07-05)

- **rerun.io** — entity paths + typed archetypes + shared time cursor; the closest existing
  embodiment of §2. Strongest single precedent.
- **MJCF/MuJoCo** — visual metadata co-located with model spec, one authoring surface (our
  §3.1 registry co-location mirrors this).
- **Simscape Mechanics Explorer / Modelica animation** — geometry params on the component feed
  both physics and rendering; viewport recomputes on change; no parallel scene file.
- **Collimator.ai** — notably does *not* have a physical viewport (plots only): this feature
  is a genuine differentiator, not table stakes.
- **Desmos / PhET / Algodoo** — bidirectional canvas↔sidebar editing, drag-with-readout,
  force/velocity vector overlays on schematics.
- **urdf-loaders (gkjohnson), MuJoCo WASM (official bindings; zalo/mujoco_wasm as legacy)** —
  browser 3D feasibility; details and caveats in `AUX_3D.md`.
- **React Flow (DOM/SVG) vs PixiJS** — §9 tech posture.
