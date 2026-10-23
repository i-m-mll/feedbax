# Ledger plan: Workspace view umbrella and children (FILED 2026-07-05)

> **Reconstruction notice:** This document was reconstructed post-hoc. The code is
> authoritative wherever it disagrees with this document.

**Status:** filed. Umbrella `f3159c7` created with all 14 children adopted (delegation
manifest refreshed), `blocks` dependency edges recorded, and a creation comment posted on the
Studio design coordination log (`93f79f2`). Issue IDs:

| C# | Issue | Title |
|---|---|---|
| — | `f3159c7` | Umbrella: Studio Workspace view — representation contract, authoring, and playback |
| — | `946c859` | (adopted) spatial representation + objective authoring contract |
| — | `6a094c9` | (adopted) runtime playback + validation overlays |
| C1 | `fd5143f` | Representation contract schemas + registry plumbing |
| C2 | `e2cc8ec` | Built-in representation declarations (mechanics + tasks) |
| C3 | `9534977` | Scene resolver + SVG renderer (authoring mode) |
| C4 | `81534f4` | Stage scoping + WorkspaceViewState persistence |
| C5 | `79dc3d1` | Sampled task preview |
| C6 | `6ec401b` | Objective authoring interactions + canonicalization |
| C7 | `525249b` | Muscle-geometry consolidation (Python) |
| C8 | `957f465` | workspace_replay artifact product + retention compile |
| C9 | `854d05f` | Playback UI + transport |
| C10 | `3156ec2` | Live overlay generalization (selector-keyed frames) |
| C11 | `d4124ac` | Comparison overlay (runs/checkpoints) |
| C12 | `b1ecca4` | 3D representation strategy hedges (analysis-only) |

The proposal text below is retained as the rationale of record for the filing.

## Existing ledger state

- `946c859` (open, feature) — Workspace spatial representation + objective authoring
  contract. Matches DESIGN §§2–4, 7.3. Its single comment already sketches
  `SpatialRepresentationSpec`/`WorkspaceRendererSpec`/`WorkspaceOverlaySpec`; DESIGN refines
  these into archetypes + anchors + pose sources.
- `6a094c9` (open, feature) — Workspace runtime playback + validation overlays. Matches
  DESIGN §§5–6. Explicitly non-blocking on `946c859` beyond shared entity ids/frames.
- `a3e46d5` (closed umbrella) — their former parent; a new umbrella is needed.
- Cross-cutting neighbors: `0c46f69` (bottom-pane pipeline spec — stage scoping),
  `bd5aefb` (task timelines/value authoring — scrubber/epoch coordination),
  `c6c6da0`/`e33f487` (manifests — replay provenance), `d90b3e5` (runs vs sessions),
  `6116155` (canvas architecture), `f68cf66` (provider/plugin contract).

## Proposed umbrella

**`umbrella: Workspace view — representation contract, authoring, and playback`**
Motivating question: *what contract lets any mechanics/task/objective component declare its
geometric presence so the Workspace tab is a faithful, to-scale, animatable projection of
specs and artifacts across authoring, preview, and playback?*
Body: link `DESIGN.md`/`AUX_3D.md`, Children table per convention. Adopt `946c859` and
`6a094c9` as children (`mandible umbrella adopt`). Comment creation on `93f79f2` (Studio
design coordination), not on rlrmp coords.

## Proposed children (beyond the two adopted)

| # | Working title | Scope (DESIGN ref) | Depends on |
|---|---|---|---|
| C1 | Representation contract schemas + registry plumbing | `RepresentationSpec` (archetypes, anchors, bindings, style channels, frame providers) as versioned Pydantic contracts; optional Studio facet on `DeclaredComponent`, projected to `ComponentDefinition`; catalog serving; TS mirrors + `parseContract` wiring; unrepresented-placeholder fallback. (§§2–3) | — (anchor child; refines `946c859`) |
| C2 | Built-in representation declarations | `point_body`, `planar_chain`, `muscle_path` declarations for PointMass, TwoLinkArm, MuscledArm/AnalyticalMusculoskeletalPlant, templates; task representations for SimpleReaches + DelayedReaches (markers, regions, distribution glyphs, canonical goal roles, temporality derivation). (§3.2, §3.4) | C1, C7 |
| C3 | Scene resolver + SVG workspace renderer (authoring mode) | Resolution pipeline, `ResolvedScene`, world-frame viewport (pan/zoom/scale bar, scale-invariant glyphs), entity selection/hover cross-linking, replacement of hardcoded `WorkspaceProjection`, validation surfacing (incl. reachability envelope). (§3.5, §7.1, §7.4) | C1, C2 |
| C4 | Stage scoping + `WorkspaceViewState` persistence | Bottom-stage → top-pane scenario/objective scoping made real; versioned view-state schema + snapshot validation + migrations. (§5, §8; coordinate with `0c46f69`) | C3 |
| C5 | Sampled preview | Trial-sampling endpoint (no controller), seed/reseed UI, ghosted instances, distribution-glyph interplay. (§6.4) | C3 |
| C6 | Objective authoring interactions | Drag-to-create objective terms, anchor interaction roles, canonicalization + snap UX, violet token promotion across views. (§4, §7.3; core of `946c859` acceptance) | C3 |
| C7 | Muscle-geometry consolidation (Python) | Single canonical source for 6-muscle/2-link defaults; reconcile `MuscledArm` defaults, `TwoLinkArmMuscleGeometry.default_six_muscle`, `default_6muscle_2link_topology`/`default_muscle_config`; public accessors for muscle path geometry. (§3.4) | — (parallel; unblocks C2 muscles) |
| C8 | Replay product + retention compile | `workspace_replay` artifact product schema; scene→`RetainedObservableSpec` compilation; server-side anchor resolution; manifest linkage; NPZ-importer downgrade path. (§2.3, §6.1; core of `6a094c9`; coordinate `c6c6da0`) | C1; manifests lane |
| C9 | Playback UI + transport | Revived transport (rAF/scrubber/keyboard) with `TrialTimeline` epoch bands, event ticks, loss-window bands, dt-honoring stepping; trial/aggregate selection semantics; ghosting; mode banner. (§5–6.3, §7.4) | C3, C8 |
| C10 | Live overlay generalization | `training_trajectory` → selector-keyed frame event (versioned migration); live badging. (§6.2) | C8 (frame vocab) |
| C11 | Comparison overlay | Multi-run/checkpoint paired playback on shared trials. (§7.4) | C9 |
| C12 | 3D strategy note (analysis-only) | Adopt/record `AUX_3D.md` hedges as decisions; no implementation. | — |

Direct-manipulation param editing (§7.2) is deliberately *not* a numbered child yet: it
spans C3/C5/C6 and has the open rest-pose question (DESIGN §10.1) — slice it after C3 lands
and the question is answered.

## Sequencing sketch

C1 → C2 → C3 is the critical path to "the mockup is replaced by something real."
C7 and C12 are parallel-anytime. C5 and C6 branch off C3 independently (C5 is the cheapest
visible win; C6 discharges most of `946c859`). C8 → C9 → {C10, C11} is the `6a094c9` lane and
can start (C8 schema work) as soon as C1 stabilizes anchor selectors.

## Policy notes

- Every new durable schema (C1, C4, C8, C10) needs schema id + version + migration-or-reject
  per repo policy, with focused transition tests.
- Implementation work requires ledger issues before code; this plan is the pre-work artifact.
- `946c859`/`6a094c9` acceptance criteria remain authoritative; DESIGN maps onto them rather
  than replacing them. If any child's slice proves the two issues' data contracts should merge
  or re-split, record that on the umbrella, per their own bodies.

## Open questions to resolve at umbrella-filing time

DESIGN §10 list; minimally: rest-pose ownership (blocks direct-manipulation slicing), trial
identity in artifacts (blocks C8 schema), task-representation home (blocks C2 task half),
`biomechanics_spec` fate (C1 decision).
