# Pipeline UX Redesign — Ledger Map

> **Reconstruction notice:** This document was reconstructed post-hoc. The code is
> authoritative wherever it disagrees with this document.

Status: FILED 2026-07-06. Companion to `pipeline_ux_redesign.md` (section references like
§4.2 point there).

Umbrella `efa2093` (child of `4007bf2`) with 12 adopted children:
P1=`9aa8ff2` · P2=`c199a9c` · P3=`e276739` · P4=`3a6d02e` · P5=`82a15e5` · P6=`067dc59` ·
P7=`717e8fb` · P8=`82a09d8` · P9=`12e49a2` · P10=`0ede698` · P11=`4b52e88` · P12=`76785fb`.
Decision-log comment on `93f79f2`; cross-ref comments on `4007bf2`, `96ac771`, `0c46f69`,
`d90b3e5`; related links per the table below.

## Where proposals land on existing issues

| Issue | State | What this design adds to it |
|---|---|---|
| `4007bf2` pipeline completion umbrella | open | Parent for everything below; the redesign is its UI half. |
| `0c46f69` Studio pipeline workspace spec | open | §4 core model (pending manifests, selections-as-queries, one SelectionContext, projections Table/Lineage/Queue) is a concrete proposal for this spec's next revision; its 2026-07-04 "rails note" (surface skipped/not_applicable, two dependency granularities) is answered by §6 Analysis + §7. |
| `96ac771` training-run authoring & execution-target UX | open, fresh | §4.2 axes/matrix/expansion, §5 ValueSpec editor + bulk verbs, §6 Train tab, billable spec-lock gate. This issue's acceptance slice ("sweep of 0 and 1e-5 → two planned rows with a dynamic column") is exactly §4.2 steps 1–4. Likely needs splitting: frontend matrix UX vs backend expansion (B2). |
| `d90b3e5` runs vs sessions/tabs | open | The §6 Train layout (table + focused-run detail) is the remaining UI-column work this issue tracks; subsume or close into the Train-tab child. |
| `e33f487` manifests as canonical lineage + DB indexing | open | B1 (run-summary/index API) is this issue's scope; §4.1 makes it the load-bearing substrate for the whole pane. |
| `bd5aefb` ValueSpec umbrella | closed (substrate only) | §5 is the successor design: Value×Variation split, enumerable axis forms, eligibility-matrix enforcement, typed lowering (D5). File a new child rather than reopening. |
| `4d1558c` report template layer | open | §6 Report deliberately stays minimal until this lands; no new scope claimed. |
| `946c859` / `6a094c9` workspace spatial + playback | open | Integration seam only: §4.4 provenance mode (`SelectionContext.focusedId` + frozen-snapshot projection) is the contract the workspace-view lane should consume for animation. Cross-link, don't merge. |
| `f68cf66` / `51832b9` Mandible provider/mapping | open/paused | Untouched; queue projection (§6) should later surface Mandible run-status checkpoints, but nothing here depends on it. |
| `589535c` import/export packets | closed | §8.4 ingest UI consumes it (new child below). |
| `3300ced` / `196bd09` conditions + expressions | closed | Consumed by §4.3, §6 Analysis, B4, B5. No new backend scope beyond the `top-k per group` predicate primitive and the dry-run wrapper. |

## Children (filed under `efa2093`; scopes as specced in the issue bodies)

Labels: `feature`; sequencing in dependency order.

1. **P1 — Pending-manifest staging + run index API** — `9aa8ff2` (B1; enables everything). Staging writes
   `status:"pending"` manifests with deterministic content-addressed training-run ids; typed
   run-summary endpoints over the manifest index; retire `runAPI.ts` stubs and
   `pipelineCollections.ts` metadata reshaping; GC story for cancelled/stale pending manifests.
   Durable-schema note: id scheme + any `TrainingRunManifest` field additions need migration
   rules.
2. **P2 — Sweep axes, matrix expansion, run sets** — `c199a9c` (B2; depends P1). ValueSpec sweep axes →
   expansion (cross/zip/manual) → pending manifests + `TrainingRunSetManifest.axes`;
   auto-labels from varied axes; align with worker `AxisRole "authored_sweep"`; reference
   semantics from `config/batch.py` zip/broadcast. Migration rule for the set-manifest schema
   bump.
3. **P3 — ValueSpec v2** — `e276739` (B3; co-designed with P2; successor to `bd5aefb`). Value×Variation
   schema split, enumerable forms (list, lin/log range, sampler+n), per-field eligibility
   matrix, typed lowering of non-constant component params (D5 fix), editor rebuild (anchored
   popover, structured editors, consequence footer). Schema version + migrations + accept/
   migrate/reject tests.
4. **P4 — Train tab rebuild + TrainingPanel dissolution** — `3a6d02e` (frontend; depends P1–P3; subsumes
   `d90b3e5` residual UI). Matrix builder, ghost-row preview, unified table (pending/running/
   completed), set grouping + status strips, focused-run detail sidebar, bulk-edit verbs
   (Keep/Set/Distribute/Cross), launch wiring through the plan/run machinery (P2-principle:
   one launch path).
5. **P5 — SelectionContext + provenance mode** — `82a15e5` (frontend+contract; depends P1). Single
   selection store replacing `runStore` duality; hover/click bidirectional sync with pin;
   read-only frozen-snapshot projection of the top pane with diff badges, Promote-to-draft,
   Restage-from-here. Publishes the seam consumed by `946c859`/`6a094c9`.
6. **P6 — Selections as queries** — `067dc59` (B4; depends P1). SelectionSpec explicit/query/frozen;
   chip-based predicate authoring; `top-k by metric per group` primitive; refresh-matches diff +
   reprocess tri-state (missing / missing+failed / all / stale).
7. **P7 — Eval staging & execution wiring** — `717e8fb` (B6; depends P1, P6; parts of `96ac771`'s
   execution-target scope). Eval matrix = run selection × condition axes × checkpoint policy
   (`CheckpointSelectionManifest`); cache-hit preview via content-addressed eval ids; enable
   "Run selected".
8. **P8 — Bundle binding UI + dry-run surface** — `82a09d8` (B5; depends P6). Bundle cards with predicate
   binding; dry-run endpoint (run_condition + depends_on_roles vs current inputs);
   skipped/not_applicable/missing chips with reason strings everywhere.
9. **P9 — Queue projection + execution targets** — `12e49a2` (B7; depends P1). Cross-stage pending queue,
   target assignment, reorder/pause/cancel, explicit supersession events, billable spec-lock
   confirmation gate; wire RunPod execution backend into web orchestration.
10. **P10 — Lineage projection** — `0ede698` (depends P1, P5). Manifest DAG view over `ParentRef`s;
    click-through to stage tabs; staleness/skip rendering on nodes.
11. **P11 — Staleness semantics** — `4b52e88` (B8; depends P1). Spec-hash divergence detection
    (draft-changed, upstream-superseded), `superseded_by` links, stale filters feeding the
    reprocess tri-state. Never auto-invalidates.
12. **P12 — Compare mode + ingest UI** — `76785fb` (smaller; depends P1). N-selection param/metric diff with
    identical-fields collapse; manifest-packet ingest surface making the `import` stage real
    (rlrmp-trained runs into Studio).

Reasonable first wave: P1 + P2 + P3 (substrate), with P4 immediately behind — that wave alone
delivers the user-visible acceptance slice already written on `96ac771`.

## Coordination notes

- DONE 2026-07-06: decision-log comment on `93f79f2`; children-filed comment on `4007bf2`;
  decomposition note on `96ac771`; pointer on `0c46f69`; subsumption note on `d90b3e5`;
  related links created (P1↔`e33f487`, P2↔`96ac771`/`bd5aefb`, P3↔`bd5aefb`,
  P4↔`d90b3e5`/`96ac771`, P5↔`946c859`/`6a094c9`, P7↔`96ac771`, P8↔`0c46f69`,
  P9↔`96ac771`); `efa2093` linked as child of `4007bf2`.
- §10 questions resolved 2026-07-06 (see design doc §10): pending manifests confirmed;
  TrainingPanel organs distribute (panel is orphaned code); frozen-snapshot query selections
  with manual refresh; replicates within-run with per-ValueSpec scope; seed sweeps unified as
  scope-`sweep` on the master PRNG key; "matrix"/"pending" naming. Remaining design residue
  for the P1 spec: the pending-manifest GC story.
- User calls left open in ledger comments: whether `96ac771` closes when the first wave lands
  or folds into `efa2093`; `d90b3e5` closure when `3a6d02e` lands.
- Durable-schema changes (P1 ids, P2 set axes, P3 ValueSpec v2, P6 SelectionSpec) each carry
  the repo's migration-rule + focused-test requirement; the specs must name old-version
  accept/migrate/reject behavior explicitly.
