# Run-matrix harness

Feedbax uses the same vocabulary for downstream variation as it does for training:
a **base** is the shared request, a **row** is one named condition, a **delta** is an
ordered typed override, and a **source** identifies where a derived value came from.
Rows resolve their deltas first and then evaluate their per-row derivations. Output and
spec paths use `row_id` by default; callers may explicitly override a path where needed.

## Schemas

- Evaluation matrices use `feedbax.spec.evaluation_run_matrix.v3`. Version 2 added
  authenticated staged-parent bindings. Version 3 combines those bindings with either
  explicit base-plus-row authoring or a content-pinned JSON base plus ordered axes of
  named delta sets. The registry migrates v1 through v2 to v3 and v2 directly to v3;
  v0 and unknown versions remain rejected.
- Evaluation matrix deltas use `feedbax.spec.evaluation_run_matrix_delta.v1`. Such a
  document pins one whole parent document by canonical hash — either a direct evaluation
  matrix or another delta spec — and applies ordered composition layers to it. Parent
  resolution is repository-root confined, hash verified, cycle safe, and enforces
  ancestor-write acknowledgements; the flattened terminal document is validated as an
  ordinary `feedbax.spec.evaluation_run_matrix.v3` matrix and executes unchanged.
  Composition provenance is recorded as
  `feedbax.manifest.evaluation_matrix_composition_provenance.v1`.
- Analysis bundles use `feedbax.spec.analysis_bundle.v3`. The registry mechanically
  migrates v2 stages to the shared-base/per-stage-patch representation and rejects
  unsupported versions.

These identities are registered in `feedbax.contracts.migrations`; consumers should
route durable payloads through that registry rather than validating an arbitrary model
directly.

Axis products compile in authored axis/value order to the same explicit rows consumed
by the harness. Generated row IDs join axis/value IDs, and collisions, repeated delta
paths, invalid JSON values, missing bases, path escapes, and hash mismatches fail closed.
Each axis-authored row manifest embeds `feedbax.manifest.evaluation_axis_expansion_provenance.v1`
under `matrix_harness.axis_expansion`, including the authored matrix hash, pinned-base
authority, ordered coordinates, canonical row order, and canonical payload hashes.
Explicit-row manifests do not gain this metadata.

## Harness responsibilities

`MatrixMaterializerHarness` owns the condition loop, row expansion boundary, row-id
output roots, manifest-path collection, content-addressed custody, regeneration specs,
and a standard Markdown note. Evaluation-matrix execution is the first adapter. The
structural diff APIs compare either two resolved rows or regenerated and archived
payloads, returning deterministic JSON-style paths.

Plugins are loaded before matrix validation and execution. Figure, facet, and
computation recipe keys remain fail-closed: an unknown key raises an error that lists
the registered choices.

## Flat-spec escape hatch

A pipeline may bypass row expansion only by supplying a non-empty reason. The harness
records that reason in regeneration metadata and the visible Markdown note. This makes
exceptional flat execution auditable rather than a silent fallback.
