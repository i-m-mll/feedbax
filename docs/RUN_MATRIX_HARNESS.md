# Run-matrix harness

Feedbax uses the same vocabulary for downstream variation as it does for training:
a **base** is the shared request, a **row** is one named condition, a **delta** is an
ordered typed override, and a **source** identifies where a derived value came from.
Rows resolve their deltas first and then evaluate their per-row derivations. Output and
spec paths use `row_id` by default; callers may explicitly override a path where needed.

## Schemas

- Evaluation matrices use `feedbax.spec.evaluation_run_matrix.v1`. It is a new schema
  family and rejects unknown IDs or versions.
- Analysis bundles use `feedbax.spec.analysis_bundle.v3`. The registry mechanically
  migrates v2 stages to the shared-base/per-stage-patch representation and rejects
  unsupported versions.

These identities are registered in `feedbax.contracts.migrations`; consumers should
route durable payloads through that registry rather than validating an arbitrary model
directly.

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
