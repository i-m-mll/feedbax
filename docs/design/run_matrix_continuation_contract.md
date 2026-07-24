# Run-matrix continuation-authoring contract

This document names the contract for authoring a *continuation* training run
matrix: a new matrix that extends a previous, already-locked one (for example,
to add more training batches, widen a sweep, or attach a new row) without
copying its authored content by hand. It covers `TrainingRunMatrixSpec` and
its supporting types in `feedbax/contracts/run_matrix.py`; the recursive
intent-composition model those types share (`CompositionDelta` /
`flatten_composition`) is documented separately in
`docs/TRAINING_RUN_COMPOSITION.md`.

## The contract

1. **Compose/flatten.** A matrix `base` resolves to one payload document,
   either inline, or by reading and content-verifying an authored or
   resolved-output reference (`resolve_base_payload_with_attribution` in
   `feedbax/training/run_matrix.py`). If the referenced document is itself an
   authored matrix, its own base is resolved recursively first.
2. **Canonicalize, then hash/pin the base.** Ordered composition deltas
   (`TrainingRunMatrixSpec.deltas`, applied by `apply_composition_deltas`) are
   flattened onto the resolved document. The result is canonicalized and its
   content hash becomes the base's pinned identity
   (`AuthoredIntentMatrixBaseSpec.content_hash` /
   `ResolvedOutputMatrixBaseSpec.resolved_root_hash`). A continuation pins a
   **new** base hash for its (possibly patched) document; it does not reuse
   the ancestor's hash unless the document is byte-identical.
3. **Apply permitted row overrides.** Each `MatrixRow.overrides` is an ordered
   list of `OverridePatch` records applied to the pinned base by
   `apply_override_patches`. Overrides are JSON-Patch-like `add` / `replace` /
   `remove` operations addressed by dotted paths (`a.b.2.c`); numeric path
   segments index into lists.
4. **Row identity derives from the patched result.** A row's
   `authored_payload_hash` and downstream `planned_run_id` are computed from
   its own overridden payload (`_lower_authored_row` /
   `_planned_run_id` in `feedbax/training/run_matrix.py`), never from the
   base hash directly. **A row does not inherit the base hash unchanged** —
   even a row with an empty `overrides` list still hashes its own resolved
   payload, so identity always reflects what that row actually runs.

Composition deltas and row overrides both patch through the same
`OverridePatch` primitive and the same `_apply_patch` engine in
`feedbax/contracts/run_matrix.py`, so the following override semantics apply
uniformly whether you are patching a training-run payload for a row or
patching a serialized matrix-spec document itself while authoring a
continuation.

## Override-patch semantics

- `add` at an existing key/index is rejected (`already exists`); `add` at a
  path whose parent does not yet exist is rejected (`missing key/index`).
- `replace` and `remove` require the target key/index to already exist;
  otherwise they fail closed with a message naming the exact path.
- **List append.** `add` targeting a list accepts two append tokens, following
  JSON-Patch (RFC 6902 section 4.1): a numeric index exactly equal to the
  list's current length, or the literal `-` token. Both mean "insert after the
  last element." Any other index — in range or beyond it — is a positional
  target, not an append, and beyond-range indices are still rejected.
  Append works at any nesting depth, including a new entry appended to a
  `patches` list nested inside a `deltas` element.

Before this contract, `add` only accepted indices already in range, so
appending a new element to a list (for example, a new patch onto an existing
composition delta's `patches` array) was impossible without replacing the
whole list. That gap is what made continuations awkward to author: extending
one field of an existing delta forced re-authoring its entire `patches` list.

## Worked example

An ancestor matrix's `deltas`, as a plain serialized document:

```json
{
  "deltas": [
    {
      "layer_id": "warm_restart",
      "patches": [
        {"path": "training_config.n_batches", "op": "replace", "value": 150}
      ]
    }
  ]
}
```

Author the continuation by *appending* a new patch to the existing delta's
`patches` list, instead of replacing the whole array:

```json
{
  "path": "deltas.0.patches.1",
  "op": "add",
  "value": {"path": "training_config.batch_size", "op": "add", "value": 64}
}
```

Applying that one override patch via `apply_override_patches` yields a
distinct continuation document whose `warm_restart` delta now carries both
patches, leaving the ancestor document untouched. Canonicalizing and hashing
that document pins the continuation's new base identity; flattening its
deltas onto a base payload of
`{"training_config": {"learning_rate": 0.01, "n_batches": 100}}` produces
`{"training_config": {"learning_rate": 0.01, "n_batches": 150, "batch_size": 64}}`.

A row then applies its own override on top of that pinned base — for example,
replacing `training_config.learning_rate` with `0.02` — and its identity is
computed from that patched result, not from the base hash.

This example is executed, not merely prose: `tests/test_run_matrix_materialization.py::test_continuation_matrix_authoring_contract_worked_example`
runs the same steps end to end and asserts the same values. Keep the test and
this section in sync when either changes.

## Deferred: typed builder API

A typed builder API for authoring override patches and composition deltas
(so callers write attribute-style updates instead of dotted-path JSON-Patch
records) was considered while writing this contract. It is deliberately
deferred: `OverridePatch` and `MatrixCompositionDelta` are durable, portable,
content-hashed wire records, and the dotted-path/JSON-Patch representation is
what canonicalizes and hashes. A builder would be sugar over construction,
not a new authoring capability, and no experiment currently needs one. Add it
only when a real authoring pain point demonstrates the need, per the
feedbax factoring norm of extending existing landed surfaces over speculative
abstraction.
