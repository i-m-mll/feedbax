# Training-run intent composition

Training programs are authored as recursive composition nodes. Each node has exactly one typed
parent reference and an ordered list of local deltas; named lanes are reusable labels over those
nodes, while matrix rows remain terminal expansions. Authored parents are canonical-content
pinned. Resolved-output parents and checkpoint parents are immutable evidence and are never
treated as editable authored layers.

Execution dependencies are separate from intent deltas. Fork-from-checkpoint, continuation
reconciliation, lineage correction, stopped-row state, durable slot transforms, and task-identity
gates are discriminated records. They may constrain or explain execution, but cannot silently
rewrite authored semantics.

## Validation boundary

Only the flattened terminal intent is validated against the target training schema. Intermediate
lanes may be intentionally incomplete. Patch preconditions are nevertheless checked at every
layer, and `resolve_base_payload_with_attribution` returns the flattened payload together with the
last layer that wrote each path, so validation tooling can attribute a terminal error. A child
writing a path already written by an ancestor must list that path in its explicit acknowledgement
set.

A layer crosses a schema-family boundary only by declaring both `schema_id` and `schema_version`
on its delta and patching the payload's `schema_id` and `schema_version` to exactly those declared
values. Patching either identity field without a declaration fails closed; declaring a boundary
whose resulting flattened identity does not match also fails closed. Thus a declaration is an
enforced transition, not descriptive metadata.

## Identity and lineage

The authored-envelope hash covers ordered pinned parent identities, ordered local deltas,
selectors, seeds, declared sources, and execution-dependency declarations. The composed-intent
hash covers canonical flattened intent before execution. Therefore rebasing or squashing changes
the authored envelope while preserving the composed identity when effective intent is unchanged.
The execution hash remains a separate identity over resolved semantics and immutable inputs.

Archival lineage is an append-only, content-pinned DAG, not the nested authoring tree. Graft and
correction events retain the original edge and state whether the correction supersedes it for
interpretation or starts a new execution. A rematerialized result always receives a new execution
identity, and children pinned to an old parent hash never move implicitly.

The contract APIs provide flattening, path attribution, layered semantic diffs, equal-semantics
stack comparisons, and near-duplicate-lane detection. They are library APIs; command-line and
analysis/report presentation are deliberately outside this lane.
