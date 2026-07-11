# Training-run resolved-semantics snapshot v1

Schema identity: `feedbax.spec.training_run_resolved_semantics`, version
`feedbax.spec.training_run_resolved_semantics.v1`.

A snapshot is a JSON object with `schema_id`, `schema_version`, `root_hash`, and
`nodes`. `nodes` maps lowercase SHA-256 hashes to nodes. A node is exactly one of:

- `{"type":"scalar","value":<JSON scalar>}`;
- `{"type":"array","children":[<child hash>, ...]}`; or
- `{"type":"object","children":{"key":<child hash>, ...}}`.

Every node hash is SHA-256 over the node encoded with Feedbax's training-spec
canonical JSON rule: UTF-8, sorted string object keys, compact separators,
exact integers, and finite floats encoded with Python's shortest
correctly-round-tripping representation. Negative zero normalizes to positive
zero. Non-string keys, non-finite floats, and unsupported values are rejected.
The root hash names the complete decoded JSON value. Object and array
nodes always reference children by hash, so identical subtrees occur once in the
table. The decoder must reject missing nodes, cycles, and unknown node types.

`feedbax.contracts.resolved_snapshot_decoder` is deliberately pure standard
library code and is the normative small decoder implementation.

Custody writers require callers to provide an untracked or gitignored custody
root. They verify existing digest-path bytes before reusing an object.
