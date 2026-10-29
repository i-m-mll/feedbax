# Authoring budgets

`feedbax.spec.authoring_budget.v2` bounds one authored envelope. Feedbax owns
measurement and the closed cap vocabulary; the project owns the per-layer cap
values. Every layer must state `max_bytes`, `max_scalar_bytes`, `max_ast_nodes`,
`max_depth`, `max_items`, `max_object_keys`, and `max_assertions`. The training
layer may additionally state the enforced `project_caps.max_rows` bound.

`max_bytes` measures the complete parsed document using the existing
`feedbax.contracts.authored_canonical.canonical_bytes` authority
(`canonical_json_v1`). Headers, authenticated references, assertions, and deltas
all count. Sorted keys and compact JSON separators make source key order,
indentation, key spacing, and the final newline irrelevant to this measurement.
The existing Unicode escaping of `canonical_json_v1` remains part of the byte
count. This change does not change any content-pin or execution-identity domain.

There is no line cap in v2. Syntax-node and depth caps bound structure, and
canonical bytes bound total serialized content. A line cap adds no independent
bound on authored intent. Scalar, array, object, assertion, and optional authored
row caps continue to apply. Duplicate keys, nonfinite constants, invalid UTF-8,
and the existing authored source-format violations still refuse before any
project model sees the document.

`feedbax init` emits v2 policies. Its evaluation defaults are 8,192 canonical
bytes and 512 syntax nodes, enough for one ordinary two-arm comparator expressed
as an existing evaluation delta with exact receipt references. The scalar,
depth, item, object-key, and assertion defaults remain unchanged. These defaults
apply to newly initialized projects; existing project policies are explicit
resources and are never rewritten by a compiler.

## Policy transition

The loader and structured-spec registry reject v1 with an actionable diagnostic;
there is no automatic migration that silently reinterprets a source-byte policy.
To author a v2 policy:

1. Remove `max_lines` from every layer section.
2. Review `max_bytes` using `canonical_json_v1` measurements of the project's
   authored corpus. Preserve other bounds unless separately justified.
3. Declare `schema_version: feedbax.spec.authoring_budget.v2` and update the
   project's policy identity and explanatory metadata as appropriate.

A v2 section retaining `max_lines` refuses as an unknown cap. Unsupported schema
versions also fail closed. Large generated documents remain compact; this budget
change permits readable small hand-authored specs and does not prescribe
pretty-printing machine output.

Co-Authored-By: Codex (GPT) <codex@openai.com>
