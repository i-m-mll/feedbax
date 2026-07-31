# Composable figure specifications

`FigureCompositionSpec` removes repeated figure structure without introducing a
new patch language. It content-pins one ordinary `FigureSpec` (or another
composition envelope) and applies the same ordered `MatrixCompositionDelta`
layers used by Feedbax analysis and training authoring. Any FigureSpec field can
be composed, including `panels` and `trace_families`.

Resolve before rendering or review:

```console
feedbax-figure resolve figures/velocity.figure-composition.json --repo-root .
```

The command prints only the fully resolved, ordinary current FigureSpec as
sorted JSON. Add `--with-lineage` to include the authored-envelope hash,
resolved semantic hash, root content pin, ordered layers, and per-path
attribution. The corresponding Python API is
`feedbax.analysis.figures.resolve_figure_spec`.

Execution uses this same resolver:

```console
feedbax-figure run figures/velocity.figure-composition.json --repo-root .
```

The figure manifest's `figure_spec` is the resolved ordinary FigureSpec. Its
regeneration specs retain the authored `FigureCompositionSpec` and
`FigureCompositionProvenance`. Provenance includes every verified whole source
document and selected payload in root-to-leaf order, so regeneration does not
depend on the original files remaining present. Lineage remains visible without
entering the resolved semantic identity.

## Compact runnable example

Run this from a repository root. It writes a shared base and one child envelope,
then resolves the child without rendering:

```python
import json
from pathlib import Path

from feedbax.analysis.figures import resolve_figure_spec
from feedbax.contracts.figures import (
    FigureCompositionDelta,
    FigureCompositionSpec,
    FigureSpec,
    PanelSpec,
)
from feedbax.contracts.manifest import OverridePatch, canonical_json_bytes, sha256_bytes
from feedbax.contracts.matrix_core import ContentPinnedJsonBase

root = Path("figure-example")
root.mkdir(exist_ok=True)
base = FigureSpec(
    name="shared",
    assembler="feedbax.grid_figure",
    panels=[PanelSpec(name="main", title="Shared title")],
)
base_payload = base.model_dump(mode="json", exclude_none=True)
(root / "base.json").write_text(
    json.dumps(base_payload, indent=2) + "\n", encoding="utf-8"
)

composed = FigureCompositionSpec(
    parent=ContentPinnedJsonBase(
        ref="base.json",
        sha256=sha256_bytes(canonical_json_bytes(base_payload)),
    ),
    deltas=[
        FigureCompositionDelta(
            layer_id="velocity",
            patches=[
                OverridePatch(op="replace", path="name", value="velocity"),
                OverridePatch(
                    op="replace", path="panels.0.title", value="Velocity profiles"
                ),
            ],
        )
    ],
)
resolved = resolve_figure_spec(composed, repo_root=root)
print(resolved.figure_spec.model_dump_json(indent=2, exclude_none=True))
print("authored:", resolved.authored_identity_sha256)
print("resolved:", resolved.resolved_identity_sha256)
```

Layers apply root-to-child and in declaration order. Replacing a path already
written by an ancestor requires listing the overlapping subtree or leaf in
`acknowledges_ancestor_paths`; prefix overlap is recognized in both directions.
Otherwise resolution fails. Layer attribution is qualified by its composition
envelope identity, so repeated readable layer IDs in nested envelopes remain
unambiguous. A delta may not
change any top-level or nested self-versioned schema identity without the
shared delta mechanism's explicit schema-boundary contract. Figure composition
does not silently repair an invalid nested schema.

An absent-only structural extension under the same root schema uses the
figure-specific `FigureCompositionDelta.same_schema_structural_additions`.
Each entry is a `SameSchemaStructuralAddition` naming the exact added dotted
path and the exact `schema_id`/`schema_version` carried there or assigned to
that typed path by the figure resolver. The declarations must match the
complete set of identities introduced by that layer, and every declaration
must lie under an `add` patch. Added panels declare the stable
`feedbax.spec.figure_panel.v1` identity in the delta; the resolver validates the
added value through `PanelSpec`, while the resolved panel remains byte-shape
compatible and carries no new fields. Replacements, missing or
invalid identities, sibling paths, undeclared nested identities, and root
schema drift fail closed.

## Supported authoring roots

Composition is supported by `resolve_figure_spec`, `coerce_figure_spec`,
`execute_figure_spec`, the `feedbax-figure resolve` and `run` commands, and
`AnalysisBundleSpec` v6 figure stages. Bundle execution resolves relative
content pins only beneath its explicit, trusted `repo_root`; omitting that root
for a composed stage fails closed. Ordered reports consume the resolved ordinary
FigureSpec embedded in a FigureManifest, so their validation boundary and
identity pins remain unchanged.

The Studio API deliberately rejects `FigureCompositionSpec` with the typed
`figure_composition_not_supported_in_studio` error. Studio accepts a resolved
ordinary FigureSpec v2 and never accepts a client-controlled filesystem root.

## Identity and migration rules

- `feedbax.spec.figure.v2` remains the unchanged ordinary resolved schema, with
  its existing validation, migration, and rejection policy.
- `feedbax.spec.figure_composition.v2` is authored lineage, not resolved figure
  semantics. Its v1 predecessor migrates without semantic change to v2, which
  scopes structural addition declarations to `FigureCompositionDelta`.
  Composition identity excludes the readable parent `ref`; the content hash is
  authoritative.
- The resolved identity is the canonical hash of the ordinary current
  FigureSpec only. Two envelopes that resolve to identical semantics therefore
  share a resolved identity even if their authored lineage differs.
- `feedbax.spec.figure_runtime_binding.v2` records distinct authored source or
  envelope identity and resolved FigureSpec identity. Its v1 migration preserves
  the old field's actual resolved-hash meaning and marks the unavailable authored
  identity explicitly rather than relabeling it.
- A root may use `SourceDocumentInheritance` to content-pin a shared object once
  and graft it into absent dotted targets. Canonical non-negative list indices
  use the same rules as `ContentPinnedJsonBase.payload_path`; collisions and
  malformed, negative, or out-of-range indices fail closed.
- Parent schema mismatches, content-pin failures, unknown templates, cycles,
  unacknowledged overrides, and nested schema-boundary violations fail closed.

::: feedbax.analysis.figures.resolve_figure_spec

::: feedbax.contracts.figures.FigureCompositionSpec
