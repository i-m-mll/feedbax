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

## How the figure layers fit together

Each layer has one job. Keeping those jobs separate prevents a house style from
silently changing scientific meaning, while still allowing figures to reuse
almost everything:

| Layer | Owns | Does not own |
| --- | --- | --- |
| Analysis output | Values and their scientific meaning | Plot layout or styling |
| Figure composition | Pinned reusable figure structure and acknowledged overrides | Arbitrary code executed after rendering |
| `FigureSpec` | Inputs, panels, traces, axis meaning, and explicit exceptions | Project discovery or process-global defaults |
| `FigureTemplate` | Reusable figure-family slots and assembler defaults | Scientific transformations of input values |
| Plotly template | Fonts, colors, title spacing, automargins, and other appearance | Which axes are scientifically comparable |
| Grid assembler | Panel geometry and row-label or legend gutters Plotly cannot infer | A second styling system |

An analysis should compute and retain scientific results; an authored figure
root should describe how those results become a durable figure. Native Plotly
templates remain the appearance authority. Feedbax contracts add scientific
meaning, reproducible composition, validation, identity, and provenance around
that rendering path.

Scientific meaning follows the same authenticated path as the plotted value:

1. An analysis records an exact `AnalysisOutputDescriptor` for each described
   product field. Its `ValueSchema` owns `quantity`, `units`, and optional
   `frame`; those fields are scientific content in the
   `AnalysisDataProduct` identity, not arbitrary metadata.
2. A semantics-preserving operation uses
   `preserve_analysis_output_descriptor` to name one exact source product and
   path and one exact output path. A meaning-changing operation declares its
   output descriptor. Feedbax does not inspect Python or infer unit algebra.
3. `FigureDataProductArtifactPayload` selects the exact authenticated product.
   A direct `ValueQuery` into that selected payload carries the matching
   descriptor with the value. Filtering, coercion, defaults, and other
   meaning-changing expressions do not inherit it implicitly.
4. Each registered trace constructor declares which data arguments contribute
   to each coordinate. Compatible contributions derive the panel's axis class;
   quantity, unit, and frame must all agree. The existing semantic range
   reducer then groups only equal classes.

An explicit `FigureSpec` axis quantity, unit, or frame is an assertion when a
descriptor exists, and a disagreement fails before the constructor renders.
For descriptor-free retained products or literal values, an explicit axis is
the recorded source of otherwise missing meaning. Under a semantic range
policy, any data-bearing axis that remains unknown fails clearly. Omitting
`output_descriptors` keeps existing product identity bytes unchanged and means
unknown; no field-name or display-label inference occurs.

One other boundary remains important in the current API:

- A reusable template or composition affects only figures that explicitly
  reference it. Feedbax does not currently discover a project-wide policy from
  the working directory or apply a process-global Plotly setting. Applying one
  policy to every governed figure therefore requires each figure root to share
  that pinned authority.

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

## Semantic ranges and grid layout

An axis may declare an exact `quantity` and `unit` pair. Both are opaque
scientific identities, not display labels: `{"quantity": "force", "unit":
"N"}` matches only that exact pair, and an axis for torque in `N` is a different
class. A figure's optional `semantic_range_policy` selects either `"all"` or a
list of at most 64 exact pairs and supplies one finite nonnegative
`padding_fraction`. During grid assembly, every selected class with at least two
eligible Cartesian axes is lowered to the existing `DataDerivedAxisRange`
reducer. An explicit axis `range` or `derived_range` is an exception and always
wins; a selected class absent from one figure is a no-op. Generated groups are
separate for x and y and are allocated away from every authored group name. No
units are converted.

The semantic class describes the *plotted coordinate*, not the intervention or
source experiment. For example, a response to a force perturbation may still
plot position error in metres. Quantity and unit must therefore agree with the
values bound to that axis.

Grid figures accept serialized Plotly templates through their existing
`assembler_params.template` input. Plotly validates the template, and the
ordinary appearance values remain inheritable and pinnable through the existing
`FigureTemplate` composition path:

```json
{
  "width": 720,
  "margin": {"l": 80, "r": 140},
  "template": {
    "layout": {
      "xaxis": {"automargin": true, "title": {"standoff": 14}},
      "yaxis": {"automargin": true, "title": {"standoff": 14}},
      "legend": {"xanchor": "left"}
    }
  },
  "row_label_gutter": 28,
  "legend_gutter": 32
}
```

`PanelSpec.row_label` uses Plotly's native row-title annotation, separately from
the panel title. `row_label_gutter` reserves a pixel-wide strip outside the plot
area and centres row labels in it; `legend_gutter` places the legend that many
pixels beyond the reserved strip. Legend placement is calculated from the
actual plot width after left and right margins. An explicit `legend_x` wins.
Axis automargin, title standoff, fonts, colors, and other ordinary appearance
remain native Plotly-template values and are never injected afterward.

## Identity and migration rules

- `feedbax.spec.figure.v2` remains the unchanged ordinary resolved schema, with
  its existing validation, migration, and rejection policy.
- `feedbax.spec.semantic_axis_range_policy.v1` is the only supported version of
  its optional nested declaration; v0 and unknown versions reject.
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
