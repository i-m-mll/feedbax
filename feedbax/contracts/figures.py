"""Declarative figure contracts.

These models describe figure provenance and construction inputs. Plotly JSON is
the rendered product; these contracts are the durable source of truth.

Additive changelog, 2026-07-24: ``FigureSpec`` accepts optional
``trace_families``. A family declares one trace plus an ordered index set and
expands to the traces an equivalent hand-enumerated spec declares. The field is
an optional None-default, so canonical JSON and figure-manifest identity bytes
for pre-existing per-trace specs are unchanged.

Additive changelog, 2026-07-24: ``FigureSpec`` accepts an optional
``colorbar``, the key that makes a figure's color mapping readable. It is
declared either standalone or bound to a declared trace family, and is also an
optional None-default, so pre-existing identity bytes are again unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from feedbax.contracts.expressions import Expr, ValueExpr
from feedbax.contracts.manifest import ArtifactRef, ParentRef, SpecPayload, StrictModel
from feedbax.contracts.selection import ManifestPredicate


FIGURE_SPEC_SCHEMA_ID = "feedbax.spec.figure"
FIGURE_SPEC_SCHEMA_VERSION = "feedbax.spec.figure.v2"
FIGURE_INPUT_AUTHORITY_SCHEMA_ID = "feedbax.spec.figure_input_authority"
FIGURE_INPUT_AUTHORITY_SCHEMA_VERSION = "feedbax.spec.figure_input_authority.v1"
FIGURE_TEMPLATE_SCHEMA_ID = "feedbax.spec.figure_template"
FIGURE_TEMPLATE_SCHEMA_VERSION = "feedbax.spec.figure_template.v1"
FIGURE_PIECE_SCHEMA_ID = "feedbax.spec.figure_piece"
FIGURE_PIECE_SCHEMA_VERSION = "feedbax.spec.figure_piece.v1"
FIGURE_TRACE_FAMILY_SCHEMA_ID = "feedbax.spec.figure_trace_family"
FIGURE_TRACE_FAMILY_SCHEMA_VERSION = "feedbax.spec.figure_trace_family.v1"
FIGURE_COLORBAR_SCHEMA_ID = "feedbax.spec.figure_colorbar"
FIGURE_COLORBAR_SCHEMA_VERSION = "feedbax.spec.figure_colorbar.v1"

#: The only substitution token the figure contract understands. Trace families
#: are indexed substitution, deliberately not a templating or expression
#: language: no formatting, no arithmetic, no conditionals.
INDEX_TOKEN = "{index}"

SlotMultiplicity = Literal["one", "per_facet", "many"]
FacetTarget = Literal["figures", "panels"]
ColorscaleSpec: TypeAlias = str | list[str] | list[tuple[float, str]]


class AxisLabels(StrictModel):
    """Per-panel axis labels."""

    x: str | None = None
    y: str | None = None


class PanelAxis(StrictModel):
    """Declarative settings for one panel axis."""

    type: Literal["linear", "log"] | None = None
    range: tuple[float, float] | None = None


class TraceBinding(StrictModel):
    """One logical trace group in a declarative figure."""

    name: str
    constructor: str
    data: dict[str, ValueExpr | Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    piece: str | None = None
    include_when: Expr | None = None
    required: bool = False
    panel: str | None = None


def _name_collisions(names: Sequence[str]) -> list[str]:
    """Return every name that appears more than once, in stable order."""
    return sorted({name for name, count in Counter(names).items() if count > 1})


def map_family_strings(value: Any, transform: Callable[[str], str]) -> Any:
    """Apply ``transform`` to every string inside a trace-family declaration.

    The traversal is the single source of truth for what indexed substitution
    reaches: string values, model fields that were explicitly set, and entries
    of mappings and sequences. Mapping keys are constructor argument names and
    panel identifiers, so they never participate; non-string leaves such as
    numeric literals and array payloads pass through untouched.
    """
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, BaseModel):
        return type(value).model_validate(
            {
                name: map_family_strings(getattr(value, name), transform)
                for name in value.model_fields_set
            }
        )
    if isinstance(value, Mapping):
        return {key: map_family_strings(item, transform) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(map_family_strings(item, transform) for item in value)
    if isinstance(value, list):
        return [map_family_strings(item, transform) for item in value]
    return value


def validate_index_text(text: str) -> str:
    """Reject any brace syntax other than the exact index token."""
    residue = text.replace(INDEX_TOKEN, "")
    if "{" in residue or "}" in residue:
        raise ValueError(
            f"trace family strings support only the {INDEX_TOKEN} token and are not a "
            f"templating language; got {text!r}"
        )
    return text


def substitute_index(value: Any, index: int | str) -> Any:
    """Substitute one index value for every index token in a declaration."""

    def _substitute(text: str) -> str:
        return validate_index_text(text).replace(INDEX_TOKEN, str(index))

    return map_family_strings(value, _substitute)


class TraceFamilyRange(StrictModel):
    """Half-open integer index range, enumerated exactly like ``range``."""

    start: int = 0
    stop: int
    step: int = 1

    @model_validator(mode="after")
    def _validate_range(self) -> "TraceFamilyRange":
        if self.step == 0:
            raise ValueError("TraceFamilyRange step must be nonzero")
        if not self.values():
            raise ValueError(
                f"TraceFamilyRange start={self.start}, stop={self.stop}, step={self.step} "
                "enumerates no indices"
            )
        return self

    def values(self) -> tuple[int, ...]:
        """Return the ordered indices this range enumerates."""
        return tuple(range(self.start, self.stop, self.step))


class TraceFamilyIndex(StrictModel):
    """The ordered index set one trace family expands over."""

    values: list[int | str] | None = None
    range: TraceFamilyRange | None = None

    @model_validator(mode="after")
    def _validate_index_set(self) -> "TraceFamilyIndex":
        if (self.values is None) == (self.range is None):
            raise ValueError("TraceFamilyIndex requires exactly one of values or range")
        resolved = self.resolve()
        if not resolved:
            raise ValueError("TraceFamilyIndex must enumerate at least one index")
        # Substitution stringifies the index, so 1 and "1" are the same index.
        collisions = _name_collisions([str(index) for index in resolved])
        if collisions:
            raise ValueError(f"TraceFamilyIndex indices collide after substitution: {collisions}")
        return self

    def resolve(self) -> tuple[int | str, ...]:
        """Return the ordered index values, deterministically."""
        if self.range is not None:
            return self.range.values()
        return tuple(self.values or ())


class TraceFamily(StrictModel):
    """One trace declaration expanded deterministically over an index set.

    Every string in ``trace`` may carry the ``{index}`` token: data paths, the
    trace name, the panel, and param values. Expansion substitutes the index
    into those strings, in index order, and produces exactly the bindings an
    equivalent hand-enumerated spec declares.

    ``colorscale`` gives the family a deterministic color position per index,
    sampled once for the whole family. It is the declaration a colorbar key
    composes with: the colorbar reads the family's expanded index/color pairs
    rather than re-deriving colors of its own.
    """

    schema_id: str = FIGURE_TRACE_FAMILY_SCHEMA_ID
    schema_version: str = FIGURE_TRACE_FAMILY_SCHEMA_VERSION
    name: str
    index: TraceFamilyIndex
    trace: TraceBinding
    colorscale: ColorscaleSpec | None = None

    @model_validator(mode="after")
    def _validate_family(self) -> "TraceFamily":
        if self.schema_id != FIGURE_TRACE_FAMILY_SCHEMA_ID:
            raise ValueError(f"unsupported TraceFamily schema_id: {self.schema_id!r}")
        if self.schema_version != FIGURE_TRACE_FAMILY_SCHEMA_VERSION:
            raise ValueError(f"unsupported TraceFamily schema_version: {self.schema_version!r}")

        indexed = False

        def _scan(text: str) -> str:
            nonlocal indexed
            indexed = indexed or INDEX_TOKEN in text
            return validate_index_text(text)

        map_family_strings(self.trace, _scan)
        if not indexed:
            raise ValueError(
                f"TraceFamily {self.name!r} declares no {INDEX_TOKEN} token, so its members "
                "would not vary with the index"
            )
        if self.colorscale is not None and "color" in self.trace.params:
            raise ValueError(
                f"TraceFamily {self.name!r} declares both a colorscale and an explicit "
                "trace params color"
            )
        collisions = _name_collisions(self.expanded_trace_names())
        if collisions:
            raise ValueError(
                f"TraceFamily {self.name!r} expands to colliding trace names: {collisions}"
            )
        return self

    def expanded_trace_names(self) -> tuple[str, ...]:
        """Return the member trace names this family expands to, in index order."""
        return tuple(
            substitute_index(self.trace.name, index) for index in self.index.resolve()
        )


class FigureColorbar(StrictModel):
    """The declared key for a figure's color mapping.

    A colorbar is declared in one of two forms. The standalone form names an
    explicit ``colorscale`` and the ``range`` of values it spans. The bound
    form names a declared trace ``family`` instead: the family's expanded
    ``(index, color)`` pairs already are its color key, so the bound form is
    resolved by reading them rather than by re-sampling the colorscale, and the
    key cannot disagree with the traces it describes.

    With a family, ``range`` is optional and relabels the family's index domain
    onto the values those indices stand for, mapping the smallest index to
    ``range[0]`` and the largest to ``range[1]`` — a fan of knot traces indexed
    ``0..59`` can therefore be keyed by the conditioning level it sweeps.
    Omitted, the index domain is its own value domain.
    """

    schema_id: str = FIGURE_COLORBAR_SCHEMA_ID
    schema_version: str = FIGURE_COLORBAR_SCHEMA_VERSION
    title: str | None = None
    colorscale: ColorscaleSpec | None = None
    range: tuple[float, float] | None = None
    family: str | None = None

    @model_validator(mode="after")
    def _validate_colorbar(self) -> "FigureColorbar":
        if self.schema_id != FIGURE_COLORBAR_SCHEMA_ID:
            raise ValueError(f"unsupported FigureColorbar schema_id: {self.schema_id!r}")
        if self.schema_version != FIGURE_COLORBAR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigureColorbar schema_version: {self.schema_version!r}"
            )
        if self.family is None:
            if self.colorscale is None or self.range is None:
                raise ValueError(
                    "FigureColorbar requires either a trace family or both an explicit "
                    "colorscale and range"
                )
        elif self.colorscale is not None:
            raise ValueError(
                f"FigureColorbar bound to trace family {self.family!r} must not declare its "
                "own colorscale; the family's assigned colors are the key"
            )
        if self.range is not None and self.range[0] >= self.range[1]:
            raise ValueError(f"FigureColorbar range must increase; got {self.range}")
        return self


class PanelSpec(StrictModel):
    """One subplot cell-group."""

    name: str
    title: str | ValueExpr | None = None
    axes_labels: AxisLabels | None = None
    x_axis: PanelAxis | None = None
    y_axis: PanelAxis | None = None
    row: int | None = None
    col: int | None = None


class FigureArtifactPayload(StrictModel):
    """One authority-bound artifact decoded before figure execution effects."""

    name: str
    authority: Literal["artifact_provider"] = "artifact_provider"
    manifest_role: str
    artifact_role: str
    artifact_provider: str
    media_type: str = "application/json"
    manifest_status: Literal["completed"] = "completed"
    payload_schema_id: str | None = None
    payload_schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_provider(self) -> "FigureArtifactPayload":
        if not self.artifact_provider:
            raise ValueError("FigureArtifactPayload artifact_provider must be nonempty")
        return self


class FigureInputAuthority(StrictModel):
    """Portable authority requirements for one exact figure parent."""

    schema_id: str = FIGURE_INPUT_AUTHORITY_SCHEMA_ID
    schema_version: str = FIGURE_INPUT_AUTHORITY_SCHEMA_VERSION
    parent: ParentRef
    artifact_payloads: list[FigureArtifactPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_identity(self) -> "FigureInputAuthority":
        if self.schema_id != FIGURE_INPUT_AUTHORITY_SCHEMA_ID:
            raise ValueError(f"unsupported FigureInputAuthority schema_id: {self.schema_id!r}")
        if self.schema_version != FIGURE_INPUT_AUTHORITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported FigureInputAuthority schema_version: "
                f"{self.schema_version!r}"
            )
        names = [payload.name for payload in self.artifact_payloads]
        if len(names) != len(set(names)):
            raise ValueError("FigureInputAuthority artifact payload names must be unique")
        return self


class SlotSpec(StrictModel):
    """Declared binding hole for a reusable figure template."""

    name: str
    constructor: str
    required: bool = True
    params_defaults: dict[str, Any] = Field(default_factory=dict)
    multiplicity: SlotMultiplicity = "one"


class FigureTemplate(StrictModel):
    """Registered, data-free figure shape."""

    schema_id: str = FIGURE_TEMPLATE_SCHEMA_ID
    schema_version: str = FIGURE_TEMPLATE_SCHEMA_VERSION
    name: str
    description: str
    assembler: str
    assembler_params: dict[str, Any] = Field(default_factory=dict)
    slots: list[SlotSpec] = Field(default_factory=list)
    default_pieces: list[str] = Field(default_factory=list)
    facet_by: list[str] = Field(default_factory=list)
    facet_target: FacetTarget = "figures"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "FigureTemplate":
        if self.schema_id != FIGURE_TEMPLATE_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigureTemplate schema_id: {self.schema_id!r}, "
                f"expected {FIGURE_TEMPLATE_SCHEMA_ID!r}"
            )
        if self.schema_version != FIGURE_TEMPLATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigureTemplate schema_version: {self.schema_version!r}, "
                f"expected {FIGURE_TEMPLATE_SCHEMA_VERSION!r}"
            )
        if len(set(self.facet_by)) != len(self.facet_by):
            raise ValueError("FigureTemplate facet_by entries must be unique")
        if any(slot.multiplicity == "per_facet" for slot in self.slots) and not self.facet_by:
            raise ValueError(
                "FigureTemplate per_facet slots require at least one facet_by dimension"
            )
        return self


class FigurePiece(StrictModel):
    """Registered reusable, precomputed trace ingredient."""

    schema_id: str = FIGURE_PIECE_SCHEMA_ID
    schema_version: str = FIGURE_PIECE_SCHEMA_VERSION
    name: str
    description: str
    artifact_ref: ArtifactRef | None = None
    manifest_predicate: ManifestPredicate | None = None
    generator_spec: SpecPayload | None = None
    data_path: str | None = None
    label: str
    constructor: str
    style: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source_and_schema(self) -> "FigurePiece":
        if self.schema_id != FIGURE_PIECE_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigurePiece schema_id: {self.schema_id!r}, "
                f"expected {FIGURE_PIECE_SCHEMA_ID!r}"
            )
        if self.schema_version != FIGURE_PIECE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigurePiece schema_version: {self.schema_version!r}, "
                f"expected {FIGURE_PIECE_SCHEMA_VERSION!r}"
            )
        sources = [
            self.artifact_ref is not None,
            self.manifest_predicate is not None,
            self.generator_spec is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "FigurePiece requires exactly one source: artifact_ref, "
                "manifest_predicate, or generator_spec"
            )
        return self


class FigureSpec(StrictModel):
    """Executable declarative figure document."""

    schema_id: str = FIGURE_SPEC_SCHEMA_ID
    schema_version: str = FIGURE_SPEC_SCHEMA_VERSION
    name: str
    template: str | None = None
    assembler: str | None = None
    assembler_params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[ParentRef] = Field(default_factory=list)
    input_authorities: list[FigureInputAuthority] = Field(default_factory=list)
    slot_bindings: dict[str, TraceBinding | list[TraceBinding]] = Field(default_factory=dict)
    traces: list[TraceBinding] = Field(default_factory=list)
    # None-default so per-trace specs authored before trace families keep their
    # exact canonical JSON, spec payload, and figure-manifest identity bytes.
    trace_families: list[TraceFamily] | None = None
    colorbar: FigureColorbar | None = None
    panels: list[PanelSpec] = Field(default_factory=list)
    pieces: list[str] = Field(default_factory=list)
    exclude_pieces: list[str] = Field(default_factory=list)
    facet_bindings: dict[str, ValueExpr] = Field(default_factory=dict)
    figure_routing: dict[str, Any] = Field(default_factory=dict)
    run_condition: Expr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_and_shape(self) -> "FigureSpec":
        if self.schema_id != FIGURE_SPEC_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigureSpec schema_id: {self.schema_id!r}, "
                f"expected {FIGURE_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != FIGURE_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigureSpec schema_version: {self.schema_version!r}, "
                f"expected {FIGURE_SPEC_SCHEMA_VERSION!r}"
            )
        if self.template is None and self.assembler is None:
            raise ValueError("FigureSpec without template requires assembler")
        authority_parents = [authority.parent for authority in self.input_authorities]
        if len(authority_parents) != len({parent.model_dump_json() for parent in authority_parents}):
            raise ValueError("FigureSpec input_authorities contain a duplicate exact ParentRef")
        unknown = [parent for parent in authority_parents if parent not in self.inputs]
        if unknown:
            raise ValueError("FigureSpec input authority parent must exactly match a declared input")
        self._validate_trace_families()
        self._validate_colorbar_binding()
        return self

    def _validate_trace_families(self) -> None:
        families = self.trace_families or []
        if not families:
            return
        family_collisions = _name_collisions([family.name for family in families])
        if family_collisions:
            raise ValueError(f"FigureSpec trace_families names collide: {family_collisions}")
        expanded = [name for family in families for name in family.expanded_trace_names()]
        declared = {trace.name for trace in self.traces}
        collisions = sorted(
            set(_name_collisions(expanded)) | (set(expanded) & declared)
        )
        if collisions:
            raise ValueError(
                f"FigureSpec trace family expansion collides with other trace names: {collisions}"
            )

    def _validate_colorbar_binding(self) -> None:
        colorbar = self.colorbar
        if colorbar is None or colorbar.family is None:
            return
        families = {family.name: family for family in (self.trace_families or [])}
        family = families.get(colorbar.family)
        if family is None:
            raise ValueError(
                f"FigureSpec colorbar names undeclared trace family {colorbar.family!r}"
            )
        if family.colorscale is None:
            raise ValueError(
                f"FigureSpec colorbar trace family {family.name!r} declares no colorscale, "
                "so it assigns no colors for the key to read"
            )
        indices = family.index.resolve()
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(
                f"FigureSpec colorbar trace family {family.name!r} has non-numeric indices; "
                "a colorbar keys a numeric index domain"
            )
        if len(indices) < 2:
            raise ValueError(
                f"FigureSpec colorbar trace family {family.name!r} enumerates one index, "
                "which spans no range for a colorbar to key"
            )
