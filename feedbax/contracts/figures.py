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

Additive changelog, 2026-07-24: ``TraceFamily`` accepts optional
``legend_index`` (the one member that carries the family's legend entry),
``values`` (the physical quantity each member stands for, which positions its
color and enables the ``{value}`` substitution token), and
``interleave_group`` (families that expand one member per index position
instead of as per-family blocks). All three are optional None-defaults, so a
family declared without them expands exactly as before and every pre-existing
spec keeps its canonical JSON and figure-manifest identity bytes.

Additive changelog, 2026-07-27: ``FigureSpec.input_authorities`` accepts the
versioned ``FigureInputRoleAuthority`` form. It selects exactly one already
declared input by role and resolves to that input's complete ``ParentRef``;
the existing exact-parent authority form and its canonical bytes are unchanged.

Additive changelog, 2026-07-27: ``FigureSpec.slot_families`` accepts optional
versioned ``FigureSlotFamily`` declarations. Each declaration binds one
template slot from an explicit row table and expands to ordinary
``TraceBinding`` values before template validation. The field is an optional
None-default, so pre-existing FigureSpec and FigureTemplate identity bytes and
schema versions are unchanged.

Additive changelog, 2026-07-27: figure-constructor data may carry the versioned
``PerturbationTiming`` contract. It declares bounded, nominal, or full-trial
applicability plus the authored sample schedule. Constructor data remains
unchanged when timing is absent.

Additive changelog, 2026-07-28: ``PanelSpec.equal_data_aspect`` accepts the
versioned ``EqualDataAspect`` declaration. It requests exact 1:1 linear data
units from a supporting panel assembler. The field is an optional None-default,
so existing panel and figure identity bytes are unchanged.

Additive changelog, 2026-07-28: ``FigureColorbar.placement`` accepts the
versioned ``ColorbarPanelPlacement`` declaration. It sizes and positions a
colorbar relative to one resolved grid panel. The field is an optional
None-default, so existing colorbar and figure identity bytes are unchanged.
"""

from __future__ import annotations

import math
import re
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
FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_ID = "feedbax.spec.figure_input_role_authority"
FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_VERSION = "feedbax.spec.figure_input_role_authority.v1"
FIGURE_TEMPLATE_SCHEMA_ID = "feedbax.spec.figure_template"
FIGURE_TEMPLATE_SCHEMA_VERSION = "feedbax.spec.figure_template.v1"
FIGURE_PIECE_SCHEMA_ID = "feedbax.spec.figure_piece"
FIGURE_PIECE_SCHEMA_VERSION = "feedbax.spec.figure_piece.v1"
FIGURE_TRACE_FAMILY_SCHEMA_ID = "feedbax.spec.figure_trace_family"
FIGURE_TRACE_FAMILY_SCHEMA_VERSION = "feedbax.spec.figure_trace_family.v1"
FIGURE_SLOT_FAMILY_SCHEMA_ID = "feedbax.spec.figure_slot_family"
FIGURE_SLOT_FAMILY_SCHEMA_VERSION = "feedbax.spec.figure_slot_family.v1"
FIGURE_COLORBAR_SCHEMA_ID = "feedbax.spec.figure_colorbar"
FIGURE_COLORBAR_SCHEMA_VERSION = "feedbax.spec.figure_colorbar.v1"
COLORBAR_PANEL_PLACEMENT_SCHEMA_ID = "feedbax.spec.colorbar_panel_placement"
COLORBAR_PANEL_PLACEMENT_SCHEMA_VERSION = "feedbax.spec.colorbar_panel_placement.v1"
PERTURBATION_TIMING_SCHEMA_ID = "feedbax.spec.perturbation_timing"
PERTURBATION_TIMING_SCHEMA_VERSION = "feedbax.spec.perturbation_timing.v1"
EQUAL_DATA_ASPECT_SCHEMA_ID = "feedbax.spec.equal_data_aspect"
EQUAL_DATA_ASPECT_SCHEMA_VERSION = "feedbax.spec.equal_data_aspect.v1"
FIGURE_RUNTIME_BINDING_SCHEMA_ID = "feedbax.spec.figure_runtime_binding"
FIGURE_RUNTIME_BINDING_SCHEMA_VERSION = "feedbax.spec.figure_runtime_binding.v1"
FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_ID = "feedbax.spec.figure_data_product_payload"
FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_VERSION = "feedbax.spec.figure_data_product_payload.v1"

#: The two substitution tokens the figure contract understands. Trace families
#: are indexed substitution, deliberately not a templating or expression
#: language: no arithmetic, no conditionals, no tokens beyond these. ``{index}``
#: substitutes the member's index verbatim. ``{value}`` substitutes the physical
#: value the member stands for and is available only when the family declares
#: ``values``; it accepts a trailing format spec (``{value:.3f}``) because a
#: float is unreadable without one, which is why an index — already exact —
#: takes none.
INDEX_TOKEN = "{index}"
VALUE_TOKEN = "{value}"

#: Single source of what substitution recognizes: the exact index token, or the
#: value token with an optional brace-free format spec.
SUBSTITUTION_PATTERN = re.compile(r"\{index\}|\{value(?::([^{}]*))?\}")

SlotMultiplicity = Literal["one", "per_facet", "many"]
FacetTarget = Literal["figures", "panels"]
ColorscaleSpec: TypeAlias = str | list[str] | list[tuple[float, str]]


class PerturbationTiming(StrictModel):
    """Authored perturbation applicability and discrete sample schedule.

    ``bounded`` timing names the first affected sample and the number of
    affected samples. ``nominal`` and ``full_trial`` are explicit no-marker
    cases and therefore forbid bounded interval fields. ``sample_count`` is
    always required so every consuming constructor can authenticate the
    schedule against the coordinates it actually plots.
    """

    schema_id: str = PERTURBATION_TIMING_SCHEMA_ID
    schema_version: str = PERTURBATION_TIMING_SCHEMA_VERSION
    applicability: Literal["bounded", "nominal", "full_trial"]
    sample_count: int = Field(gt=0)
    start_index: int | None = Field(default=None, ge=0)
    duration: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_schedule(self) -> "PerturbationTiming":
        if self.schema_id != PERTURBATION_TIMING_SCHEMA_ID:
            raise ValueError(f"unsupported PerturbationTiming schema_id: {self.schema_id!r}")
        if self.schema_version != PERTURBATION_TIMING_SCHEMA_VERSION:
            raise ValueError(
                "unsupported PerturbationTiming schema_version: "
                f"{self.schema_version!r}"
            )
        if self.applicability == "bounded":
            if self.start_index is None or self.duration is None:
                raise ValueError(
                    "bounded PerturbationTiming requires start_index and duration"
                )
            if self.start_index + self.duration > self.sample_count:
                raise ValueError(
                    "bounded PerturbationTiming exceeds sample_count: "
                    f"start_index {self.start_index} + duration {self.duration} "
                    f"> sample_count {self.sample_count}"
                )
        elif self.start_index is not None or self.duration is not None:
            raise ValueError(
                f"{self.applicability} PerturbationTiming forbids start_index and duration"
            )
        return self

    def affected_range(self) -> tuple[int, int] | None:
        """Return the half-open affected sample range, or no bounded range."""
        if self.applicability != "bounded":
            return None
        assert self.start_index is not None
        assert self.duration is not None
        return self.start_index, self.start_index + self.duration


class AxisLabels(StrictModel):
    """Per-panel axis labels."""

    x: str | None = None
    y: str | None = None


class PanelAxis(StrictModel):
    """Declarative settings for one panel axis."""

    type: Literal["linear", "log"] | None = None
    range: tuple[float, float] | None = None


class EqualDataAspect(StrictModel):
    """Request exact 1:1 linear data units for one panel."""

    schema_id: str = EQUAL_DATA_ASPECT_SCHEMA_ID
    schema_version: str = EQUAL_DATA_ASPECT_SCHEMA_VERSION
    ratio: Literal[1] = 1

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "EqualDataAspect":
        if self.schema_id != EQUAL_DATA_ASPECT_SCHEMA_ID:
            raise ValueError(
                f"unsupported EqualDataAspect schema_id: {self.schema_id!r}"
            )
        if self.schema_version != EQUAL_DATA_ASPECT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported EqualDataAspect schema_version: "
                f"{self.schema_version!r}"
            )
        return self


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

    @model_validator(mode="after")
    def _validate_perturbation_timing_identity(self) -> "TraceBinding":
        raw = self.data.get("perturbation_timing")
        if raw is None or isinstance(raw, PerturbationTiming):
            return self
        if not isinstance(raw, Mapping):
            raise ValueError("TraceBinding perturbation_timing data must be a mapping")
        if raw.get("schema_id") != PERTURBATION_TIMING_SCHEMA_ID:
            raise ValueError(
                "TraceBinding perturbation_timing requires explicit schema_id "
                f"{PERTURBATION_TIMING_SCHEMA_ID!r}"
            )
        if raw.get("schema_version") != PERTURBATION_TIMING_SCHEMA_VERSION:
            raise ValueError(
                "TraceBinding perturbation_timing requires explicit schema_version "
                f"{PERTURBATION_TIMING_SCHEMA_VERSION!r}"
            )
        return self


class FigureSlotFamilyRow(StrictModel):
    """One explicit row in a compact template-slot family.

    Every supported varying field is named directly. Nullable fields are still
    required in serialized rows, so an omitted label, color, marker, or panel
    cannot be mistaken for an authoring oversight; use ``null`` deliberately.
    ``data_paths`` maps constructor data argument names to existing figure
    value expressions and provides no interpolation or expression syntax of
    its own.
    """

    name: str
    panel: str | None
    data_paths: dict[str, ValueExpr]
    label: str | None
    color: str | None
    marker: str | None


class FigureSlotFamily(StrictModel):
    """A constrained row family that supplies one figure-template slot."""

    schema_id: str = FIGURE_SLOT_FAMILY_SCHEMA_ID
    schema_version: str = FIGURE_SLOT_FAMILY_SCHEMA_VERSION
    name: str
    slot: str
    rows: list[FigureSlotFamilyRow] = Field(min_length=1)
    constructor: str = ""
    data: dict[str, ValueExpr | Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    required: bool = False

    @model_validator(mode="after")
    def _validate_family(self) -> "FigureSlotFamily":
        if self.schema_id != FIGURE_SLOT_FAMILY_SCHEMA_ID:
            raise ValueError(f"unsupported FigureSlotFamily schema_id: {self.schema_id!r}")
        if self.schema_version != FIGURE_SLOT_FAMILY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FigureSlotFamily schema_version: {self.schema_version!r}"
            )
        if not self.name:
            raise ValueError("FigureSlotFamily name must be nonempty")
        if not self.slot:
            raise ValueError("FigureSlotFamily slot must be nonempty")
        collisions = _name_collisions([row.name for row in self.rows])
        if collisions:
            raise ValueError(
                f"FigureSlotFamily {self.name!r} rows have duplicate trace names: {collisions}"
            )
        for row in self.rows:
            overlap = sorted(set(self.data) & set(row.data_paths))
            if overlap:
                raise ValueError(
                    f"FigureSlotFamily {self.name!r} row {row.name!r} data paths collide "
                    f"with common data keys: {overlap}"
                )
            row_params = {
                key
                for key, value in (
                    ("label", row.label),
                    ("color", row.color),
                    ("marker_symbol", row.marker),
                )
                if value is not None
            }
            param_overlap = sorted(set(self.params) & row_params)
            if param_overlap:
                raise ValueError(
                    f"FigureSlotFamily {self.name!r} row {row.name!r} fields collide "
                    f"with common params: {param_overlap}"
                )
        return self

    def expand(self) -> tuple[TraceBinding, ...]:
        """Expand rows deterministically into ordinary concrete bindings."""
        bindings: list[TraceBinding] = []
        for row in self.rows:
            row_params = {
                key: value
                for key, value in (
                    ("label", row.label),
                    ("color", row.color),
                    ("marker_symbol", row.marker),
                )
                if value is not None
            }
            bindings.append(
                TraceBinding(
                    name=row.name,
                    constructor=self.constructor,
                    data={**self.data, **row.data_paths},
                    params={**self.params, **row_params},
                    required=self.required,
                    panel=row.panel,
                )
            )
        return tuple(bindings)


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


def validate_index_text(text: str, *, values_declared: bool = False) -> str:
    """Reject any brace syntax other than the supported substitution tokens.

    ``values_declared`` says whether the family owning ``text`` declares
    per-member values. It defaults to false, so a ``{value}`` token is rejected
    unless the family that would give it meaning declares one value per member.
    """
    residue = SUBSTITUTION_PATTERN.sub("", text)
    if "{" in residue or "}" in residue:
        raise ValueError(
            f"trace family strings support only the {INDEX_TOKEN} and {VALUE_TOKEN} tokens "
            f"and are not a templating language; got {text!r}"
        )
    if not values_declared and any(
        match.group(0) != INDEX_TOKEN for match in SUBSTITUTION_PATTERN.finditer(text)
    ):
        raise ValueError(
            f"trace family strings carry the {VALUE_TOKEN} token only when the family "
            f"declares per-member values; got {text!r}"
        )
    return text


def substitute_index(value: Any, index: int | str, member_value: float | None = None) -> Any:
    """Substitute one member's index, and its value when it has one, into a declaration.

    Both tokens are replaced in a single pass, so a substituted index can never
    be re-read as a token.
    """

    def _replace(match: re.Match[str]) -> str:
        if match.group(0) == INDEX_TOKEN:
            return str(index)
        # validate_index_text has already rejected a value token without a value.
        return format(member_value, match.group(1) or "")

    def _substitute(text: str) -> str:
        validate_index_text(text, values_declared=member_value is not None)
        return SUBSTITUTION_PATTERN.sub(_replace, text)

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

    ``values`` names the physical quantity each member stands for, one value
    per index. It changes two things. Colors are then sampled at each value's
    normalized position over ``[min(values), max(values)]`` rather than at
    uniform index spacing, so a geometrically spaced sweep is colored by what
    it swept. And the ``{value}`` token becomes available, so a member's
    data path can stay keyed by index while its label reads in physical units.
    Values must be finite and distinct — a repeated value would give two
    members one color and one colorbar stop — and there must be at least two of
    them, since a single value spans no domain to position anything within.
    Their declaration order is free; position follows the value, not the order.

    ``legend_index`` names the one member that carries the family's legend
    entry. A family draws one logical series, so one entry describes it: the
    named member is expanded with ``showlegend`` true and every other member
    with it false. The family's own trace params must then not set
    ``showlegend``, for the same reason a colorscale forbids an explicit color.

    ``interleave_group`` joins this family to the other families declaring the
    same group, which expand together one member per index position rather than
    as separate blocks. Draw order is the only thing it changes.
    """

    schema_id: str = FIGURE_TRACE_FAMILY_SCHEMA_ID
    schema_version: str = FIGURE_TRACE_FAMILY_SCHEMA_VERSION
    name: str
    index: TraceFamilyIndex
    trace: TraceBinding
    colorscale: ColorscaleSpec | None = None
    values: list[float] | None = None
    legend_index: int | str | None = None
    interleave_group: str | None = None

    @model_validator(mode="after")
    def _validate_family(self) -> "TraceFamily":
        if self.schema_id != FIGURE_TRACE_FAMILY_SCHEMA_ID:
            raise ValueError(f"unsupported TraceFamily schema_id: {self.schema_id!r}")
        if self.schema_version != FIGURE_TRACE_FAMILY_SCHEMA_VERSION:
            raise ValueError(f"unsupported TraceFamily schema_version: {self.schema_version!r}")

        indices = self.index.resolve()
        self._validate_values(indices)
        self._validate_legend_index(indices)

        indexed = False
        format_specs: list[str] = []

        def _scan(text: str) -> str:
            nonlocal indexed
            validate_index_text(text, values_declared=self.values is not None)
            for match in SUBSTITUTION_PATTERN.finditer(text):
                indexed = True
                if match.group(0) != INDEX_TOKEN:
                    format_specs.append(match.group(1) or "")
            return text

        map_family_strings(self.trace, _scan)
        if not indexed:
            raise ValueError(
                f"TraceFamily {self.name!r} declares no {INDEX_TOKEN} or {VALUE_TOKEN} token, "
                "so its members would not vary with the index"
            )
        self._validate_format_specs(format_specs)
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

    def _validate_values(self, indices: Sequence[int | str]) -> None:
        if self.values is None:
            return
        if len(self.values) != len(indices):
            raise ValueError(
                f"TraceFamily {self.name!r} declares {len(self.values)} values for "
                f"{len(indices)} indices; values are one per member"
            )
        if len(self.values) < 2:
            raise ValueError(
                f"TraceFamily {self.name!r} declares one value, which spans no value domain "
                "for its member colors to be positioned within"
            )
        nonfinite = [value for value in self.values if not math.isfinite(value)]
        if nonfinite:
            raise ValueError(f"TraceFamily {self.name!r} values must be finite; got {nonfinite}")
        duplicates = sorted({value for value, count in Counter(self.values).items() if count > 1})
        if duplicates:
            raise ValueError(
                f"TraceFamily {self.name!r} values must be distinct; repeated {duplicates}"
            )

    def _validate_legend_index(self, indices: Sequence[int | str]) -> None:
        if self.legend_index is None:
            return
        if str(self.legend_index) not in {str(index) for index in indices}:
            raise ValueError(
                f"TraceFamily {self.name!r} legend_index {self.legend_index!r} is not one of "
                "its indices"
            )
        if "showlegend" in self.trace.params:
            raise ValueError(
                f"TraceFamily {self.name!r} declares both a legend_index and an explicit "
                "trace params showlegend"
            )

    def _validate_format_specs(self, format_specs: Sequence[str]) -> None:
        if not format_specs or self.values is None:
            return
        probe = self.values[0]
        for spec in format_specs:
            try:
                format(probe, spec)
            except ValueError as error:
                raise ValueError(
                    f"TraceFamily {self.name!r} value format spec {spec!r} does not format "
                    f"a value: {error}"
                ) from error

    def member_values(self) -> tuple[float | None, ...]:
        """Return one value per index, or one ``None`` per index when undeclared."""
        indices = self.index.resolve()
        if self.values is None:
            return (None,) * len(indices)
        return tuple(self.values)

    def expanded_trace_names(self) -> tuple[str, ...]:
        """Return the member trace names this family expands to, in index order."""
        return tuple(
            substitute_index(self.trace.name, index, value)
            for index, value in zip(self.index.resolve(), self.member_values(), strict=True)
        )


class ColorbarPanelPlacement(StrictModel):
    """Size and position a colorbar relative to one resolved grid panel."""

    schema_id: str = COLORBAR_PANEL_PLACEMENT_SCHEMA_ID
    schema_version: str = COLORBAR_PANEL_PLACEMENT_SCHEMA_VERSION
    panel: str = Field(min_length=1)
    length_fraction: float = Field(gt=0.0, le=1.0)
    center_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    side: Literal["left", "right"] = "right"
    offset_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_placement(self) -> "ColorbarPanelPlacement":
        if self.schema_id != COLORBAR_PANEL_PLACEMENT_SCHEMA_ID:
            raise ValueError(
                "unsupported ColorbarPanelPlacement schema_id: "
                f"{self.schema_id!r}"
            )
        if self.schema_version != COLORBAR_PANEL_PLACEMENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ColorbarPanelPlacement schema_version: "
                f"{self.schema_version!r}"
            )
        half_length = self.length_fraction / 2.0
        if not half_length <= self.center_fraction <= 1.0 - half_length:
            raise ValueError(
                "ColorbarPanelPlacement center_fraction and length_fraction "
                "would extend outside the selected panel"
            )
        return self


class FigureColorbar(StrictModel):
    """The declared key for a figure's color mapping.

    A colorbar is declared in one of two forms. The standalone form names an
    explicit ``colorscale`` and the ``range`` of values it spans. The bound
    form names a declared trace ``family`` instead: the family's expanded
    ``(index, color)`` pairs already are its color key, so the bound form is
    resolved by reading them rather than by re-sampling the colorscale, and the
    key cannot disagree with the traces it describes.

    With a family that declares no values, ``range`` is optional and relabels
    the family's index domain onto the values those indices stand for, mapping
    the smallest index to ``range[0]`` and the largest to ``range[1]`` — a fan
    of knot traces indexed ``0..59`` can therefore be keyed by the conditioning
    level it sweeps. Omitted, the index domain is its own value domain.

    With a family that declares ``values``, those values are the value domain:
    stops sit at their normalized positions and the range is exactly
    ``[min(values), max(values)]``. An explicit ``range`` is then rejected
    rather than honored, because relabeling a domain the family has already
    stated would print tick values no trace was colored by.
    """

    schema_id: str = FIGURE_COLORBAR_SCHEMA_ID
    schema_version: str = FIGURE_COLORBAR_SCHEMA_VERSION
    title: str | None = None
    colorscale: ColorscaleSpec | None = None
    range: tuple[float, float] | None = None
    family: str | None = None
    placement: ColorbarPanelPlacement | None = None

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
    equal_data_aspect: EqualDataAspect | None = None
    row: int | None = None
    col: int | None = None

    @model_validator(mode="after")
    def _validate_equal_data_aspect(self) -> "PanelSpec":
        if self.equal_data_aspect is None:
            return self
        nonlinear = [
            name
            for name, axis in (("x_axis", self.x_axis), ("y_axis", self.y_axis))
            if axis is not None and axis.type == "log"
        ]
        if nonlinear:
            raise ValueError(
                "PanelSpec equal_data_aspect requires linear axes; "
                f"nonlinear declarations: {nonlinear}"
            )
        return self


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


class FigureDataProductArtifactPayload(StrictModel):
    """Select one JSON artifact from one exact typed AnalysisDataProduct."""

    schema_id: str = FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_ID
    schema_version: str = FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_VERSION
    name: str
    authority: Literal["analysis_data_product"] = "analysis_data_product"
    manifest_role: str
    product_role: str
    product_schema_id: str
    product_schema_version: str
    artifact_role: str
    artifact_provider: str
    media_type: str = "application/json"
    manifest_status: Literal["completed"] = "completed"
    payload_schema_id: str | None = None
    payload_schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_identity(self) -> "FigureDataProductArtifactPayload":
        if self.schema_id != FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_ID:
            raise ValueError(
                "unsupported FigureDataProductArtifactPayload schema_id: "
                f"{self.schema_id!r}"
            )
        if self.schema_version != FIGURE_DATA_PRODUCT_PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                "unsupported FigureDataProductArtifactPayload schema_version: "
                f"{self.schema_version!r}"
            )
        for field_name in (
            "product_role",
            "product_schema_id",
            "product_schema_version",
            "artifact_provider",
        ):
            if not getattr(self, field_name):
                raise ValueError(
                    f"FigureDataProductArtifactPayload {field_name} must be nonempty"
                )
        return self


FigureArtifactPayloadSpec: TypeAlias = (
    FigureArtifactPayload | FigureDataProductArtifactPayload
)


class FigureInputAuthority(StrictModel):
    """Portable authority requirements for one exact figure parent."""

    schema_id: str = FIGURE_INPUT_AUTHORITY_SCHEMA_ID
    schema_version: str = FIGURE_INPUT_AUTHORITY_SCHEMA_VERSION
    parent: ParentRef
    artifact_payloads: list[FigureArtifactPayloadSpec] = Field(default_factory=list)

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


class FigureInputRoleAuthority(StrictModel):
    """Portable authority requirements for one uniquely role-addressed figure input."""

    schema_id: str = FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_ID
    schema_version: str = FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_VERSION
    input_role: str
    artifact_payloads: list[FigureArtifactPayloadSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_identity(self) -> "FigureInputRoleAuthority":
        if self.schema_id != FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigureInputRoleAuthority schema_id: {self.schema_id!r}"
            )
        if self.schema_version != FIGURE_INPUT_ROLE_AUTHORITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported FigureInputRoleAuthority schema_version: "
                f"{self.schema_version!r}"
            )
        if not self.input_role:
            raise ValueError("FigureInputRoleAuthority input_role must be nonempty")
        names = [payload.name for payload in self.artifact_payloads]
        if len(names) != len(set(names)):
            raise ValueError("FigureInputRoleAuthority artifact payload names must be unique")
        return self

    def resolve_parent(self, inputs: Sequence[ParentRef]) -> ParentRef:
        """Resolve this selector to one exact declared parent or fail closed."""
        matches_by_ref = {
            parent.model_dump_json(): parent
            for parent in inputs
            if parent.role == self.input_role
        }
        matches = list(matches_by_ref.values())
        if not matches:
            raise ValueError(
                f"FigureInputRoleAuthority input_role {self.input_role!r} matches no "
                "declared FigureSpec input"
            )
        if len(matches) != 1:
            raise ValueError(
                f"FigureInputRoleAuthority input_role {self.input_role!r} is ambiguous "
                f"across {len(matches)} declared FigureSpec inputs"
            )
        return matches[0]


FigureInputAuthoritySpec: TypeAlias = FigureInputAuthority | FigureInputRoleAuthority


class FigureRuntimeArtifactProviderBinding(StrictModel):
    """One exact parent-scoped authored-to-runtime provider binding."""

    parent: ParentRef
    authored_provider: str
    runtime_provider: str

    @model_validator(mode="after")
    def _validate_names(self) -> "FigureRuntimeArtifactProviderBinding":
        from feedbax.contracts.staged_execution import validate_staged_binding_name

        validate_staged_binding_name(self.authored_provider)
        validate_staged_binding_name(self.runtime_provider)
        return self


class FigureRuntimeBindingSpec(StrictModel):
    """Authenticated regeneration metadata kept outside authored FigureSpec identity."""

    schema_id: str = FIGURE_RUNTIME_BINDING_SCHEMA_ID
    schema_version: str = FIGURE_RUNTIME_BINDING_SCHEMA_VERSION
    authored_figure_spec_sha256: str
    inputs: list[ParentRef]
    input_authorities: list[FigureInputAuthoritySpec]
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_provider_bindings: list[FigureRuntimeArtifactProviderBinding]

    @model_validator(mode="after")
    def _validate_identity(self) -> "FigureRuntimeBindingSpec":
        if self.schema_id != FIGURE_RUNTIME_BINDING_SCHEMA_ID:
            raise ValueError(
                f"unsupported FigureRuntimeBindingSpec schema_id: {self.schema_id!r}"
            )
        if self.schema_version != FIGURE_RUNTIME_BINDING_SCHEMA_VERSION:
            raise ValueError(
                "unsupported FigureRuntimeBindingSpec schema_version: "
                f"{self.schema_version!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.authored_figure_spec_sha256):
            raise ValueError(
                "FigureRuntimeBindingSpec authored_figure_spec_sha256 must be lowercase sha256"
            )
        input_keys = {
            parent.model_dump_json(exclude_none=False) for parent in self.inputs
        }
        binding_keys: set[tuple[str, str]] = set()
        for binding in self.artifact_provider_bindings:
            parent_key = binding.parent.model_dump_json(exclude_none=False)
            if parent_key not in input_keys:
                raise ValueError(
                    "FigureRuntimeBindingSpec provider binding parent is not a runtime input"
                )
            key = (parent_key, binding.authored_provider)
            if key in binding_keys:
                raise ValueError(
                    "FigureRuntimeBindingSpec has a duplicate exact-parent provider binding"
                )
            binding_keys.add(key)
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
        slot_collisions = _name_collisions([slot.name for slot in self.slots])
        if slot_collisions:
            raise ValueError(f"FigureTemplate slot names collide: {slot_collisions}")
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
    input_authorities: list[FigureInputAuthoritySpec] = Field(default_factory=list)
    slot_bindings: dict[str, TraceBinding | list[TraceBinding]] = Field(default_factory=dict)
    # None-default preserves canonical bytes for FigureSpec documents authored
    # before compact template-slot families existed.
    slot_families: list[FigureSlotFamily] | None = None
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
        authority_parents = [
            (
                authority.parent
                if isinstance(authority, FigureInputAuthority)
                else authority.resolve_parent(self.inputs)
            )
            for authority in self.input_authorities
        ]
        if len(authority_parents) != len({parent.model_dump_json() for parent in authority_parents}):
            raise ValueError(
                "FigureSpec input_authorities resolve to a duplicate exact ParentRef"
            )
        unknown = [parent for parent in authority_parents if parent not in self.inputs]
        if unknown:
            raise ValueError("FigureSpec input authority parent must exactly match a declared input")
        self._validate_slot_families()
        self._validate_trace_families()
        self._validate_trace_names()
        self._validate_colorbar_binding()
        return self

    def _validate_slot_families(self) -> None:
        families = self.slot_families or []
        family_collisions = _name_collisions([family.name for family in families])
        if family_collisions:
            raise ValueError(f"FigureSpec slot_families names collide: {family_collisions}")
        slot_collisions = _name_collisions([family.slot for family in families])
        if slot_collisions:
            raise ValueError(
                f"FigureSpec slot_families bind the same slot more than once: {slot_collisions}"
            )
        concrete_collisions = sorted(set(self.slot_bindings) & {family.slot for family in families})
        if concrete_collisions:
            raise ValueError(
                "FigureSpec slots are bound by both slot_bindings and slot_families: "
                f"{concrete_collisions}"
            )

    def _validate_trace_families(self) -> None:
        families = self.trace_families or []
        if not families:
            return
        family_collisions = _name_collisions([family.name for family in families])
        if family_collisions:
            raise ValueError(f"FigureSpec trace_families names collide: {family_collisions}")
        self._validate_interleave_groups(families)

    def _validate_trace_names(self) -> None:
        names = [trace.name for trace in self.traces]
        for raw in self.slot_bindings.values():
            names.extend(binding.name for binding in (raw if isinstance(raw, list) else [raw]))
        names.extend(
            binding.name for family in (self.slot_families or []) for binding in family.expand()
        )
        names.extend(
            name for family in (self.trace_families or []) for name in family.expanded_trace_names()
        )
        collisions = _name_collisions(names)
        if collisions:
            raise ValueError(
                "FigureSpec trace family or slot-family expansion collides with other "
                f"trace names: {collisions}"
            )

    @staticmethod
    def _validate_interleave_groups(families: Sequence[TraceFamily]) -> None:
        groups: dict[str, list[TraceFamily]] = {}
        for family in families:
            if family.interleave_group is not None:
                groups.setdefault(family.interleave_group, []).append(family)
        for group, members in groups.items():
            if len(members) < 2:
                raise ValueError(
                    f"FigureSpec interleave group {group!r} names one trace family, which "
                    "interleaves with nothing"
                )
            counts = {len(family.index.resolve()) for family in members}
            if len(counts) > 1:
                raise ValueError(
                    f"FigureSpec interleave group {group!r} families enumerate different "
                    f"index counts {sorted(counts)}; interleaving pairs them position by position"
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
        if family.values is not None:
            # The declared values are the value domain, so the index labels play
            # no part in the key and need not be numeric.
            if colorbar.range is not None:
                raise ValueError(
                    f"FigureSpec colorbar trace family {family.name!r} declares values, which "
                    "are its value domain; an explicit range would relabel them"
                )
        elif any(not isinstance(index, int) for index in indices):
            raise ValueError(
                f"FigureSpec colorbar trace family {family.name!r} has non-numeric indices; "
                "a colorbar keys a numeric index domain"
            )
        if len(indices) < 2:
            raise ValueError(
                f"FigureSpec colorbar trace family {family.name!r} enumerates one index, "
                "which spans no range for a colorbar to key"
            )
