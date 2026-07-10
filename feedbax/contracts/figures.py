"""Declarative figure contracts.

These models describe figure provenance and construction inputs. Plotly JSON is
the rendered product; these contracts are the durable source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.expressions import Expr, ValueExpr
from feedbax.contracts.manifest import ArtifactRef, ParentRef, SpecPayload, StrictModel
from feedbax.contracts.selection import ManifestPredicate


FIGURE_SPEC_SCHEMA_ID = "feedbax.spec.figure"
FIGURE_SPEC_SCHEMA_VERSION = "feedbax.spec.figure.v1"
FIGURE_TEMPLATE_SCHEMA_ID = "feedbax.spec.figure_template"
FIGURE_TEMPLATE_SCHEMA_VERSION = "feedbax.spec.figure_template.v1"
FIGURE_PIECE_SCHEMA_ID = "feedbax.spec.figure_piece"
FIGURE_PIECE_SCHEMA_VERSION = "feedbax.spec.figure_piece.v1"

SlotMultiplicity = Literal["one", "per_facet", "many"]
FacetTarget = Literal["figures", "panels"]


class AxisLabels(StrictModel):
    """Per-panel axis labels."""

    x: str | None = None
    y: str | None = None


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


class PanelSpec(StrictModel):
    """One subplot cell-group."""

    name: str
    title: str | ValueExpr | None = None
    axes_labels: AxisLabels | None = None
    row: int | None = None
    col: int | None = None


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
    slot_bindings: dict[str, TraceBinding | list[TraceBinding]] = Field(default_factory=dict)
    traces: list[TraceBinding] = Field(default_factory=list)
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
        return self
