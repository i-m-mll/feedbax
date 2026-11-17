"""Authored Cartesian layout settings for dataset figures."""

import math
from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel


DATA_DERIVED_AXIS_RANGE_SCHEMA_ID = "feedbax.spec.data_derived_axis_range"
DATA_DERIVED_AXIS_RANGE_SCHEMA_VERSION = "feedbax.spec.data_derived_axis_range.v1"


class AxisLabels(StrictModel):
    """Per-panel axis label overrides. Cartesian panels accept x and y."""

    x: str | None = None
    y: str | None = None
    z: str | None = None


class DataDerivedAxisRange(StrictModel):
    """One rendered-data range shared by a named group of panel axes."""

    schema_id: str = DATA_DERIVED_AXIS_RANGE_SCHEMA_ID
    schema_version: str = DATA_DERIVED_AXIS_RANGE_SCHEMA_VERSION
    group: str = Field(min_length=1)
    padding_fraction: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_declaration(self) -> "DataDerivedAxisRange":
        if self.schema_id != DATA_DERIVED_AXIS_RANGE_SCHEMA_ID:
            raise ValueError(f"unsupported DataDerivedAxisRange schema_id: {self.schema_id!r}")
        if self.schema_version != DATA_DERIVED_AXIS_RANGE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported DataDerivedAxisRange schema_version: {self.schema_version!r}"
            )
        if self.group != self.group.strip():
            raise ValueError("DataDerivedAxisRange group must not have surrounding whitespace")
        if not math.isfinite(self.padding_fraction):
            raise ValueError("DataDerivedAxisRange padding_fraction must be finite")
        return self


class PanelAxis(StrictModel):
    """Declarative settings for one panel axis."""

    type: Literal["linear", "log"] | None = None
    range: tuple[float, float] | None = None
    derived_range: DataDerivedAxisRange | None = None
    quantity: str | None = Field(default=None, min_length=1, max_length=128)
    unit: str | None = Field(default=None, min_length=1, max_length=128)
    frame: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def _validate_range_authority(self) -> "PanelAxis":
        if self.derived_range is not None and self.range is not None:
            raise ValueError("PanelAxis cannot declare both range and derived_range")
        if self.derived_range is not None and self.type == "log":
            raise ValueError("PanelAxis derived_range requires a linear axis")
        if (self.quantity is None) != (self.unit is None):
            raise ValueError("PanelAxis quantity and unit must be declared together")
        if self.frame is not None and self.quantity is None:
            raise ValueError("PanelAxis frame requires quantity and unit")
        for name in ("quantity", "unit", "frame"):
            value = getattr(self, name)
            if value is not None and value != value.strip():
                raise ValueError(f"PanelAxis {name} must not have surrounding whitespace")
        return self


class ComparisonGridParams(StrictModel):
    """Defaults for panel population."""

    shared_xaxes: bool | str = False
    shared_yaxes: bool | str = "all"
    horizontal_spacing: float | None = None
    vertical_spacing: float | None = None


class FigureMarginParams(StrictModel):
    """Optional Plotly figure margins."""

    l: float | None = None  # noqa: E741 - Plotly's public margin key is "l".
    r: float | None = None
    t: float | None = None
    b: float | None = None
    pad: float | None = None
    autoexpand: bool | None = None


class FigureView(StrictModel):
    """Presentation settings for the whole figure."""

    width: int | None = None
    height: int | None = None
    title: str | None = None
    legend_tracegroupgap: int | None = None
    legend_x: float | None = None
    legend_y: float | None = None
    legend_xanchor: Literal["auto", "left", "center", "right"] | None = None
    legend_yanchor: Literal["auto", "top", "middle", "bottom"] | None = None
    margin: FigureMarginParams | None = None
    hovermode: Literal["x", "y", "closest", "x unified", "y unified", False] | None = None
    legend_gutter: float | None = Field(default=None, ge=0.0, le=512.0)
