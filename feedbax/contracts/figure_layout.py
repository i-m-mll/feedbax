"""Layout settings shared by the figure assemblers."""

from typing import Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel


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
    row_label_gutter: float | None = Field(default=None, ge=0.0, le=256.0)
    legend_gutter: float | None = Field(default=None, ge=0.0, le=512.0)
