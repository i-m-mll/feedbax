"""A rendered figure and the declarations that bind it to the Studio clock."""

from pydantic import ConfigDict, Field, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr

from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import TrialFigureTime


type FigureGroupValue = StrictStr | StrictInt | StrictFloat | StrictBool | None


class FigureTraceMeta(StrictModel):
    layer: str
    group: dict[str, FigureGroupValue]


class FigurePanelMeta(StrictModel):
    x_axis: str
    y_axis: str
    z_axis: str | None = Field(default=None, exclude_if=lambda value: value is None)
    time: TrialFigureTime | None = None


class FigureRun(StrictModel):
    job_id: str
    request_id: str


class FigureLayoutMeta(StrictModel):
    panels: dict[str, FigurePanelMeta]
    runs: dict[str, tuple[FigureRun, ...]]


class PlotlyTrace(StrictModel):
    model_config = ConfigDict(extra="allow")

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    meta: FigureTraceMeta


class PlotlyLayout(StrictModel):
    model_config = ConfigDict(extra="allow")

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    meta: FigureLayoutMeta


class FigureProduct(StrictModel):
    """Plotly's figure JSON with typed metadata for Studio's clock."""

    data: list[PlotlyTrace]
    layout: PlotlyLayout
    frames: list[dict[str, JsonValue]] | None = None
