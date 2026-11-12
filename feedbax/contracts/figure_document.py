"""Authored figure layers and encodings over labelled datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor, Name
from feedbax.contracts.figure_layout import ComparisonGridParams, FigureView
from feedbax.contracts.figures import AxisLabels, PanelAxis
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.project_inputs import NonemptyText
from feedbax.contracts.request_refs import ContentRef


class IndexSelection(StrictModel):
    index: Annotated[StrictInt, Field(ge=0)]


Selection = StrictStr | StrictInt | StrictFloat | StrictBool | None | IndexSelection


class VariableBinding(StrictModel):
    variable: Name
    select: dict[Name, Selection] = Field(default_factory=dict)


Encoding = Name | VariableBinding


class LineEncodings(StrictModel):
    x: Encoding
    y: Encoding


class BandEncodings(LineEncodings):
    lower: Encoding
    upper: Encoding


class LineStyle(StrictModel):
    label: str | None = None
    color: str | None = None
    line_dash: Literal["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"] | None = None
    line_width: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    showlegend: bool | None = None


class BandStyle(LineStyle):
    band_alpha: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)


class _Layer(StrictModel):
    name: Name
    role: Name
    panel: Name
    select: dict[Name, Selection] = Field(default_factory=dict)
    group: tuple[Name, ...] = ()


class LineLayer(_Layer):
    mark: Literal["line"]
    encode: LineEncodings
    style: LineStyle = Field(default_factory=LineStyle)


class BandLayer(_Layer):
    mark: Literal["band"]
    encode: BandEncodings
    style: BandStyle = Field(default_factory=BandStyle)


FigureLayer = Annotated[LineLayer | BandLayer, Field(discriminator="mark")]


class FigurePanel(StrictModel):
    name: Name
    row: Annotated[StrictInt, Field(gt=0)]
    col: Annotated[StrictInt, Field(gt=0)]
    title: str | None = None
    axes_labels: AxisLabels | None = None
    x_axis: PanelAxis | None = None
    y_axis: PanelAxis | None = None

    @model_validator(mode="after")
    def _check_cartesian_axes(self) -> FigurePanel:
        if self.axes_labels is not None and self.axes_labels.z is not None:
            raise ValueError("Cartesian panels accept x and y axis labels.")
        return self


class FigureDocumentView(FigureView, ComparisonGridParams):
    """Authored whole-figure and grid settings."""


class FigureDocument(StrictModel):
    """One figure composed of dataset-bound marks."""

    schema_version: Literal["feedbax.figure_document.v1"] = "feedbax.figure_document.v1"
    data: tuple[Name, ...] = Field(min_length=1)
    layout: tuple[FigurePanel, ...] = Field(min_length=1)
    layers: tuple[FigureLayer, ...] = Field(min_length=1)
    theme: Name = "plotly_white"
    view: FigureDocumentView = Field(default_factory=FigureDocumentView)
    caption: str | None = None

    @model_validator(mode="after")
    def _check_layout(self) -> FigureDocument:
        panels = {panel.name for panel in self.layout}
        cells = {(panel.row, panel.col) for panel in self.layout}
        if len(panels) != len(self.layout) or len(cells) != len(self.layout):
            raise ValueError("Figure panels require distinct names and grid cells.")
        if len(set(self.data)) != len(self.data):
            raise ValueError("Figure data roles must be distinct.")
        if len({layer.name for layer in self.layers}) != len(self.layers):
            raise ValueError("Figure layer names must be distinct.")
        for layer in self.layers:
            if layer.panel not in panels or layer.role not in self.data:
                raise ValueError(f"Layer {layer.name!r} requires a declared panel and data role.")
            if len(set(layer.group)) != len(layer.group):
                raise ValueError(f"Layer {layer.name!r} repeats a group coordinate.")
        return self


PLOTLY_FIGURE_ROLE = "figure"
PLOTLY_MEDIA_TYPE = "application/json"


class PlotlyFigurePort(StrictModel):
    """Plan declaration for an eventual saved Plotly figure product."""

    kind: Literal["plotly_figure"] = "plotly_figure"


class FigureLayerOmission(StrictModel):
    layer: Name
    reason: NonemptyText


class PlotlyFigureProduct(StrictModel):
    """A saved figure object and the omissions declared by its producer."""

    content: ContentRef
    omitted_layers: tuple[FigureLayerOmission, ...] = ()

    @model_validator(mode="after")
    def _check_product(self) -> PlotlyFigureProduct:
        if (
            self.content.tree
            or self.content.size == 0
            or self.content.media_type != PLOTLY_MEDIA_TYPE
        ):
            raise ValueError("A Plotly figure requires a nonempty JSON content object.")
        if len({item.layer for item in self.omitted_layers}) != len(self.omitted_layers):
            raise ValueError("Each omitted figure layer requires one reason.")
        return self


def layer_encodings(layer: FigureLayer) -> tuple[tuple[str, Encoding], ...]:
    """Read each mark's declared channels in order."""
    return tuple((name, getattr(layer.encode, name)) for name in type(layer.encode).model_fields)


class FigureBindingError(ValueError):
    def __init__(self, path: str, detail: str):
        super().__init__(detail)
        self.path = path


def validate_figure_bindings(
    figure: FigureDocument,
    schemas: Mapping[str, DatasetDescriptor | DatasetPort],
) -> None:
    """Check the declared fields without opening any numerical tensors."""
    missing = set(figure.data) - schemas.keys()
    if missing:
        raise FigureBindingError("data", f"Missing figure data roles: {sorted(missing)}")
    for index, layer in enumerate(figure.layers):
        path = f"layers.{index}"
        schema = schemas[layer.role]
        coords = schema.coords if isinstance(schema, DatasetDescriptor) else schema.coordinates
        fields = schema.variables | coords
        for channel, encoding in layer_encodings(layer):
            variable = encoding if isinstance(encoding, str) else encoding.variable
            if variable not in fields:
                raise FigureBindingError(
                    f"{path}.encode.{channel}",
                    f"Layer {layer.name!r} channel {channel!r} names unknown field {variable!r}.",
                )
            if isinstance(encoding, VariableBinding):
                _validate_coordinates(
                    encoding.select,
                    coords,
                    f"{path}.encode.{channel}.select",
                    f"channel {channel!r} of {layer.name!r}",
                )
        _validate_coordinates(
            layer.select, coords, path + ".select", f"selection of {layer.name!r}"
        )
        _validate_coordinates(layer.group, coords, path + ".group", f"group of {layer.name!r}")


def _validate_coordinates(
    names: Mapping[str, Selection] | tuple[str, ...],
    coords: Mapping[str, FieldDescriptor],
    path: str,
    context: str,
) -> None:
    for name in names:
        if name not in coords or len(coords[name].dims) != 1:
            raise FigureBindingError(
                path, f"Figure {context} requires a one-dimensional coordinate {name!r}."
            )
