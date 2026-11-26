"""Authored figure layers and encodings over labelled datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor, Name
from feedbax.contracts.figure_layout import AxisLabels, ComparisonGridParams, FigureView, PanelAxis
from feedbax.contracts.member_stack import MEMBER
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.project_inputs import NonemptyText
from feedbax.contracts.request_refs import ContentRef
from feedbax.plugins.declarations import validate_declaration_key


FIGURE_DOCUMENT_SCHEMA = "feedbax.figure_document.v1"


def is_figure_document_pin(payload: object) -> bool:
    """Recognize the pin fields reserved by a named figure payload."""
    return isinstance(payload, dict) and bool(payload.keys() & {"path", "sha256"})


class IndexSelection(StrictModel):
    index: Annotated[StrictInt, Field(ge=0)]


Selection = StrictStr | StrictInt | StrictFloat | StrictBool | None | IndexSelection


class VariableBinding(StrictModel):
    variable: Name
    select: dict[Name, Selection] = Field(default_factory=dict)


Encoding = Name | VariableBinding
PaletteNumber = Annotated[StrictFloat, Field(allow_inf_nan=False)]
PaletteValue = StrictStr | StrictBool | StrictInt | PaletteNumber


class ColorEncoding(StrictModel):
    """One field value per plotted group, drawn on a named figure scale."""

    field: Encoding
    scale: Name


class ContinuousColorScale(StrictModel):
    kind: Literal["continuous"]
    palette: Literal["viridis", "cividis", "plasma", "turbo"] = "viridis"
    domain: tuple[PaletteNumber, PaletteNumber] | None = None
    title: str | None = None
    legend: bool = True

    @model_validator(mode="after")
    def _check_domain(self) -> ContinuousColorScale:
        if self.domain is not None and self.domain[0] >= self.domain[1]:
            raise ValueError("A continuous color domain needs increasing bounds.")
        return self


class OrdinalColorScale(StrictModel):
    kind: Literal["ordinal"]
    palette: Literal["category10", "plotly", "dark24"] = "category10"
    domain: Annotated[tuple[PaletteValue, ...], Field(min_length=1)] | None = None
    title: str | None = None
    legend: bool = True

    @model_validator(mode="after")
    def _check_domain(self) -> OrdinalColorScale:
        if self.domain is not None and len({ordinal_key(value) for value in self.domain}) != len(self.domain):
            raise ValueError("An ordinal color domain needs distinct values.")
        return self


ColorScale = Annotated[ContinuousColorScale | OrdinalColorScale, Field(discriminator="kind")]


def ordinal_key(value: PaletteValue) -> tuple[bool, PaletteValue]:
    """A Boolean category is distinct from a numerical one."""
    return isinstance(value, bool), value


class LineEncodings(StrictModel):
    x: Encoding
    y: Encoding
    color: ColorEncoding | None = Field(default=None, exclude_if=lambda value: value is None)


class TrajectoryEncodings(LineEncodings):
    z: Encoding | None = Field(default=None, exclude_if=lambda value: value is None)


class BandEncodings(LineEncodings):
    lower: Encoding
    upper: Encoding


class ErrorBounds(StrictModel):
    """Absolute bounds of the error bar around a point's y sample."""

    lower: Encoding
    upper: Encoding


class PointsEncodings(LineEncodings):
    error_y: ErrorBounds | None = None


ReferenceValue = Encoding | Annotated[StrictFloat, Field(allow_inf_nan=False)]


class RuleEncodings(StrictModel):
    axis: Literal["x", "y"]
    at: ReferenceValue


class SpanEncodings(StrictModel):
    axis: Literal["x", "y"]
    start: ReferenceValue
    end: ReferenceValue


class LineStyle(StrictModel):
    label: str | None = None
    color: str | None = None
    line_dash: Literal["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"] | None = None
    line_width: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    showlegend: bool | None = None


class BandStyle(LineStyle):
    band_alpha: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)


class PointsStyle(LineStyle):
    marker_size: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class SpanStyle(LineStyle):
    fill_alpha: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)


class _Layer(StrictModel):
    name: Name
    role: Name
    panel: Name
    select: dict[Name, Selection] = Field(default_factory=dict)
    group: tuple[Name, ...] = ()
    """The grouping coordinates. Member grouping splits a dataset stack and leaves a single dataset whole."""


class LineLayer(_Layer):
    mark: Literal["line"]
    encode: LineEncodings
    style: LineStyle = Field(default_factory=LineStyle)


class TrajectoryLayer(_Layer):
    """A two-dimensional path or a three-dimensional path when z is encoded."""

    mark: Literal["trajectory"]
    encode: TrajectoryEncodings
    style: LineStyle = Field(default_factory=LineStyle)


class BandLayer(_Layer):
    mark: Literal["band"]
    encode: BandEncodings
    style: BandStyle = Field(default_factory=BandStyle)


class PointsLayer(_Layer):
    mark: Literal["points"]
    encode: PointsEncodings
    style: PointsStyle = Field(default_factory=PointsStyle)


class RuleLayer(_Layer):
    """One reference position across the other axis of its panel."""

    mark: Literal["rule"]
    encode: RuleEncodings
    style: LineStyle = Field(default_factory=LineStyle)


class SpanLayer(_Layer):
    """One shaded interval across the other axis of its panel."""

    mark: Literal["span"]
    encode: SpanEncodings
    style: SpanStyle = Field(default_factory=SpanStyle)


FigureLayer = Annotated[
    LineLayer | TrajectoryLayer | BandLayer | PointsLayer | RuleLayer | SpanLayer, Field(discriminator="mark"),
]


def is_spatial_layer(layer: FigureLayer) -> bool:
    return isinstance(layer, TrajectoryLayer) and layer.encode.z is not None


def color_encoding(layer: FigureLayer) -> ColorEncoding | None:
    return layer.encode.color if isinstance(layer.encode, LineEncodings) else None


class FigurePanel(StrictModel):
    name: Name
    row: Annotated[StrictInt, Field(gt=0)]
    col: Annotated[StrictInt, Field(gt=0)]
    title: str | None = None
    axes_labels: AxisLabels | None = None
    x_axis: PanelAxis | None = None
    y_axis: PanelAxis | None = None
    z_axis: PanelAxis | None = Field(default=None, exclude_if=lambda value: value is None)


class FigureDocumentView(FigureView, ComparisonGridParams):
    """Authored whole-figure and grid settings."""


class FigureDocument(StrictModel):
    """One figure composed of dataset-bound marks."""

    schema_version: Literal["feedbax.figure_document.v1"] = FIGURE_DOCUMENT_SCHEMA
    data: tuple[Name, ...] = Field(min_length=1)
    layout: tuple[FigurePanel, ...] = Field(min_length=1)
    layers: tuple[FigureLayer, ...] = Field(min_length=1)
    theme: Name = "plotly_white"
    color_scales: dict[Name, ColorScale] = Field(default_factory=dict, exclude_if=lambda value: not value)
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
        spatial = {layer.panel for layer in self.layers if is_spatial_layer(layer)}
        for panel in self.layout:
            if panel.name not in spatial and (panel.z_axis is not None or getattr(panel.axes_labels, "z", None) is not None):
                raise ValueError(f"Panel {panel.name!r} needs a trajectory with z to use a z axis.")
        for layer in self.layers:
            if layer.panel not in panels or layer.role not in self.data:
                raise ValueError(f"Layer {layer.name!r} requires a declared panel and data role.")
            if len(set(layer.group)) != len(layer.group):
                raise ValueError(f"Layer {layer.name!r} repeats a group coordinate.")
            if layer.panel in spatial and not is_spatial_layer(layer):
                raise ValueError(f"Layer {layer.name!r} must be a trajectory with z in its 3D panel.")
            if (color := color_encoding(layer)) is not None:
                if color.scale not in self.color_scales:
                    raise ValueError(f"Layer {layer.name!r} names unknown color scale {color.scale!r}.")
                if layer.style.color is not None:
                    raise ValueError(f"Layer {layer.name!r} cannot set a fixed color and a color encoding.")
        return self


FigureTemplateKey = Annotated[StrictStr, AfterValidator(validate_declaration_key)]


class FigureTemplateRef(StrictModel):
    """A figure generated from an installed declaration."""

    template: FigureTemplateKey
    settings: dict[str, JsonValue] = Field(default_factory=dict)


FigureSource = FigureDocument | FigureTemplateRef


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


def position_encodings(layer: FigureLayer) -> tuple[tuple[str, ReferenceValue], ...]:
    """Read the data channels that locate a mark, preserving paths for diagnostics."""
    match layer:
        case PointsLayer(encode=encoding):
            channels = (("x", encoding.x), ("y", encoding.y))
            if (bounds := encoding.error_y) is not None:
                return (*channels, ("error_y.lower", bounds.lower), ("error_y.upper", bounds.upper))
            return channels
        case RuleLayer(encode=encoding):
            return (("at", encoding.at),)
        case SpanLayer(encode=encoding):
            return (("start", encoding.start), ("end", encoding.end))
        case LineLayer(encode=encoding):
            return (("x", encoding.x), ("y", encoding.y))
        case TrajectoryLayer(encode=encoding):
            channels = (("x", encoding.x), ("y", encoding.y))
            return (*channels, ("z", encoding.z)) if encoding.z is not None else channels
        case BandLayer(encode=encoding):
            return (("x", encoding.x), ("y", encoding.y), ("lower", encoding.lower), ("upper", encoding.upper))


def layer_encodings(layer: FigureLayer) -> tuple[tuple[str, ReferenceValue], ...]:
    channels = position_encodings(layer)
    color = color_encoding(layer)
    return (*channels, ("color.field", color.field)) if color is not None else channels


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
        sizes = schema.dimensions if isinstance(schema, DatasetDescriptor) else {}
        fields = schema.variables | coords
        if MEMBER in layer.group and (member := coords.get(MEMBER)) is not None:
            axes = set(member.dims)
            coords = {
                name: field.model_copy(update={"dims": kept})
                if len(field.dims) > 1 and (kept := tuple(dim for dim in field.dims if dim not in axes))
                else field
                for name, field in coords.items()
            }
        for channel, encoding in layer_encodings(layer):
            if isinstance(encoding, float):
                continue
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
                    sizes,
                    f"{path}.encode.{channel}.select",
                    f"channel {channel!r} of {layer.name!r}",
                )
        _validate_coordinates(
            layer.select, coords, sizes, path + ".select", f"selection of {layer.name!r}"
        )
        _validate_coordinates(
            tuple(name for name in layer.group if name != MEMBER), coords, sizes, path + ".group", f"group of {layer.name!r}",
        )


def coordinate_dimension(field: FieldDescriptor, sizes: Mapping[str, int]) -> str | None:
    """The one dimension a coordinate labels: its only dimension, or its only one longer than one."""
    if len(field.dims) == 1:
        return field.dims[0]
    longer = [dim for dim in field.dims if sizes.get(dim, 2) != 1]
    if len(longer) == 1:
        return longer[0]
    return field.dims[-1] if field.dims and not longer else None


def _validate_coordinates(
    names: Mapping[str, Selection] | tuple[str, ...],
    coords: Mapping[str, FieldDescriptor],
    sizes: Mapping[str, int],
    path: str,
    context: str,
) -> None:
    for name in names:
        if name not in coords or coordinate_dimension(coords[name], sizes) is None:
            raise FigureBindingError(
                path, f"Figure {context} requires a one-dimensional coordinate {name!r}."
            )
