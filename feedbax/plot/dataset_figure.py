"""Draw a figure document from prepared variables in dataset coordinate order."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, assert_never

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor
from feedbax.contracts.figure_document import (
    BandLayer,
    Encoding,
    FigureDocument,
    FigureLayer,
    IndexSelection,
    LineLayer,
    Selection,
    VariableBinding,
    layer_encodings,
    validate_figure_bindings,
)
from feedbax.contracts.figure_layout import ComparisonGridParams, FigureView
from feedbax.dataset import Dataset
from feedbax.plot.colors import color_add_alpha
from feedbax.plot.constructors import GridFigureParams, PanelContent, _comparison_grid, _grid_figure


type Label = str | int | float | bool | None
type Positions = dict[str, tuple[int, ...]]


@dataclass
class _Source:
    dataset: Dataset
    descriptor: DatasetDescriptor
    values: dict[str, np.ndarray] = field(default_factory=dict)

    def read(self, name: str) -> np.ndarray:
        if name not in self.values:
            self.values[name] = np.asarray(self.dataset.data(name))
        return self.values[name]

    def coordinate(self, name: str) -> tuple[str, list[Label]]:
        (dimension,) = self.descriptor.coords[name].dims
        labels = self.read(name).tolist()
        if any(
            value is not None and not isinstance(value, (str, int, float, bool)) for value in labels
        ):
            raise ValueError(f"Coordinate {name!r} requires scalar labels.")
        return dimension, labels


def _selected(
    source: _Source,
    selections: Mapping[str, Selection],
    positions: Positions,
) -> Positions:
    result = positions.copy()
    for name, selection in selections.items():
        dimension, labels = source.coordinate(name)
        candidates = result.get(dimension, range(len(labels)))
        if isinstance(selection, IndexSelection):
            matched = tuple(index for index in candidates if index == selection.index)
        else:
            matched = tuple(
                index
                for index in candidates
                if labels[index] == selection
                and isinstance(labels[index], bool) == isinstance(selection, bool)
            )
        if not matched:
            raise ValueError(f"Selection {name!r}={selection!r} matches no coordinate values.")
        result[dimension] = matched
    return result


def _groups(
    source: _Source,
    coordinates: tuple[str, ...],
    positions: Positions,
) -> Iterator[tuple[dict[str, Label], Positions]]:
    if not coordinates:
        yield {}, positions
        return

    name, *rest = coordinates
    dimension, labels = source.coordinate(name)
    groups: dict[tuple[bool, Label], list[int]] = {}
    for index in positions.get(dimension, range(len(labels))):
        value = labels[index]
        groups.setdefault((isinstance(value, bool), value), []).append(index)
    for (_, value), indices in groups.items():
        for labels_by_name, group in _groups(
            source, tuple(rest), positions | {dimension: tuple(indices)}
        ):
            yield {name: value} | labels_by_name, group


@dataclass(frozen=True)
class _Samples:
    name: str
    field: FieldDescriptor
    dimension: str | None
    values: np.ndarray
    positions: tuple[int, ...] | None

    @property
    def label(self) -> str:
        title = self.field.quantity or self.name.replace("_", " ")
        return f"{title} ({self.field.units})" if self.field.units else title


def _encoded(source: _Source, encoding: Encoding, positions: Positions) -> _Samples:
    if isinstance(encoding, VariableBinding):
        name = encoding.variable
        positions = _selected(source, encoding.select, positions)
    else:
        name = encoding
    descriptor = source.descriptor.field(name)
    data = source.read(name)
    dimensions = list(descriptor.dims)
    for axis, dimension in reversed(tuple(enumerate(descriptor.dims))):
        if dimension in positions:
            indices = positions[dimension]
            if len(indices) == 1:
                data = np.take(data, indices[0], axis=axis)
                dimensions.pop(axis)
            else:
                data = np.take(data, indices, axis=axis)
    if len(dimensions) > 1 or not data.size:
        raise ValueError(
            f"Field {name!r} requires nonempty scalar or one-dimensional samples, got {dimensions}."
        )
    dimension = dimensions[0] if dimensions else None
    retained = positions.get(dimension) if dimension is not None else None
    if retained is not None and len(retained) == source.descriptor.dimensions[dimension]:
        retained = None
    return _Samples(name, descriptor, dimension, np.atleast_1d(data), retained)


def _draw(
    layer: FigureLayer,
    channels: Mapping[str, _Samples],
    label: str,
    color: str,
) -> tuple[go.Scatter, ...]:
    x, y = channels["x"], channels["y"]
    for name, samples in channels.items():
        if (samples.dimension, samples.values.shape, samples.positions) != (
            x.dimension,
            x.values.shape,
            x.positions,
        ):
            raise ValueError(f"Channel {name!r} does not share the x samples.")
    center = go.Scatter(
        name=label,
        legendgroup=label,
        showlegend=layer.style.showlegend,
        x=x.values,
        y=y.values,
        mode="lines",
        line={"color": color, "dash": layer.style.line_dash, "width": layer.style.line_width},
    )
    if isinstance(layer, LineLayer):
        return (center,)
    if not isinstance(layer, BandLayer):
        assert_never(layer)

    for name in ("lower", "upper"):
        field = channels[name].field
        if (field.units, field.frame) != (y.field.units, y.field.frame):
            raise ValueError(f"Band channel {name!r} must use the centre's units and frame.")
    upper, lower = (
        go.Scatter(
            name=f"{name.title()} bound",
            legendgroup=label,
            showlegend=False,
            x=x.values,
            y=channels[name].values,
            mode="lines",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
        )
        for name in ("upper", "lower")
    )
    lower.update(fill="tonexty", fillcolor=color_add_alpha(color, layer.style.band_alpha))
    return center, upper, lower


def render(
    figure: FigureDocument,
    datasets: Mapping[str, Dataset],
    theme: str | go.layout.Template | Mapping[str, Any] | None = None,
) -> go.Figure:
    """Render only the authored bindings and prepared values."""
    sources = {
        name: _Source(dataset, dataset.descriptor)
        for name, dataset in datasets.items()
        if name in figure.data
    }
    validate_figure_bindings(figure, {name: source.descriptor for name, source in sources.items()})
    canvas = go.Figure(layout={"template": figure.theme if theme is None else theme})
    colors = cycle(canvas.layout.template.layout.colorway or qualitative.Plotly)
    traces: dict[str, list[go.Scatter]] = {panel.name: [] for panel in figure.layout}
    inferred_labels: dict[str, dict[str, str]] = {}
    for layer in figure.layers:
        source = sources[layer.role]
        before = len(traces[layer.panel])
        try:
            selection = _selected(source, layer.select, {})
            for group, positions in _groups(source, layer.group, selection):
                channels = {}
                for channel, encoding in layer_encodings(layer):
                    try:
                        channels[channel] = _encoded(source, encoding, positions)
                    except ValueError as error:
                        raise ValueError(f"channel {channel!r}: {error}") from error
                label = (layer.style.label or layer.name).format_map(group)
                color = layer.style.color or next(colors)
                traces[layer.panel].extend(_draw(layer, channels, label, color))
                inferred_labels.setdefault(
                    layer.panel, {axis: channels[axis].label for axis in ("x", "y")}
                )
        except (KeyError, ValueError) as error:
            raise ValueError(f"Layer {layer.name!r}: {error}") from error
        if len(traces[layer.panel]) == before:
            raise ValueError(f"Layer {layer.name!r} has no plotted groups.")

    panels = []
    for panel in figure.layout:
        values = panel.model_dump(exclude_none=True)
        labels = inferred_labels.get(panel.name, {}) | values.pop("axes_labels", {})
        panels.append(PanelContent(**values, axes_labels=labels, traces=tuple(traces[panel.name])))
    view = figure.view
    grid = ComparisonGridParams.model_validate(
        view.model_dump(include=set(ComparisonGridParams.model_fields))
    )
    presentation = GridFigureParams.model_validate(
        view.model_dump(include=set(FigureView.model_fields))
    )
    return _grid_figure(_comparison_grid(panels, grid, figure=canvas), panels, presentation)
