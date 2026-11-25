"""Draw a figure document from prepared variables in dataset coordinate order."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, assert_never

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor
from feedbax.contracts.figure_document import (
    BandLayer,
    BandStyle,
    Encoding,
    FigureDocument,
    FigureLayer,
    IndexSelection,
    LineLayer,
    Selection,
    VariableBinding,
    coordinate_dimension,
    layer_encodings,
    validate_figure_bindings,
)
from feedbax.contracts.rollout import CannotRun
from feedbax.contracts.trial_clock import TrialMeanOptions, TrialMeanSelection
from feedbax.dataset import Dataset
from feedbax.execution.trial_clock import (
    AlignedSeries, TrialClockRecord, align_trials, aligned_sample_times, common_span, summarize_series,
)
from feedbax.plot.colors import color_add_alpha
from feedbax.plot.figure_grid import assemble_figure_grid
from feedbax.plot.figure_trial_clock import FigureTrialSource, recorded_sample_times


type Label = str | int | float | bool | None
type Positions = dict[str, tuple[int, ...]]


@dataclass
class _Source:
    dataset: Dataset
    descriptor: DatasetDescriptor
    names: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """The printed name of each value ID of a coordinate whose values name things of the project."""
    values: dict[str, np.ndarray] = field(default_factory=dict)

    def read(self, name: str) -> np.ndarray:
        if name not in self.values:
            self.values[name] = np.asarray(self.dataset.data(name))
        return self.values[name]

    def coordinate(self, name: str) -> tuple[str, list[Label]]:
        dimension = coordinate_dimension(self.descriptor.coords[name], self.descriptor.dimensions)
        if dimension is None:
            raise ValueError(f"Coordinate {name!r} labels more than one dimension.")
        labels = self.read(name).reshape(-1).tolist()
        if any(
            value is not None and not isinstance(value, (str, int, float, bool)) for value in labels
        ):
            raise ValueError(f"Coordinate {name!r} requires scalar labels.")
        return dimension, labels

    def printed(self, name: str, value: Label) -> Label:
        names = self.names.get(name)
        return names.get(value, value) if names is not None and isinstance(value, str) else value


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
                if source.printed(name, labels[index]) == selection
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
            yield {name: source.printed(name, value)} | labels_by_name, group


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
        elif data.shape[axis] == 1 and len(dimensions) > 1:
            data = np.take(data, 0, axis=axis)
            dimensions.pop(axis)
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



@dataclass(frozen=True)
class _TrialTrace:
    group: dict[str, Label]
    coordinates: dict[str, Label]
    channels: dict[str, _Samples]
    color: str
    record: TrialClockRecord
    options: TrialMeanOptions


def _trial_traces(
    layer: FigureLayer,
    rows: list[_TrialTrace],
    mean: TrialMeanSelection,
) -> list[go.Scatter]:
    """Draw an aligned thin trace for each trial and a summary for each clock condition."""
    nontrial: dict[tuple[tuple[str, Label], ...], list[_TrialTrace]] = defaultdict(list)
    for row in rows:
        nontrial[tuple((name, value) for name, value in row.coordinates.items() if name != "trial")].append(row)
    traces = []
    for coordinates, cohort in nontrial.items():
        options = cohort[0].options
        if any(row.options != options for row in cohort):
            raise ValueError("Plotted trials disagree on the available By and Align choices.")
        by_trial = {row.record.id: row for row in cohort}
        is_time = cohort[0].channels["x"].name in {"state_time", "transition_start", "transition_end"}
        for condition in align_trials((row.record for row in cohort), options, mean):
            color = by_trial[condition.trials[0].record.id].color
            series = []
            group = dict(coordinates) | {
                f"task/{name}" if name == "trial" or name in dict(coordinates) else name: (
                    canonical_json_v2_bytes(value).decode() if isinstance(value, (list, dict)) else value
                )
                for name, value in condition.values.items()
            }
            for trial in condition.trials:
                row = by_trial[trial.record.id]
                x, y = row.channels["x"], row.channels["y"]
                recorded = (
                    x.values if is_time else recorded_sample_times(trial, x.dimension, x.positions)
                )
                time = trial.times(recorded)
                if y.values.shape != time.shape or x.values.shape != time.shape:
                    raise ValueError("The plotted channels do not share the recorded trial samples.")
                samples = AlignedSeries(time, np.column_stack((x.values, y.values)), trial.alignment.span)
                series.append(samples)
                inside = (time >= trial.alignment.span.start) & (time <= trial.alignment.span.end)
                label = (layer.style.label or layer.name).format_map(row.group)
                traces.append(go.Scatter(
                    name=label, legendgroup=label, showlegend=False,
                    x=(time if is_time else x.values)[inside].tolist(), y=y.values[inside].tolist(),
                    mode="lines", line={"color": color, "dash": layer.style.line_dash, "width": 1},
                    opacity=0.3, meta={"layer": layer.name, "group": row.coordinates},
                ))
            span = common_span(item.extent for item in series)
            if span is None:
                continue
            time = aligned_sample_times([item.time for item in series], span=span)
            summary = summarize_series(series, time)
            x_mean, y_mean = np.asarray(summary.mean).T
            if is_time:
                x_mean = time
            labels = {choice.id: choice.label for choice in options.by}
            label_parts = [str(cohort[0].group[name]) for name, _ in coordinates if cohort[0].group[name] is not None]
            label_parts.extend(f"{labels[name]}={value}" for name, value in condition.values.items())
            label = " · ".join(label_parts) or layer.name
            alpha = layer.style.band_alpha if isinstance(layer, BandLayer) else BandStyle().band_alpha
            for name, values in (("upper", summary.upper), ("lower", summary.lower), ("mean", summary.mean)):
                center = name == "mean"
                traces.append(go.Scatter(
                    name=label, legendgroup=label, showlegend=center and layer.style.showlegend,
                    x=x_mean.tolist(), y=(y_mean if center else np.asarray(values)[:, 1]).tolist(), mode="lines",
                    line=(
                        {"color": color, "dash": layer.style.line_dash, "width": layer.style.line_width}
                        if center else {"color": "rgba(0,0,0,0)"}
                    ),
                    fill="tonexty" if name == "lower" else None,
                    fillcolor=color_add_alpha(color, alpha) if name == "lower" else None,
                    hoverinfo=None if center else "skip",
                    meta={"layer": layer.name, "group": group},
                ))
    return traces

def render(
    figure: FigureDocument,
    datasets: Mapping[str, Dataset],
    theme: str | go.layout.Template | Mapping[str, Any] | None = None,
    *,
    names: Mapping[str, Mapping[str, str]] | None = None,
    mean: TrialMeanSelection | None = None,
    recorded_trials: Mapping[str, Mapping[str, Dataset]] | None = None,
) -> go.Figure:
    """Render only the authored bindings and prepared values, printing ``names`` for the value IDs they cover."""
    sources = {
        name: _Source(dataset, dataset.descriptor, names or {})
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
            clock = None
            trial_rows: list[_TrialTrace] = []
            if mean is not None and "trial" in layer.group:
                if recorded_trials is None or layer.role not in recorded_trials:
                    raise ValueError(f"No retained trial clock for data role {layer.role!r}.")
                clock = FigureTrialSource(source.dataset, recorded_trials[layer.role], source.read)
            for group, positions in _groups(source, layer.group, selection):
                channels = {}
                for channel, encoding in layer_encodings(layer):
                    try:
                        channels[channel] = _encoded(source, encoding, positions)
                    except ValueError as error:
                        raise ValueError(f"channel {channel!r}: {error}") from error
                color = layer.style.color or next(colors)
                if clock is not None:
                    record, options = clock.project(positions)
                    coordinates = {}
                    for name in layer.group:
                        dimension, labels = source.coordinate(name)
                        coordinates[name] = labels[positions[dimension][0]]
                    trial_rows.append(_TrialTrace(group, coordinates, channels, color, record, options))
                else:
                    label = (layer.style.label or layer.name).format_map(group)
                    traces[layer.panel].extend(_draw(layer, channels, label, color))
                inferred_labels.setdefault(
                    layer.panel, {axis: channels[axis].label for axis in ("x", "y")}
                )
            if mean is not None and trial_rows:
                traces[layer.panel].extend(_trial_traces(layer, trial_rows, mean))
        except CannotRun:
            raise
        except (KeyError, ValueError) as error:
            raise ValueError(f"Layer {layer.name!r}: {error}") from error
        if len(traces[layer.panel]) == before:
            raise ValueError(f"Layer {layer.name!r} has no plotted groups.")

    return assemble_figure_grid(canvas, figure.layout, figure.view, traces, inferred_labels)
