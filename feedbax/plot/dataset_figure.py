"""Draw a figure document from prepared variables in dataset coordinate order."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import cycle
from types import MappingProxyType
from typing import Any, assert_never

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from pydantic import JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import DatasetDescriptor, FieldDescriptor, TrialFigureTime
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
from feedbax.contracts.member_stack import MEMBER
from feedbax.contracts.rollout import CannotRun
from feedbax.contracts.studio_figure import FigureRun, FigureTraceMeta
from feedbax.contracts.trial_clock import TrialMeanOptions, TrialMeanSelection
from feedbax.dataset import Dataset
from feedbax.execution.trial_clock import (
    AlignedSeries, TrialClockRecord, align_trials, aligned_sample_times, common_span, summarize_series, trial_origin,
)
from feedbax.plot.colors import color_add_alpha
from feedbax.plot.figure_grid import assemble_figure_grid
from feedbax.plot.figure_trial_clock import FigureRecordedDataset, FigureTrialSource


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

    def coordinate(self, name: str, member: Mapping[str, int]) -> tuple[str, list[Label]]:
        field = self.descriptor.coords[name]
        data = self.read(name)
        inner = tuple(dim for dim in field.dims if dim not in member)
        if len(field.dims) > 1 and inner != field.dims and inner:
            data = data[tuple(member.get(dim, slice(None)) for dim in field.dims)]
            field = field.model_copy(update={"dims": inner})
        dimension = coordinate_dimension(field, self.descriptor.dimensions)
        if dimension is None:
            raise ValueError(f"Coordinate {name!r} labels more than one dimension.")
        labels = data.reshape(-1).tolist()
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
    member: Mapping[str, int],
) -> Positions:
    result = positions.copy()
    for name, selection in selections.items():
        dimension, labels = source.coordinate(name, member)
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
    member: Mapping[str, int],
) -> Iterator[tuple[dict[str, Label], Positions]]:
    if not coordinates:
        yield {}, positions
        return

    name, *rest = coordinates
    dimension, labels = source.coordinate(name, member)
    groups: dict[tuple[bool, Label], list[int]] = {}
    for index in positions.get(dimension, range(len(labels))):
        value = labels[index]
        groups.setdefault((isinstance(value, bool), value), []).append(index)
    for (_, value), indices in groups.items():
        for labels_by_name, group in _groups(
            source, tuple(rest), positions | {dimension: tuple(indices)}, member,
        ):
            yield {name: value} | labels_by_name, group


def _layer_groups(
    source: _Source, layer: FigureLayer,
) -> Iterator[tuple[dict[str, Label], Positions, dict[str, int]]]:
    member = source.descriptor.coords.get(MEMBER)
    coordinates = tuple(name for name in layer.group if name != MEMBER)
    if MEMBER not in layer.group or member is None or not member.dims:
        for group, positions in _groups(source, coordinates, _selected(source, layer.select, {}, {}), {}):
            yield group, positions, {}
        return

    for indices, value in np.ndenumerate(source.read(MEMBER)):
        fixed = dict(zip(member.dims, indices, strict=True))
        positions = _selected(source, layer.select, {dim: (index,) for dim, index in fixed.items()}, fixed)
        for group, positions in _groups(source, coordinates, positions, fixed):
            yield {MEMBER: str(value)} | group, positions, fixed


@dataclass(frozen=True)
class _Samples:
    name: str
    field: FieldDescriptor
    dimension: str | None
    values: np.ndarray
    positions: tuple[int, ...] | None

    @property
    def label(self) -> str:
        title = self.field.label or self.field.quantity or self.name.replace("_", " ")
        return f"{title} ({self.field.units})" if self.field.units else title


def _encoded(
    source: _Source, encoding: Encoding, positions: Positions, member: Mapping[str, int],
) -> _Samples:
    if isinstance(encoding, VariableBinding):
        name = encoding.variable
        positions = _selected(source, encoding.select, positions, member)
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


def _check_samples(channels: Mapping[str, _Samples]) -> None:
    x = channels["x"]
    for name, samples in channels.items():
        if (samples.dimension, samples.values.shape, samples.positions) != (
            x.dimension,
            x.values.shape,
            x.positions,
        ):
            raise ValueError(f"Channel {name!r} does not share the x samples.")


def _draw(
    layer: FigureLayer,
    channels: Mapping[str, _Samples],
    label: str,
    color: str,
    *,
    group: dict[str, Label],
) -> tuple[go.Scatter, ...]:
    x, y = channels["x"], channels["y"]
    meta = FigureTraceMeta(layer=layer.name, group=group).model_dump(mode="json")
    center = go.Scatter(
        name=label,
        legendgroup=label,
        showlegend=layer.style.showlegend,
        x=x.values,
        y=y.values,
        mode="lines",
        line={"color": color, "dash": layer.style.line_dash, "width": layer.style.line_width},
        meta=meta,
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
            meta=meta,
        )
        for name in ("upper", "lower")
    )
    lower.update(fill="tonexty", fillcolor=color_add_alpha(color, layer.style.band_alpha))
    return center, upper, lower



def _trial_samples(source: _Source, x: _Samples, positions: Positions) -> _Samples | None:
    """Use the bound x clock, or the time coordinate of a trajectory's sample axis."""
    if x.field.trial_time is not None:
        return x
    if x.dimension is None:
        return None
    matching = [
        name for name, field in source.descriptor.coords.items()
        if field.trial_time is not None and x.dimension in field.dims
    ]
    if "transition_start" in matching:
        # A transition is indexed by its left edge unless the layer explicitly draws another time field.
        return _encoded(source, "transition_start", positions, {})
    if len(matching) == 1:
        return _encoded(source, matching[0], positions, {})
    raise ValueError(f"Series dimension {x.dimension!r} does not identify one recorded trial clock.")


@dataclass(frozen=True)
class _TrialTrace:
    group: dict[str, Label]
    coordinates: dict[str, Label]
    channels: dict[str, _Samples]
    color: str
    clock: _Samples | None
    record: TrialClockRecord
    options: TrialMeanOptions


@dataclass(frozen=True)
class _MeanLegend:
    layer: str
    group: dict[str, Label]
    parts: dict[str, str]
    traces: list[go.Scatter]


def _by_value_label(value: JsonValue) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, list):
        return "[" + ", ".join(map(_by_value_label, value)) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_by_value_label(item)}" for key, item in value.items()) + "}"
    return value if isinstance(value, str) else canonical_json_v2_bytes(value).decode()


def _member_labels(entries: Iterable[tuple[str, Label]]) -> dict[tuple[str, Label], str]:
    """Number members only where they would otherwise have the same legend label."""
    by_label: dict[str, dict[Label, int]] = defaultdict(dict)
    for label, member in entries:
        if member is not None:
            members = by_label[label]
            members.setdefault(member, len(members) + 1)
    return {
        (label, member): f"{label} · Run {index}"
        for label, members in by_label.items() if len(members) > 1
        for member, index in members.items()
    }


def _label_means(legends: list[_MeanLegend]) -> None:
    values: dict[str, set[bytes]] = defaultdict(set)
    for legend in legends:
        for name, value in legend.group.items():
            values[name].add(canonical_json_v2_bytes(value))
    names = [
        " · ".join(text for name, text in legend.parts.items() if name != MEMBER and len(values[name]) > 1)
        or legend.layer
        for legend in legends
    ]
    labels = _member_labels((name, legend.group.get(MEMBER)) for name, legend in zip(names, legends, strict=True))
    for name, legend in zip(names, legends, strict=True):
        label = labels.get((name, legend.group.get(MEMBER)), name)
        # Distinct groups may round to the same label. Keep their legend controls independent.
        legend_group = canonical_json_v2_bytes([label, legend.group]).decode()
        for trace in legend.traces:
            trace.update(name=label, legendgroup=legend_group)


def _trial_traces(
    layer: FigureLayer,
    rows: list[_TrialTrace],
    mean: TrialMeanSelection,
    legends: list[_MeanLegend],
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
        is_time = cohort[0].channels["x"].field.trial_time is not None
        by_labels = {choice.id: choice.label for choice in options.by}
        for condition in align_trials((row.record for row in cohort), options, mean):
            color = by_trial[condition.trials[0].record.id].color
            series = []
            group = dict(coordinates)
            parts = {name: str(cohort[0].group[name]) for name in group if cohort[0].group[name] is not None}
            for name, value in condition.values.items():
                key = f"task/{name}" if name == "trial" or name in group else name
                group[key] = canonical_json_v2_bytes(value).decode() if isinstance(value, (list, dict)) else value
                parts[key] = f"{by_labels[name]}={_by_value_label(value)}"
            for trial in condition.trials:
                row = by_trial[trial.record.id]
                x, y = row.channels["x"], row.channels["y"]
                clock = row.clock
                if clock is None:
                    time = np.array([0.0])
                else:
                    source_time = clock.field.trial_time
                    assert source_time is not None
                    recorded = clock.values + trial_origin(trial.record, source_time.zero_epoch)
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
                    opacity=0.3, meta=FigureTraceMeta(layer=layer.name, group=row.coordinates).model_dump(mode="json"),
                ))
            span = common_span(item.extent for item in series)
            if span is None:
                continue
            time = aligned_sample_times([item.time for item in series], span=span)
            summary = summarize_series(series, time)
            x_mean, y_mean = np.asarray(summary.mean).T
            if is_time:
                x_mean = time
            alpha = layer.style.band_alpha if isinstance(layer, BandLayer) else BandStyle().band_alpha
            summary_traces = []
            for name, values in (("upper", summary.upper), ("lower", summary.lower), ("mean", summary.mean)):
                center = name == "mean"
                summary_traces.append(go.Scatter(
                    showlegend=center and layer.style.showlegend,
                    x=x_mean.tolist(), y=(y_mean if center else np.asarray(values)[:, 1]).tolist(), mode="lines",
                    line=(
                        {"color": color, "dash": layer.style.line_dash, "width": layer.style.line_width}
                        if center else {"color": "rgba(0,0,0,0)"}
                    ),
                    fill="tonexty" if name == "lower" else None,
                    fillcolor=color_add_alpha(color, alpha) if name == "lower" else None,
                    hoverinfo=None if center else "skip",
                    meta=FigureTraceMeta(layer=layer.name, group=group).model_dump(mode="json"),
                ))
            traces.extend(summary_traces)
            legends.append(_MeanLegend(layer.name, group, parts, summary_traces))
    return traces

def render(
    figure: FigureDocument,
    datasets: Mapping[str, Dataset],
    theme: str | go.layout.Template | Mapping[str, Any] | None = None,
    *,
    names: Mapping[str, Mapping[str, str]] | None = None,
    mean: TrialMeanSelection | None = None,
    recorded_trials: Mapping[str, Sequence[FigureRecordedDataset]] | None = None,
    runs: Mapping[str, tuple[FigureRun, ...]] = MappingProxyType({}),
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
    panel_times: dict[str, TrialFigureTime | None] = {}
    mean_panel_times: dict[str, TrialFigureTime] = {}
    mean_legends: list[_MeanLegend] = []
    for layer in figure.layers:
        source = sources[layer.role]
        before = len(traces[layer.panel])
        try:
            clock = None
            trial_rows: list[_TrialTrace] = []
            layer_legends: list[tuple[str, Label, tuple[go.Scatter, ...]]] = []
            if "trial" in layer.group and mean is not None:
                if recorded_trials is None or layer.role not in recorded_trials:
                    raise ValueError(f"No retained trial clock for data role {layer.role!r}.")
                clock = FigureTrialSource(source.dataset, recorded_trials[layer.role], source.read)
            for group, positions, member in _layer_groups(source, layer):
                channels = {}
                for channel, encoding in layer_encodings(layer):
                    try:
                        channels[channel] = _encoded(source, encoding, positions, member)
                    except ValueError as error:
                        raise ValueError(f"channel {channel!r}: {error}") from error
                _check_samples(channels)
                printed = {name: source.printed(name, value) for name, value in group.items()}
                if "trial" in group:
                    dimension, _ = source.coordinate("trial", member)
                    printed["trial"] = positions[dimension][0] + 1
                color = layer.style.color or next(colors)
                if clock is not None:
                    record, options = clock.project(positions)
                    trial_rows.append(_TrialTrace(
                        printed, group, channels, color, _trial_samples(source, channels["x"], positions), record, options,
                    ))
                else:
                    label = (layer.style.label or layer.name).format_map(printed)
                    drawn = _draw(layer, channels, label, color, group=group)
                    layer_legends.append((label, group.get(MEMBER), drawn))
                    traces[layer.panel].extend(drawn)
                time = channels["x"].field.trial_time
                if panel_times.setdefault(layer.panel, time) != time:
                    panel_times[layer.panel] = None
                inferred_labels.setdefault(
                    layer.panel, {axis: channels[axis].label for axis in ("x", "y")}
                )
            member_labels = _member_labels((label, member) for label, member, _ in layer_legends)
            for label, member, drawn in layer_legends:
                display = member_labels.get((label, member), label)
                if display != label:
                    drawn[0].update(name=display)
                    for trace in drawn:
                        trace.update(legendgroup=display)
            if mean is not None and trial_rows:
                traces[layer.panel].extend(_trial_traces(layer, trial_rows, mean, mean_legends))
                if trial_rows[0].channels["x"].field.trial_time is not None:
                    mean_panel_times[layer.panel] = TrialFigureTime(zero_epoch=mean.align)
        except CannotRun:
            raise
        except (KeyError, ValueError) as error:
            raise ValueError(f"Layer {layer.name!r}: {error}") from error
        if len(traces[layer.panel]) == before:
            raise ValueError(f"Layer {layer.name!r} has no plotted groups.")

    _label_means(mean_legends)
    return assemble_figure_grid(
        canvas, figure.layout, figure.view, traces, inferred_labels,
        panel_times=panel_times | mean_panel_times, runs=runs,
    )
