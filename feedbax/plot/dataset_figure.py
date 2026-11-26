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
    PointsLayer,
    ReferenceValue,
    RuleLayer,
    Selection,
    SpanLayer,
    TrajectoryLayer,
    VariableBinding,
    color_encoding,
    coordinate_dimension,
    ordinal_key,
    position_encodings,
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
from feedbax.plot.figure_colors import ColorSample, ResolvedColorScale, resolve_color_scale
from feedbax.plot.figure_grid import AxisReference, FigureTrace, assemble_figure_grid
from feedbax.plot.figure_trial_clock import FigureRecordedDataset, FigureTrialSource


type Label = str | int | float | bool | None
type Positions = dict[str, tuple[int, ...]]
type Group = tuple[dict[str, Label], Positions, dict[str, int]]


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

    def group_names(self, group: Mapping[str, Label]) -> dict[str, Label]:
        labels = {name: self.printed(name, value) for name, value in group.items()}
        member = group.get(MEMBER)
        if "condition" in labels and isinstance(member, str):
            # Different saved Evaluate rows can share one semantic condition ID.
            if name := self.names.get(MEMBER, {}).get(member):
                labels["condition"] = name
        return labels


class _EmptyGroupLabel(str):
    def __repr__(self) -> str:
        return ""

    def __format__(self, spec: str) -> str:
        return ""


def _layer_label(layer: FigureLayer, group: Mapping[str, Label]) -> str:
    template = layer.style.label or layer.name
    if None not in group.values():
        return template.format_map(group)
    values = {name: _EmptyGroupLabel() if value is None else value for name, value in group.items()}
    return " ".join(template.format_map(values).split())


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
) -> Iterator[Group]:
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


def _color_sample(source: _Source, encoding: Encoding, group: Group, title: str | None) -> ColorSample:
    _, positions, member = group
    sample = _encoded(source, encoding, positions, member)
    value = sample.values.item(0)
    if not isinstance(value, (str, bool, int, float)) or not np.all(sample.values == value):
        raise ValueError("A color channel requires one scalar value per plotted group.")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("A color channel requires finite numerical values.")
    shown = str(source.printed(sample.name, value))
    axis = sample.field.label or sample.field.quantity
    if axis and (title or sample.label).partition(" (")[0].casefold() == axis.casefold():
        shown = shown.removeprefix(axis + " ")
    return ColorSample(value, shown, sample.label, sample.field.units)


def _color_groups(
    figure: FigureDocument, sources: Mapping[str, _Source],
) -> tuple[dict[str, list[tuple[Group, ColorSample]]], dict[str, ResolvedColorScale], dict[str, FigureLayer]]:
    by_scale: dict[str, list[ColorSample]] = defaultdict(list)
    grouped = {}
    first_layer = {}
    for layer in figure.layers:
        if (color := color_encoding(layer)) is None:
            continue
        source = sources[layer.role]
        try:
            title = figure.color_scales[color.scale].title
            entries = [(group, _color_sample(source, color.field, group, title)) for group in _layer_groups(source, layer)]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Layer {layer.name!r} color field: {error}") from error
        grouped[layer.name] = entries
        by_scale[color.scale].extend(sample for _, sample in entries)
        first_layer.setdefault(color.scale, layer)
    scales = {}
    for name, settings in figure.color_scales.items():
        if samples := by_scale.get(name):
            try:
                scales[name] = resolve_color_scale(settings, samples)
            except ValueError as error:
                raise ValueError(f"Color scale {name!r}: {error}") from error
    return grouped, scales, first_layer


def _color_guides(
    canvas: go.Figure, figure: FigureDocument, scales: Mapping[str, ResolvedColorScale], layers: Mapping[str, FigureLayer],
) -> None:
    panels = {panel.name: panel for panel in figure.layout}
    guides = tuple(name for name in scales if figure.color_scales[name].legend)
    legend = canvas.layout.legend
    if guides and legend.x is None and legend.y is None and legend.orientation is None:
        legend.update(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top")
    for index, name in enumerate(guides):
        scale, layer = scales[name], layers[name]
        panel = panels[layer.panel]
        axis = "coloraxis" if index == 0 else f"coloraxis{index + 1}"
        bar = scale.axis["colorbar"] | {
            "x": 1.02, "y": 1 - (index + 0.5) / len(guides), "len": 0.95 / len(guides),
        }
        canvas.update_layout(**{axis: scale.axis | {"colorbar": bar}})
        spatial = isinstance(layer, TrajectoryLayer) and layer.encode.z is not None
        trace = go.Scatter3d if spatial else go.Scatter
        coordinates = {"x": [None, None], "y": [None, None]} | ({"z": [None, None]} if spatial else {})
        canvas.add_trace(trace(
            **coordinates, name="", mode="markers", showlegend=False, hoverinfo="skip",
            marker={"color": scale.limits, "coloraxis": axis, "opacity": 0},
            meta=FigureTraceMeta(layer=layer.name, group={}).model_dump(mode="json"),
        ), row=panel.row, col=panel.col)


def _check_samples(channels: Mapping[str, _Samples]) -> None:
    x = channels["x"]
    for name, samples in channels.items():
        if (samples.dimension, samples.values.shape, samples.positions) != (
            x.dimension,
            x.values.shape,
            x.positions,
        ):
            raise ValueError(f"Channel {name!r} does not share the x samples.")


def _check_bounds(channels: Mapping[str, _Samples], names: tuple[str, ...]) -> None:
    y = channels["y"].field
    for name in names:
        field = channels[name].field
        if (field.units, field.frame) != (y.units, y.frame):
            raise ValueError(f"Channel {name!r} must use the y channel's units and frame.")


def _path_trace(
    values: Mapping[str, Sequence[float] | np.ndarray],
    layer: LineLayer | BandLayer | TrajectoryLayer, label: str, color: str, group: dict[str, Label], *, thin: bool = False,
) -> FigureTrace:
    trace = go.Scatter3d if "z" in values else go.Scatter
    return trace(
        **values, name=label, legendgroup=label, showlegend=False if thin else layer.style.showlegend,
        mode="lines", line={"color": color, "dash": layer.style.line_dash, "width": 1 if thin else layer.style.line_width},
        opacity=0.3 if thin else None,
        meta=FigureTraceMeta(layer=layer.name, group=group).model_dump(mode="json"),
    )


def _draw(
    layer: LineLayer | BandLayer | PointsLayer | TrajectoryLayer,
    channels: Mapping[str, _Samples],
    label: str,
    color: str,
    *,
    group: dict[str, Label],
) -> tuple[FigureTrace, ...]:
    x, y = channels["x"], channels["y"]
    if isinstance(layer, TrajectoryLayer):
        return (_path_trace({name: channel.values for name, channel in channels.items()}, layer, label, color, group),)
    meta = FigureTraceMeta(layer=layer.name, group=group).model_dump(mode="json")
    if isinstance(layer, PointsLayer):
        error_y = None
        if layer.encode.error_y is not None:
            _check_bounds(channels, ("error_y.lower", "error_y.upper"))
            up = channels["error_y.upper"].values - y.values
            down = y.values - channels["error_y.lower"].values
            if not (np.isfinite(up) & np.isfinite(down) & (up >= 0) & (down >= 0)).all():
                raise ValueError("Error bar bounds must be finite and enclose their point.")
            error_y = {"type": "data", "symmetric": False, "color": color, "array": up, "arrayminus": down}
        return (go.Scatter(
            name=label,
            legendgroup=label,
            showlegend=layer.style.showlegend,
            x=x.values,
            y=y.values,
            mode="markers",
            marker={"color": color, "size": layer.style.marker_size},
            error_y=error_y,
            meta=meta,
        ),)
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

    _check_bounds(channels, ("lower", "upper"))
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


def _position(
    source: _Source, value: ReferenceValue, positions: Positions, member: Mapping[str, int],
) -> np.ndarray:
    if isinstance(value, float):
        return np.array([value])
    return _encoded(source, value, positions, member).values


def _references(
    source: _Source,
    layer: RuleLayer | SpanLayer,
    positions: Positions,
    member: Mapping[str, int],
    label: str,
    color: str,
) -> tuple[AxisReference, ...]:
    """Each rule position, or each span between its start and end, read for one group."""
    line = {"color": color, "dash": layer.style.line_dash, "width": layer.style.line_width}
    legend = layer.style.showlegend is not False
    if isinstance(layer, RuleLayer):
        at = _position(source, layer.encode.at, positions, member)
        return tuple(
            AxisReference(layer.encode.axis, float(value), float(value), label, legend, line, None)
            for value in at
        )
    starts, ends = (
        _position(source, value, positions, member) for value in (layer.encode.start, layer.encode.end)
    )
    if starts.shape != ends.shape and starts.size != 1 and ends.size != 1:
        raise ValueError("Span start and end do not share their samples.")
    if (ends < starts).any():
        raise ValueError("Span end must not precede its start.")
    fill = color_add_alpha(color, layer.style.fill_alpha)
    edge = line if layer.style.line_width is not None else line | {"width": 0}
    return tuple(
        AxisReference(layer.encode.axis, float(start), float(end), label, legend, edge, fill)
        for start, end in zip(*np.broadcast_arrays(starts, ends), strict=True)
    )


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
    traces: list[FigureTrace]


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
            values[name].add(canonical_json_v2_bytes(legend.parts.get(name, value) if name == "condition" else value))
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
    layer: LineLayer | BandLayer | TrajectoryLayer,
    rows: list[_TrialTrace],
    mean: TrialMeanSelection,
    legends: list[_MeanLegend],
) -> list[FigureTrace]:
    """Draw each trial and its clock-aligned summary on the same position axes."""
    nontrial: dict[tuple[tuple[str, Label], ...], list[_TrialTrace]] = defaultdict(list)
    for row in rows:
        nontrial[tuple((name, value) for name, value in row.coordinates.items() if name != "trial")].append(row)
    traces = []
    has_color = color_encoding(layer) is not None
    for coordinates, cohort in nontrial.items():
        options = cohort[0].options
        if any(row.options != options for row in cohort):
            raise ValueError("Plotted trials disagree on the available By and Align choices.")
        by_trial = {row.record.id: row for row in cohort}
        is_time = cohort[0].channels["x"].field.trial_time is not None
        by_labels = {choice.id: choice.label for choice in options.by}
        for condition in align_trials((row.record for row in cohort), options, mean):
            colors = {by_trial[trial.record.id].color for trial in condition.trials}
            color = (
                by_trial[condition.trials[0].record.id].color if not has_color or len(colors) == 1 else "#666666"
            )
            series = []
            group = dict(coordinates)
            parts = {name: str(cohort[0].group[name]) for name in group if cohort[0].group[name] is not None}
            for name, value in condition.values.items():
                key = f"task/{name}" if name == "trial" or name in group else name
                group[key] = canonical_json_v2_bytes(value).decode() if isinstance(value, (list, dict)) else value
                parts[key] = f"{by_labels[name]}={_by_value_label(value)}"
            for trial in condition.trials:
                row = by_trial[trial.record.id]
                positions = {axis: row.channels[axis].values for axis in ("x", "y", "z") if axis in row.channels}
                clock = row.clock
                if clock is None:
                    time = np.array([0.0])
                else:
                    source_time = clock.field.trial_time
                    assert source_time is not None
                    recorded = clock.values + trial_origin(trial.record, source_time.zero_epoch)
                    time = trial.times(recorded)
                if any(values.shape != time.shape for values in positions.values()):
                    raise ValueError("The plotted channels do not share the recorded trial samples.")
                samples = AlignedSeries(time, np.column_stack(tuple(positions.values())), trial.alignment.span)
                series.append(samples)
                inside = (time >= trial.alignment.span.start) & (time <= trial.alignment.span.end)
                label = _layer_label(layer, row.group)
                if is_time:
                    positions["x"] = time
                traces.append(_path_trace(
                    {axis: values[inside].tolist() for axis, values in positions.items()}, layer, label,
                    row.color if has_color else color, row.coordinates, thin=True,
                ))
            span = common_span(item.extent for item in series)
            if span is None:
                continue
            time = aligned_sample_times([item.time for item in series], span=span)
            summary = summarize_series(series, time)
            if isinstance(layer, TrajectoryLayer):
                axes = ("x", "y", "z") if layer.encode.z is not None else ("x", "y")
                mean_path = dict(zip(axes, np.asarray(summary.mean).T, strict=True))
                if is_time:
                    mean_path["x"] = time
                trace = _path_trace({axis: value.tolist() for axis, value in mean_path.items()}, layer, layer.name, color, group)
                traces.append(trace)
                legends.append(_MeanLegend(layer.name, group, parts, [trace]))
                continue
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
    color_groups, scales, guide_layers = _color_groups(figure, sources)
    traces: dict[str, list[FigureTrace]] = {panel.name: [] for panel in figure.layout}
    references: dict[str, list[AxisReference]] = {panel.name: [] for panel in figure.layout}
    inferred_labels: dict[str, dict[str, str]] = {}
    panel_times: dict[str, TrialFigureTime | None] = {}
    mean_panel_times: dict[str, TrialFigureTime] = {}
    mean_legends: list[_MeanLegend] = []
    for layer in figure.layers:
        source = sources[layer.role]
        before = len(traces[layer.panel]) + len(references[layer.panel])
        try:
            if isinstance(layer, RuleLayer | SpanLayer):
                for group, positions, member in _layer_groups(source, layer):
                    printed = source.group_names(group)
                    label = _layer_label(layer, printed)
                    references[layer.panel].extend(
                        _references(source, layer, positions, member, label, layer.style.color or next(colors))
                    )
            else:
                clock = None
                trial_rows: list[_TrialTrace] = []
                layer_legends: list[tuple[str, Label, tuple[FigureTrace, ...]]] = []
                color_spec = color_encoding(layer)
                if "trial" in layer.group and mean is not None and not isinstance(layer, PointsLayer):
                    if recorded_trials is None or layer.role not in recorded_trials:
                        raise ValueError(f"No retained trial clock for data role {layer.role!r}.")
                    clock = FigureTrialSource(source.dataset, recorded_trials[layer.role], source.read)
                groups = color_groups[layer.name] if color_spec is not None else (
                    (group, None) for group in _layer_groups(source, layer)
                )
                for (group, positions, member), palette_value in groups:
                    channels = {}
                    for channel, encoding in position_encodings(layer):
                        try:
                            channels[channel] = _encoded(source, encoding, positions, member)
                        except ValueError as error:
                            raise ValueError(f"channel {channel!r}: {error}") from error
                    _check_samples(channels)
                    if isinstance(layer, TrajectoryLayer) and any(
                        channel.values.dtype.kind not in "fiu" for channel in channels.values()
                    ):
                        raise ValueError("A trajectory requires numerical position channels.")
                    printed = source.group_names(group)
                    if "trial" in group:
                        dimension, _ = source.coordinate("trial", member)
                        printed["trial"] = positions[dimension][0] + 1
                    color = (
                        scales[color_spec.scale].colors[ordinal_key(palette_value.value)]
                        if color_spec is not None and palette_value is not None else layer.style.color or next(colors)
                    )
                    if clock is not None:
                        record, options = clock.project(positions)
                        trial_rows.append(_TrialTrace(
                            printed, group, channels, color, _trial_samples(source, channels["x"], positions), record, options,
                        ))
                    else:
                        label = _layer_label(layer, printed)
                        drawn = _draw(layer, channels, label, color, group=group)
                        layer_legends.append((label, group.get(MEMBER), drawn))
                        traces[layer.panel].extend(drawn)
                    time = channels["x"].field.trial_time
                    if panel_times.setdefault(layer.panel, time) != time:
                        panel_times[layer.panel] = None
                    inferred_labels.setdefault(
                        layer.panel, {axis: channels[axis].label for axis in ("x", "y", "z") if axis in channels}
                    )
                member_labels = _member_labels((label, member) for label, member, _ in layer_legends)
                for label, member, drawn in layer_legends:
                    display = member_labels.get((label, member), label)
                    if display != label:
                        drawn[0].update(name=display)
                        for trace in drawn:
                            trace.update(legendgroup=display)
                if mean is not None and trial_rows and not isinstance(layer, PointsLayer):
                    traces[layer.panel].extend(_trial_traces(layer, trial_rows, mean, mean_legends))
                    if trial_rows[0].channels["x"].field.trial_time is not None:
                        mean_panel_times[layer.panel] = TrialFigureTime(zero_epoch=mean.align)
        except CannotRun:
            raise
        except (KeyError, ValueError) as error:
            raise ValueError(f"Layer {layer.name!r}: {error}") from error
        if len(traces[layer.panel]) + len(references[layer.panel]) == before:
            raise ValueError(f"Layer {layer.name!r} has no plotted groups.")

    _label_means(mean_legends)
    canvas = assemble_figure_grid(
        canvas, figure.layout, figure.view, traces, inferred_labels,
        references=references, panel_times=panel_times | mean_panel_times, runs=runs,
    )
    _color_guides(canvas, figure, scales, guide_layers)
    return canvas
