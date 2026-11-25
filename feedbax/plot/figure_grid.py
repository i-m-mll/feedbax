"""Assemble Cartesian and spatial panels of a dataset figure."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feedbax.contracts.dataset import TrialFigureTime
from feedbax.contracts.figure_document import FigureDocumentView, FigurePanel
from feedbax.contracts.figure_layout import DataDerivedAxisRange, FigureView, PanelAxis
from feedbax.contracts.studio_figure import FigureLayoutMeta, FigurePanelMeta, FigureRun
from feedbax.plot.ranges import calculate_array_minmax


type FigureTrace = go.Scatter | go.Scatter3d
type AxisDimension = Literal["x", "y", "z"]


@dataclass(frozen=True)
class AxisReference:
    """A rule (``start == end``) or a span of one axis, drawn across the other axis of its panel."""

    axis: Literal["x", "y"]
    start: float
    end: float
    label: str
    legend: bool
    line: Mapping[str, Any]
    fill: str | None
    """The span's fill colour; a rule has none."""

    def shape(self, refs: Mapping[Literal["x", "y"], str], *, first: bool) -> dict[str, Any]:
        across = "y" if self.axis == "x" else "x"
        return {
            "type": "line" if self.fill is None else "rect",
            f"{self.axis}ref": refs[self.axis], f"{self.axis}0": self.start, f"{self.axis}1": self.end,
            f"{across}ref": f"{refs[across]} domain", f"{across}0": 0, f"{across}1": 1,
            "line": dict(self.line),
            "fillcolor": self.fill,
            "layer": "above" if self.fill is None else "below",
            "name": self.label,
            "legendgroup": self.label,
            "showlegend": self.legend and first,
        }


@dataclass(frozen=True)
class _Axis:
    panel: FigurePanel
    dimension: AxisDimension
    layout: go.layout.XAxis | go.layout.YAxis | go.layout.scene.XAxis | go.layout.scene.YAxis | go.layout.scene.ZAxis
    settings: PanelAxis | None
    scene: str | None = None

    @property
    def reference(self) -> str:
        if self.scene is not None:
            return f"{self.scene}.{self.dimension}axis"
        return self.layout.plotly_name.replace("axis", "", 1)

    @property
    def declaration(self) -> DataDerivedAxisRange | None:
        return self.settings.derived_range if self.settings is not None else None


def _check_matched_ranges(axes: Sequence[_Axis]) -> None:
    by_ref = {axis.reference: axis for axis in axes}
    matches: dict[str, set[str]] = {ref: set() for ref in by_ref}
    for ref, axis in by_ref.items():
        target = getattr(axis.layout, "matches", None)
        if target in matches:
            matches[ref].add(target)
            matches[target].add(ref)
    remaining = set(matches)
    while remaining:
        pending = [remaining.pop()]
        component = []
        while pending:
            ref = pending.pop()
            component.append(by_ref[ref])
            neighbours = matches[ref] & remaining
            remaining -= neighbours
            pending.extend(neighbours)
        groups = {axis.declaration.group for axis in component if axis.declaration is not None}
        if not groups:
            continue
        if len(groups) > 1:
            raise ValueError(
                f"A Plotly shared axis spans multiple data-derived range groups: {sorted(groups)}."
            )
        undeclared = sorted(
            axis.panel.name for axis in component if axis.declaration is None
        )
        if undeclared:
            (group,) = groups
            raise ValueError(
                f"Data-derived range group {group!r} matches undeclared panels {undeclared}."
            )


def _extents(
    dimension: AxisDimension, traces: Sequence[FigureTrace], references: Sequence[AxisReference],
) -> Iterator[Sequence[float]]:
    """The values each trace and reference reaches on one axis, error bars included."""
    for trace in traces:
        values = getattr(trace, dimension)
        yield values
        error = getattr(trace, f"error_{dimension}")
        if error.array is not None:
            center = np.asarray(values, dtype=float)
            yield center + np.asarray(error.array, dtype=float)
            yield center - np.asarray(error.arrayminus, dtype=float)
    for reference in references:
        if reference.axis == dimension:
            yield (reference.start, reference.end)


def _set_derived_ranges(
    axes: Sequence[_Axis], traces: Mapping[str, Sequence[FigureTrace]], references: Mapping[str, Sequence[AxisReference]],
) -> None:
    groups: dict[str, list[tuple[_Axis, DataDerivedAxisRange]]] = {}
    for axis in axes:
        if (declaration := axis.declaration) is not None:
            groups.setdefault(declaration.group, []).append((axis, declaration))
    if not groups:
        return

    _check_matched_ranges(axes)
    for group, members in groups.items():
        if len(members) < 2:
            raise ValueError(f"Data-derived range group {group!r} requires at least two panel axes.")
        dimensions = {axis.dimension for axis, _ in members}
        if len(dimensions) != 1:
            raise ValueError(f"Data-derived range group {group!r} mixes different axes.")
        paddings = {declaration.padding_fraction for _, declaration in members}
        if len(paddings) != 1:
            raise ValueError(f"Data-derived range group {group!r} has conflicting padding fractions.")
        (dimension,) = dimensions
        (padding,) = paddings
        extent = calculate_array_minmax(
            (
                values for axis, _ in members
                for values in _extents(dimension, traces[axis.panel.name], references.get(axis.panel.name, ()))
            ),
            padding=padding,
        )
        for axis, _ in members:
            axis.layout.range = extent


def _set_view(figure: go.Figure, view: FigureDocumentView) -> None:
    layout = view.model_dump(
        include=set(FigureView.model_fields) - {"legend_gutter"}, exclude_none=True,
    )
    if "title" in layout:
        layout["title_text"] = layout.pop("title")
    if view.legend_gutter is not None and view.legend_x is None:
        if view.width is None:
            raise ValueError("legend_gutter requires an explicit figure width")
        margins = layout.get("margin", {})
        template = figure.layout.template.layout.margin
        inherited = {side: getattr(template, side) for side in ("l", "r")}
        plot_width = view.width - sum(
            margins.get(side, value if value is not None else 80)
            for side, value in inherited.items()
        )
        if plot_width <= 0:
            raise ValueError("figure width must exceed its left and right margins")
        layout["legend_x"] = 1.0 + view.legend_gutter / plot_width
    figure.update_layout(**layout)


def assemble_figure_grid(
    figure: go.Figure,
    panels: Sequence[FigurePanel],
    view: FigureDocumentView,
    traces: Mapping[str, Sequence[FigureTrace]],
    inferred_labels: Mapping[str, Mapping[str, str]],
    *,
    references: Mapping[str, Sequence[AxisReference]] = MappingProxyType({}),
    panel_times: Mapping[str, TrialFigureTime | None] = MappingProxyType({}),
    runs: Mapping[str, tuple[FigureRun, ...]] = MappingProxyType({}),
) -> go.Figure:
    rows = max(panel.row for panel in panels)
    cols = max(panel.col for panel in panels)
    titles = [""] * (rows * cols)
    for panel in panels:
        titles[(panel.row - 1) * cols + panel.col - 1] = panel.title or panel.name
    spatial = {(panel.row - 1, panel.col - 1) for panel in panels
               if any(isinstance(trace, go.Scatter3d) for trace in traces[panel.name])}
    specs = [
        [{"type": "scene"} if (row, col) in spatial else {"type": "xy"} for col in range(cols)]
        for row in range(rows)
    ] if spatial else None
    figure = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles, figure=figure, specs=specs,
        shared_xaxes=view.shared_xaxes, shared_yaxes=view.shared_yaxes,
        horizontal_spacing=view.horizontal_spacing, vertical_spacing=view.vertical_spacing,
    )
    axes = []
    panel_meta = {}
    legends: set[str] = set()
    for panel in panels:
        for trace in traces[panel.name]:
            figure.add_trace(trace, row=panel.row, col=panel.col)
        subplot = figure.get_subplot(panel.row, panel.col)
        scene = subplot.plotly_name if isinstance(subplot, go.layout.Scene) else None
        x = _Axis(panel, "x", subplot.xaxis, panel.x_axis, scene)
        y = _Axis(panel, "y", subplot.yaxis, panel.y_axis, scene)
        z = _Axis(panel, "z", subplot.zaxis, panel.z_axis, scene) if scene is not None else None
        for reference in references.get(panel.name, ()):
            figure.add_shape(reference.shape({"x": x.reference, "y": y.reference}, first=reference.label not in legends))
            legends.add(reference.label)
        labels = inferred_labels.get(panel.name, {})
        for axis in (x, y, z) if z is not None else (x, y):
            label = getattr(panel.axes_labels, axis.dimension, None)
            axis.layout.title.text = label if label is not None else labels.get(axis.dimension)
            if axis.settings is not None:
                axis.layout.update(**axis.settings.model_dump(include={"type", "range"}, exclude_none=True))
            axes.append(axis)
        panel_meta[panel.name] = FigurePanelMeta(
            x_axis=x.reference, y_axis=y.reference, z_axis=z.reference if z is not None else None,
            time=panel_times.get(panel.name),
        )
    _set_derived_ranges(axes, traces, references)
    _set_view(figure, view)
    figure.layout.meta = FigureLayoutMeta(panels=panel_meta, runs=dict(runs)).model_dump(
        mode="json", exclude_none=True,
    )
    return figure
