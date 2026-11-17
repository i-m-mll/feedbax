"""Assemble the Cartesian panels of a dataset figure."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feedbax.contracts.figure_document import FigureDocumentView, FigurePanel
from feedbax.contracts.figure_layout import DataDerivedAxisRange, FigureView, PanelAxis
from feedbax.plot.ranges import calculate_array_minmax


@dataclass(frozen=True)
class _Axis:
    panel: FigurePanel
    dimension: Literal["x", "y"]
    layout: go.layout.XAxis | go.layout.YAxis
    settings: PanelAxis | None

    @property
    def reference(self) -> str:
        return self.layout.plotly_name.replace("axis", "", 1)

    @property
    def declaration(self) -> DataDerivedAxisRange | None:
        return self.settings.derived_range if self.settings is not None else None


def _check_matched_ranges(axes: Sequence[_Axis]) -> None:
    by_ref = {axis.reference: axis for axis in axes}
    matches: dict[str, set[str]] = {ref: set() for ref in by_ref}
    for ref, axis in by_ref.items():
        target = axis.layout.matches
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


def _set_derived_ranges(
    axes: Sequence[_Axis], traces: Mapping[str, Sequence[go.Scatter]],
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
            raise ValueError(f"Data-derived range group {group!r} mixes x and y axes.")
        paddings = {declaration.padding_fraction for _, declaration in members}
        if len(paddings) != 1:
            raise ValueError(f"Data-derived range group {group!r} has conflicting padding fractions.")
        (dimension,) = dimensions
        (padding,) = paddings
        extent = calculate_array_minmax(
            (getattr(trace, dimension) for axis, _ in members for trace in traces[axis.panel.name]),
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
    traces: Mapping[str, Sequence[go.Scatter]],
    inferred_labels: Mapping[str, Mapping[str, str]],
) -> go.Figure:
    rows = max(panel.row for panel in panels)
    cols = max(panel.col for panel in panels)
    titles = [""] * (rows * cols)
    for panel in panels:
        titles[(panel.row - 1) * cols + panel.col - 1] = panel.title or panel.name
    figure = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles, figure=figure,
        shared_xaxes=view.shared_xaxes, shared_yaxes=view.shared_yaxes,
        horizontal_spacing=view.horizontal_spacing, vertical_spacing=view.vertical_spacing,
    )
    axes = []
    for panel in panels:
        for trace in traces[panel.name]:
            figure.add_trace(trace, row=panel.row, col=panel.col)
        subplot = figure.get_subplot(panel.row, panel.col)
        labels = inferred_labels.get(panel.name, {})
        for dimension, layout, settings in (
            ("x", subplot.xaxis, panel.x_axis),
            ("y", subplot.yaxis, panel.y_axis),
        ):
            label = getattr(panel.axes_labels, dimension, None)
            layout.title.text = label if label is not None else labels.get(dimension)
            if settings is not None:
                layout.update(**settings.model_dump(include={"type", "range"}, exclude_none=True))
            axes.append(_Axis(panel, dimension, layout, settings))
    _set_derived_ranges(axes, traces)
    _set_view(figure, view)
    return figure
