"""Registered constructors for declarative figure execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any, Literal

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from feedbax.contracts.manifest import StrictModel
from feedbax.plot.colors import color_add_alpha

ConstructorTier = Literal["trace", "panel", "figure", "custom_figure"]


class EmptyParams(StrictModel):
    """Default params model for constructors without typed params."""


class ProfileParams(StrictModel):
    """Defaults for one-dimensional profile traces."""

    label: str | None = None
    color: str | None = None
    error_bars_alpha: float = 0.2
    n_std_plot: float = 1.0
    stride_curves: int = 1
    mode: Literal["std", "curves"] = "std"


class Trajectory2DParams(StrictModel):
    """Defaults for 2D trajectory traces."""

    label: str | None = None
    color: str | None = None
    stride: int = 1
    show_mean: bool = True
    opacity: float = 0.4
    line_width: float = 0.75
    mean_line_width: float = 2.5


class EndpointMarkerParams(StrictModel):
    """Defaults for endpoint marker traces."""

    label: str = "Endpoints"
    color: str = "rgb(25, 25, 25)"
    marker_size: int = 7
    straight_guides: bool = True


class ComparisonGridParams(StrictModel):
    """Defaults for panel population."""

    shared_xaxes: bool | str = False
    shared_yaxes: bool | str = "all"
    horizontal_spacing: float | None = None
    vertical_spacing: float | None = None


class GridFigureParams(StrictModel):
    """Defaults for whole-figure layout."""

    panel_constructor: str = "feedbax.comparison_grid"
    width: int | None = None
    height: int | None = None
    title: str | None = None
    legend_tracegroupgap: int | None = None


class Trajectories2DRowParams(StrictModel):
    """House style for 2D effector trajectory rows."""

    panel_constructor: str = "feedbax.comparison_grid"
    width_base: int = 100
    width_per_panel: int = 300
    height: int = 400
    legend_tracegroupgap: int = 1


@dataclass(frozen=True)
class PanelContent:
    """Resolved traces and layout metadata for one panel."""

    name: str
    traces: tuple[Any, ...] = ()
    title: str | None = None
    row: int | None = None
    col: int | None = None
    axes_labels: Mapping[str, str | None] | None = None


TraceConstructor = Callable[[Mapping[str, Any], StrictModel], Sequence[Any]]
PanelConstructor = Callable[[Sequence[PanelContent], StrictModel], go.Figure]
FigureConstructor = Callable[[go.Figure, Sequence[PanelContent], StrictModel], go.Figure]
CustomFigureConstructor = Callable[[Mapping[str, Any], StrictModel], go.Figure]


@dataclass(frozen=True)
class FigureConstructorRegistration:
    """One registered code-bearing figure constructor."""

    key: str
    tier: ConstructorTier
    callable: TraceConstructor | PanelConstructor | FigureConstructor | CustomFigureConstructor
    params_model: type[StrictModel] = EmptyParams
    description: str = ""
    version: str = "v1"

    def params(self, value: Mapping[str, Any] | None = None) -> StrictModel:
        return self.params_model.model_validate(dict(value or {}))


_CONSTRUCTORS: dict[str, FigureConstructorRegistration] = {}
_TEMPLATES: dict[str, Any] = {}
_PIECES: dict[str, Any] = {}
_TYPE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


def _validate_namespaced_type_key(type_key: str, *, field: str) -> str:
    if not isinstance(type_key, str):
        raise ValueError(f"{field} must be a string; got {type(type_key).__name__}")
    if not type_key.strip():
        raise ValueError(f"{field} must not be empty")
    if type_key != type_key.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")
    if not _TYPE_KEY_PATTERN.fullmatch(type_key):
        raise ValueError(
            f"{field} must be a lowercase dotted key '<package>.<name>'; got {type_key!r}"
        )
    return type_key


def register_figure_constructor(
    key: str,
    *,
    tier: ConstructorTier,
    constructor: (
        TraceConstructor | PanelConstructor | FigureConstructor | CustomFigureConstructor
    ),
    params_model: type[StrictModel] = EmptyParams,
    description: str,
    version: str = "v1",
    replace: bool = False,
) -> None:
    """Register a constructor under a stable namespaced key."""
    key = _validate_namespaced_type_key(key, field="figure constructor")
    if key in _CONSTRUCTORS and not replace:
        raise ValueError(f"Figure constructor {key!r} is already registered")
    _CONSTRUCTORS[key] = FigureConstructorRegistration(
        key=key,
        tier=tier,
        callable=constructor,
        params_model=params_model,
        description=description,
        version=version,
    )


def get_figure_constructor(key: str, *, tier: ConstructorTier | None = None) -> FigureConstructorRegistration:
    """Return a registered constructor, optionally enforcing its tier."""
    try:
        registration = _CONSTRUCTORS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_CONSTRUCTORS)) or "none"
        raise ValueError(
            f"Figure constructor {key!r} is not registered. "
            f"Registered constructors: {available}."
        ) from exc
    if tier is not None and registration.tier != tier:
        raise ValueError(
            f"Figure constructor {key!r} has tier {registration.tier!r}, "
            f"expected {tier!r}"
        )
    return registration


def registered_figure_constructors(*, tier: ConstructorTier | None = None) -> tuple[FigureConstructorRegistration, ...]:
    """Return registered constructors sorted by key."""
    registrations = sorted(_CONSTRUCTORS.values(), key=lambda item: item.key)
    if tier is not None:
        registrations = [item for item in registrations if item.tier == tier]
    return tuple(registrations)


def register_figure_template(template: Any, *, replace: bool = False) -> None:
    """Register a FigureTemplate-like object by name."""
    name = _validate_namespaced_type_key(template.name, field="figure template")
    if name in _TEMPLATES and not replace:
        raise ValueError(f"Figure template {name!r} is already registered")
    _TEMPLATES[name] = template


def get_figure_template(name: str) -> Any:
    try:
        return _TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TEMPLATES)) or "none"
        raise ValueError(
            f"Figure template {name!r} is not registered. "
            f"Registered templates: {available}."
        ) from exc


def registered_figure_templates() -> tuple[Any, ...]:
    return tuple(_TEMPLATES[name] for name in sorted(_TEMPLATES))


def register_figure_piece(piece: Any, *, replace: bool = False) -> None:
    """Register a FigurePiece-like object by name."""
    name = _validate_namespaced_type_key(piece.name, field="figure piece")
    if name in _PIECES and not replace:
        raise ValueError(f"Figure piece {name!r} is already registered")
    _PIECES[name] = piece


def get_figure_piece(name: str) -> Any:
    try:
        return _PIECES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PIECES)) or "none"
        raise ValueError(
            f"Figure piece {name!r} is not registered. Registered pieces: {available}."
        ) from exc


def registered_figure_pieces() -> tuple[Any, ...]:
    return tuple(_PIECES[name] for name in sorted(_PIECES))


def constructor_catalog() -> list[dict[str, Any]]:
    """Return a JSON-serializable constructor catalog."""
    return [
        {
            "key": item.key,
            "tier": item.tier,
            "description": item.description,
            "version": item.version,
            "params_schema": item.params_model.model_json_schema(),
        }
        for item in registered_figure_constructors()
    ]


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _profile_band(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = ProfileParams.model_validate(params.model_dump())
    y = _array(data.get("y", data.get("values", [])))
    if y.ndim == 1:
        y = y[None, :]
    x = _array(data.get("x", np.arange(y.shape[-1])))
    mean = _array(data.get("mean", np.nanmean(y, axis=0)))
    std = _array(data.get("std", np.nanstd(y, axis=0)))
    upper = _array(data.get("upper", mean + p.n_std_plot * std))
    lower = _array(data.get("lower", mean - p.n_std_plot * std))
    label = p.label or str(data.get("label", "Profile"))
    color = p.color or str(data.get("color", "rgb(31,119,180)"))
    return [
        go.Scatter(
            name=label,
            legendgroup=label,
            x=x,
            y=mean,
            mode="lines",
            line={"color": color},
        ),
        go.Scatter(
            name="Upper bound",
            legendgroup=label,
            x=x,
            y=upper,
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            name="Lower bound",
            legendgroup=label,
            x=x,
            y=lower,
            line={"color": "rgba(255,255,255,0)"},
            fill="tonexty",
            fillcolor=color_add_alpha(color, p.error_bars_alpha / max(1.0, y.shape[0] ** 0.5)),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]


def _profile_curves(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = ProfileParams.model_validate(params.model_dump())
    y = _array(data.get("y", data.get("values", [])))
    if y.ndim == 1:
        y = y[None, :]
    x = _array(data.get("x", np.arange(y.shape[-1])))
    label = p.label or str(data.get("label", "Profile"))
    color = p.color or str(data.get("color", "rgb(31,119,180)"))
    traces: list[Any] = []
    for index, curve in enumerate(y[:: max(1, p.stride_curves)]):
        traces.append(
            go.Scatter(
                name=label,
                legendgroup=label,
                showlegend=index == 0,
                x=x,
                y=curve,
                mode="lines",
                line={"color": color, "width": 0.5},
            )
        )
    return traces


def _trajectory_2d(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = Trajectory2DParams.model_validate(params.model_dump())
    trajectories = _array(data.get("trajectories", data.get("y", [])))
    if trajectories.ndim == 2:
        trajectories = trajectories[None, :, :]
    label = p.label or str(data.get("label", "Trajectory"))
    color = p.color or str(data.get("color", "rgb(31,119,180)"))
    traces: list[Any] = []
    for index, traj in enumerate(trajectories[:: max(1, p.stride)]):
        traces.append(
            go.Scatter(
                name=label,
                legendgroup=label,
                showlegend=index == 0,
                x=traj[:, 0],
                y=traj[:, 1],
                mode="lines",
                opacity=p.opacity,
                line={"color": color, "width": p.line_width},
            )
        )
    if p.show_mean and trajectories.shape[0] > 1:
        mean = np.nanmean(trajectories, axis=0)
        traces.append(
            go.Scatter(
                name=f"{label} mean",
                legendgroup=label,
                x=mean[:, 0],
                y=mean[:, 1],
                mode="lines",
                line={"color": color, "width": p.mean_line_width},
            )
        )
    return traces


def _endpoint_markers(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = EndpointMarkerParams.model_validate(params.model_dump())
    endpoints = _array(data.get("endpoints", []))
    if endpoints.ndim == 1:
        endpoints = endpoints[None, :]
    traces: list[Any] = [
        go.Scatter(
            name=p.label,
            x=endpoints[:, 0],
            y=endpoints[:, 1],
            mode="markers",
            marker={"color": p.color, "size": p.marker_size, "symbol": "circle-open"},
        )
    ]
    if p.straight_guides:
        for endpoint in endpoints:
            traces.append(
                go.Scatter(
                    name=f"{p.label} guide",
                    x=[0, endpoint[0]],
                    y=[0, endpoint[1]],
                    mode="lines",
                    line={"color": p.color, "dash": "dot", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return traces


def _empty_annotation(_data: Mapping[str, Any], _params: StrictModel) -> Sequence[Any]:
    return []


def _comparison_grid(panels: Sequence[PanelContent], params: StrictModel) -> go.Figure:
    p = ComparisonGridParams.model_validate(params.model_dump())
    panel_list = list(panels) or [PanelContent(name="main")]
    max_row = max((panel.row or index + 1) for index, panel in enumerate(panel_list))
    max_col = max((panel.col or 1) for panel in panel_list)
    titles = [panel.title or panel.name for panel in panel_list]
    fig = make_subplots(
        rows=max_row,
        cols=max_col,
        subplot_titles=titles,
        shared_xaxes=p.shared_xaxes,
        shared_yaxes=p.shared_yaxes,
        horizontal_spacing=p.horizontal_spacing,
        vertical_spacing=p.vertical_spacing,
    )
    for index, panel in enumerate(panel_list):
        row = panel.row or index + 1
        col = panel.col or 1
        for trace in panel.traces:
            fig.add_trace(trace, row=row, col=col)
        if panel.axes_labels:
            fig.update_xaxes(title_text=panel.axes_labels.get("x"), row=row, col=col)
            fig.update_yaxes(title_text=panel.axes_labels.get("y"), row=row, col=col)
    return fig


def _grid_figure(fig: go.Figure, panels: Sequence[PanelContent], params: StrictModel) -> go.Figure:
    p = GridFigureParams.model_validate(params.model_dump())
    updates: dict[str, Any] = {}
    if p.width is not None:
        updates["width"] = p.width
    if p.height is not None:
        updates["height"] = p.height
    if p.title is not None:
        updates["title_text"] = p.title
    if p.legend_tracegroupgap is not None:
        updates["legend_tracegroupgap"] = p.legend_tracegroupgap
    if updates:
        fig.update_layout(**updates)
    return fig


def _trajectories_2d_row(fig: go.Figure, panels: Sequence[PanelContent], params: StrictModel) -> go.Figure:
    p = Trajectories2DRowParams.model_validate(params.model_dump())
    n_panels = max(1, len(panels))
    fig.update_layout(
        width=p.width_base + n_panels * p.width_per_panel,
        height=p.height,
        legend_tracegroupgap=p.legend_tracegroupgap,
    )
    return fig


def register_default_figure_constructors() -> None:
    """Install Feedbax's built-in constructor set idempotently."""
    defaults: list[tuple[str, ConstructorTier, Callable[..., Any], type[StrictModel], str]] = [
        ("feedbax.profile_band", "trace", _profile_band, ProfileParams, "Mean line with standard-deviation band."),
        ("feedbax.profile_curves", "trace", _profile_curves, ProfileParams, "Per-trial profile curves."),
        ("feedbax.trajectory_2d", "trace", _trajectory_2d, Trajectory2DParams, "2D trajectory traces."),
        ("feedbax.endpoint_markers", "trace", _endpoint_markers, EndpointMarkerParams, "Endpoint markers and optional straight guides."),
        ("feedbax.hline", "trace", _empty_annotation, EmptyParams, "Horizontal annotation placeholder."),
        ("feedbax.vrect", "trace", _empty_annotation, EmptyParams, "Vertical-region annotation placeholder."),
        ("feedbax.comparison_grid", "panel", _comparison_grid, ComparisonGridParams, "N-panel comparison grid."),
        ("feedbax.grid_figure", "figure", _grid_figure, GridFigureParams, "Generic grid figure layout finalizer."),
        ("feedbax.trajectories_2d_row", "figure", _trajectories_2d_row, Trajectories2DRowParams, "2D trajectory row house style."),
    ]
    for key, tier, constructor, params_model, description in defaults:
        register_figure_constructor(
            key,
            tier=tier,  # type: ignore[arg-type]
            constructor=constructor,  # type: ignore[arg-type]
            params_model=params_model,
            description=description,
            replace=True,
        )


register_default_figure_constructors()
