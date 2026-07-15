"""Registered constructors for declarative figure execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
import re
from typing import Any, Literal

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from feedbax.contracts.manifest import StrictModel
from feedbax.plot.colors import color_add_alpha, sample_colorscale_unique

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
    colorscale: str | list[str] | list[tuple[float, str]] | None = None
    colorscale_key: str | None = None
    colorscale_axis: int = 0
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


class HLineParams(StrictModel):
    """Defaults for a horizontal panel annotation."""

    y: float | None = None
    color: str = "grey"
    dash: str = "dot"
    width: float = 1.0
    opacity: float = 1.0


class VRectParams(StrictModel):
    """Defaults for a vertical-region panel annotation."""

    x0: float | None = None
    x1: float | None = None
    fillcolor: str = "grey"
    opacity: float = 0.2
    line_color: str | None = None
    line_width: float = 0.0


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


def get_figure_constructor(
    key: str, *, tier: ConstructorTier | None = None
) -> FigureConstructorRegistration:
    """Return a registered constructor, optionally enforcing its tier."""
    try:
        registration = _CONSTRUCTORS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_CONSTRUCTORS)) or "none"
        raise ValueError(
            f"Figure constructor {key!r} is not registered. Registered constructors: {available}."
        ) from exc
    if tier is not None and registration.tier != tier:
        raise ValueError(
            f"Figure constructor {key!r} has tier {registration.tier!r}, expected {tier!r}"
        )
    return registration


def registered_figure_constructors(
    *, tier: ConstructorTier | None = None
) -> tuple[FigureConstructorRegistration, ...]:
    """Return registered constructors sorted by key."""
    registrations = sorted(_CONSTRUCTORS.values(), key=lambda item: item.key)
    if tier is not None:
        registrations = [item for item in registrations if item.tier == tier]
    return tuple(registrations)


def unregister_figure_constructor(key: str) -> None:
    """Remove a registered constructor, if present.

    This is primarily useful for plugin lifecycle management and for tests that
    must restore the process-global registry after registering a downstream
    figure recipe.
    """
    _CONSTRUCTORS.pop(key, None)


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
        try:
            template = load_figure_template(name)
        except FileNotFoundError:
            available = ", ".join(sorted(_TEMPLATES)) or "none"
            raise ValueError(
                f"Figure template {name!r} is not registered and no package YAML resource "
                f"was found. Registered templates: {available}."
            ) from exc
        register_figure_template(template)
        return template


def registered_figure_templates() -> tuple[Any, ...]:
    """Return registered figure recipes sorted by their namespaced key."""
    return tuple(_TEMPLATES[name] for name in sorted(_TEMPLATES))


def unregister_figure_template(name: str) -> None:
    """Remove a registered figure recipe, if present."""
    _TEMPLATES.pop(name, None)


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
        try:
            piece = load_figure_piece(name)
        except FileNotFoundError:
            available = ", ".join(sorted(_PIECES)) or "none"
            raise ValueError(
                f"Figure piece {name!r} is not registered and no package YAML resource "
                f"was found. Registered pieces: {available}."
            ) from exc
        register_figure_piece(piece)
        return piece


def registered_figure_pieces() -> tuple[Any, ...]:
    """Return registered figure pieces sorted by their namespaced key."""
    return tuple(_PIECES[name] for name in sorted(_PIECES))


def unregister_figure_piece(name: str) -> None:
    """Remove a registered figure piece, if present."""
    _PIECES.pop(name, None)


def _resource_file(
    key: str,
    *,
    directory: str,
    registry: Any,
) -> tuple[str, Any]:
    """Resolve one package figure-resource key to an importlib resource."""
    if "/" in key:
        package_name, resource_name = key.split("/", 1)
        if not resource_name:
            raise ValueError(f"Empty figure resource name after package {package_name!r}")
        metadata = registry.get_package_metadata(package_name)
        candidates = [(package_name, metadata, resource_name)]
    else:
        package_prefix, separator, dotted_name = key.partition(".")
        candidates = []
        if separator:
            try:
                metadata = registry.get_package_metadata(package_prefix)
            except (KeyError, ValueError):
                pass
            else:
                candidates.append((package_prefix, metadata, dotted_name))
        if not candidates:
            single = registry.single_package_name()
            if single is not None:
                candidates.append((single, registry.get_package_metadata(single), key))
            else:
                candidates.extend(
                    (package_name, metadata, key)
                    for package_name, metadata in registry.iter_package_metadata()
                )

    matches: list[tuple[str, Any]] = []
    for package_name, metadata, resource_name in candidates:
        root = f"{metadata.package_module.__name__}.{metadata.config_resource_root}.{directory}"
        try:
            root_files = resources.files(root)
        except (FileNotFoundError, ModuleNotFoundError):
            continue
        for suffix in (".yml", ".yaml"):
            resource = root_files.joinpath(f"{resource_name}{suffix}")
            if resource.is_file():
                matches.append((package_name, resource))
    if not matches:
        raise FileNotFoundError(
            f"Figure resource {key!r} not found under any registered package {directory!r} root"
        )
    if len(matches) > 1:
        options = ", ".join(f"{package}/{key}" for package, _resource in matches)
        raise ValueError(f"Figure resource {key!r} is ambiguous; use one of {options}")
    return matches[0]


def _load_figure_resource(key: str, *, directory: str, model: type[Any], registry: Any) -> Any:
    from feedbax.config.yaml import get_yaml_loader

    _package_name, resource = _resource_file(key, directory=directory, registry=registry)
    with resource.open("r", encoding="utf-8") as stream:
        payload = get_yaml_loader(typ="safe").load(stream) or {}
    return model.model_validate(payload)


def load_figure_template(key: str, *, registry: Any | None = None) -> Any:
    """Load and validate a FigureTemplate YAML package resource."""
    from feedbax.contracts.figures import FigureTemplate

    if registry is None:
        from feedbax.plugins import _get_experiment_registry

        registry = _get_experiment_registry()

    return _load_figure_resource(
        key,
        directory="figure_templates",
        model=FigureTemplate,
        registry=registry,
    )


def load_figure_piece(key: str, *, registry: Any | None = None) -> Any:
    """Load and validate a FigurePiece YAML package resource."""
    from feedbax.contracts.figures import FigurePiece

    if registry is None:
        from feedbax.plugins import _get_experiment_registry

        registry = _get_experiment_registry()

    return _load_figure_resource(
        key,
        directory="figure_pieces",
        model=FigurePiece,
        registry=registry,
    )


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
    has_mean = "mean" in data
    has_std = "std" in data
    if has_mean != has_std:
        raise ValueError("profile_band requires both 'mean' and 'std' when either is supplied")

    if has_mean:
        mean = _array(data["mean"])
        std = _array(data["std"])
        error_bars_alpha = p.error_bars_alpha
    else:
        raw_key = "y" if "y" in data else "values" if "values" in data else None
        if raw_key is None:
            raise ValueError(
                "profile_band requires both 'mean' and 'std' or raw 'y'/'values' samples"
            )
        y = _array(data[raw_key])
        if y.ndim == 1:
            y = y[None, :]
        mean = np.nanmean(y, axis=0)
        std = np.nanstd(y, axis=0)
        error_bars_alpha = p.error_bars_alpha / max(1.0, y.shape[0] ** 0.5)

    x = _array(data["x"] if "x" in data else np.arange(mean.shape[-1]))
    upper = _array(data["upper"] if "upper" in data else mean + p.n_std_plot * std)
    lower = _array(data["lower"] if "lower" in data else mean - p.n_std_plot * std)
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
            fillcolor=color_add_alpha(color, error_bars_alpha),
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
    if trajectories.ndim < 3 or trajectories.shape[-1] != 2:
        raise ValueError(
            "trajectory_2d requires trajectories with shape (..., time, 2), "
            f"got {trajectories.shape}"
        )
    label = p.label or str(data.get("label", "Trajectory"))
    colorscale = p.colorscale
    if colorscale is None and p.colorscale_key is not None:
        colorscales = data.get("colorscales")
        if not isinstance(colorscales, Mapping) or p.colorscale_key not in colorscales:
            raise ValueError(
                f"trajectory_2d colorscale_key {p.colorscale_key!r} is not present in data.colorscales"
            )
        colorscale = colorscales[p.colorscale_key]
    if colorscale is None:
        colorscale = data.get("colorscale")

    batch_shape = trajectories.shape[:-2]
    axis = p.colorscale_axis if p.colorscale_axis >= 0 else len(batch_shape) + p.colorscale_axis
    if axis < 0 or axis >= len(batch_shape):
        raise ValueError(
            f"trajectory_2d colorscale_axis {p.colorscale_axis} is outside batch shape {batch_shape}"
        )
    trajectories = np.moveaxis(trajectories, axis, 0)[:: max(1, p.stride)]
    group_count = trajectories.shape[0]
    curves_per_group = int(np.prod(trajectories.shape[1:-2], dtype=int)) or 1
    trajectories = trajectories.reshape((-1, *trajectories.shape[-2:]))
    if p.color is not None:
        group_colors = [p.color] * group_count
    elif colorscale is not None:
        group_colors = list(sample_colorscale_unique(colorscale, group_count, colortype="rgb"))
    else:
        group_colors = [str(data.get("color", "rgb(31,119,180)"))] * group_count
    colors = [color for color in group_colors for _ in range(curves_per_group)]
    traces: list[Any] = []
    for index, (traj, color) in enumerate(zip(trajectories, colors, strict=True)):
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
                line={"color": group_colors[0], "width": p.mean_line_width},
            )
        )
    return traces


def _endpoint_markers(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = EndpointMarkerParams.model_validate(params.model_dump())
    trajectories_value = data.get("trajectories")
    starts_value = data.get("starts")
    endpoints_value = data.get("endpoints")
    if trajectories_value is not None:
        trajectories = _array(trajectories_value)
        if trajectories.ndim < 2 or trajectories.shape[-1] != 2:
            raise ValueError(
                "endpoint_markers trajectories must have shape (..., time, 2), "
                f"got {trajectories.shape}"
            )
        trajectories = trajectories.reshape((-1, *trajectories.shape[-2:]))
        starts = trajectories[:, 0, :]
        endpoints = trajectories[:, -1, :]
    else:
        endpoints = _array(endpoints_value if endpoints_value is not None else [])
        starts = _array(starts_value) if starts_value is not None else None
        if endpoints.ndim >= 3 and endpoints.shape[0] == 2 and starts is None:
            starts = endpoints[0].reshape((-1, 2))
            endpoints = endpoints[1].reshape((-1, 2))
    if endpoints.ndim == 1:
        endpoints = endpoints[None, :]
    if endpoints.ndim != 2 or endpoints.shape[-1] != 2:
        raise ValueError(
            f"endpoint_markers requires endpoints with shape (n, 2), got {endpoints.shape}"
        )
    if starts is not None:
        if starts.ndim == 1:
            starts = starts[None, :]
        starts = np.broadcast_to(starts, endpoints.shape)
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
        if starts is None:
            raise ValueError(
                "endpoint_markers straight_guides requires trajectory data or explicit starts; "
                "guides are never inferred from the plot origin"
            )
        for start, endpoint in zip(starts, endpoints, strict=True):
            traces.append(
                go.Scatter(
                    name=f"{p.label} guide",
                    x=[start[0], endpoint[0]],
                    y=[start[1], endpoint[1]],
                    mode="lines",
                    line={"color": p.color, "dash": "dot", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return traces


def _hline(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = HLineParams.model_validate(params.model_dump())
    y = p.y if p.y is not None else data.get("y")
    if y is None:
        raise ValueError("hline requires y in data or params")
    return [
        go.layout.Shape(
            type="line",
            x0=0,
            x1=1,
            xref="x domain",
            y0=float(y),
            y1=float(y),
            yref="y",
            line={"color": p.color, "dash": p.dash, "width": p.width},
            opacity=p.opacity,
        )
    ]


def _vrect(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = VRectParams.model_validate(params.model_dump())
    x0 = p.x0 if p.x0 is not None else data.get("x0")
    x1 = p.x1 if p.x1 is not None else data.get("x1")
    if x0 is None or x1 is None:
        raise ValueError("vrect requires x0 and x1 in data or params")
    return [
        go.layout.Shape(
            type="rect",
            x0=float(x0),
            x1=float(x1),
            xref="x",
            y0=0,
            y1=1,
            yref="y domain",
            fillcolor=p.fillcolor,
            opacity=p.opacity,
            line={"color": p.line_color or p.fillcolor, "width": p.line_width},
        )
    ]


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
            if isinstance(trace, go.layout.Shape):
                fig.add_shape(trace, row=row, col=col)
            else:
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


def _trajectories_2d_row(
    fig: go.Figure, panels: Sequence[PanelContent], params: StrictModel
) -> go.Figure:
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
        (
            "feedbax.profile_band",
            "trace",
            _profile_band,
            ProfileParams,
            "Mean line with standard-deviation band.",
        ),
        (
            "feedbax.profile_curves",
            "trace",
            _profile_curves,
            ProfileParams,
            "Per-trial profile curves.",
        ),
        (
            "feedbax.trajectory_2d",
            "trace",
            _trajectory_2d,
            Trajectory2DParams,
            "2D trajectory traces.",
        ),
        (
            "feedbax.endpoint_markers",
            "trace",
            _endpoint_markers,
            EndpointMarkerParams,
            "Endpoint markers with trajectory-derived straight guides.",
        ),
        ("feedbax.hline", "trace", _hline, HLineParams, "Horizontal panel annotation."),
        ("feedbax.vrect", "trace", _vrect, VRectParams, "Vertical-region panel annotation."),
        (
            "feedbax.comparison_grid",
            "panel",
            _comparison_grid,
            ComparisonGridParams,
            "N-panel comparison grid.",
        ),
        (
            "feedbax.grid_figure",
            "figure",
            _grid_figure,
            GridFigureParams,
            "Generic grid figure layout finalizer.",
        ),
        (
            "feedbax.trajectories_2d_row",
            "figure",
            _trajectories_2d_row,
            Trajectories2DRowParams,
            "2D trajectory row house style.",
        ),
    ]
    changed_versions = {
        "feedbax.trajectory_2d": "v2",
        "feedbax.endpoint_markers": "v2",
        "feedbax.hline": "v2",
        "feedbax.vrect": "v2",
    }
    for key, tier, constructor, params_model, description in defaults:
        register_figure_constructor(
            key,
            tier=tier,  # type: ignore[arg-type]
            constructor=constructor,  # type: ignore[arg-type]
            params_model=params_model,
            description=description,
            version=changed_versions.get(key, "v1"),
            replace=True,
        )


register_default_figure_constructors()
