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

from feedbax.contracts.figures import ColorscaleSpec, FigureColorbar, PerturbationTiming
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


class ProfileBandParams(ProfileParams):
    """Style and visibility controls for a profile mean and band."""

    line_dash: str | None = None
    line_width: float | None = None
    showlegend: bool | None = None
    show_band: bool = True


class TrajectoryStartMarkerParams(StrictModel):
    """Marker on the first sample of every plotted trajectory.

    A ``color`` of ``None`` means each marker inherits the color of the
    trajectory it belongs to, so markers follow a continuous colorscale
    instead of collapsing to one flat color.

    ``symbol`` is passed to Plotly unchanged and is validated by the trace type
    that draws it. The 3D symbol vocabulary is the smaller one, so a symbol a 2D
    panel accepts is not automatically available to a 3D one.
    """

    label: str | None = None
    color: str | None = None
    size: float = 8.0
    symbol: str = "circle"
    showlegend: bool | None = None


class TrajectoryParams(StrictModel):
    """Encodings shared by the 2D and 3D trajectory constructors.

    These are the channels a trajectory carries regardless of how many spatial
    dimensions it is drawn in: which color it takes and from where, how many
    curves are drawn, whether a mean is added, and how the line and its start
    marker are styled.
    """

    label: str | None = None
    color: str | None = None
    colorscale: ColorscaleSpec | None = None
    colorscale_key: str | None = None
    colorscale_axis: int = 0
    stride: int = 1
    show_mean: bool = True
    showlegend: bool | None = None
    opacity: float = 0.4
    line_dash: str | None = None
    line_width: float = 0.75
    mean_line_width: float = 2.5
    start_marker: TrajectoryStartMarkerParams | None = None


class Trajectory2DParams(TrajectoryParams):
    """Defaults for 2D trajectory traces."""

    affected_color: str = "rgba(255, 160, 160, 0.65)"
    affected_line_width: float = 6.0
    affected_marker_size: float = 10.0


class Trajectory3DParams(TrajectoryParams):
    """Defaults for 3D trajectory traces.

    A 3D trajectory carries exactly the shared encodings and adds none of its
    own. The perturbation underlay the 2D constructor draws is deliberately
    absent: it is an authored annotation of when a perturbation applied, and no
    3D consumer has asked for one, so ``trajectory_3d`` rejects perturbation
    timing rather than accepting and ignoring it.
    """


class EndpointMarkerParams(StrictModel):
    """Defaults for endpoint marker traces."""

    label: str = "Endpoints"
    color: str = "rgb(25, 25, 25)"
    marker_size: int = 7
    straight_guides: bool = True
    showlegend: bool | None = None


class ScalarScatterParams(StrictModel):
    """Style controls for a markers-only scalar scatter with error bars.

    The constructor draws one marker per x value and never a connecting line;
    ``mode`` is fixed to ``"markers"`` and is deliberately not exposed here.
    """

    label: str | None = None
    color: str | None = None
    marker_symbol: str = "circle"
    marker_size: float = 8.0
    error_bar_thickness: float = 1.5
    error_bar_width: float | None = None
    error_bar_color: str | None = None
    showlegend: bool | None = None


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
    event_line_dash: str = "solid"
    event_line_width: float = 1.5


class ComparisonGridParams(StrictModel):
    """Defaults for panel population."""

    shared_xaxes: bool | str = False
    shared_yaxes: bool | str = "all"
    horizontal_spacing: float | None = None
    vertical_spacing: float | None = None


class FigureMarginParams(StrictModel):
    """Optional Plotly figure margins."""

    l: float | None = None
    r: float | None = None
    t: float | None = None
    b: float | None = None
    pad: float | None = None
    autoexpand: bool | None = None


class GridFigureParams(StrictModel):
    """Defaults for whole-figure layout."""

    panel_constructor: str = "feedbax.comparison_grid"
    width: int | None = None
    height: int | None = None
    title: str | None = None
    legend_tracegroupgap: int | None = None
    legend_x: float | None = None
    legend_y: float | None = None
    legend_xanchor: Literal["auto", "left", "center", "right"] | None = None
    legend_yanchor: Literal["auto", "top", "middle", "bottom"] | None = None
    margin: FigureMarginParams | None = None
    hovermode: Literal["x", "y", "closest", "x unified", "y unified", False] | None = None
    template: str | None = None
    # Resolved by figure execution from FigureSpec.colorbar; not authored here.
    colorbar: FigureColorbar | None = None


class Trajectories2DRowParams(StrictModel):
    """House style for 2D effector trajectory rows."""

    panel_constructor: str = "feedbax.comparison_grid"
    width_base: int = 100
    width_per_panel: int = 300
    height: int = 400
    legend_tracegroupgap: int = 1


@dataclass(frozen=True)
class PanelContent:
    """Resolved traces and layout metadata for one panel.

    ``panel_type`` of ``None`` is the Cartesian panel, which is what every panel
    was before scene panels existed; ``"scene"`` is the 3D one.
    """

    name: str
    traces: tuple[Any, ...] = ()
    title: str | None = None
    row: int | None = None
    col: int | None = None
    axes_labels: Mapping[str, str | None] | None = None
    x_axis: Mapping[str, Any] | None = None
    y_axis: Mapping[str, Any] | None = None
    equal_data_aspect: Mapping[str, Any] | None = None
    z_axis: Mapping[str, Any] | None = None
    panel_type: str | None = None
    row_span: int | None = None
    col_span: int | None = None


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
    p = ProfileBandParams.model_validate(params.model_dump())
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
    line: dict[str, Any] = {"color": color}
    if p.line_dash is not None:
        line["dash"] = p.line_dash
    if p.line_width is not None:
        line["width"] = p.line_width
    mean_trace_kwargs: dict[str, Any] = {}
    if p.showlegend is not None:
        mean_trace_kwargs["showlegend"] = p.showlegend
    mean_trace = go.Scatter(
        name=label,
        legendgroup=label,
        x=x,
        y=mean,
        mode="lines",
        line=line,
        **mean_trace_kwargs,
    )
    if not p.show_band:
        return [mean_trace]
    return [
        mean_trace,
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


#: Plotly coordinate argument names, in the order a trajectory's columns take.
_COORDINATE_NAMES = ("x", "y", "z")

#: The scatter trace type that draws points of each spatial dimensionality.
_SCATTER_BY_DIM: dict[int, type[Any]] = {2: go.Scatter, 3: go.Scatter3d}


@dataclass(frozen=True)
class _TrajectoryCurves:
    """Curves, per-curve colors, and label prepared for a trajectory constructor.

    This is the shared core of the trajectory constructors: everything that
    depends on the authored encodings rather than on how many spatial
    dimensions the traces are drawn in.
    """

    curves: np.ndarray
    colors: list[str]
    group_colors: list[str]
    label: str


def _coordinates(points: np.ndarray, dim: int) -> dict[str, np.ndarray]:
    """Return the Plotly coordinate arguments for one array of plotted points."""
    return {name: points[..., axis] for axis, name in enumerate(_COORDINATE_NAMES[:dim])}


def _trajectory_array(data: Mapping[str, Any], *, constructor: str, dim: int) -> np.ndarray:
    """Return authored trajectories as a batched ``(..., time, dim)`` array."""
    trajectories = _array(data.get("trajectories", data.get("y", [])))
    if trajectories.ndim == 2:
        trajectories = trajectories[None, :, :]
    if trajectories.ndim < 3 or trajectories.shape[-1] != dim:
        raise ValueError(
            f"{constructor} requires trajectories with shape (..., time, {dim}), "
            f"got {trajectories.shape}"
        )
    return trajectories


def _trajectory_curves(
    trajectories: np.ndarray,
    data: Mapping[str, Any],
    p: TrajectoryParams,
    *,
    constructor: str,
) -> _TrajectoryCurves:
    """Resolve authored colors and flatten a trajectory batch into drawn curves."""
    label = p.label or str(data.get("label", "Trajectory"))
    colorscale = p.colorscale
    if colorscale is None and p.colorscale_key is not None:
        colorscales = data.get("colorscales")
        if not isinstance(colorscales, Mapping) or p.colorscale_key not in colorscales:
            raise ValueError(
                f"{constructor} colorscale_key {p.colorscale_key!r} is not present in data.colorscales"
            )
        colorscale = colorscales[p.colorscale_key]
    if colorscale is None:
        colorscale = data.get("colorscale")

    batch_shape = trajectories.shape[:-2]
    axis = p.colorscale_axis if p.colorscale_axis >= 0 else len(batch_shape) + p.colorscale_axis
    if axis < 0 or axis >= len(batch_shape):
        raise ValueError(
            f"{constructor} colorscale_axis {p.colorscale_axis} is outside batch shape {batch_shape}"
        )
    trajectories = np.moveaxis(trajectories, axis, 0)[:: max(1, p.stride)]
    group_count = trajectories.shape[0]
    curves_per_group = int(np.prod(trajectories.shape[1:-2], dtype=int)) or 1
    curves = trajectories.reshape((-1, *trajectories.shape[-2:]))
    group_colors: list[str]
    if p.color is not None:
        group_colors = [p.color] * group_count
    elif colorscale is not None:
        group_colors = [
            str(color)
            for color in sample_colorscale_unique(colorscale, group_count, colortype="rgb")
        ]
    else:
        group_colors = [str(data.get("color", "rgb(31,119,180)"))] * group_count
    colors = [color for color in group_colors for _ in range(curves_per_group)]
    return _TrajectoryCurves(
        curves=curves,
        colors=colors,
        group_colors=group_colors,
        label=label,
    )


def _trajectory_line(p: TrajectoryParams, color: str, width: float) -> dict[str, Any]:
    """Return one trajectory line style, carrying dash only when authored."""
    line: dict[str, Any] = {"color": color, "width": width}
    if p.line_dash is not None:
        line["dash"] = p.line_dash
    return line


def _trajectory_start_marker(prepared: _TrajectoryCurves, p: TrajectoryParams) -> list[Any]:
    """Return the authored start-marker trace, or nothing when none is authored."""
    if p.start_marker is None or prepared.curves.shape[0] == 0:
        return []
    marker = p.start_marker
    marker_label = marker.label or f"{prepared.label} start"
    return [
        _point_marker_trace(
            prepared.curves[:, 0, :],
            name=marker_label,
            color=marker.color if marker.color is not None else prepared.colors,
            size=marker.size,
            symbol=marker.symbol,
            legendgroup=marker_label,
            showlegend=marker.showlegend,
        )
    ]


def _trajectory_2d(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = Trajectory2DParams.model_validate(params.model_dump())
    trajectories = _trajectory_array(data, constructor="trajectory_2d", dim=2)
    timing = _perturbation_timing(data, constructor="trajectory_2d")
    if timing is not None and trajectories.shape[-2] != timing.sample_count:
        raise ValueError(
            "trajectory_2d perturbation timing sample_count "
            f"{timing.sample_count} does not match every plotted trajectory length "
            f"{trajectories.shape[-2]}"
        )
    prepared = _trajectory_curves(trajectories, data, p, constructor="trajectory_2d")
    label = prepared.label
    traces: list[Any] = []
    for index, (traj, color) in enumerate(zip(prepared.curves, prepared.colors, strict=True)):
        scientific_trace = go.Scatter(
            name=label,
            legendgroup=label,
            showlegend=index == 0 and p.showlegend is not False,
            **_coordinates(traj, 2),
            mode="lines",
            opacity=p.opacity,
            line=_trajectory_line(p, color, p.line_width),
        )
        traces.extend(
            _trajectory_with_affected_underlay(
                scientific_trace,
                traj,
                timing,
                color=p.affected_color,
                width=p.affected_line_width,
                marker_size=p.affected_marker_size,
            )
        )
    if p.show_mean and prepared.curves.shape[0] > 1:
        mean = np.nanmean(prepared.curves, axis=0)
        mean_trace_kwargs: dict[str, Any] = {}
        if p.showlegend is not None:
            mean_trace_kwargs["showlegend"] = p.showlegend
        mean_trace = go.Scatter(
            name=f"{label} mean",
            legendgroup=label,
            **_coordinates(mean, 2),
            mode="lines",
            line=_trajectory_line(p, prepared.group_colors[0], p.mean_line_width),
            **mean_trace_kwargs,
        )
        traces.extend(
            _trajectory_with_affected_underlay(
                mean_trace,
                mean,
                timing,
                color=p.affected_color,
                width=p.affected_line_width,
                marker_size=p.affected_marker_size,
            )
        )
    traces.extend(_trajectory_start_marker(prepared, p))
    return traces


def _trajectory_3d(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = Trajectory3DParams.model_validate(params.model_dump())
    trajectories = _trajectory_array(data, constructor="trajectory_3d", dim=3)
    if data.get("perturbation_timing") is not None:
        raise ValueError(
            "trajectory_3d draws no perturbation underlay, so perturbation_timing data "
            "would be silently ignored; annotate the perturbation in a 2D panel instead"
        )
    prepared = _trajectory_curves(trajectories, data, p, constructor="trajectory_3d")
    label = prepared.label
    traces: list[Any] = []
    for index, (traj, color) in enumerate(zip(prepared.curves, prepared.colors, strict=True)):
        traces.append(
            go.Scatter3d(
                name=label,
                legendgroup=label,
                showlegend=index == 0 and p.showlegend is not False,
                **_coordinates(traj, 3),
                mode="lines",
                opacity=p.opacity,
                line=_trajectory_line(p, color, p.line_width),
            )
        )
    if p.show_mean and prepared.curves.shape[0] > 1:
        mean = np.nanmean(prepared.curves, axis=0)
        mean_trace_kwargs: dict[str, Any] = {}
        if p.showlegend is not None:
            mean_trace_kwargs["showlegend"] = p.showlegend
        traces.append(
            go.Scatter3d(
                name=f"{label} mean",
                legendgroup=label,
                **_coordinates(mean, 3),
                mode="lines",
                line=_trajectory_line(p, prepared.group_colors[0], p.mean_line_width),
                **mean_trace_kwargs,
            )
        )
    traces.extend(_trajectory_start_marker(prepared, p))
    return traces


def _trajectory_with_affected_underlay(
    scientific_trace: go.Scatter,
    trajectory: np.ndarray,
    timing: PerturbationTiming | None,
    *,
    color: str,
    width: float,
    marker_size: float,
) -> list[go.Scatter]:
    """Return an optional affected-segment underlay immediately before its trace."""
    if timing is None or timing.applicability != "bounded":
        return [scientific_trace]
    affected_range = timing.affected_range()
    assert affected_range is not None
    start, stop = affected_range
    affected = trajectory[start:stop]
    singleton = affected.shape[0] == 1
    underlay = go.Scatter(
        name=f"{scientific_trace.name} affected segment",
        legendgroup=scientific_trace.legendgroup,
        showlegend=False,
        hoverinfo="skip",
        x=affected[:, 0],
        y=affected[:, 1],
        mode="markers" if singleton else "lines",
        line={"color": color, "width": width},
        marker={"color": color, "size": marker_size} if singleton else None,
    )
    return [underlay, scientific_trace]


def _point_marker_trace(
    points: np.ndarray,
    *,
    name: str,
    color: Any,
    size: float,
    symbol: str,
    legendgroup: str | None = None,
    showlegend: bool | None = None,
) -> Any:
    """Return one markers-only trace over an array of plotted points.

    The points' own last axis states how many spatial dimensions they occupy and
    therefore which scatter type draws them: two columns are a Cartesian
    ``Scatter``, three are a ``Scatter3d``.

    Args:
        points: Marker positions with shape ``(n, 2)`` or ``(n, 3)``.
        name: Trace name, which is also the legend entry text.
        color: One Plotly color for every marker, or one color per point.
        size: Marker size.
        symbol: Plotly marker symbol.
        legendgroup: Optional legend group; omitted when ``None``.
        showlegend: Optional legend visibility; Plotly's default when ``None``.
    """
    dim = points.shape[-1]
    scatter_type = _SCATTER_BY_DIM.get(dim)
    if scatter_type is None:
        raise ValueError(
            f"marker points must have 2 or 3 columns, got shape {points.shape}"
        )
    optional: dict[str, Any] = {}
    if legendgroup is not None:
        optional["legendgroup"] = legendgroup
    if showlegend is not None:
        optional["showlegend"] = showlegend
    return scatter_type(
        name=name,
        **_coordinates(points, dim),
        mode="markers",
        marker={"color": color, "size": size, "symbol": symbol},
        **optional,
    )


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
        _point_marker_trace(
            endpoints,
            name=p.label,
            color=p.color,
            size=p.marker_size,
            symbol="circle-open",
            showlegend=p.showlegend,
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


def _scalar_scatter(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = ScalarScatterParams.model_validate(params.model_dump())
    if "x" not in data or "y" not in data:
        raise ValueError("scalar_scatter requires both 'x' and 'y' data bindings")
    x = _array(data["x"])
    y = _array(data["y"])
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"scalar_scatter requires 1-D 'x' and 'y'; got shapes {x.shape} and {y.shape}"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"scalar_scatter 'x' and 'y' lengths differ: {x.shape[0]} vs {y.shape[0]}"
        )
    label = p.label or str(data.get("label", "Value"))
    color = p.color or str(data.get("color", "rgb(31,119,180)"))
    marker: dict[str, Any] = {
        "symbol": p.marker_symbol,
        "size": p.marker_size,
        "color": color,
    }
    scatter_kwargs: dict[str, Any] = {
        "name": label,
        "legendgroup": label,
        "x": x,
        "y": y,
        "mode": "markers",
        "marker": marker,
    }
    if p.showlegend is not None:
        scatter_kwargs["showlegend"] = p.showlegend
    if "y_err" in data:
        y_err = _array(data["y_err"])
        if y_err.ndim != 1:
            raise ValueError(f"scalar_scatter 'y_err' must be 1-D; got shape {y_err.shape}")
        if y_err.shape[0] != y.shape[0]:
            raise ValueError(
                f"scalar_scatter 'y_err' length {y_err.shape[0]} does not match "
                f"'y' length {y.shape[0]}"
            )
        error_y: dict[str, Any] = {
            "type": "data",
            "array": y_err,
            "symmetric": True,
            "thickness": p.error_bar_thickness,
            "color": p.error_bar_color or color,
        }
        if p.error_bar_width is not None:
            error_y["width"] = p.error_bar_width
        scatter_kwargs["error_y"] = error_y
    return [go.Scatter(**scatter_kwargs)]


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
    timing = _perturbation_timing(data, constructor="vrect")
    if timing is not None:
        if p.x0 is not None or p.x1 is not None or "x0" in data or "x1" in data:
            raise ValueError(
                "vrect perturbation timing cannot be combined with explicit x0 or x1"
            )
        x = _perturbation_coordinates(data, timing)
        affected_range = timing.affected_range()
        if affected_range is None:
            return []
        start, stop = affected_range
        if timing.duration == 1:
            position = float(x[start])
            return [
                go.layout.Shape(
                    type="line",
                    x0=position,
                    x1=position,
                    xref="x",
                    y0=0,
                    y1=1,
                    yref="y domain",
                    line={
                        "color": p.line_color or p.fillcolor,
                        "dash": p.event_line_dash,
                        "width": p.event_line_width,
                    },
                    opacity=p.opacity,
                )
            ]
        return [
            go.layout.Shape(
                type="rect",
                x0=float(x[start]),
                x1=float(x[stop - 1]),
                xref="x",
                y0=0,
                y1=1,
                yref="y domain",
                fillcolor=p.fillcolor,
                opacity=p.opacity,
                line={"color": p.line_color or p.fillcolor, "width": p.line_width},
            )
        ]
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


def _perturbation_timing(
    data: Mapping[str, Any],
    *,
    constructor: str,
) -> PerturbationTiming | None:
    raw = data.get("perturbation_timing")
    if raw is None:
        return None
    try:
        return PerturbationTiming.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"{constructor} perturbation_timing is invalid: {exc}") from exc


def _perturbation_coordinates(
    data: Mapping[str, Any],
    timing: PerturbationTiming,
) -> np.ndarray:
    if "x" not in data:
        raise ValueError("vrect perturbation timing requires plotted x coordinates")
    x = _array(data["x"])
    if x.ndim != 1:
        raise ValueError(
            f"vrect perturbation timing requires 1-D x coordinates, got shape {x.shape}"
        )
    if x.shape[0] != timing.sample_count:
        raise ValueError(
            "vrect perturbation timing sample_count "
            f"{timing.sample_count} does not match plotted x coordinate length {x.shape[0]}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("vrect perturbation timing requires finite x coordinates")
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            "vrect perturbation timing requires strictly increasing x coordinates"
        )
    return x


def _panel_cell(panel: PanelContent, index: int) -> tuple[int, int]:
    """Return the grid cell a panel starts at."""
    return panel.row or index + 1, panel.col or 1


def _subplot_specs(
    panel_list: Sequence[PanelContent],
    *,
    rows: int,
    cols: int,
) -> list[list[dict[str, Any] | None]] | None:
    """Return the Plotly ``specs`` grid, or ``None`` when every cell is plain.

    A grid of plain Cartesian cells is exactly what ``make_subplots`` builds
    without ``specs`` at all, so it is left unstated. Anything else — a scene
    panel or a panel spanning several cells — is stated explicitly, and the
    cells a span covers are ``None`` because Plotly reserves them for the
    spanning panel rather than giving them subplots of their own.
    """
    if not any(
        panel.panel_type not in (None, "xy")
        or (panel.row_span or 1) > 1
        or (panel.col_span or 1) > 1
        for panel in panel_list
    ):
        return None
    grid: list[list[dict[str, Any] | None]] = [
        [{"type": "xy"} for _ in range(cols)] for _ in range(rows)
    ]
    claimed: dict[tuple[int, int], str] = {}
    for index, panel in enumerate(panel_list):
        row, col = _panel_cell(panel, index)
        row_span = panel.row_span or 1
        col_span = panel.col_span or 1
        if row + row_span - 1 > rows or col + col_span - 1 > cols:
            raise ValueError(
                f"panel {panel.name!r} spans {row_span}x{col_span} cells from "
                f"row {row}, col {col}, which leaves the {rows}x{cols} grid"
            )
        for covered_row in range(row, row + row_span):
            for covered_col in range(col, col + col_span):
                owner = claimed.get((covered_row, covered_col))
                if owner is not None:
                    raise ValueError(
                        f"panels {owner!r} and {panel.name!r} both claim grid cell "
                        f"row {covered_row}, col {covered_col}"
                    )
                claimed[(covered_row, covered_col)] = panel.name
                grid[covered_row - 1][covered_col - 1] = None
        spec: dict[str, Any] = {"type": panel.panel_type or "xy"}
        if row_span > 1:
            spec["rowspan"] = row_span
        if col_span > 1:
            spec["colspan"] = col_span
        grid[row - 1][col - 1] = spec
    return grid


def _update_scene_panel(fig: go.Figure, panel: PanelContent, *, row: int, col: int) -> None:
    """Apply one scene panel's labels, axis settings, and aspect mode.

    A scene subplot carries its axes inside ``layout.scene`` rather than as
    Cartesian layout axes, so ``update_xaxes`` silently reaches nothing here and
    ``update_scenes`` is the surface that does.
    """
    scene: dict[str, Any] = {}
    if panel.axes_labels:
        for name in _COORDINATE_NAMES:
            title = panel.axes_labels.get(name)
            if title is not None:
                scene[f"{name}axis_title_text"] = title
    for name, axis in (
        ("x", panel.x_axis),
        ("y", panel.y_axis),
        ("z", panel.z_axis),
    ):
        if axis:
            scene[f"{name}axis"] = dict(axis)
    if panel.equal_data_aspect:
        # A scene has no per-axis scale anchor; equal data units are its aspect mode.
        scene["aspectmode"] = "data"
    if scene:
        fig.update_scenes(**scene, row=row, col=col)


def _update_cartesian_panel(fig: go.Figure, panel: PanelContent, *, row: int, col: int) -> None:
    """Apply one Cartesian panel's labels, axis settings, and aspect ratio."""
    if panel.axes_labels:
        fig.update_xaxes(title_text=panel.axes_labels.get("x"), row=row, col=col)
        fig.update_yaxes(title_text=panel.axes_labels.get("y"), row=row, col=col)
    if panel.x_axis:
        fig.update_xaxes(**panel.x_axis, row=row, col=col)
    if panel.y_axis:
        fig.update_yaxes(**panel.y_axis, row=row, col=col)
    if panel.equal_data_aspect:
        subplot = fig.get_subplot(row, col)
        if subplot is None or not hasattr(subplot, "xaxis"):
            raise ValueError(
                "equal_data_aspect requires a Cartesian x/y subplot at "
                f"row {row}, col {col}"
            )
        shared_axes = []
        for axis_name, axis, layout_axes in (
            ("x", subplot.xaxis, fig.select_xaxes()),
            ("y", subplot.yaxis, fig.select_yaxes()),
        ):
            axis_ref = axis.plotly_name.replace("axis", "", 1)
            if axis.matches is not None or any(
                other.matches == axis_ref for other in layout_axes
            ):
                shared_axes.append(axis_name)
        if shared_axes:
            raise ValueError(
                "equal_data_aspect cannot constrain shared axes at "
                f"row {row}, col {col}: {shared_axes}; set the corresponding "
                "shared_xaxes/shared_yaxes parameter to False"
            )
        x_axis_ref = subplot.xaxis.plotly_name.replace("axis", "", 1)
        fig.update_yaxes(
            scaleanchor=x_axis_ref,
            scaleratio=panel.equal_data_aspect["ratio"],
            row=row,
            col=col,
        )


def _comparison_grid(panels: Sequence[PanelContent], params: StrictModel) -> go.Figure:
    p = ComparisonGridParams.model_validate(params.model_dump())
    panel_list = list(panels) or [PanelContent(name="main")]
    max_row = max(
        (panel.row or index + 1) + (panel.row_span or 1) - 1
        for index, panel in enumerate(panel_list)
    )
    max_col = max((panel.col or 1) + (panel.col_span or 1) - 1 for panel in panel_list)
    titles = [panel.title or panel.name for panel in panel_list]
    fig = make_subplots(
        rows=max_row,
        cols=max_col,
        subplot_titles=titles,
        shared_xaxes=p.shared_xaxes,
        shared_yaxes=p.shared_yaxes,
        horizontal_spacing=p.horizontal_spacing,
        vertical_spacing=p.vertical_spacing,
        specs=_subplot_specs(panel_list, rows=max_row, cols=max_col),
    )
    for index, panel in enumerate(panel_list):
        row, col = _panel_cell(panel, index)
        for trace in panel.traces:
            if isinstance(trace, go.layout.Shape):
                fig.add_shape(trace, row=row, col=col)
            else:
                fig.add_trace(trace, row=row, col=col)
        if panel.panel_type == "scene":
            _update_scene_panel(fig, panel, row=row, col=col)
        else:
            _update_cartesian_panel(fig, panel, row=row, col=col)
    return fig


def _colorbar_carrier(
    colorbar: FigureColorbar,
    layout: Mapping[str, Any] | None = None,
) -> go.Scatter:
    """Return the point-free trace that draws a resolved colorbar.

    Plotly draws a colorbar from a trace that carries a colorscale, so the key
    is one scatter with no plotted points: it adds no legend entry, no hover
    target, and no data to the panels it sits beside.
    """
    if colorbar.family is not None:
        raise ValueError(
            f"colorbar bound to trace family {colorbar.family!r} reached the assembler "
            "with its family colors unresolved"
        )
    if colorbar.colorscale is None or colorbar.range is None:
        raise ValueError("a rendered colorbar requires both a colorscale and a range")
    low, high = colorbar.range
    marker: dict[str, Any] = {
        "color": [low],
        "colorscale": colorbar.colorscale,
        "cmin": low,
        "cmax": high,
        "showscale": True,
    }
    colorbar_layout = dict(layout or {})
    if colorbar.title is not None:
        colorbar_layout["title"] = colorbar.title
    if colorbar_layout:
        marker["colorbar"] = colorbar_layout
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker=marker,
    )


def _subplot_paper_domains(
    subplot: Any,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Return one subplot's paper-coordinate x and y domains, whatever its type.

    A Cartesian subplot states its domains on its two layout axes; a scene
    states one ``domain`` covering the whole cube. Both are the same rectangle
    of the paper, which is all a colorbar needs to be placed beside a panel.
    """
    if subplot is None:
        return None, None
    if isinstance(subplot, go.layout.Scene):
        domain: Any = subplot.domain
        if domain is None:
            return None, None
        return tuple(domain.x or ()), tuple(domain.y or ())
    if not hasattr(subplot, "xaxis") or not hasattr(subplot, "yaxis"):
        return None, None
    return tuple(subplot.xaxis.domain or ()), tuple(subplot.yaxis.domain or ())


def _panel_colorbar_layout(
    fig: go.Figure,
    panels: Sequence[PanelContent],
    colorbar: FigureColorbar,
) -> dict[str, Any]:
    """Resolve panel-relative colorbar placement into Plotly paper coordinates."""
    placement = colorbar.placement
    if placement is None:
        return {}
    matches = [
        (index, panel)
        for index, panel in enumerate(panels)
        if panel.name == placement.panel
    ]
    if len(matches) != 1:
        raise ValueError(
            "colorbar placement requires exactly one panel named "
            f"{placement.panel!r}; found {len(matches)}"
        )
    index, panel = matches[0]
    row, col = _panel_cell(panel, index)
    subplot = fig.get_subplot(row, col)
    x_domain, y_domain = _subplot_paper_domains(subplot)
    if x_domain is None or y_domain is None:
        raise ValueError(
            "colorbar placement requires a panel with resolved paper domains; "
            f"panel {placement.panel!r} is at row {row}, col {col}"
        )
    if len(x_domain) != 2 or len(y_domain) != 2:
        raise ValueError(
            "colorbar placement requires resolved two-point x/y domains for panel "
            f"{placement.panel!r}"
        )
    x_low, x_high = x_domain
    y_low, y_high = y_domain
    width = x_high - x_low
    height = y_high - y_low
    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"colorbar placement panel {placement.panel!r} has a non-positive domain"
        )
    if placement.side == "right":
        x = x_high + placement.offset_fraction * width
        x_anchor = "left"
    else:
        x = x_low - placement.offset_fraction * width
        x_anchor = "right"
    return {
        "len": placement.length_fraction * height,
        "lenmode": "fraction",
        "x": x,
        "xanchor": x_anchor,
        "y": y_low + placement.center_fraction * height,
        "yanchor": "middle",
    }


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
    if p.legend_x is not None:
        updates["legend_x"] = p.legend_x
    if p.legend_y is not None:
        updates["legend_y"] = p.legend_y
    if p.legend_xanchor is not None:
        updates["legend_xanchor"] = p.legend_xanchor
    if p.legend_yanchor is not None:
        updates["legend_yanchor"] = p.legend_yanchor
    if p.margin is not None:
        updates["margin"] = p.margin.model_dump(exclude_none=True)
    if p.hovermode is not None:
        updates["hovermode"] = p.hovermode
    if p.template is not None:
        updates["template"] = p.template
    if updates:
        fig.update_layout(**updates)
    if p.colorbar is not None:
        fig.add_trace(
            _colorbar_carrier(
                p.colorbar,
                _panel_colorbar_layout(fig, panels, p.colorbar),
            )
        )
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
            ProfileBandParams,
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
            "feedbax.trajectory_3d",
            "trace",
            _trajectory_3d,
            Trajectory3DParams,
            "3D trajectory traces for a scene panel.",
        ),
        (
            "feedbax.endpoint_markers",
            "trace",
            _endpoint_markers,
            EndpointMarkerParams,
            "Endpoint markers with trajectory-derived straight guides.",
        ),
        (
            "feedbax.scalar_scatter",
            "trace",
            _scalar_scatter,
            ScalarScatterParams,
            "Markers-only scalar scatter with optional symmetric per-point y error bars.",
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
        "feedbax.profile_band": "v2",
        "feedbax.trajectory_2d": "v5",
        "feedbax.endpoint_markers": "v3",
        "feedbax.hline": "v2",
        "feedbax.vrect": "v3",
        "feedbax.comparison_grid": "v4",
        "feedbax.grid_figure": "v5",
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
