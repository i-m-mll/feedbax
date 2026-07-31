"""3D trajectory constructor, scene panel, and panel-span contract tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import plotly.graph_objs as go
import pytest
from pydantic import ValidationError

from feedbax.analysis.figures import (
    FigureSpecExecutionError,
    execute_figure_spec,
    figure_manifest_plotly_json,
)
from feedbax.contracts.figures import (
    DEFAULT_SCENE_CAMERA_DISTANCE,
    SCENE_CAMERA_SCHEMA_ID,
    FigureSpec,
    PanelSpec,
    SceneCamera,
    TraceBinding,
    TraceFamily,
    TraceFamilyIndex,
    TraceFamilyRange,
)
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    ParentRef,
    safe_manifest_key,
    spec_payload,
    write_manifest,
)
from feedbax.plot.constructors import (
    PanelContent,
    _subplot_specs,
    get_figure_constructor,
)

pytestmark = [pytest.mark.feedbax_contract]

#: The first consumer's shape: four cardinal directions, distinguished by line
#: dash because color is already spent on the continuous conditioning variable,
#: at eleven levels of that variable.
DIRECTION_DASHES = {"east": "solid", "north": "dash", "west": "dot", "south": "dashdot"}
LEVEL_COUNT = 11
#: A geometrically spaced sweep, so uniform index spacing and value positioning
#: cannot accidentally agree.
LEVELS = [round(0.2 * 1.2**step, 6) for step in range(LEVEL_COUNT)]
COLORSCALE = "Viridis"


def _trajectory_3d(*, figure_registry) -> Any:
    return get_figure_constructor("feedbax.trajectory_3d", tier="trace", registry=figure_registry)


def _curve(direction_index: int, level_index: int) -> list[list[float]]:
    """Return one short 3D path that varies with both authored dimensions."""
    angle = direction_index * math.pi / 2.0
    scale = 1.0 + level_index
    return [
        [0.0, 0.0, 0.0],
        [math.cos(angle), math.sin(angle), 0.5 * scale],
        [2.0 * math.cos(angle) * scale, 2.0 * math.sin(angle) * scale, scale],
    ]


def _pca_input(root: Path) -> ParentRef:
    """Write the hidden-state payload the first consumer's figure reads."""
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:pc-trajectories",
        status="completed",
        metadata={
            direction: [_curve(direction_index, level) for level in range(LEVEL_COUNT)]
            for direction_index, direction in enumerate(DIRECTION_DASHES)
        },
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "feedbax.test"}),
    )
    write_manifest(manifest, root=root)
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        uri=f"manifests/analysis_runs/{safe_manifest_key(manifest.id)}.json",
    )


def test_trajectory_3d_carries_the_shared_trajectory_encodings(application_registry_bundle) -> None:
    constructor = _trajectory_3d(figure_registry=application_registry_bundle.figures)
    traces = constructor.callable(
        {
            "trajectories": [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            ],
            "colorscales": {"condition": COLORSCALE},
        },
        constructor.params(
            {
                "label": "east",
                "colorscale_key": "condition",
                "show_mean": False,
                "line_dash": "dashdot",
                "line_width": 2.0,
                "opacity": 0.6,
                "start_marker": {"label": "Reach start", "size": 10.0, "showlegend": False},
            }
        ),
    )

    assert [type(trace) for trace in traces] == [go.Scatter3d] * 4
    lines, marker = traces[:3], traces[-1]
    assert [list(trace.z) for trace in lines] == [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
    assert {trace.line.dash for trace in lines} == {"dashdot"}
    assert {trace.line.width for trace in lines} == {2.0}
    assert {trace.opacity for trace in lines} == {0.6}
    assert [trace.showlegend for trace in lines] == [True, False, False]
    assert len({trace.line.color for trace in lines}) == 3
    assert marker.mode == "markers"
    assert list(marker.marker.color) == [trace.line.color for trace in lines]
    assert marker.marker.size == 10.0
    assert marker.showlegend is False
    assert list(marker.z) == [0.0, 0.0, 0.0]


def test_trajectory_3d_default_params_leave_style_and_markers_untouched(
    application_registry_bundle,
) -> None:
    constructor = _trajectory_3d(figure_registry=application_registry_bundle.figures)
    assert constructor.version == "v1"
    params = constructor.params()
    assert params.line_dash is None
    assert params.start_marker is None

    traces = constructor.callable(
        {"trajectories": [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]]},
        params,
    )

    assert [trace.name for trace in traces] == ["Trajectory", "Trajectory", "Trajectory mean"]
    assert [trace.line.dash for trace in traces] == [None, None, None]
    assert list(traces[-1].z) == [0.0, 1.5]


def test_trajectory_3d_rejects_two_column_trajectories(application_registry_bundle) -> None:
    constructor = _trajectory_3d(figure_registry=application_registry_bundle.figures)

    with pytest.raises(ValueError, match=r"shape \(\.\.\., time, 3\)"):
        constructor.callable({"trajectories": [[[0, 0], [1, 1]]]}, constructor.params())


def test_trajectory_3d_rejects_perturbation_timing_rather_than_ignoring_it(
    application_registry_bundle,
) -> None:
    constructor = _trajectory_3d(figure_registry=application_registry_bundle.figures)

    with pytest.raises(ValueError, match="draws no perturbation underlay"):
        constructor.callable(
            {
                "trajectories": [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
                "perturbation_timing": {
                    "schema_id": "feedbax.spec.perturbation_timing",
                    "schema_version": "feedbax.spec.perturbation_timing.v1",
                    "applicability": "full_trial",
                    "sample_count": 3,
                },
            },
            constructor.params(),
        )


def test_trajectory_3d_rejects_a_symbol_only_the_2d_vocabulary_has(
    application_registry_bundle,
) -> None:
    constructor = _trajectory_3d(figure_registry=application_registry_bundle.figures)

    with pytest.raises(ValueError, match="symbol"):
        constructor.callable(
            {"trajectories": [[[0, 0, 0], [1, 1, 1]]]},
            constructor.params({"show_mean": False, "start_marker": {"symbol": "triangle-up"}}),
        )


def test_plain_panels_state_no_subplot_specs() -> None:
    """A grid of ordinary Cartesian cells is exactly what Plotly builds by default."""
    plain = [PanelContent(name="left", row=1, col=1), PanelContent(name="right", row=1, col=2)]
    assert _subplot_specs(plain, rows=1, cols=2) is None

    with_scene = [*plain, PanelContent(name="cube", row=2, col=1, panel_type="scene")]
    assert _subplot_specs(with_scene, rows=2, cols=2) == [
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "scene"}, {"type": "xy"}],
    ]


def test_panel_span_covers_its_cells_and_leaves_them_without_subplots() -> None:
    panels = [
        PanelContent(name="left", row=1, col=1),
        PanelContent(name="right", row=1, col=2),
        PanelContent(name="cube", row=2, col=1, panel_type="scene", col_span=2),
    ]

    assert _subplot_specs(panels, rows=2, cols=2) == [
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "scene", "colspan": 2}, None],
    ]


def test_overlapping_panel_spans_are_rejected() -> None:
    panels = [
        PanelContent(name="wide", row=1, col=1, col_span=2),
        PanelContent(name="narrow", row=1, col=2),
    ]

    with pytest.raises(ValueError, match="both claim grid cell"):
        _subplot_specs(panels, rows=1, cols=2)


def test_panel_span_leaving_the_grid_is_rejected() -> None:
    panels = [PanelContent(name="wide", row=1, col=1, col_span=3)]

    with pytest.raises(ValueError, match="leaves the 1x2 grid"):
        _subplot_specs(panels, rows=1, cols=2)


def test_panel_spec_rejects_a_third_axis_without_a_scene_panel() -> None:
    with pytest.raises(ValidationError, match="only a panel_type='scene' panel has"):
        PanelSpec(name="plane", z_axis={"range": (0.0, 1.0)})

    with pytest.raises(ValidationError, match="only a panel_type='scene' panel has"):
        PanelSpec(name="plane", axes_labels={"x": "PC1", "y": "PC2", "z": "PC3"})

    with pytest.raises(ValidationError, match="only a panel_type='scene' panel has"):
        PanelSpec(name="plane", camera={"azimuth": 45.0, "elevation": 20.0})

    scene = PanelSpec(
        name="cube",
        panel_type="scene",
        axes_labels={"x": "PC1", "y": "PC2", "z": "PC3"},
        z_axis={"range": (0.0, 1.0)},
    )
    assert scene.axes_labels is not None and scene.axes_labels.z == "PC3"


def test_scene_panel_renders_a_3d_trace_with_axis_titles_and_data_aspect(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    spec = FigureSpec(
        name="scene-panel",
        assembler="feedbax.grid_figure",
        panels=[
            PanelSpec(name="plane", row=1, col=1, axes_labels={"x": "PC1", "y": "PC2"}),
            PanelSpec(
                name="cube",
                row=2,
                col=1,
                col_span=2,
                panel_type="scene",
                axes_labels={"x": "PC1", "y": "PC2", "z": "PC3"},
                equal_data_aspect={},
            ),
        ],
        traces=[
            TraceBinding(
                name="plane-reach",
                constructor="feedbax.trajectory_2d",
                panel="plane",
                data={"trajectories": [[[0.0, 0.0], [1.0, 1.0]]]},
            ),
            TraceBinding(
                name="cube-reach",
                constructor="feedbax.trajectory_3d",
                panel="cube",
                data={"trajectories": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]},
            ),
        ],
    )

    manifest, _path = execute_figure_spec(
        spec, root=tmp_path, registry=application_registry_bundle.figures
    )
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    types = [trace["type"] for trace in rendered["data"]]
    assert types == ["scatter", "scatter3d"]
    scene = rendered["layout"]["scene"]
    assert scene["aspectmode"] == "data"
    assert scene["xaxis"]["title"]["text"] == "PC1"
    assert scene["yaxis"]["title"]["text"] == "PC2"
    assert scene["zaxis"]["title"]["text"] == "PC3"
    # The spanning scene owns the whole second row of the paper.
    assert tuple(scene["domain"]["x"]) == (0.0, 1.0)
    assert manifest.constructor_versions["feedbax.trajectory_3d"] == "v1"


def _scene_camera_spec(camera: Any | None) -> FigureSpec:
    """One data-aspect scene panel, with or without an authored viewpoint."""
    panel: dict[str, Any] = {
        "name": "cube",
        "row": 1,
        "col": 1,
        "panel_type": "scene",
        "axes_labels": {"x": "PC1", "y": "PC2", "z": "PC3"},
        "equal_data_aspect": {},
    }
    if camera is not None:
        panel["camera"] = camera
    return FigureSpec(
        name="scene-camera",
        assembler="feedbax.grid_figure",
        panels=[PanelSpec(**panel)],
        traces=[
            TraceBinding(
                name="cube-reach",
                constructor="feedbax.trajectory_3d",
                panel="cube",
                data={"trajectories": [[[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]]]},
            )
        ],
    )


def _rendered_scene(spec: FigureSpec, root: Path, *, figure_registry) -> dict[str, Any]:
    manifest, _path = execute_figure_spec(spec, root=root, registry=figure_registry)
    rendered = figure_manifest_plotly_json(manifest)
    assert rendered is not None
    return rendered["layout"]["scene"]


def test_omitting_the_camera_leaves_the_renderer_default_view(
    tmp_path: Path, application_registry_bundle
) -> None:
    """A scene that states no viewpoint carries no camera key anywhere.

    The authored identity projection is the exclude-none dump
    ``execute_figure_spec`` hashes, so an undeclared camera contributes no
    identity bytes and no rendered layout key.
    """
    spec = _scene_camera_spec(None)
    authored = spec.model_dump(mode="json", exclude_none=True)

    assert "camera" not in authored["panels"][0]
    scene = _rendered_scene(spec, tmp_path, figure_registry=application_registry_bundle.figures)
    assert "camera" not in scene
    assert scene["aspectmode"] == "data"


def test_authored_camera_lowers_to_an_eye_and_composes_with_the_data_aspect(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    """The viewpoint reaches the scene without disturbing its aspect mode."""
    scene = _rendered_scene(
        _scene_camera_spec(
            {
                "azimuth": 0.0,
                "elevation": 90.0,
                "distance": 2.0,
                "projection": "orthographic",
            }
        ),
        tmp_path,
        figure_registry=application_registry_bundle.figures,
    )

    # Straight down the vertical axis: the degenerate case for an authored `up`.
    assert scene["camera"]["eye"] == pytest.approx({"x": 0.0, "y": 0.0, "z": 2.0}, abs=1e-12)
    assert scene["camera"]["up"] == pytest.approx({"x": -1.0, "y": 0.0, "z": 0.0}, abs=1e-12)
    assert scene["camera"]["projection"] == {"type": "orthographic"}
    # The aspect mode is the shape of the box and survives the camera untouched.
    assert scene["aspectmode"] == "data"
    assert scene["zaxis"]["title"]["text"] == "PC3"


def test_camera_omits_projection_when_it_is_not_authored(
    tmp_path: Path, application_registry_bundle
) -> None:
    scene = _rendered_scene(
        _scene_camera_spec({"azimuth": 135.0, "elevation": -20.0}),
        tmp_path,
        figure_registry=application_registry_bundle.figures,
    )

    assert "projection" not in scene["camera"]
    eye = scene["camera"]["eye"]
    assert math.hypot(eye["x"], eye["y"], eye["z"]) == pytest.approx(DEFAULT_SCENE_CAMERA_DISTANCE)


def test_scene_camera_has_explicit_schema_identity() -> None:
    camera = SceneCamera(azimuth=45.0, elevation=35.0)
    assert camera.schema_id == SCENE_CAMERA_SCHEMA_ID
    assert camera.schema_version == "feedbax.spec.scene_camera.v1"

    with pytest.raises(ValidationError, match="unsupported SceneCamera schema_version"):
        SceneCamera(
            azimuth=45.0,
            elevation=35.0,
            schema_version="feedbax.spec.scene_camera.v0",
        )


def test_scene_camera_up_is_perpendicular_to_every_view_direction() -> None:
    """The blank-render case — up parallel to the view — cannot be stated."""
    for azimuth in (-180.0, -37.5, 0.0, 90.0, 359.0):
        for elevation in (-90.0, -45.0, 0.0, 12.5, 90.0):
            camera = SceneCamera(azimuth=azimuth, elevation=elevation)
            eye = camera.eye_vector()
            up = camera.up_vector()
            assert sum(a * b for a, b in zip(eye, up, strict=True)) == pytest.approx(0.0, abs=1e-12)
            assert math.hypot(*up) == pytest.approx(1.0)


def test_scene_camera_default_distance_reproduces_the_renderer_default_view() -> None:
    """Authoring the default view explicitly gives back the default eye."""
    default = SceneCamera.from_eye(1.25, 1.25, 1.25)

    assert default.azimuth == pytest.approx(45.0)
    assert default.elevation == pytest.approx(math.degrees(math.asin(1 / math.sqrt(3))))
    assert default.distance == pytest.approx(DEFAULT_SCENE_CAMERA_DISTANCE)
    assert default.eye_vector() == pytest.approx((1.25, 1.25, 1.25))

    # Plotly's default `up` of (0, 0, 1) resolves to its own projection onto the
    # view plane, which is exactly the derived up direction.
    eye = default.eye_vector()
    norm = math.hypot(*eye)
    unit_eye = tuple(component / norm for component in eye)
    projected = tuple(
        vertical - unit_eye[2] * component
        for vertical, component in zip((0.0, 0.0, 1.0), unit_eye, strict=True)
    )
    projected_norm = math.hypot(*projected)
    assert default.up_vector() == pytest.approx(
        tuple(component / projected_norm for component in projected)
    )


def test_scene_camera_round_trips_an_eye_read_off_a_rendered_figure() -> None:
    for eye in ((1.6, -0.9, 0.4), (-2.0, 0.0, 2.0), (0.0, 0.0, 1.5)):
        assert SceneCamera.from_eye(*eye).eye_vector() == pytest.approx(eye, abs=1e-12)


def test_scene_camera_rejects_viewpoints_that_would_render_nothing() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        SceneCamera(azimuth=0.0, elevation=0.0, distance=0.0)

    with pytest.raises(ValidationError, match="less than or equal to 90"):
        SceneCamera(azimuth=0.0, elevation=91.0)

    with pytest.raises(ValidationError, match=r"\['azimuth'\] must be finite"):
        SceneCamera(azimuth=math.inf, elevation=0.0)

    with pytest.raises(ValidationError, match=r"\['distance'\] must be finite"):
        SceneCamera(azimuth=0.0, elevation=0.0, distance=math.inf)

    with pytest.raises(ValueError, match="names no viewpoint"):
        SceneCamera.from_eye(0.0, 0.0, 0.0)


def test_scene_panel_refuses_a_cartesian_trace(tmp_path: Path, application_registry_bundle) -> None:
    spec = FigureSpec(
        name="scene-panel-mismatch",
        assembler="feedbax.grid_figure",
        panels=[PanelSpec(name="cube", row=1, col=1, panel_type="scene")],
        traces=[
            TraceBinding(
                name="plane-reach",
                constructor="feedbax.trajectory_2d",
                panel="cube",
                data={"trajectories": [[[0.0, 0.0], [1.0, 1.0]]]},
            )
        ],
    )

    with pytest.raises(FigureSpecExecutionError) as failure:
        execute_figure_spec(spec, root=tmp_path, registry=application_registry_bundle.figures)
    assert "not compatible with subplot type 'scene'" in str(failure.value.__cause__)


def test_panel_grid_declarations_require_a_grid_panel_assembler(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec = FigureSpec(
        name="scene-without-grid-assembler",
        assembler="feedbax.trajectories_2d_row",
        assembler_params={"panel_constructor": "feedbax.test_panel_assembler"},
        panels=[PanelSpec(name="cube", row=1, col=1, panel_type="scene")],
        traces=[
            TraceBinding(
                name="cube-reach",
                constructor="feedbax.trajectory_3d",
                panel="cube",
                data={"trajectories": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]},
            )
        ],
    )
    from feedbax.plot.constructors import register_figure_constructor

    register_figure_constructor(
        "feedbax.test_panel_assembler",
        tier="panel",
        constructor=lambda _panels, _params: go.Figure(),
        description="panel assembler without grid composition",
        registry=application_registry_bundle.figures,
    )
    with pytest.raises(FigureSpecExecutionError) as failure:
        execute_figure_spec(spec, root=tmp_path, registry=application_registry_bundle.figures)
    assert "declares ['panel_type']" in str(failure.value.__cause__)


def test_colorbar_is_placed_beside_a_scene_panel(
    tmp_path: Path, application_registry_bundle
) -> None:
    spec = FigureSpec(
        name="scene-colorbar",
        assembler="feedbax.grid_figure",
        panels=[PanelSpec(name="cube", row=1, col=1, panel_type="scene")],
        colorbar={
            "title": "level",
            "colorscale": COLORSCALE,
            "range": (0.0, 1.0),
            "placement": {
                "panel": "cube",
                "length_fraction": 0.8,
                "side": "right",
                "offset_fraction": 0.05,
            },
        },
        traces=[
            TraceBinding(
                name="cube-reach",
                constructor="feedbax.trajectory_3d",
                panel="cube",
                data={"trajectories": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]},
            )
        ],
    )

    manifest, _path = execute_figure_spec(
        spec, root=tmp_path, registry=application_registry_bundle.figures
    )
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    carrier = rendered["data"][-1]
    colorbar = carrier["marker"]["colorbar"]
    # The scene spans the whole paper, so the bar sits just outside its right edge.
    assert colorbar["x"] == pytest.approx(1.05)
    assert colorbar["xanchor"] == "left"
    assert colorbar["len"] == pytest.approx(0.8)
    assert colorbar["title"] == {"text": "level"}


def _direction_family(direction: str, dash: str) -> TraceFamily:
    """One reach direction as eleven value-positioned single-trajectory calls."""
    return TraceFamily(
        name=f"{direction}-levels",
        index=TraceFamilyIndex(range=TraceFamilyRange(stop=LEVEL_COUNT)),
        colorscale=COLORSCALE,
        values=LEVELS,
        legend_index=0,
        trace=TraceBinding(
            name=f"{direction} {{index}}",
            constructor="feedbax.trajectory_3d",
            panel="cube",
            required=True,
            data={"trajectories": {"item": "analysis", "path": f"metadata.{direction}.{{index}}"}},
            params={
                "label": direction,
                "show_mean": False,
                "line_dash": dash,
                "start_marker": {"label": f"{direction} start", "size": 10.0, "showlegend": False},
            },
        ),
    )


def test_first_consumer_figure_renders_end_to_end(
    tmp_path: Path, application_registry_bundle
) -> None:
    """Two 2D panels above one full-width 3D panel, with the S4 encodings.

    Forty-four trajectories reach the scene panel as forty-four separate
    constructor calls, which is what value-positioned color requires: each level
    is colored at its own fraction of the swept range rather than at uniform
    index spacing. Direction is carried by line dash, the legend holds one entry
    per direction, every trajectory's first sample carries a marker in its own
    color, and one shared colorbar keys the conditioning variable.
    """
    analysis = _pca_input(tmp_path)
    spec = FigureSpec(
        name="s4-hidden-state-overlay",
        assembler="feedbax.grid_figure",
        assembler_params={"shared_yaxes": False, "width": 900, "height": 900},
        inputs=[analysis],
        panels=[
            PanelSpec(
                name="pc12",
                row=1,
                col=1,
                axes_labels={"x": "PC1", "y": "PC2"},
                equal_data_aspect={},
            ),
            PanelSpec(
                name="pc23",
                row=1,
                col=2,
                axes_labels={"x": "PC2", "y": "PC3"},
                equal_data_aspect={},
            ),
            PanelSpec(
                name="cube",
                row=2,
                col=1,
                col_span=2,
                panel_type="scene",
                axes_labels={"x": "PC1", "y": "PC2", "z": "PC3"},
                equal_data_aspect={},
            ),
        ],
        colorbar={
            "title": "conditioning level",
            "family": "east-levels",
            "placement": {"panel": "cube", "length_fraction": 0.8, "offset_fraction": 0.02},
        },
        trace_families=[
            _direction_family(direction, dash) for direction, dash in DIRECTION_DASHES.items()
        ],
    )

    manifest, _path = execute_figure_spec(
        spec, root=tmp_path, registry=application_registry_bundle.figures
    )
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    traces = rendered["data"]
    # One line and one start marker per level per direction, plus the colorbar carrier.
    assert len(traces) == len(DIRECTION_DASHES) * LEVEL_COUNT * 2 + 1
    scientific = traces[:-1]
    lines = [trace for trace in scientific if trace.get("mode") == "lines"]
    markers = [trace for trace in scientific if trace.get("mode") == "markers"]
    assert {trace["type"] for trace in lines} == {"scatter3d"}
    assert {trace["type"] for trace in markers} == {"scatter3d"}

    # One legend entry per direction, and nothing else.
    assert [trace["name"] for trace in traces if trace.get("showlegend")] == list(DIRECTION_DASHES)
    for direction, dash in DIRECTION_DASHES.items():
        block = [trace for trace in lines if trace["name"] == direction]
        assert len(block) == LEVEL_COUNT
        assert {trace["line"]["dash"] for trace in block} == {dash}

    # The same level takes the same color in every direction, positioned by value.
    by_level = list(
        zip(
            *[
                [trace["line"]["color"] for trace in lines if trace["name"] == direction]
                for direction in DIRECTION_DASHES
            ],
            strict=True,
        )
    )
    assert all(len(set(level_colors)) == 1 for level_colors in by_level)
    level_colors = [level[0] for level in by_level]
    assert len(set(level_colors)) == LEVEL_COUNT

    # Each start marker inherits the color of the single trajectory it marks.
    assert [trace["marker"]["color"][0] for trace in markers if trace["name"] == "east start"] == (
        level_colors
    )

    # The colorbar's stops are the family's own value-positioned assignment.
    stops = traces[-1]["marker"]["colorscale"]
    assert [color for _position, color in stops] == level_colors
    span = LEVELS[-1] - LEVELS[0]
    assert [position for position, _color in stops] == pytest.approx(
        [(level - LEVELS[0]) / span for level in LEVELS]
    )
    assert traces[-1]["marker"]["cmin"] == pytest.approx(LEVELS[0])
    assert traces[-1]["marker"]["cmax"] == pytest.approx(LEVELS[-1])

    scene = rendered["layout"]["scene"]
    assert scene["aspectmode"] == "data"
    assert scene["zaxis"]["title"]["text"] == "PC3"
    assert tuple(scene["domain"]["x"]) == (0.0, 1.0)
    assert rendered["layout"]["yaxis"]["scaleanchor"] == "x"
    assert rendered["layout"]["yaxis2"]["scaleanchor"] == "x2"
