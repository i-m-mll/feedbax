from __future__ import annotations

import base64
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from pydantic import ValidationError

import feedbax.plugins
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.figures import (
    FIGURE_RENDER_MEDIA_TYPES,
    FIGURE_RENDER_ROLE,
    FigureSpecExecutionError,
    execute_figure_spec,
    figure_manifest_plotly_json,
)
from feedbax.analysis.bundles import AnalysisBundleSpec, BundleStageSpec, StageArtifactDependency
from feedbax.contracts.figures import (
    FIGURE_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_VERSION,
    FigurePiece,
    FigureSpec,
    FigureTemplate,
    SlotSpec,
    TraceBinding,
)
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    StrictModel,
    canonical_json_bytes,
    load_manifest,
    spec_payload,
    write_manifest,
    safe_manifest_key,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.plot.constructors import (
    PanelContent,
    constructor_catalog,
    get_figure_constructor,
    get_figure_piece,
    get_figure_template,
    load_figure_piece,
    load_figure_template,
    register_figure_piece,
    register_figure_constructor,
    register_figure_template,
    registered_figure_constructors,
)
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

pytestmark = [pytest.mark.feedbax_contract]


def _plotly_array_values(value: object) -> list[float]:
    if isinstance(value, dict):
        return np.frombuffer(
            base64.b64decode(value["bdata"]),
            dtype=np.dtype(value["dtype"]),
        ).tolist()
    return list(value)  # type: ignore[arg-type]


def _analysis_manifest(root: Path) -> AnalysisRunManifest:
    product = AnalysisDataProduct(
        product_schema_id="test.figure_payload",
        product_schema_version="test.figure_payload.v1",
        role="profiles",
        logical_name="profiles",
        producer_manifest_id="feedbax-analysis-run:test",
        materialization={
            "x": [0, 1, 2],
            "y": [[1, 2, 3], [2, 3, 4]],
            "optional": [5, 6, 7],
        },
    )
    manifest = AnalysisRunManifest(
        id="feedbax-analysis-run:test",
        status="completed",
        metadata={
            "x": [0, 1, 2],
            "y": [[1, 2, 3], [2, 3, 4]],
            "series": {
                "slow": [[1, 2, 3], [2, 3, 4]],
                "fast": [[4, 3, 2], [3, 2, 1]],
                "flat": [[2, 2, 2], [2, 2, 2]],
            },
        },
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "feedbax.test"}),
        produced_data=[product],
    )
    write_manifest(manifest, root=root)
    return manifest


def _contained_analysis_ref(manifest: AnalysisRunManifest) -> ParentRef:
    return ParentRef(
        kind=manifest.kind,
        id=manifest.id,
        role="analysis",
        uri=f"manifests/analysis_runs/{safe_manifest_key(manifest.id)}.json",
    )


def test_figure_spec_schema_identity_rejects_old_versions() -> None:
    current = FigureSpec(name="demo", assembler="feedbax.grid_figure")
    assert current.schema_id == FIGURE_SPEC_SCHEMA_ID
    assert current.schema_version == FIGURE_SPEC_SCHEMA_VERSION

    for old_version in ("feedbax.spec.figure.v0", "feedbax.spec.figure.v1"):
        with pytest.raises(ValidationError, match="unsupported FigureSpec schema_version"):
            FigureSpec.model_validate(
                {
                    "schema_id": FIGURE_SPEC_SCHEMA_ID,
                    "schema_version": old_version,
                    "name": "old",
                    "assembler": "feedbax.grid_figure",
                }
            )

    with pytest.raises(UnsupportedSpecVersion):
        spec_payload(
            "FigureSpec",
            {
                "schema_id": FIGURE_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.spec.figure.v0",
                "name": "old",
                "assembler": "feedbax.grid_figure",
            },
        )
    assert default_spec_registry.current_version("FigureSpec") == FIGURE_SPEC_SCHEMA_VERSION


def test_constructor_registry_validates_tiers_and_duplicates() -> None:
    keys = {item.key for item in registered_figure_constructors()}
    assert "feedbax.profile_band" in keys
    assert get_figure_constructor("feedbax.profile_band", tier="trace").tier == "trace"

    with pytest.raises(ValueError, match="expected 'panel'"):
        get_figure_constructor("feedbax.profile_band", tier="panel")

    def trace(_data, _params):
        return []

    register_figure_constructor(
        "feedbax.test_trace",
        tier="trace",
        constructor=trace,
        description="test trace",
        replace=True,
    )
    with pytest.raises(ValueError, match="already registered"):
        register_figure_constructor(
            "feedbax.test_trace",
            tier="trace",
            constructor=trace,
            description="test trace",
        )


def test_constructor_versions_are_reported_in_catalog_and_manifest(tmp_path: Path) -> None:
    expected = {
        "feedbax.profile_band": "v2",
        "feedbax.profile_curves": "v1",
        "feedbax.comparison_grid": "v2",
        "feedbax.grid_figure": "v3",
    }
    catalog_versions = {item["key"]: item["version"] for item in constructor_catalog()}
    assert {key: catalog_versions[key] for key in expected} == expected

    spec = FigureSpec(
        name="constructor-versions",
        assembler="feedbax.grid_figure",
        traces=[
            TraceBinding(
                name="band",
                constructor="feedbax.profile_band",
                data={"y": [[1.0, 2.0]]},
            ),
            TraceBinding(
                name="curves",
                constructor="feedbax.profile_curves",
                data={"y": [[1.0, 2.0]]},
            ),
        ],
    )
    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    assert manifest.constructor_versions == expected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("line_dash", "dash"),
        ("line_width", 2.0),
        ("showlegend", False),
        ("show_band", False),
    ],
)
def test_profile_band_only_params_are_scoped_to_profile_band(key: str, value: object) -> None:
    band = get_figure_constructor("feedbax.profile_band", tier="trace")
    curves = get_figure_constructor("feedbax.profile_curves", tier="trace")

    assert getattr(band.params({key: value}), key) == value
    with pytest.raises(ValidationError, match=key):
        curves.params({key: value})


def test_profile_band_uses_supplied_statistics_without_touching_raw_samples() -> None:
    class ExplosiveRawSamples:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("raw samples were evaluated")

    constructor = get_figure_constructor("feedbax.profile_band", tier="trace")
    params = constructor.params({"error_bars_alpha": 0.6, "n_std_plot": 2.0})
    derived = {"mean": [2.0, 4.0, 6.0], "std": [0.5, 1.0, 1.5]}

    for data in (derived, {**derived, "y": ExplosiveRawSamples()}):
        traces = constructor.callable(data, params)
        np.testing.assert_allclose(traces[0].x, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(traces[0].y, [2.0, 4.0, 6.0])
        np.testing.assert_allclose(traces[1].y, [3.0, 6.0, 9.0])
        np.testing.assert_allclose(traces[2].y, [1.0, 2.0, 3.0])
        assert traces[2].fillcolor == "rgba(31,119,180, 0.6)"


def test_profile_band_raw_samples_preserve_statistics_and_alpha_scaling() -> None:
    constructor = get_figure_constructor("feedbax.profile_band", tier="trace")
    params = constructor.params({"error_bars_alpha": 0.8, "n_std_plot": 2.0})
    samples = np.array(
        [
            [1.0, np.nan, 3.0],
            [3.0, 5.0, 7.0],
            [5.0, 7.0, 11.0],
            [7.0, 9.0, 15.0],
        ]
    )

    traces = constructor.callable({"y": samples}, params)
    expected_mean = np.nanmean(samples, axis=0)
    expected_std = np.nanstd(samples, axis=0)
    np.testing.assert_allclose(traces[0].y, expected_mean)
    np.testing.assert_allclose(traces[1].y, expected_mean + 2.0 * expected_std)
    np.testing.assert_allclose(traces[2].y, expected_mean - 2.0 * expected_std)
    assert traces[2].fillcolor == "rgba(31,119,180, 0.4)"

    single_traces = constructor.callable({"values": [1.0, 2.0, 3.0]}, params)
    np.testing.assert_allclose(single_traces[0].y, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(single_traces[1].y, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(single_traces[2].y, [1.0, 2.0, 3.0])
    assert single_traces[2].fillcolor == "rgba(31,119,180, 0.8)"


def test_profile_band_applies_mean_style_and_can_suppress_band(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="styled-mean-only-profile",
        assembler="feedbax.grid_figure",
        traces=[
            TraceBinding(
                name="profile",
                constructor="feedbax.profile_band",
                data={"y": [[1, 2, 3], [2, 3, 4]]},
                params={
                    "line_dash": "dashdot",
                    "line_width": 3.5,
                    "showlegend": False,
                    "show_band": False,
                },
            )
        ],
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    assert len(rendered["data"]) == 1
    assert rendered["data"][0]["line"]["dash"] == "dashdot"
    assert rendered["data"][0]["line"]["width"] == 3.5
    assert rendered["data"][0]["showlegend"] is False


@pytest.mark.parametrize(
    "data",
    [
        {"mean": [1.0, 2.0], "y": [[1.0, 2.0]]},
        {"std": [0.1, 0.2]},
    ],
)
def test_profile_band_rejects_partial_derived_statistics(data) -> None:
    constructor = get_figure_constructor("feedbax.profile_band", tier="trace")

    with pytest.raises(ValueError, match="requires both 'mean' and 'std'"):
        constructor.callable(data, constructor.params())


def test_profile_band_rejects_missing_statistics_and_samples() -> None:
    constructor = get_figure_constructor("feedbax.profile_band", tier="trace")

    with pytest.raises(ValueError, match="or raw 'y'/'values' samples"):
        constructor.callable({}, constructor.params())


def test_profile_band_rejects_invalid_plotly_dash() -> None:
    constructor = get_figure_constructor("feedbax.profile_band", tier="trace")

    with pytest.raises(ValueError, match="dash"):
        constructor.callable(
            {"mean": [1.0, 2.0], "std": [0.1, 0.2]},
            constructor.params({"line_dash": "not-a-plotly-dash"}),
        )


def test_execute_figure_spec_records_optional_omission_and_custody(tmp_path: Path) -> None:
    manifest = _analysis_manifest(tmp_path)
    spec = FigureSpec(
        name="profile-demo",
        assembler="feedbax.grid_figure",
        inputs=[_contained_analysis_ref(manifest)],
        panels=[{"name": "main", "title": "Demo"}],
        traces=[
            TraceBinding(
                name="present",
                constructor="feedbax.profile_band",
                panel="main",
                required=True,
                data={
                    "x": {"item": "analysis", "path": "metadata.x"},
                    "y": {"item": "analysis", "path": "metadata.y"},
                },
            ),
            TraceBinding(
                name="missing-optional",
                constructor="feedbax.profile_band",
                panel="main",
                required=False,
                data={"y": {"item": "analysis", "path": "missing.path"}},
            ),
        ],
    )

    figure_manifest, path = execute_figure_spec(spec, root=tmp_path)
    loaded = load_manifest(path)
    assert loaded == figure_manifest
    assert figure_manifest.kind == "FigureManifest"
    assert figure_manifest.status == "completed"
    assert any(artifact.role == FIGURE_RENDER_ROLE for artifact in figure_manifest.artifacts)
    assert {record.name: record.status for record in figure_manifest.binding_records} == {
        "present": "included",
        "missing-optional": "omitted",
    }
    assert figure_manifest.expression_results_digest


@pytest.mark.parametrize(
    ("render_format", "expected_media_type"),
    [
        ("json", "application/json"),
        ("html", "text/html"),
        ("png", "image/png"),
        ("pdf", "application/pdf"),
        ("svg", "image/svg+xml"),
    ],
)
def test_execute_figure_spec_labels_render_artifact_with_written_media_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    render_format: str,
    expected_media_type: str,
) -> None:
    """The stored render artifact carries the media type of the format actually written."""

    def _write_image_without_kaleido(self: go.Figure, path: str, *args, **kwargs) -> None:
        Path(path).write_bytes(b"render-bytes")

    monkeypatch.setattr(go.Figure, "write_image", _write_image_without_kaleido)

    manifest = _analysis_manifest(tmp_path)
    spec = FigureSpec(
        name=f"render-media-type-{render_format}",
        assembler="feedbax.grid_figure",
        inputs=[_contained_analysis_ref(manifest)],
        panels=[{"name": "main"}],
        traces=[
            TraceBinding(
                name="profile",
                constructor="feedbax.profile_band",
                panel="main",
                data={
                    "x": {"item": "analysis", "path": "metadata.x"},
                    "y": {"item": "analysis", "path": "metadata.y"},
                },
            )
        ],
        figure_routing={"render_format": render_format},
    )

    figure_manifest, _path = execute_figure_spec(spec, root=tmp_path)
    renders = [
        artifact
        for artifact in figure_manifest.artifacts
        if artifact.role == FIGURE_RENDER_ROLE
        and artifact.metadata.get("format") == render_format
    ]

    assert len(renders) == 1
    assert renders[0].media_type == expected_media_type
    assert renders[0].media_type in FIGURE_RENDER_MEDIA_TYPES
    assert figure_manifest_plotly_json(figure_manifest) is not None


def test_execute_figure_spec_routes_panel_and_figure_assembler_params(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="routed-assembler-params",
        assembler="feedbax.grid_figure",
        assembler_params={
            "width": 1320,
            "height": 520,
            "title": "Declarative whole-figure layout",
            "legend_tracegroupgap": 1,
            "shared_yaxes": False,
            "horizontal_spacing": 0.2,
        },
        panels=[
            {"name": "left", "row": 1, "col": 1},
            {"name": "right", "row": 1, "col": 2},
        ],
        traces=[
            TraceBinding(
                name="left-profile",
                constructor="feedbax.profile_band",
                panel="left",
                data={"y": [[1, 2, 3], [2, 3, 4]]},
            ),
            TraceBinding(
                name="right-profile",
                constructor="feedbax.profile_band",
                panel="right",
                data={"y": [[4, 5, 6], [5, 6, 7]]},
            ),
        ],
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    layout = rendered["layout"]
    assert layout["width"] == 1320
    assert layout["height"] == 520
    assert layout["title"]["text"] == "Declarative whole-figure layout"
    assert layout["legend"]["tracegroupgap"] == 1
    assert layout["xaxis"]["domain"] == [0.0, 0.4]
    assert layout["xaxis2"]["domain"] == [0.6000000000000001, 1.0]
    assert "matches" not in layout["yaxis2"]


def test_execute_figure_spec_emits_panel_axes_and_figure_chrome(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="panel-axes-and-figure-chrome",
        assembler="feedbax.grid_figure",
        assembler_params={
            "shared_yaxes": False,
            "legend_x": 1.05,
            "legend_y": 0.5,
            "legend_xanchor": "left",
            "legend_yanchor": "middle",
            "margin": {
                "l": 12,
                "r": 30,
                "t": 40,
                "b": 20,
                "pad": 3,
                "autoexpand": False,
            },
            "hovermode": "x unified",
            "template": "plotly_white",
        },
        panels=[
            {
                "name": "left",
                "row": 1,
                "col": 1,
                "x_axis": {"range": [0, 5]},
                "y_axis": {"type": "log", "range": [-3, 1]},
            },
            {"name": "right", "row": 1, "col": 2, "y_axis": {"range": [-1, 2]}},
        ],
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    layout = rendered["layout"]
    assert layout["xaxis"]["range"] == [0.0, 5.0]
    assert layout["yaxis"]["type"] == "log"
    assert layout["yaxis"]["range"] == [-3.0, 1.0]
    assert layout["yaxis2"]["range"] == [-1.0, 2.0]
    assert layout["legend"] == {
        "x": 1.05,
        "xanchor": "left",
        "y": 0.5,
        "yanchor": "middle",
    }
    assert layout["margin"] == {
        "autoexpand": False,
        "b": 20.0,
        "l": 12.0,
        "pad": 3.0,
        "r": 30.0,
        "t": 40.0,
    }
    assert layout["hovermode"] == "x unified"
    assert layout["template"]["layout"]["paper_bgcolor"] == "white"


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"legend_xanchor": "outside"}, "legend_xanchor"),
        ({"hovermode": "diagonal"}, "hovermode"),
        ({"margin": {"top": 10}}, "margin.top"),
    ],
)
def test_grid_figure_rejects_invalid_typed_chrome(params, match: str) -> None:
    constructor = get_figure_constructor("feedbax.grid_figure", tier="figure")

    with pytest.raises(ValidationError, match=match):
        constructor.params(params)


@pytest.mark.parametrize(("params", "present"), [({}, False), ({"hovermode": False}, True)])
def test_grid_figure_distinguishes_omitted_and_false_hovermode(params, present: bool) -> None:
    panel_constructor = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    figure_constructor = get_figure_constructor("feedbax.grid_figure", tier="figure")
    fig = panel_constructor.callable([], panel_constructor.params())

    rendered = figure_constructor.callable(
        fig, [], figure_constructor.params(params)
    ).to_plotly_json()

    assert ("hovermode" in rendered["layout"]) is present
    if present:
        assert rendered["layout"]["hovermode"] is False


def test_panel_axis_rejects_invalid_type_and_range_shape() -> None:
    with pytest.raises(ValidationError, match="panels.0.y_axis.type"):
        FigureSpec(
            name="invalid-axis-type",
            assembler="feedbax.grid_figure",
            panels=[{"name": "main", "y_axis": {"type": "polar"}}],
        )
    with pytest.raises(ValidationError, match="panels.0.y_axis.range"):
        FigureSpec(
            name="invalid-axis-range",
            assembler="feedbax.grid_figure",
            panels=[{"name": "main", "y_axis": {"range": [0, 1, 2]}}],
        )
    with pytest.raises(ValidationError, match="panels.0.x_axis.secondary_y"):
        FigureSpec(
            name="invalid-axis-extra-key",
            assembler="feedbax.grid_figure",
            panels=[{"name": "main", "x_axis": {"secondary_y": True}}],
        )


def test_panel_only_assembler_params_preserve_constructor_payload(tmp_path: Path) -> None:
    spec = FigureSpec(
        name="panel-only-assembler-params",
        assembler="feedbax.grid_figure",
        assembler_params={"shared_yaxes": False, "horizontal_spacing": 0.2},
        panels=[
            {"name": "left", "row": 1, "col": 1},
            {"name": "right", "row": 1, "col": 2},
        ],
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    panel_contents = [
        PanelContent(name="left", row=1, col=1),
        PanelContent(name="right", row=1, col=2),
    ]
    panel_registration = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    figure_registration = get_figure_constructor("feedbax.grid_figure", tier="figure")
    expected = panel_registration.callable(
        panel_contents,
        panel_registration.params(spec.assembler_params),
    )
    expected = figure_registration.callable(
        expected,
        panel_contents,
        figure_registration.params(),
    )

    assert rendered is not None
    assert canonical_json_bytes(rendered) == canonical_json_bytes(json.loads(expected.to_json()))


def test_execute_figure_spec_rejects_unowned_assembler_param_without_render(
    tmp_path: Path,
) -> None:
    spec = FigureSpec(
        name="unknown-assembler-param",
        assembler="feedbax.grid_figure",
        assembler_params={"unknown_layout_field": True},
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)

    assert exc_info.value.manifest.status == "failed"
    assert exc_info.value.manifest.artifacts == []
    assert "unknown_layout_field" in exc_info.value.manifest.failure["message"]
    assert not (tmp_path / "figures").exists()


def test_execute_figure_spec_rejects_assembler_param_owned_by_both_models(
    tmp_path: Path,
) -> None:
    class AmbiguousPanelParams(StrictModel):
        shared_setting: str | None = None

    class AmbiguousFigureParams(StrictModel):
        panel_constructor: str = "feedbax.test_ambiguous_panel"
        shared_setting: str | None = None

    panel = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    figure = get_figure_constructor("feedbax.grid_figure", tier="figure")
    register_figure_constructor(
        "feedbax.test_ambiguous_panel",
        tier="panel",
        constructor=panel.callable,
        params_model=AmbiguousPanelParams,
        description="Panel constructor with an intentionally ambiguous parameter.",
        replace=True,
    )
    register_figure_constructor(
        "feedbax.test_ambiguous_figure",
        tier="figure",
        constructor=figure.callable,
        params_model=AmbiguousFigureParams,
        description="Figure constructor with an intentionally ambiguous parameter.",
        replace=True,
    )
    spec = FigureSpec(
        name="ambiguous-assembler-param",
        assembler="feedbax.test_ambiguous_figure",
        assembler_params={
            "panel_constructor": "feedbax.test_ambiguous_panel",
            "shared_setting": "conflict",
        },
    )

    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)

    assert exc_info.value.manifest.status == "failed"
    assert exc_info.value.manifest.artifacts == []
    assert "shared_setting" in exc_info.value.manifest.failure["message"]
    assert "owned by both" in exc_info.value.manifest.failure["message"]
    assert not (tmp_path / "figures").exists()


def test_existing_panel_only_spec_preserves_rendered_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pio.templates, "default", "plotly_white")
    spec = FigureSpec(
        name="bundle-figure",
        assembler="feedbax.grid_figure",
        traces=[
            TraceBinding(
                name="demo",
                constructor="feedbax.profile_band",
                data={"y": [[1, 2, 3], [2, 3, 4]]},
            )
        ],
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    assert rendered is not None
    assert hashlib.sha256(canonical_json_bytes(rendered)).hexdigest() == (
        "7402f9fb0a4fb64e94a3cfcd336c709a87393c4d9aa29488e0342f83f15dab9e"
    )


def test_execute_figure_spec_required_absence_fails_with_manifest(tmp_path: Path) -> None:
    manifest = _analysis_manifest(tmp_path)
    spec = FigureSpec(
        name="profile-required-missing",
        assembler="feedbax.grid_figure",
        inputs=[_contained_analysis_ref(manifest)],
        traces=[
            TraceBinding(
                name="missing-required",
                constructor="feedbax.profile_band",
                required=True,
                data={"y": {"item": "analysis", "path": "missing.path"}},
            )
        ],
    )
    with pytest.raises(FigureSpecExecutionError) as exc_info:
        execute_figure_spec(spec, root=tmp_path)
    assert exc_info.value.manifest.status == "failed"
    assert exc_info.value.manifest.failure["type"] == "ExpressionPathMissing"


def test_execute_figure_spec_fans_per_facet_binding_out_to_panels(tmp_path: Path) -> None:
    manifest = _analysis_manifest(tmp_path)
    register_figure_template(
        FigureTemplate(
            name="feedbax.test_faceted_panels",
            description="Test per-facet panel fan-out.",
            assembler="feedbax.grid_figure",
            slots=[
                SlotSpec(
                    name="profiles",
                    constructor="feedbax.profile_band",
                    multiplicity="per_facet",
                )
            ],
            facet_by=["condition"],
            facet_target="panels",
        ),
        replace=True,
    )
    spec = FigureSpec(
        name="faceted-panels",
        template="feedbax.test_faceted_panels",
        inputs=[_contained_analysis_ref(manifest)],
        panels=[
            {
                "name": "facet-base",
                "x_axis": {"type": "log", "range": [-2, 1]},
                "y_axis": {"type": "linear", "range": [-1, 4]},
            }
        ],
        slot_bindings={
            "profiles": TraceBinding(
                name="profile",
                constructor="feedbax.profile_band",
                data={
                    "y": {"item": "facet_values", "path": "condition"},
                    "label": {"item": "condition"},
                },
            )
        },
        facet_bindings={
            "condition": {"item": "analysis", "path": "metadata.series"},
        },
    )

    figure_manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(figure_manifest)

    assert rendered is not None
    for suffix in ("", "2", "3"):
        assert rendered["layout"][f"xaxis{suffix}"]["type"] == "log"
        assert rendered["layout"][f"xaxis{suffix}"]["range"] == [-2.0, 1.0]
        assert rendered["layout"][f"yaxis{suffix}"]["type"] == "linear"
        assert rendered["layout"][f"yaxis{suffix}"]["range"] == [-1.0, 4.0]
    assert [annotation["text"] for annotation in rendered["layout"]["annotations"]] == [
        "condition=slow",
        "condition=fast",
        "condition=flat",
    ]
    assert {trace["name"] for trace in rendered["data"] if trace.get("showlegend", True)} == {
        "slow",
        "fast",
        "flat",
    }
    assert [(record.panel, record.status) for record in figure_manifest.binding_records] == [
        ("condition=slow", "included"),
        ("condition=fast", "included"),
        ("condition=flat", "included"),
    ]


def test_execute_figure_spec_fans_facets_out_to_separate_renders(tmp_path: Path) -> None:
    manifest = _analysis_manifest(tmp_path)
    register_figure_template(
        FigureTemplate(
            name="feedbax.test_faceted_figures",
            description="Test separate-figure facet fan-out.",
            assembler="feedbax.grid_figure",
            slots=[
                SlotSpec(
                    name="profiles",
                    constructor="feedbax.profile_band",
                    multiplicity="per_facet",
                )
            ],
            facet_by=["condition"],
            facet_target="figures",
        ),
        replace=True,
    )
    spec = FigureSpec(
        name="faceted-figures",
        template="feedbax.test_faceted_figures",
        inputs=[_contained_analysis_ref(manifest)],
        slot_bindings={
            "profiles": TraceBinding(
                name="profile",
                constructor="feedbax.profile_band",
                data={"y": [[1, 2, 3]], "label": {"item": "condition"}},
            )
        },
        facet_bindings={"condition": {"item": "analysis", "path": "metadata.x"}},
    )

    figure_manifest, _path = execute_figure_spec(spec, root=tmp_path)
    plotly_artifacts = [
        artifact
        for artifact in figure_manifest.artifacts
        if artifact.role == FIGURE_RENDER_ROLE and artifact.metadata.get("format") == "plotly-json"
    ]

    assert [artifact.metadata["facet"] for artifact in plotly_artifacts] == [
        "condition=0",
        "condition=1",
        "condition=2",
    ]


def test_artifact_backed_piece_supplies_trace_data(tmp_path: Path) -> None:
    piece_payload = tmp_path / "piece.json"
    piece_payload.write_text(
        json.dumps({"payload": {"x": [0, 1, 2], "y": [[1, 2, 3], [2, 3, 4]]}}),
        encoding="utf-8",
    )
    register_figure_piece(
        FigurePiece(
            name="feedbax.test_piece",
            description="Test artifact-backed piece",
            artifact_ref=ArtifactRef(
                role="figure_piece",
                logical_name="piece.json",
                media_type="application/json",
                uri=str(piece_payload),
            ),
            data_path="payload",
            label="Piece",
            constructor="feedbax.profile_band",
            style={"color": "rgb(31,119,180)"},
        ),
        replace=True,
    )
    spec = FigureSpec(
        name="piece-demo",
        assembler="feedbax.grid_figure",
        pieces=["feedbax.test_piece"],
    )
    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    assert manifest.status == "completed"
    assert manifest.resolved_pieces[0].name == "feedbax.test_piece"
    assert manifest.binding_records[0].status == "included"
    assert rendered is not None
    assert _plotly_array_values(rendered["data"][0]["x"]) == [0.0, 1.0, 2.0]
    assert _plotly_array_values(rendered["data"][0]["y"]) == [1.5, 2.5, 3.5]


def test_canonical_artifact_backed_piece_supplies_exact_trace_data(tmp_path: Path) -> None:
    payload = {"payload": {"x": [10, 20, 30], "y": [[4, 5, 6], [7, 8, 9]]}}
    expected_bytes = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="test.canonical_figure_piece"),
        root=tmp_path,
        index_manifest=False,
    )
    artifact = context.record_json_artifact(
        payload,
        role="figure_piece",
        logical_name="canonical-piece.json",
    )
    register_figure_piece(
        FigurePiece(
            name="feedbax.test_canonical_piece",
            description="Canonical artifact-backed piece",
            artifact_ref=artifact,
            data_path="payload",
            label="Canonical piece",
            constructor="feedbax.profile_band",
        ),
        replace=True,
    )
    spec = FigureSpec(
        name="canonical-piece-demo",
        assembler="feedbax.grid_figure",
        pieces=["feedbax.test_canonical_piece"],
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path)
    rendered = figure_manifest_plotly_json(manifest)

    assert artifact.uri == artifact.artifact_id
    assert ImmutableArtifactBlobProvider(tmp_path).get_bytes(artifact) == expected_bytes
    assert manifest.status == "completed"
    assert rendered is not None
    assert _plotly_array_values(rendered["data"][0]["x"]) == [10.0, 20.0, 30.0]
    assert _plotly_array_values(rendered["data"][0]["y"]) == [5.5, 6.5, 7.5]


def test_endpoint_markers_derive_curved_reach_guides_from_trajectory() -> None:
    endpoint_constructor = get_figure_constructor("feedbax.endpoint_markers", tier="trace")
    assert endpoint_constructor.version == "v2"
    curved = [[[2, 3], [3, 7], [5, 8]]]
    endpoint_traces = endpoint_constructor.callable(
        {"trajectories": curved},
        endpoint_constructor.params(),
    )
    assert list(endpoint_traces[0].x) == [5]
    assert list(endpoint_traces[0].y) == [8]
    assert list(endpoint_traces[1].x) == [2, 5]
    assert list(endpoint_traces[1].y) == [3, 8]


def test_hline_and_vrect_emit_plotly_shapes() -> None:
    hline = get_figure_constructor("feedbax.hline", tier="trace")
    hline_shapes = hline.callable({"y": 2.5}, hline.params())
    assert hline_shapes[0].type == "line"
    assert hline_shapes[0].y0 == hline_shapes[0].y1 == 2.5

    vrect = get_figure_constructor("feedbax.vrect", tier="trace")
    vrect_shapes = vrect.callable({"x0": 4, "x1": 7}, vrect.params())
    assert vrect_shapes[0].type == "rect"
    assert (vrect_shapes[0].x0, vrect_shapes[0].x1) == (4, 7)
    panel = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    annotated = panel.callable(
        [PanelContent(name="annotations", traces=(*hline_shapes, *vrect_shapes))],
        panel.params(),
    )
    assert [shape.type for shape in annotated.layout.shapes] == ["line", "rect"]


def test_trajectory_2d_resolves_colorscale_key() -> None:
    trajectory = get_figure_constructor("feedbax.trajectory_2d", tier="trace")
    trajectory_traces = trajectory.callable(
        {
            "trajectories": [
                [[0, 0], [1, 1]],
                [[0, 0], [2, 2]],
            ],
            "colorscales": {"condition": "Viridis"},
        },
        trajectory.params({"colorscale_key": "condition", "show_mean": False}),
    )
    assert trajectory_traces[0].line.color != trajectory_traces[1].line.color


def test_load_figure_template_and_piece_from_package_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "toy_figure_pkg"
    template_root = package_root / "config" / "figure_templates"
    piece_root = package_root / "config" / "figure_pieces"
    template_root.mkdir(parents=True)
    piece_root.mkdir(parents=True)
    for path in (
        package_root / "__init__.py",
        package_root / "config" / "__init__.py",
        template_root / "__init__.py",
        piece_root / "__init__.py",
    ):
        path.write_text("", encoding="utf-8")
    (template_root / "profiles.yml").write_text(
        """
name: toy.profiles
description: Toy profile template.
assembler: feedbax.grid_figure
slots:
  - name: profile
    constructor: feedbax.profile_band
""",
        encoding="utf-8",
    )
    piece_payload = tmp_path / "piece.json"
    piece_payload.write_text("{}", encoding="utf-8")
    (piece_root / "baseline.yaml").write_text(
        f"""
name: toy.baseline
description: Toy baseline piece.
artifact_ref:
  role: figure_piece
  logical_name: piece.json
  media_type: application/json
  uri: {piece_payload}
label: Baseline
constructor: feedbax.profile_band
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for module_name in list(sys.modules):
        if module_name == "toy_figure_pkg" or module_name.startswith("toy_figure_pkg."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    package = importlib.import_module("toy_figure_pkg")
    registry = ExperimentRegistry()
    registry.register_package(
        "toy",
        package,
        parts=[],
        analysis_module_root="analysis",
        training_module_root="training",
        config_resource_root="config",
    )

    template = load_figure_template("toy/profiles", registry=registry)
    piece = load_figure_piece("toy/baseline", registry=registry)

    assert template.name == "toy.profiles"
    assert template.slots[0].constructor == "feedbax.profile_band"
    assert piece.name == "toy.baseline"
    assert piece.artifact_ref is not None
    assert piece.artifact_ref.uri == str(piece_payload)

    monkeypatch.setattr(feedbax.plugins, "_EXPERIMENT_REGISTRY", registry)
    assert get_figure_template("toy.profiles").name == "toy.profiles"
    assert get_figure_piece("toy.baseline").name == "toy.baseline"


def test_bundle_figure_topology_and_default_render_role() -> None:
    figure = FigureSpec(
        name="bundle-figure",
        assembler="feedbax.grid_figure",
        traces=[
            TraceBinding(
                name="demo",
                constructor="feedbax.profile_band",
                data={"y": [[1, 2, 3], [2, 3, 4]]},
            )
        ],
    )
    bundle = AnalysisBundleSpec(
        name="figure-bundle",
        stages=[
            BundleStageSpec(name="fig", kind="figure", figure=figure),
            BundleStageSpec(
                name="report",
                kind="report",
                report_type="feedbax.bundle_summary",
                depends_on_roles=[StageArtifactDependency(stage="fig", role=FIGURE_RENDER_ROLE)],
            ),
        ],
    )
    assert bundle.stages[0].kind == "figure"

    with pytest.raises(ValidationError, match="figure stages are leaves"):
        AnalysisBundleSpec(
            name="bad-figure-bundle",
            stages=[
                BundleStageSpec(name="fig", kind="figure", figure=figure),
                BundleStageSpec(
                    name="analysis",
                    kind="analysis",
                    analysis_type="feedbax.test",
                    depends_on=["fig"],
                ),
            ],
        )
