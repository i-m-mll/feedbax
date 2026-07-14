from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

import feedbax.plugins
from feedbax.analysis.figures import (
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
    ArtifactRef,
    ParentRef,
    load_manifest,
    spec_payload,
    write_manifest,
    safe_manifest_key,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.plot.constructors import (
    PanelContent,
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

pytestmark = [pytest.mark.feedbax_contract]


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
    assert manifest.status == "completed"
    assert manifest.resolved_pieces[0].name == "feedbax.test_piece"
    assert manifest.binding_records[0].status == "included"


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
