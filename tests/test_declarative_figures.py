from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis.figures import FIGURE_RENDER_ROLE, FigureSpecExecutionError, execute_figure_spec
from feedbax.analysis.bundles import AnalysisBundleSpec, BundleStageSpec, StageArtifactDependency
from feedbax.contracts.figures import (
    FIGURE_SPEC_SCHEMA_ID,
    FIGURE_SPEC_SCHEMA_VERSION,
    FigurePiece,
    FigureSpec,
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
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.plot.constructors import (
    get_figure_constructor,
    register_figure_piece,
    register_figure_constructor,
    registered_figure_constructors,
)

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
        metadata={"x": [0, 1, 2], "y": [[1, 2, 3], [2, 3, 4]]},
        analysis_spec=spec_payload("AnalysisRunSpec", {"analysis_type": "feedbax.test"}),
        produced_data=[product],
    )
    write_manifest(manifest, root=root)
    return manifest


def test_figure_spec_schema_identity_rejects_old_versions() -> None:
    current = FigureSpec(name="demo", assembler="feedbax.grid_figure")
    assert current.schema_id == FIGURE_SPEC_SCHEMA_ID
    assert current.schema_version == FIGURE_SPEC_SCHEMA_VERSION

    with pytest.raises(ValidationError, match="unsupported FigureSpec schema_version"):
        FigureSpec.model_validate(
            {
                "schema_id": FIGURE_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.spec.figure.v0",
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
        inputs=[ParentRef(kind=manifest.kind, id=manifest.id, role="analysis")],
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
        inputs=[ParentRef(kind=manifest.kind, id=manifest.id, role="analysis")],
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
                depends_on_roles=[
                    StageArtifactDependency(stage="fig", role=FIGURE_RENDER_ROLE)
                ],
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
