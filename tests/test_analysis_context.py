from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from feedbax.analysis.context import (
    AnalysisArtifactFile,
    AnalysisRunContext,
    parent_ref_from_evaluation_manifest,
)
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.analysis.materialization import (
    AnalysisArtifactGroup,
    ContextMaterializationPending,
    ContextMaterializer,
    ExistingAnalysisArtifact,
    MaterializationResult,
)
from feedbax.manifest import (
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    REGENERATION_SPEC_SCHEMA_ID,
    RegenerationCommand,
    RegenerationSpec,
    analysis_run_manifest_id,
    load_manifest,
)
from feedbax.manifest_index import rebuild_manifest_index
from tests.analysis_fixtures import (
    ARTIFACT_PRODUCER_CALLS,
    ToyAnalysis,
    build_toy_artifact_analyses,
    build_toy_analysis_data,
    execute_toy_evaluation,
    reset_artifact_producer_calls,
)


def test_headless_analysis_context_writes_manifest_figures_and_rebuildable_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_analysis",
        inputs=[eval_ref],
        params={"outputs": ["toy"]},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        fig_dump_formats=("json",),
        issues=["52c7728"],
    )

    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"toy": ToyAnalysis(variant="toy", cache_result=True)},
        build_toy_analysis_data(),
        context,
        fig_dump_formats=["json"],
    )

    assert all_results["toy"]["value"] == 3
    assert context.manifest_id == analysis_run_manifest_id(spec)
    assert context.manifest_path is not None
    manifest = load_manifest(context.manifest_path)
    assert manifest.kind == "AnalysisRunManifest"
    assert manifest.status == "completed"
    assert manifest.inputs == [eval_ref]
    assert manifest.provenance.parents == [eval_ref]
    assert manifest.provenance.issues == ["52c7728"]
    assert manifest.analysis_spec.inline["inputs"][0]["id"] == eval_manifest.id
    assert manifest.summary_metrics["figure_count"] == 1
    assert manifest.summary_metrics["artifact_count"] == 1
    assert manifest.summary_metrics["analysis_count"] == 1
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].role == "figure"
    assert manifest.artifacts[0].media_type == "application/json"
    assert Path(manifest.artifacts[0].uri).exists()
    assert list(context.results_cache_dir.glob("*.pkl"))

    index_path = rebuild_manifest_index(tmp_path)
    with sqlite3.connect(index_path) as conn:
        manifest_row = conn.execute(
            "SELECT kind, status FROM manifests WHERE id = ?",
            (context.manifest_id,),
        ).fetchone()
        edge_row = conn.execute(
            """
            SELECT parent_kind, parent_id, role
            FROM lineage_edges
            WHERE child_id = ?
            """,
            (context.manifest_id,),
        ).fetchone()
        artifact_row = conn.execute(
            """
            SELECT role, logical_name, media_type
            FROM artifacts
            WHERE manifest_id = ?
            """,
            (context.manifest_id,),
        ).fetchone()

    assert manifest_row == ("AnalysisRunManifest", "completed")
    assert edge_row == ("EvaluationRunManifest", eval_manifest.id, "evaluation_run")
    assert artifact_row == ("figure", "toy/toy_toy_analysis_0.json", "application/json")


def test_requested_outputs_empty_intersection_raises_clear_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_analysis",
        inputs=[eval_ref],
        params={"requested_outputs": ["missing"]},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        fig_dump_formats=("json",),
    )

    with pytest.raises(ValueError, match="requested_outputs=\\['missing'\\]") as excinfo:
        run_analyses_with_context(
            {"toy": ToyAnalysis(variant="toy", cache_result=True)},
            build_toy_analysis_data(),
            context,
            requested_outputs={"missing"},
        )

    assert "available_analysis_keys=['toy']" in str(excinfo.value)
    assert context.manifest_path is None


def test_requested_outputs_partial_intersection_runs_matching_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_analysis",
        inputs=[eval_ref],
        params={"requested_outputs": ["toy", "missing"]},
    )
    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        fig_dump_formats=("json",),
    )

    all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"toy": ToyAnalysis(variant="toy", cache_result=True)},
        build_toy_analysis_data(),
        context,
        fig_dump_formats=["json"],
        requested_outputs={"toy", "missing"},
    )

    assert set(all_analyses) == {"toy"}
    assert all_results["toy"]["value"] == 3
    manifest = load_manifest(context.manifest_path)
    assert manifest.status == "completed"
    assert manifest.summary_metrics["analysis_count"] == 1


def test_analysis_context_records_grouped_artifacts_cache_and_downstream_consumption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    eval_manifest, eval_path = execute_toy_evaluation(tmp_path)
    eval_ref = parent_ref_from_evaluation_manifest(
        eval_manifest.id,
        uri=str(eval_path),
    )
    spec = AnalysisRunSpec(
        analysis_type="toy_artifact_analysis",
        inputs=[eval_ref],
        params={"outputs": ["artifact_consumer"]},
    )
    reset_artifact_producer_calls()

    context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        issues=["c3bce34"],
    )
    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        build_toy_artifact_analyses(),
        build_toy_analysis_data(),
        context,
    )

    assert ARTIFACT_PRODUCER_CALLS["count"] == 1
    assert all_results["artifact_consumer"] == {
        "consumed_value": 4,
        "array_total": 15,
        "group_id": f"{context.manifest_id}:toy-artifact-group",
    }
    assert context.manifest_path is not None
    manifest = load_manifest(context.manifest_path)
    assert manifest.kind == "AnalysisRunManifest"
    assert manifest.summary_metrics["artifact_count"] == 2
    assert manifest.summary_metrics["figure_count"] == 0
    assert manifest.summary_metrics["analysis_count"] == 2

    artifacts_by_role = {artifact.role: artifact for artifact in manifest.artifacts}
    summary_ref = artifacts_by_role["analysis_summary"]
    arrays_ref = artifacts_by_role["analysis_arrays"]
    assert summary_ref.logical_name == "toy/summary.json"
    assert summary_ref.media_type == "application/json"
    assert arrays_ref.logical_name == "toy/arrays.npz"
    assert arrays_ref.media_type == "application/x-npz"
    assert (
        summary_ref.metadata["artifact_group"]["id"] == arrays_ref.metadata["artifact_group"]["id"]
    )
    assert summary_ref.metadata["artifact_group"]["member_role"] == "summary"
    assert arrays_ref.metadata["artifact_group"]["member_role"] == "bulk_arrays"
    assert arrays_ref.metadata["arrays"]["values"]["role"] == "toy_value_series"
    assert (tmp_path / summary_ref.metadata["relative_path"]).exists()
    assert (tmp_path / arrays_ref.metadata["relative_path"]).exists()
    assert Path(summary_ref.uri).exists()
    assert Path(arrays_ref.uri).exists()
    assert list(context.results_cache_dir.glob("*.pkl"))

    index_path = rebuild_manifest_index(tmp_path)
    with sqlite3.connect(index_path) as conn:
        artifact_rows = conn.execute(
            """
            SELECT role, logical_name, media_type
            FROM artifacts
            WHERE manifest_id = ?
            ORDER BY logical_name
            """,
            (context.manifest_id,),
        ).fetchall()

    assert artifact_rows == [
        ("analysis_arrays", "toy/arrays.npz", "application/x-npz"),
        ("analysis_summary", "toy/summary.json", "application/json"),
    ]

    cached_context = AnalysisRunContext(
        spec=spec,
        root=tmp_path,
        issues=["c3bce34"],
    )
    _all_analyses, cached_results, _all_figs = run_analyses_with_context(
        build_toy_artifact_analyses(),
        build_toy_analysis_data(),
        cached_context,
    )

    assert ARTIFACT_PRODUCER_CALLS["count"] == 1
    assert cached_results["artifact_consumer"] == all_results["artifact_consumer"]
    cached_manifest = load_manifest(cached_context.manifest_path)
    assert cached_manifest.summary_metrics["artifact_count"] == 2
    assert {artifact.artifact_id for artifact in cached_manifest.artifacts} == {
        summary_ref.artifact_id,
        arrays_ref.artifact_id,
    }


def test_context_materializer_emits_json_payload_with_explicit_compute_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    spec = AnalysisRunSpec(
        analysis_type="toy_context_materializer",
        params={"requested_outputs": ["materializer"]},
    )
    context = AnalysisRunContext(spec=spec, root=tmp_path)

    def materialize(run_context: AnalysisRunContext) -> dict[str, object]:
        return {
            "kind": "toy.materialized.v1",
            "manifest_id": run_context.manifest_id,
            "value": 17,
        }

    analysis = ContextMaterializer(
        materializer=materialize,
        artifact_role="toy_materialized_payload",
        logical_name="toy/materialized.json",
        schema_boundary="toy-owned payload",
    )
    pending = analysis.compute(build_toy_analysis_data())

    assert isinstance(pending, ContextMaterializationPending)
    assert pending.status == "pending_context_artifact_emission"

    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"materializer": analysis},
        build_toy_analysis_data(),
        context,
    )

    assert all_results["materializer"]["kind"] == "toy.materialized.v1"
    manifest = load_manifest(context.manifest_path)
    assert manifest.summary_metrics["artifact_count"] == 1
    payload_ref = manifest.artifacts[0]
    assert payload_ref.role == "toy_materialized_payload"
    assert payload_ref.logical_name == "toy/materialized.json"
    assert payload_ref.metadata["schema_boundary"] == "toy-owned payload"
    payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
    assert payload == all_results["materializer"]


def test_context_materializer_records_embedded_refs_groups_and_regeneration_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FEEDBAX_WEB_DATA", raising=False)
    spec = AnalysisRunSpec(
        analysis_type="toy_context_materializer_rich",
        inputs=[ParentRef(kind="EvaluationRunManifest", id="eval-rich")],
    )
    context = AnalysisRunContext(spec=spec, root=tmp_path)

    def materialize(run_context: AnalysisRunContext) -> MaterializationResult:
        existing_path = run_context.results_cache_dir / "existing-summary.json"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text('{"summary": true}\n', encoding="utf-8")

        group_dir = run_context.results_cache_dir / "bulk"
        group_dir.mkdir(parents=True, exist_ok=True)
        bulk_path = group_dir / "unit.npz"
        np.savez_compressed(bulk_path, values=np.asarray([1, 2, 3], dtype=np.int64))

        external_ref = ArtifactRef(
            role="downstream_existing_ref",
            logical_name="external/ref.json",
            artifact_id="artifact://external/ref",
            media_type="application/json",
            uri="https://example.invalid/ref.json",
        )
        regeneration = RegenerationSpec(
            command=RegenerationCommand(argv=["python", "make_payload.py"]),
            parameters={"analysis_type": run_context.spec.analysis_type},
            inputs=list(run_context.spec.inputs),
            outputs=[
                ArtifactRef(
                    role="toy_materialized_payload",
                    logical_name="toy/rich-materialized.json",
                )
            ],
        )
        return MaterializationResult(
            payload={
                "kind": "toy.rich-materialized.v1",
                "nested": {
                    "refs": [external_ref],
                },
            },
            payload_metadata={"payload_schema": "toy.rich-materialized.v1"},
            existing_artifacts=[
                ExistingAnalysisArtifact(
                    path=existing_path,
                    role="toy_existing_summary",
                    logical_name="toy/existing-summary.json",
                    media_type="application/json",
                )
            ],
            artifact_groups=[
                AnalysisArtifactGroup(
                    group_id="toy_bulk_group",
                    metadata={"description": "opaque toy bulk group"},
                    members=[
                        AnalysisArtifactFile(
                            path=bulk_path,
                            role="toy_bulk_arrays",
                            logical_name="toy/bulk/unit.npz",
                            media_type="application/x-npz",
                            group_role="bulk_arrays",
                            metadata={"arrays": {"values": {"role": "toy_series"}}},
                        )
                    ],
                )
            ],
            regeneration_specs=[regeneration],
        )

    analysis = ContextMaterializer(
        materializer=materialize,
        artifact_role="toy_materialized_payload",
        logical_name="toy/rich-materialized.json",
        schema_boundary="toy-owned payload",
    )
    _all_analyses, all_results, _all_figs = run_analyses_with_context(
        {"materializer": analysis},
        build_toy_analysis_data(),
        context,
    )

    manifest = load_manifest(context.manifest_path)
    assert all_results["materializer"]["nested"]["refs"][0]["artifact_id"] == (
        "artifact://external/ref"
    )
    assert manifest.summary_metrics["artifact_count"] == 4

    artifacts_by_role = {artifact.role: artifact for artifact in manifest.artifacts}
    payload_ref = artifacts_by_role["toy_materialized_payload"]
    external_ref = artifacts_by_role["downstream_existing_ref"]
    existing_ref = artifacts_by_role["toy_existing_summary"]
    bulk_ref = artifacts_by_role["toy_bulk_arrays"]

    assert json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))["nested"]["refs"][0][
        "artifact_id"
    ] == external_ref.artifact_id
    assert Path(existing_ref.uri).exists()
    assert Path(bulk_ref.uri).exists()
    assert bulk_ref.metadata["artifact_group"]["id"] == "toy_bulk_group"
    assert bulk_ref.metadata["artifact_group"]["member_role"] == "bulk_arrays"
    assert bulk_ref.metadata["artifact_group"]["metadata"] == {
        "description": "opaque toy bulk group"
    }
    assert bulk_ref.metadata["arrays"]["values"]["role"] == "toy_series"

    assert len(manifest.regeneration_specs) == 1
    regeneration_payload = manifest.regeneration_specs[0]
    assert regeneration_payload.kind == "RegenerationSpec"
    assert regeneration_payload.schema_id == REGENERATION_SPEC_SCHEMA_ID
    assert regeneration_payload.inline["parameters"] == {
        "analysis_type": "toy_context_materializer_rich"
    }
    assert regeneration_payload.inline["outputs"][0]["logical_name"] == (
        "toy/rich-materialized.json"
    )
