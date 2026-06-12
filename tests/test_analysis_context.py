from __future__ import annotations

import sqlite3
from pathlib import Path

from feedbax.analysis.context import AnalysisRunContext, parent_ref_from_evaluation_manifest
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.manifest import (
    AnalysisRunSpec,
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
    assert summary_ref.metadata["artifact_group"]["id"] == arrays_ref.metadata[
        "artifact_group"
    ]["id"]
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
    assert {
        artifact.artifact_id for artifact in cached_manifest.artifacts
    } == {summary_ref.artifact_id, arrays_ref.artifact_id}
