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
    ToyAnalysis,
    build_toy_analysis_data,
    execute_toy_evaluation,
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
