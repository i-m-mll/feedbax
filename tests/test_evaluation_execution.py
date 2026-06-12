from __future__ import annotations

import sqlite3
from pathlib import Path

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.manifest import (
    EvaluationRunSpec,
    ParentRef,
    evaluation_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
)
from feedbax.manifest_index import rebuild_manifest_index


def test_evaluation_run_spec_executes_headless_and_reuses_manifest_cache(tmp_path: Path):
    calls: list[str] = []
    parent = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:toy",
        role="training_run",
    )
    spec = EvaluationRunSpec(
        evaluation_type="toy_eval",
        inputs=[parent],
        params={"n_trials": 3},
    )
    spec_path = tmp_path / "evaluation-spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def recipe(
        run_spec: EvaluationRunSpec,
        root: Path,
        states_path: Path,
    ) -> EvaluationRecipeResult:
        calls.append(str(root))
        return EvaluationRecipeResult(
            states={"training_run_ids": [ref.id for ref in run_spec.inputs]},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
            metadata={"states_path_seen": str(states_path)},
        )

    register_evaluation_recipe("toy_eval", recipe, replace=True)
    try:
        manifest, path = execute_evaluation_run_spec(
            spec_path,
            root=tmp_path,
            issues=["8f40e2d"],
        )
        assert manifest.status == "completed"
        assert path.exists()
        assert manifest.id == evaluation_run_manifest_id(spec)
        assert manifest.evaluation_spec.inline["evaluation_type"] == "toy_eval"
        assert manifest.input_training_runs == [parent]
        assert manifest.provenance.parents == [parent]
        assert manifest.provenance.issues == ["8f40e2d"]
        assert manifest.summary_metrics["n_trials"] == 3

        cache_path = evaluation_states_cache_path(manifest.id, root=tmp_path)
        assert cache_path.exists()
        assert manifest.metadata["cache"]["states_path"] == str(cache_path)

        loaded = load_manifest(path)
        assert loaded.id == manifest.id

        rerun_manifest, rerun_path = execute_evaluation_run_spec(spec, root=tmp_path)
        assert rerun_path == path
        assert rerun_manifest.id == manifest.id
        assert rerun_manifest.summary_metrics["n_trials"] == 3
        assert rerun_manifest.summary_metrics["input_training_runs"] == 1
        assert "states_cache_hit" not in rerun_manifest.summary_metrics
        assert rerun_manifest.metadata["cache"]["states_cache_hit"] is True
        assert calls == [str(tmp_path)]

        index_path = rebuild_manifest_index(tmp_path)
        with sqlite3.connect(index_path) as conn:
            row = conn.execute(
                "SELECT kind, status FROM manifests WHERE id = ?",
                (manifest.id,),
            ).fetchone()
            edge = conn.execute(
                """
                SELECT parent_kind, parent_id, role
                FROM lineage_edges
                WHERE child_id = ?
                """,
                (manifest.id,),
            ).fetchone()
        assert row == ("EvaluationRunManifest", "completed")
        assert edge == ("TrainingRunManifest", parent.id, "training_run")
    finally:
        unregister_evaluation_recipe("toy_eval")
