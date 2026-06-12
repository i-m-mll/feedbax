from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import numpy as np
import plotly.graph_objects as go

from feedbax.analysis.analysis import AbstractAnalysis, AnalysisRef
from feedbax.analysis.context import AnalysisArtifactFile, AnalysisRunContext
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.manifest import EvaluationRunManifest, EvaluationRunSpec, ParentRef
from feedbax.types import AnalysisInputData, TreeNamespace


TOY_EVALUATION_TYPE = "feedbax_test_toy_eval"
ARTIFACT_PRODUCER_CALLS = {"count": 0}


class ToyAnalysis(AbstractAnalysis):
    """Tiny reusable analysis for manifest-canonical execution tests."""

    def compute(self, data: AnalysisInputData, **kwargs):
        return {"value": data.states["value"] + 1}

    def make_figs(self, data: AnalysisInputData, *, result, **kwargs):
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[data.states["value"], result["value"]])
        return {"main": fig}

    def _params_to_save(self, hps, *, result, **kwargs):
        return {"result_value": result["value"]}


class ToyArtifactProducer(AbstractAnalysis):
    """Toy analysis that emits a logical JSON+NPZ artifact group."""

    def compute(self, data: AnalysisInputData, **kwargs):
        ARTIFACT_PRODUCER_CALLS["count"] += 1
        value = data.states["value"] + 2
        return {
            "value": value,
            "values": np.asarray([value, value + 1, value + 2], dtype=np.int64),
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **kwargs,
    ):
        group_id = f"{context.manifest_id}:toy-artifact-group"
        artifact_dir = context.results_cache_dir / "toy_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        summary_path = artifact_dir / "toy-summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "kind": "toy-summary",
                    "value": result["value"],
                    "array_count": int(result["values"].size),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        arrays_path = artifact_dir / "toy-arrays.npz"
        np.savez(arrays_path, values=result["values"])

        summary_ref, arrays_ref = context.record_artifact_group(
            group_id=group_id,
            metadata={"description": "opaque toy artifact group"},
            members=[
                AnalysisArtifactFile(
                    path=summary_path,
                    role="analysis_summary",
                    logical_name="toy/summary.json",
                    media_type="application/json",
                    group_role="summary",
                    metadata={"payload_kind": "toy.summary.v1"},
                ),
                AnalysisArtifactFile(
                    path=arrays_path,
                    role="analysis_arrays",
                    logical_name="toy/arrays.npz",
                    media_type="application/x-npz",
                    group_role="bulk_arrays",
                    metadata={
                        "arrays": {
                            "values": {
                                "role": "toy_value_series",
                                "dtype": str(result["values"].dtype),
                            }
                        }
                    },
                ),
            ],
        )
        return {
            **result,
            "artifact_refs": {
                "summary": summary_ref,
                "arrays": arrays_ref,
            },
        }


class ToyArtifactConsumer(AbstractAnalysis):
    """Toy downstream analysis that consumes an artifact-producing result."""

    def compute(self, data: AnalysisInputData, produced, **kwargs):
        summary_ref = produced["artifact_refs"]["summary"]
        arrays_ref = produced["artifact_refs"]["arrays"]
        summary = json.loads(Path(summary_ref.uri).read_text(encoding="utf-8"))
        arrays = np.load(Path(arrays_ref.uri))
        return {
            "consumed_value": summary["value"],
            "array_total": int(arrays["values"].sum()),
            "group_id": summary_ref.metadata["artifact_group"]["id"],
        }


def reset_artifact_producer_calls() -> None:
    ARTIFACT_PRODUCER_CALLS["count"] = 0


def build_toy_artifact_analyses() -> dict[str, AbstractAnalysis]:
    consumer = eqx.tree_at(
        lambda analysis: analysis._extra_inputs,
        ToyArtifactConsumer(variant="toy"),
        {"produced": AnalysisRef("artifact_producer")},
    )
    return {
        "artifact_producer": ToyArtifactProducer(variant="toy", cache_result=True),
        "artifact_consumer": consumer,
    }


def build_toy_analysis_data(value: int = 2) -> AnalysisInputData:
    """Return minimal data accepted by ``AbstractAnalysis`` execution."""
    return AnalysisInputData(
        models={},
        tasks={},
        states={"value": value},
        hps={"toy": TreeNamespace(task=TreeNamespace(eval_n=1))},
        extras={},
    )


def execute_toy_evaluation(root: Path) -> tuple[EvaluationRunManifest, Path]:
    """Write a tiny upstream evaluation manifest for analysis lineage tests."""
    parent = ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:toy",
        role="training_run",
    )
    spec = EvaluationRunSpec(
        evaluation_type=TOY_EVALUATION_TYPE,
        inputs=[parent],
        params={"n_trials": 1},
    )

    def recipe(
        run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"training_run_ids": [ref.id for ref in run_spec.inputs]},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
        )

    register_evaluation_recipe(TOY_EVALUATION_TYPE, recipe, replace=True)
    try:
        return execute_evaluation_run_spec(spec, root=root, issues=["8f40e2d"])
    finally:
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)
