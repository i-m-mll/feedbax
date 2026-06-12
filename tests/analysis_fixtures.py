from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.manifest import EvaluationRunManifest, EvaluationRunSpec, ParentRef
from feedbax.types import AnalysisInputData, TreeNamespace


TOY_EVALUATION_TYPE = "feedbax_test_toy_eval"


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
