from __future__ import annotations

from pathlib import Path

import numpy as np

from feedbax.analysis.evaluation import EvaluationRecipeResult, execute_evaluation_run_spec
from feedbax.analysis.specs import AnalysisRecipeResult, execute_analysis_run_spec
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    evaluation_states_cache_path,
    load_manifest,
)
from feedbax.plugins.application import new_application_registry_bundle
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


def test_analysis_recompute_uses_injected_evaluation_registry(tmp_path: Path) -> None:
    registries = new_application_registry_bundle(local_component_source=None)
    evaluation_type = "tests.injected_eval"
    analysis_type = "tests.injected_analysis"
    evaluation_calls: list[int] = []

    def evaluate(spec, _root, _states_path, _execution_context):
        value = int(spec.params["value"])
        evaluation_calls.append(value)
        return EvaluationRecipeResult(
            states={"value": np.asarray(value, dtype=np.int32)},
            summary_metrics={"value": value},
            metadata={"states_schema": "tests.injected_states.v1"},
        )

    def analyze(_spec, _root, inputs, _execution_context):
        value = int(inputs[0].states["value"])
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="injected", cache_result=True)},
            data=build_toy_analysis_data(value=value),
        )

    registries.evaluation_recipes.register(evaluation_type, evaluate)
    registries.analysis_recipes.register(analysis_type, analyze)
    evaluation, evaluation_path = execute_evaluation_run_spec(
        EvaluationRunSpec(
            evaluation_type=evaluation_type,
            inputs=[
                ParentRef(
                    kind="TrainingRunManifest",
                    id="feedbax-training-run:injected",
                    role="training_run",
                )
            ],
            params={"value": 5},
        ),
        registry=registries.evaluation_recipes,
        root=tmp_path,
        force=True,
    )
    states_path = evaluation_states_cache_path(evaluation.id, root=tmp_path)
    states_path.unlink()

    manifest, path = execute_analysis_run_spec(
        AnalysisRunSpec(
            analysis_type=analysis_type,
            inputs=[
                ParentRef(
                    kind="EvaluationRunManifest",
                    id=evaluation.id,
                    role="evaluation_run",
                    uri=str(evaluation_path),
                )
            ],
            params={"requested_outputs": ["toy"]},
        ),
        registry=registries.analysis_recipes,
        evaluation_registry=registries.evaluation_recipes,
        root=tmp_path,
        fig_dump_formats=("json",),
        experiment_registry=registries.experiment_packages,
    )

    assert evaluation_calls == [5, 5]
    assert states_path.exists()
    assert manifest.status == "completed"
    assert load_manifest(path).id == manifest.id
