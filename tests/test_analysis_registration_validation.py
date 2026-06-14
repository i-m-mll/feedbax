from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis import (
    AnalysisRecipeProtocol,
    EvaluationRecipeProtocol,
    RecipeValidationError,
    validate_analysis_recipe,
    validate_evaluation_recipe,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.contracts.manifest import AnalysisRunSpec, EvaluationRunSpec
from feedbax.plugins import discovery
from feedbax.plugins.registry import ExperimentRegistry
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


def test_public_validation_protocols_are_importable_from_stable_paths() -> None:
    assert AnalysisRecipeProtocol is not None
    assert EvaluationRecipeProtocol is not None
    assert validate_analysis_recipe is not None
    assert validate_evaluation_recipe is not None


class CustomEvaluationRecipe:
    """Non-vmap-style callable recipe used by downstream custom evaluations."""

    def __init__(self, value: int):
        self.value = value

    def __call__(
        self,
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": self.value},
            summary_metrics={"value": self.value},
        )


def test_valid_custom_evaluation_recipe_registers() -> None:
    recipe = CustomEvaluationRecipe(value=3)

    register_evaluation_recipe("custom_non_vmap_eval", recipe, replace=True)
    try:
        validated = validate_evaluation_recipe("custom_non_vmap_eval", recipe)
        assert validated is recipe
    finally:
        unregister_evaluation_recipe("custom_non_vmap_eval")


def test_evaluation_recipe_registration_rejects_non_callable_and_names_type() -> None:
    with pytest.raises(RecipeValidationError, match="Evaluation recipe 'broken_eval'.*callable"):
        register_evaluation_recipe("broken_eval", object(), replace=True)


def test_evaluation_recipe_registration_rejects_bad_signature_and_names_type() -> None:
    def recipe(_run_spec: EvaluationRunSpec) -> EvaluationRecipeResult:
        return EvaluationRecipeResult()

    with pytest.raises(
        RecipeValidationError,
        match="Evaluation recipe 'bad_eval'.*three positional arguments.*states_path",
    ):
        register_evaluation_recipe("bad_eval", recipe, replace=True)


def test_discovery_reraises_recipe_validation_with_package_and_type(monkeypatch) -> None:
    class BrokenEntryPoint:
        name = "broken_pkg"

        def load(self):
            def register(_registry: ExperimentRegistry) -> None:
                register_evaluation_recipe("broken_discovered_eval", object(), replace=True)

            return register

    monkeypatch.setattr(
        discovery.importlib.metadata,
        "entry_points",
        lambda group: [BrokenEntryPoint()],
    )

    with pytest.raises(
        RuntimeError,
        match="broken_pkg.*Evaluation recipe 'broken_discovered_eval'.*callable",
    ):
        discovery.discover_experiment_packages(registry=ExperimentRegistry())


def test_valid_analysis_recipe_registers() -> None:
    def recipe(
        _run_spec: AnalysisRunSpec,
        _root: Path,
        _inputs: list[object],
    ) -> AnalysisRecipeResult:
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy")},
            data=build_toy_analysis_data(value=1),
        )

    register_analysis_recipe("valid_analysis", recipe, replace=True)
    try:
        validated = validate_analysis_recipe("valid_analysis", recipe)
        assert validated is recipe
    finally:
        unregister_analysis_recipe("valid_analysis")


def test_analysis_recipe_registration_rejects_non_callable_and_names_type() -> None:
    with pytest.raises(RecipeValidationError, match="Analysis recipe 'broken_analysis'.*callable"):
        register_analysis_recipe("broken_analysis", object(), replace=True)


def test_analysis_recipe_registration_rejects_bad_signature_and_names_type() -> None:
    def recipe(_run_spec: AnalysisRunSpec, _root: Path) -> AnalysisRecipeResult:
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy")},
            data=build_toy_analysis_data(value=1),
        )

    with pytest.raises(
        RecipeValidationError,
        match="Analysis recipe 'bad_analysis'.*three positional arguments.*inputs",
    ):
        register_analysis_recipe("bad_analysis", recipe, replace=True)
