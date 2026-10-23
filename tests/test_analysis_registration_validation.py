from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis import (
    AnalysisRecipeProtocol,
    EvaluationRecipeProtocol,
    ReportRecipeProtocol,
    RecipeValidationError,
    validate_analysis_recipe,
    validate_evaluation_recipe,
    validate_report_recipe,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeRegistry,
    EvaluationRecipeResult,
)
from feedbax.analysis.specs import (
    AnalysisRecipeRegistry,
    AnalysisRecipeResult,
)
from feedbax.analysis.reports import ReportRecipeResult
from feedbax.contracts.manifest import AnalysisRunSpec, EvaluationRunSpec, ReportSpec
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


def test_public_validation_protocols_are_importable_from_stable_paths() -> None:
    assert AnalysisRecipeProtocol is not None
    assert EvaluationRecipeProtocol is not None
    assert ReportRecipeProtocol is not None
    assert validate_analysis_recipe is not None
    assert validate_evaluation_recipe is not None
    assert validate_report_recipe is not None


class CustomEvaluationRecipe:
    """Non-vmap-style callable recipe used by downstream custom evaluations."""

    def __init__(self, value: int):
        self.value = value

    def __call__(
        self,
        _run_spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": self.value},
            summary_metrics={"value": self.value},
        )


def test_valid_custom_evaluation_recipe_registers() -> None:
    recipe = CustomEvaluationRecipe(value=3)
    registry = EvaluationRecipeRegistry()

    registry.register("testpkg.custom_non_vmap_eval", recipe)
    assert registry.get("testpkg.custom_non_vmap_eval") is recipe


def test_recipe_type_keys_reject_bare_names() -> None:
    recipe = CustomEvaluationRecipe(value=3)
    registry = EvaluationRecipeRegistry()

    with pytest.raises(RecipeValidationError, match="<package>\\.<name>"):
        registry.register("custom_non_vmap_eval", recipe)


def test_recipe_type_keys_accept_feedbax_and_downstream_namespaces() -> None:
    recipe = CustomEvaluationRecipe(value=3)
    registry = EvaluationRecipeRegistry()

    registry.register("feedbax.custom_non_vmap_eval", recipe)
    registry.register("rlrmp.standard_matrix_evaluation", recipe)
    assert registry.get("feedbax.custom_non_vmap_eval") is recipe
    assert registry.get("rlrmp.standard_matrix_evaluation") is recipe


def test_evaluation_recipe_registration_rejects_non_callable_and_names_type() -> None:
    registry = EvaluationRecipeRegistry()
    with pytest.raises(
        RecipeValidationError,
        match="Evaluation recipe 'testpkg.broken_eval'.*callable",
    ):
        registry.register("testpkg.broken_eval", object())


def test_evaluation_recipe_registration_rejects_bad_signature_and_names_type() -> None:
    registry = EvaluationRecipeRegistry()

    def recipe(_run_spec: EvaluationRunSpec) -> EvaluationRecipeResult:
        return EvaluationRecipeResult()

    with pytest.raises(
        RecipeValidationError,
        match="Evaluation recipe 'testpkg.bad_eval'.*four positional arguments.*states_path",
    ):
        registry.register("testpkg.bad_eval", recipe)


def test_registry_preserves_recipe_validation_context() -> None:
    registry = EvaluationRecipeRegistry()
    with pytest.raises(
        RecipeValidationError,
        match="Evaluation recipe 'testpkg.broken_discovered_eval'.*callable",
    ):
        registry.register("testpkg.broken_discovered_eval", object())


def test_valid_analysis_recipe_registers() -> None:
    registry = AnalysisRecipeRegistry()

    def recipe(
        _run_spec: AnalysisRunSpec,
        _root: Path,
        _inputs: list[object],
        _execution_context,
    ) -> AnalysisRecipeResult:
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy")},
            data=build_toy_analysis_data(value=1),
        )

    registry.register("testpkg.valid_analysis", recipe)
    assert registry.get("testpkg.valid_analysis") is recipe


def test_analysis_recipe_registration_rejects_non_callable_and_names_type() -> None:
    registry = AnalysisRecipeRegistry()
    with pytest.raises(
        RecipeValidationError,
        match="Analysis recipe 'testpkg.broken_analysis'.*callable",
    ):
        registry.register("testpkg.broken_analysis", object())


def test_analysis_recipe_registration_rejects_bad_signature_and_names_type() -> None:
    registry = AnalysisRecipeRegistry()

    def recipe(_run_spec: AnalysisRunSpec, _root: Path) -> AnalysisRecipeResult:
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy")},
            data=build_toy_analysis_data(value=1),
        )

    with pytest.raises(
        RecipeValidationError,
        match="Analysis recipe 'testpkg.bad_analysis'.*four positional arguments.*inputs",
    ):
        registry.register("testpkg.bad_analysis", recipe)


def test_valid_report_recipe_validates() -> None:
    def recipe(
        _report_spec: ReportSpec,
        _root: Path,
        _inputs: list[object],
    ) -> ReportRecipeResult:
        return ReportRecipeResult()

    assert validate_report_recipe("testpkg.valid_report", recipe) is recipe


def test_report_recipe_validation_rejects_bad_signature_and_names_type() -> None:
    def recipe(_report_spec: ReportSpec, _root: Path) -> ReportRecipeResult:
        return ReportRecipeResult()

    with pytest.raises(
        RecipeValidationError,
        match="Report recipe 'testpkg.bad_report'.*three positional arguments.*inputs",
    ):
        validate_report_recipe("testpkg.bad_report", recipe)
