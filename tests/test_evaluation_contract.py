from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.analysis.evaluation import (
    EvaluationRecipeRegistry,
    EvaluationRecipeResult,
)
from feedbax.contracts.base import ParentRef
from feedbax.contracts.manifest import EvaluationRunSpec
from feedbax.contracts.migrations import SpecSchemaFamily, SpecSchemaRegistry
from feedbax.testing.evaluation_contract import (
    check_evaluation_recipe,
    evaluation_params_schema_family_id,
)

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]


EVALUATION_TYPE = "testpkg.dummy_eval"
STATES_SCHEMA = "testpkg.states.dummy_eval.v1"


def _parent() -> ParentRef:
    return ParentRef(
        kind="TrainingRunManifest",
        id="feedbax-training-run:dummy",
        role="training_run",
    )


def _spec(evaluation_type: str = EVALUATION_TYPE) -> EvaluationRunSpec:
    return EvaluationRunSpec(
        evaluation_type=evaluation_type,
        inputs=[_parent()],
        params={"n_trials": 2},
    )


def _registry_with_dummy_family() -> SpecSchemaRegistry:
    registry = SpecSchemaRegistry()
    family_id = evaluation_params_schema_family_id(EVALUATION_TYPE)
    registry.register_family(
        SpecSchemaFamily(
            kind=family_id,
            schema_id=family_id,
            current_version=f"{family_id}.v1",
            description="Dummy downstream eval params for conformance tests.",
        )
    )
    return registry


def _evaluation_registry(recipe) -> EvaluationRecipeRegistry:
    registry = EvaluationRecipeRegistry()
    registry.register(EVALUATION_TYPE, recipe)
    return registry


def test_evaluation_contract_helper_passes_for_dummy_recipe(tmp_path: Path) -> None:
    calls: list[str] = []

    def recipe(
        spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        calls.append(spec.evaluation_type)
        return EvaluationRecipeResult(
            states={"training_run_ids": [ref.id for ref in spec.inputs]},
            summary_metrics={"n_trials": spec.params["n_trials"]},
            metadata={"states_schema": STATES_SCHEMA},
        )

    report = check_evaluation_recipe(
        EVALUATION_TYPE,
        _spec,
        evaluation_registry=_evaluation_registry(recipe),
        root=tmp_path,
        schema_registry=_registry_with_dummy_family(),
    )

    assert report.evaluation_type == EVALUATION_TYPE
    assert report.params_schema_family == "testpkg.spec.evaluation.dummy_eval"
    assert not report.params_schema_waived
    assert report.states_schema == STATES_SCHEMA
    assert report.cache_round_trip
    assert report.failure_manifest_id != report.manifest_id
    assert calls == [EVALUATION_TYPE]


def test_evaluation_contract_helper_accepts_explicit_params_schema_waiver(
    tmp_path: Path,
) -> None:
    def recipe(
        _spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": 1},
            metadata={"states_schema": STATES_SCHEMA},
        )

    report = check_evaluation_recipe(
        EVALUATION_TYPE,
        _spec,
        evaluation_registry=_evaluation_registry(recipe),
        root=tmp_path,
        schema_registry=SpecSchemaRegistry(),
        params_schema_waiver="params family lands in a later downstream change",
    )

    assert report.params_schema_waived


def test_evaluation_contract_helper_rejects_missing_params_family_without_waiver(
    tmp_path: Path,
) -> None:
    def recipe(
        _spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": 1},
            metadata={"states_schema": STATES_SCHEMA},
        )

    with pytest.raises(AssertionError, match="params schema family is not registered"):
        check_evaluation_recipe(
            EVALUATION_TYPE,
            _spec,
            evaluation_registry=_evaluation_registry(recipe),
            root=tmp_path,
            schema_registry=SpecSchemaRegistry(),
        )


def test_evaluation_contract_helper_rejects_missing_states_schema(
    tmp_path: Path,
) -> None:
    def recipe(
        _spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(states={"value": 1})

    with pytest.raises(AssertionError, match="metadata\\['states_schema'\\]"):
        check_evaluation_recipe(
            EVALUATION_TYPE,
            _spec,
            evaluation_registry=_evaluation_registry(recipe),
            root=tmp_path,
            schema_registry=_registry_with_dummy_family(),
        )


def test_evaluation_contract_helper_rejects_spec_type_mismatch(tmp_path: Path) -> None:
    def recipe(
        _spec: EvaluationRunSpec,
        _root: Path,
        _states_path: Path,
        _execution_context,
    ) -> EvaluationRecipeResult:
        return EvaluationRecipeResult(
            states={"value": 1},
            metadata={"states_schema": STATES_SCHEMA},
        )

    with pytest.raises(AssertionError, match="expected 'testpkg.dummy_eval'"):
        check_evaluation_recipe(
            EVALUATION_TYPE,
            lambda: _spec("testpkg.other_eval"),
            evaluation_registry=_evaluation_registry(recipe),
            root=tmp_path,
            schema_registry=_registry_with_dummy_family(),
        )
