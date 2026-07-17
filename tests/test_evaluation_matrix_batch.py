from __future__ import annotations

from pathlib import Path

import jax.tree as jt
import numpy as np
import pytest

from feedbax.analysis.evaluation import (
    EvaluationBatchExecution,
    EvaluationBatchRowError,
    EvaluationRecipeResult,
    EvaluationRunMatrixSpec,
    execute_evaluation_run_matrix,
    load_evaluation_states,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
from feedbax.analysis.specs import (
    register_analysis_recipe,
    resolve_analysis_inputs,
    unregister_analysis_recipe,
)
from feedbax.contracts.manifest import AnalysisRunSpec, EvaluationRunSpec, OverridePatch
from feedbax.contracts.matrix_core import MatrixRow


EVALUATION_TYPE = "feedbax.test.batched_matrix"
ANALYSIS_TYPE = "feedbax.test.batched_matrix_analysis"


def _matrix() -> EvaluationRunMatrixSpec:
    return EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type=EVALUATION_TYPE,
            params={"gain": 1.0, "states_custody": "durable"},
        ),
        rows=[
            MatrixRow(
                row_id=f"row-{index}",
                deltas=[OverridePatch(path="params.gain", value=float(index + 1))],
            )
            for index in range(3)
        ],
    )


def _result(gain: float) -> EvaluationRecipeResult:
    states = {"value": np.asarray([gain, gain + 1], dtype=np.float32)}
    return EvaluationRecipeResult(
        states=states,
        summary_metrics={"total": float(states["value"].sum())},
        metadata={"states_schema": "feedbax.test.batched_matrix_states.v1"},
    )


def test_batched_matrix_matches_default_and_round_trips_require_durable(
    tmp_path: Path,
) -> None:
    scalar_calls: list[float] = []
    batch_calls: list[tuple[str, ...]] = []

    def scalar_recipe(spec, _root, _states_path, _execution_context):
        scalar_calls.append(spec.params["gain"])
        return _result(spec.params["gain"])

    def batch_recipe(items, _execution_context):
        batch_calls.append(tuple(item.row_id for item in items))
        gains = np.asarray([item.spec.params["gain"] for item in items])
        return [_result(float(gain)) for gain in gains]

    register_evaluation_recipe(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=batch_recipe,
        replace=True,
    )
    try:
        default = execute_evaluation_run_matrix(_matrix(), root=tmp_path / "default")
        batched = execute_evaluation_run_matrix(
            _matrix(),
            root=tmp_path / "batched",
            batch=EvaluationBatchExecution(),
        )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert scalar_calls == [1.0, 2.0, 3.0]
    assert batch_calls == [("row-0", "row-1", "row-2")]
    for default_row, batched_row in zip(default.rows, batched.rows, strict=True):
        assert default_row.row_id == batched_row.row_id
        assert default_row.result.summary_metrics == batched_row.result.summary_metrics
        default_states = load_evaluation_states(
            default_row.result, root=tmp_path / "default" / default_row.row_id
        )
        batched_states = load_evaluation_states(
            batched_row.result, root=tmp_path / "batched" / batched_row.row_id
        )
        np.testing.assert_array_equal(default_states["value"], batched_states["value"])

    register_analysis_recipe(
        ANALYSIS_TYPE,
        lambda *_args: None,
        replace=True,
        evaluation_states_structure=lambda _: jt.structure(
            {"value": np.asarray([0.0, 0.0], dtype=np.float32)}
        ),
    )
    try:
        for row in batched.rows:
            authority = authenticated_manifest_ref(
                row.result,
                row.manifest_path,
                "evaluation_run",
            )
            resolved = resolve_analysis_inputs(
                AnalysisRunSpec(
                    analysis_type=ANALYSIS_TYPE,
                    inputs=[authority],
                    evaluation_states_policy="require_durable",
                ),
                root=tmp_path / "batched" / row.row_id,
            )[0]
            np.testing.assert_array_equal(
                resolved.states["value"],
                load_evaluation_states(
                    row.result, root=tmp_path / "batched" / row.row_id
                )["value"],
            )
    finally:
        unregister_analysis_recipe(ANALYSIS_TYPE)


def test_batched_matrix_fails_closed_with_typed_row_diagnostic(tmp_path: Path) -> None:
    def scalar_recipe(spec, _root, _states_path, _execution_context):
        return _result(spec.params["gain"])

    def failing_batch(items, _execution_context):
        raise EvaluationBatchRowError(items[1].row_id, ValueError("injected failure"))

    register_evaluation_recipe(
        EVALUATION_TYPE,
        scalar_recipe,
        batch_recipe=failing_batch,
        replace=True,
    )
    output = tmp_path / "batched"
    try:
        with pytest.raises(EvaluationBatchRowError, match="row-1.*injected failure") as exc_info:
            execute_evaluation_run_matrix(
                _matrix(),
                root=output,
                batch=EvaluationBatchExecution(),
            )
    finally:
        unregister_evaluation_recipe(EVALUATION_TYPE)

    assert exc_info.value.row_id == "row-1"
    assert not output.exists()
