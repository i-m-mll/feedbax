from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    EvaluationRunMatrixSpec,
    execute_evaluation_run_matrix,
    materialize_evaluation_run_matrix,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.contracts.expressions import ValueQuery
from feedbax.contracts.manifest import (
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EvaluationRunSpec,
    OverridePatch,
)
from feedbax.contracts.matrix_core import MatrixRow, RowDerivation, derive_row_path
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry


def _matrix() -> EvaluationRunMatrixSpec:
    return EvaluationRunMatrixSpec(
        base=EvaluationRunSpec(
            evaluation_type="example.evaluate",
            training_run_ids=["train-a"],
            params={"gain": 1.0, "derived_gain": 0.0},
        ),
        rows=[
            MatrixRow(
                row_id="control",
                deltas=[OverridePatch(path="params.gain", value=2.0)],
                derivations=[
                    RowDerivation(
                        output_path="params.derived_gain",
                        query=ValueQuery(item="row", path="params.gain"),
                    )
                ],
            ),
            MatrixRow(
                row_id="treatment",
                deltas=[OverridePatch(path="params.gain", value=3.0)],
                output_path="custom/result.json",
                spec_path="custom/request.json",
            ),
        ],
    )


def test_evaluation_matrix_applies_deltas_before_per_row_derivation() -> None:
    rows = materialize_evaluation_run_matrix(_matrix())

    assert [row.row_id for row in rows] == ["control", "treatment"]
    assert rows[0].payload.params == {"gain": 2.0, "derived_gain": 2.0}
    assert rows[1].payload.params == {"gain": 3.0, "derived_gain": 0.0}
    assert rows[0].output_path == "control/output.json"
    assert rows[0].spec_path == "control/spec.json"
    assert rows[1].output_path == "custom/result.json"
    assert rows[1].spec_path == "custom/request.json"


def test_evaluation_matrix_requires_unique_path_safe_rows() -> None:
    with pytest.raises(ValidationError, match="row_id values must be unique"):
        EvaluationRunMatrixSpec(
            base=EvaluationRunSpec(evaluation_type="example.evaluate"),
            rows=[MatrixRow(row_id="same"), MatrixRow(row_id="same")],
        )
    with pytest.raises(ValidationError, match="not path-safe"):
        MatrixRow(row_id="not/a/row")
    with pytest.raises(ValueError, match="must be relative"):
        derive_row_path("row", explicit_path="/tmp/output.json")


def test_evaluation_matrix_schema_accepts_current_and_rejects_legacy() -> None:
    payload = _matrix().model_dump(mode="json")
    result = default_spec_registry.migrate("EvaluationRunMatrixSpec", payload)

    assert result.schema_id == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID
    assert result.target_version == EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert not result.migrated

    legacy = {**payload, "schema_version": "feedbax.spec.evaluation_run_matrix.v0"}
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        default_spec_registry.migrate("EvaluationRunMatrixSpec", legacy)


def test_evaluation_matrix_schema_identity_is_pinned() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["schema_id"] = "example.spec.evaluation_matrix"
    with pytest.raises(ValidationError, match="unsupported EvaluationRunMatrixSpec schema_id"):
        EvaluationRunMatrixSpec.model_validate(payload)


def test_evaluation_matrix_executes_through_harness(tmp_path: Path) -> None:
    def recipe(spec, _root, _states_path):
        return EvaluationRecipeResult(summary_metrics={"gain": spec.params["gain"]})

    register_evaluation_recipe("example.evaluate", recipe)
    try:
        result = execute_evaluation_run_matrix(_matrix(), root=tmp_path)
    finally:
        unregister_evaluation_recipe("example.evaluate")

    assert [row.row_id for row in result.rows] == ["control", "treatment"]
    assert all(row.manifest_path is not None and row.manifest_path.exists() for row in result.rows)
    assert result.rows[0].result.metadata["matrix_harness"]["row_id"] == "control"
    assert "regeneration_spec" in result.rows[0].result.metadata["matrix_harness"]
    assert {artifact.role for artifact in result.rows[0].result.artifacts} >= {
        "regeneration_spec",
        "resolved_row_spec",
    }


def test_public_exports_include_matrix_and_harness_apis() -> None:
    import feedbax.analysis as analysis
    import feedbax.contracts as contracts

    assert {
        "EvaluationRunMatrixSpec",
        "MatrixMaterializerHarness",
        "execute_evaluation_run_matrix",
    } <= set(analysis.__all__)
    assert {"MatrixRow", "RowDerivation", "RowMatrixSpec"} <= set(contracts.__all__)
