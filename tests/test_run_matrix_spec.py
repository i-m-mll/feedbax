from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.contracts.manifest import OverridePatch, TrainingSweepAxis
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    MatrixBaseSpec,
    TrainingRunMatrixSpec,
)


def _minimal_spec() -> dict[str, object]:
    return {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "matrix",
        "base": {"inline": {"schema_id": "feedbax.spec.training_run"}},
        "rows": [{"row_id": "row_a", "overrides": []}],
    }


def test_run_matrix_spec_accepts_explicit_rows_and_axes_modes() -> None:
    explicit = TrainingRunMatrixSpec.model_validate(_minimal_spec())
    axes_payload = _minimal_spec()
    axes_payload.pop("rows")
    axes_payload["axes"] = [
        {
            "id": "lr",
            "path": "training_config.learning_rate",
            "variation": {"kind": "explicit", "values": [0.1, 0.01]},
        }
    ]

    axes = TrainingRunMatrixSpec.model_validate(axes_payload)

    assert explicit.schema_id == TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
    assert axes.axes[0] == TrainingSweepAxis.model_validate(axes_payload["axes"][0])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.__setitem__("schema_id", "wrong"), "/schema_id"),
        (lambda payload: payload.__setitem__("schema_version", "old"), "/schema_version"),
        (lambda payload: payload.__setitem__("axes", [{"id": "x", "path": "a.b", "variation": {"kind": "explicit", "values": [1]}}]), "mutually exclusive"),
        (lambda payload: payload.__setitem__("rows", []), "mutually exclusive"),
        (lambda payload: payload["rows"].append({"row_id": "row_a", "overrides": []}), "unique"),
        (lambda payload: payload.__setitem__("rows", [{"row_id": "bad/row", "overrides": []}]), "path-safe"),
        (lambda payload: payload.__setitem__("derivations", [{"output_path": "a", "query": {"item": "x"}}, {"output_path": "a", "query": {"item": "x"}}]), "collide"),
    ],
)
def test_run_matrix_spec_rejects_invalid_shapes(mutator, message: str) -> None:
    payload = _minimal_spec()
    mutator(payload)

    with pytest.raises(ValidationError, match=message):
        TrainingRunMatrixSpec.model_validate(payload)


def test_base_ref_rejects_absolute_paths_and_requires_one_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        MatrixBaseSpec.model_validate({})
    with pytest.raises(ValidationError, match="repo-relative"):
        MatrixBaseSpec.model_validate({"ref": "/tmp/base.json"})


def test_override_patch_remove_has_no_value_and_add_replace_require_value() -> None:
    assert OverridePatch.model_validate({"path": "a.b", "op": "remove"}).value is None
    with pytest.raises(ValidationError, match="must not carry value"):
        OverridePatch.model_validate({"path": "a.b", "op": "remove", "value": 1})
    with pytest.raises(ValidationError, match="requires value"):
        OverridePatch.model_validate({"path": "a.b", "op": "replace"})


def test_run_matrix_spec_old_version_rejection_policy_is_registered() -> None:
    result = default_spec_registry.migrate(
        "TrainingRunMatrixSpec",
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        },
    )
    assert not result.migrated

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "TrainingRunMatrixSpec",
            {"schema_version": "feedbax.spec.training_run_matrix.v0"},
        )
