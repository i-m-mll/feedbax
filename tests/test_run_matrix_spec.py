from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from feedbax.contracts.manifest import OverridePatch, TrainingSweepAxis
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5,
    MatrixBaseSpec,
    TrainingRunMatrixSpec,
    TrainingRunMatrixSpecV5,
)


def _minimal_spec() -> dict[str, object]:
    return {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "matrix",
        "base": {"kind": "inline", "inline": {"schema_id": "feedbax.spec.training_run"}},
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
        (lambda payload: payload.__setitem__("schema_version", "old"), "schema_version"),
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
    adapter = TypeAdapter(MatrixBaseSpec)
    with pytest.raises(ValidationError, match="union_tag_not_found"):
        adapter.validate_python({})
    with pytest.raises(ValidationError, match="repo-relative"):
        adapter.validate_python(
            {
                "kind": "authored_intent",
                "ref": "/tmp/base.json",
                "content_hash": "0" * 64,
            }
        )


def test_override_patch_remove_has_no_value_and_add_replace_require_value() -> None:
    assert OverridePatch.model_validate({"path": "a.b", "op": "remove"}).value is None
    with pytest.raises(ValidationError, match="must not carry value"):
        OverridePatch.model_validate({"path": "a.b", "op": "remove", "value": 1})
    with pytest.raises(ValidationError, match="requires value"):
        OverridePatch.model_validate({"path": "a.b", "op": "replace"})


def test_run_matrix_spec_migrations_accept_current_and_reject_unsupported_versions() -> None:
    result = default_spec_registry.migrate(
        "TrainingRunMatrixSpec",
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        },
    )
    assert not result.migrated

    migrated_v4 = default_spec_registry.migrate(
        "TrainingRunMatrixSpec",
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": "feedbax.spec.training_run_matrix.v4",
            "name": "no-base-derivations",
            "base": {"kind": "inline", "inline": {"value": 1}},
            "rows": [{"row_id": "row_a", "overrides": []}],
        },
    )
    assert migrated_v4.migrated
    assert migrated_v4.target_version == TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert migrated_v4.payload["schema_version"] == TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
    assert [record.migration_id for record in migrated_v4.migration_records] == [
        "training-run-matrix-v4-to-v5-per-row-derivations",
        "training-run-matrix-v5-to-v6-closed-fork-authority",
    ]

    with pytest.raises(ValueError, match="base-only derivation semantics are ambiguous"):
        default_spec_registry.migrate(
            "TrainingRunMatrixSpec",
            {
                "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.spec.training_run_matrix.v4",
                "derivations": [{"output_path": "a", "query": {"item": "x"}}],
            },
        )

    migrated = default_spec_registry.migrate(
        "TrainingRunMatrixSpec",
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": "feedbax.spec.training_run_matrix.v1",
            "base": {"inline": {"value": 1}},
        },
    )
    assert migrated.payload["base"] == {"kind": "inline", "inline": {"value": 1}}
    assert migrated.payload["deltas"] == []
    assert migrated.payload["execution_dependencies"] == []

    migrated_v2 = default_spec_registry.migrate(
        "TrainingRunMatrixSpec",
        {
            "schema_version": "feedbax.spec.training_run_matrix.v2",
            "base": {"kind": "inline", "inline": {"value": 1}},
        },
    )
    assert migrated_v2.target_version == TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION

    pinned = default_spec_registry.migrate(
        "TrainingRunMatrixSpec",
        {
            "schema_version": "feedbax.spec.training_run_matrix.v1",
            "base": {"ref": "base.json", "sha256": "a" * 64},
        },
    )
    assert pinned.payload["base"] == {
        "kind": "authored_intent",
        "ref": "base.json",
        "content_hash": "a" * 64,
        "pin_algorithm": "legacy_raw_sha256",
    }
    with pytest.raises(ValueError, match="does not identify checkpoint custody"):
        default_spec_registry.migrate("TrainingRunMatrixSpec", {
            "schema_version": "feedbax.spec.training_run_matrix.v3",
            "fork": {"source_run_id": "legacy-run"},
        })

    with pytest.raises(ValueError, match="unpinned"):
        default_spec_registry.migrate(
            "TrainingRunMatrixSpec",
            {
                "schema_version": "feedbax.spec.training_run_matrix.v1",
                "base": {"ref": "base.json"},
            },
        )

    for invalid_base in (
        {"inline": {}, "ref": "base.json"},
        {"inline": {}, "sha256": "a" * 64},
    ):
        with pytest.raises(ValueError, match="cannot carry"):
            default_spec_registry.migrate(
                "TrainingRunMatrixSpec",
                {
                    "schema_version": "feedbax.spec.training_run_matrix.v1",
                    "base": invalid_base,
                },
            )

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "TrainingRunMatrixSpec",
            {"schema_version": "feedbax.spec.training_run_matrix.v0"},
        )

    with pytest.raises(UnsupportedSpecVersion, match="No Feedbax structured spec migration path"):
        default_spec_registry.migrate(
            "TrainingRunMatrixSpec",
            {"schema_version": "feedbax.spec.training_run_matrix.v999"},
        )


def test_matrix_v5_remains_exact_and_migrates_only_authentic_execution_hashes() -> None:
    payload = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5,
        "name": "legacy-fork",
        "base": {"kind": "inline", "inline": {"value": 1}},
        "rows": [{"row_id": "target", "overrides": []}],
        "execution_dependencies": [
            {
                "kind": "fork_from_selected_checkpoint",
                "source_execution_hash": "a" * 64,
                "source_row_id": "source",
                "checkpoint_transaction_id": "transaction",
                "checkpoint_root_hash": "b" * 64,
            }
        ],
    }
    legacy = TrainingRunMatrixSpecV5.model_validate(payload)
    assert legacy.model_dump(mode="json", exclude_none=True) == {
        **payload,
        "deltas": [],
        "sources": [],
        "derivations": [],
        "axes": [],
        "combination": {"mode": "cross", "groups": [], "manual_coordinates": [], "metadata": {}},
        "tags": [],
        "metadata": {},
        "rows": [{"row_id": "target", "overrides": [], "metadata": {}}],
        "execution_dependencies": [{**payload["execution_dependencies"][0], "slot_transforms": []}],
    }

    migrated = default_spec_registry.migrate("TrainingRunMatrixSpec", payload)
    dependency = migrated.payload["execution_dependencies"][0]
    assert "source_execution_hash" not in dependency
    assert dependency["source_authority"] == {
        "kind": "execution_hash",
        "execution_hash": "a" * 64,
    }
    TrainingRunMatrixSpec.model_validate(migrated.payload)

    missing = {
        **payload,
        "execution_dependencies": [
            {
                key: value
                for key, value in payload["execution_dependencies"][0].items()
                if key != "source_execution_hash"
            }
        ],
    }
    with pytest.raises(ValueError, match="never synthesizes execution identity"):
        default_spec_registry.migrate("TrainingRunMatrixSpec", missing)


@pytest.mark.parametrize("tolerance", [-1.0, float("inf"), float("nan")])
def test_matrix_v6_rejects_invalid_absolute_lr_tolerance(tolerance: float) -> None:
    payload = _minimal_spec()
    payload["fork"] = {
        "lr_continuation": "continue",
        "parity": "require",
        "absolute_lr_tolerance": tolerance,
    }
    with pytest.raises(ValidationError, match="finite and nonnegative"):
        TrainingRunMatrixSpec.model_validate(payload)


def test_matrix_v6_binds_resolved_parent_and_target_only_authority() -> None:
    payload = _minimal_spec()
    payload["base"] = {
        "kind": "resolved_output",
        "ref": "artifact-blob:source",
        "resolved_root_hash": "c" * 64,
        "row_id": "source",
        "checkpoint_transaction_id": "transaction",
    }
    payload["execution_dependencies"] = [
        {
            "kind": "fork_from_selected_checkpoint",
            "source_authority": {
                "kind": "resolved_output_root",
                "resolved_root_hash": "c" * 64,
            },
            "source_row_id": "source",
            "checkpoint_transaction_id": "transaction",
            "checkpoint_root_hash": "d" * 64,
            "source_barrier": "after_segment",
            "slot_transforms": [
                {
                    "transform_id": "generic.initialize_slot",
                    "version": "v1",
                    "implementation_sha256": "e" * 64,
                    "stage": "target_post",
                    "target_row_id": "row_a",
                    "slot": "adaptive_state",
                    "target_only": {
                        "method_ref": "generic/method/v1",
                        "slot_identity": "f" * 64,
                    },
                }
            ],
        }
    ]
    assert TrainingRunMatrixSpec.model_validate(payload).base.row_id == "source"

    payload["execution_dependencies"][0]["source_row_id"] = "other"
    with pytest.raises(ValidationError, match="selected checkpoint drift"):
        TrainingRunMatrixSpec.model_validate(payload)
