from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from feedbax.contracts.manifest import TrainingRunManifest, spec_payload
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    STANDARD_SUPERVISED_METHOD_REF,
    TRAINING_RUN_SPEC_SCHEMA_ID,
    TRAINING_RUN_SPEC_SCHEMA_VERSION,
    TRAINING_RUN_SPEC_SCHEMA_VERSION_V1,
    TRAINING_RUN_SPEC_SCHEMA_VERSION_V2,
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.integrations.provider import provider_manifest, validate_training_run_spec


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _training_run_payload() -> dict[str, object]:
    worker = WorkerExecutionSpec(
        method_contract=standard_supervised_method_contract(),
        effective_phase=standard_supervised_effective_phase_spec(),
    )
    spec = TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=worker,
    )
    return spec.model_dump(mode="json")


def test_training_run_spec_current_version_round_trips_and_embeds_in_manifest() -> None:
    payload = _training_run_payload()

    spec = TrainingRunSpec.model_validate(payload)
    round_tripped = TrainingRunSpec.model_validate(spec.model_dump(mode="json"))
    registry_result = default_spec_registry.migrate("TrainingRunSpec", payload)
    embedded = spec_payload("TrainingRunSpec", payload)
    manifest = TrainingRunManifest(id="feedbax-training-run:test", training_spec=embedded)

    assert spec.schema_id == TRAINING_RUN_SPEC_SCHEMA_ID
    assert spec.schema_version == TRAINING_RUN_SPEC_SCHEMA_VERSION
    assert spec.on_nan == "raise"
    assert spec.method_ref.key == STANDARD_SUPERVISED_METHOD_REF
    assert round_tripped == spec
    assert registry_result.payload == payload
    assert not registry_result.migrated
    assert embedded.schema_id == TRAINING_RUN_SPEC_SCHEMA_ID
    assert embedded.schema_version == TRAINING_RUN_SPEC_SCHEMA_VERSION
    assert manifest.training_spec == embedded


def test_training_config_grad_clip_default_and_explicit_null_round_trip() -> None:
    default_config = TrainingConfig.model_validate({})
    unclipped_config = TrainingConfig.model_validate({"grad_clip": None})
    payload = unclipped_config.model_dump(mode="json")

    assert default_config.grad_clip == 1.0
    assert unclipped_config.grad_clip is None
    assert payload["grad_clip"] is None
    assert TrainingConfig.model_validate(payload).grad_clip is None


def test_training_run_spec_old_version_rejection_policy_is_explicit() -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        default_spec_registry.migrate(
            "TrainingRunSpec",
            {"schema_version": "feedbax.spec.training_run.v0"},
        )

    message = str(excinfo.value)
    assert "family='TrainingRunSpec'" in message
    assert "schema_id='feedbax.spec.training_run'" in message
    assert "migration_intentionally_absent=yes" in message


def test_training_run_spec_v1_migrates_through_v3_with_fail_loud_nan_policy() -> None:
    payload = _training_run_payload()
    payload.pop("on_nan")
    payload["schema_version"] = TRAINING_RUN_SPEC_SCHEMA_VERSION_V1

    result = default_spec_registry.migrate("TrainingRunSpec", payload)

    assert result.source_version == TRAINING_RUN_SPEC_SCHEMA_VERSION_V1
    assert result.target_version == TRAINING_RUN_SPEC_SCHEMA_VERSION
    assert result.payload["schema_version"] == TRAINING_RUN_SPEC_SCHEMA_VERSION
    assert result.payload["on_nan"] == "raise"
    assert [record.migration_id for record in result.migration_records] == [
        "training-run-spec-v1-to-v2-nan-policy",
        "training-run-spec-v2-to-v3-mapped-axis-vocabulary",
    ]
    assert TrainingRunSpec.model_validate(result.payload).on_nan == "raise"


def test_training_run_spec_v2_migrates_embedded_worker_contracts_to_v2() -> None:
    payload = _training_run_payload()
    payload["schema_version"] = TRAINING_RUN_SPEC_SCHEMA_VERSION_V2
    worker = payload["worker_execution"]
    assert isinstance(worker, dict)
    worker.pop("mapping_levels", None)
    for field_name in ("method_contract", "effective_phase"):
        embedded = worker[field_name]
        assert isinstance(embedded, dict)
        embedded["schema_version"] = "feedbax.spec.worker.execution_program.v1"
        phase_program = embedded["phase_program"]
        assert isinstance(phase_program, dict)
        phase_program["schema_version"] = "feedbax.spec.worker.execution_program.v1"

    result = default_spec_registry.migrate("TrainingRunSpec", payload)

    assert result.payload["schema_version"] == TRAINING_RUN_SPEC_SCHEMA_VERSION
    migrated_worker = result.payload["worker_execution"]
    assert migrated_worker["mapping_levels"] is None
    for field_name in ("method_contract", "effective_phase"):
        embedded = migrated_worker[field_name]
        assert embedded["schema_version"] == "feedbax.spec.worker.execution_program.v2"
        assert (
            embedded["phase_program"]["schema_version"]
            == "feedbax.spec.worker.execution_program.v2"
        )
    assert TrainingRunSpec.model_validate(result.payload).worker_execution.mapping_levels is None


def test_standard_supervised_method_payload_validates_and_round_trips() -> None:
    payload = _training_run_payload()
    result = validate_training_run_spec(payload)
    method_payload = payload["method_payload"]
    method_result = default_spec_registry.migrate(
        "StandardSupervisedMethodPayload",
        {
            "schema_version": STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            **method_payload["payload"],
        },
    )

    assert result.valid
    assert method_payload["schema_id"] == STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID
    assert method_payload["schema_version"] == STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION
    assert method_result.target_version == STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION
    assert not method_result.migrated


def test_training_run_spec_unknown_method_ref_fails_with_available_registry_keys() -> None:
    payload = _training_run_payload()
    payload["method_ref"]["name"] = "unknown"

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    message = str(excinfo.value)
    assert "/method_ref" in message
    assert "unknown method_ref 'feedbax/unknown/v1'" in message
    assert STANDARD_SUPERVISED_METHOD_REF in message


def test_training_run_spec_unsupported_method_payload_version_fails_closed() -> None:
    payload = _training_run_payload()
    payload["method_payload"]["schema_version"] = (
        "feedbax.spec.training_method.standard_supervised_payload.v0"
    )

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    message = str(excinfo.value)
    assert "/method_payload/schema_version" in message
    assert "migration_intentionally_absent=yes" in message
    assert STANDARD_SUPERVISED_METHOD_REF in message


def test_training_run_spec_rejects_behavior_significant_untyped_extensions() -> None:
    payload = _training_run_payload()
    payload["method_extensions"]["behavior"] = {"jit_static": ["learning_rate"]}

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    message = str(excinfo.value)
    assert "method_extensions.behavior" in message
    assert "Extra inputs are not permitted" in message


def test_training_run_spec_missing_worker_declaration_fails_pre_launch() -> None:
    payload = _training_run_payload()
    del payload["worker_execution"]

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    assert "worker_execution" in str(excinfo.value)


def test_training_run_spec_partial_request_fails_pre_launch() -> None:
    payload = _training_run_payload()
    del payload["task"]

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    assert "task" in str(excinfo.value)


def test_training_run_spec_worker_method_ref_must_match_request_method_ref() -> None:
    payload = _training_run_payload()
    payload = copy.deepcopy(payload)
    payload["worker_execution"]["method_contract"]["method_ref"] = "feedbax/other/v1"

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    message = str(excinfo.value)
    assert "/worker_execution/method_contract/method_ref" in message
    assert STANDARD_SUPERVISED_METHOD_REF in message


def test_descriptor_rejects_embedded_contract_or_effective_phase_drift() -> None:
    contract_payload = _training_run_payload()
    contract_payload["worker_execution"]["method_contract"]["axes"][0]["role"] = "environment"
    contract_spec = TrainingRunSpec.model_validate(contract_payload)

    with pytest.raises(ValueError, match="must exactly match.*payload-compiled"):
        _ = contract_spec.resolved_method

    phase_payload = _training_run_payload()
    phase_payload["worker_execution"]["effective_phase"]["axes"][0]["role"] = "environment"
    phase_spec = TrainingRunSpec.model_validate(phase_payload)

    with pytest.raises(ValueError, match="effective_phase must exactly match"):
        _ = phase_spec.resolved_method


def test_method_resolution_cache_revalidates_model_copy_updates() -> None:
    spec = TrainingRunSpec.model_validate(_training_run_payload())
    original_resolution = spec.resolved_method
    assert spec.resolved_method is original_resolution

    payload = spec.method_payload.model_copy(
        update={
            "payload": {
                **spec.method_payload.payload,
                "gradient_clip": 0.25,
            }
        }
    )
    changed = spec.model_copy(update={"method_payload": payload})

    assert changed.resolved_method is not original_resolution
    assert changed.resolved_method.payload.gradient_clip == 0.25

    invalid_contract = spec.worker_execution.method_contract.model_copy(
        update={"method_ref": "feedbax/other/v1"}
    )
    invalid_worker = spec.worker_execution.model_copy(
        update={"method_contract": invalid_contract}
    )
    invalid = spec.model_copy(update={"worker_execution": invalid_worker})

    with pytest.raises(ValueError, match="method_contract/method_ref must match"):
        _ = invalid.resolved_method


def test_provider_manifest_advertises_training_run_spec_as_execution_input() -> None:
    manifest = provider_manifest()

    assert manifest.capabilities["validate_training_spec"].input_schema == "TrainingRunSpec"
    assert manifest.capabilities["start_training_run"].input_schema == "TrainingRunSpec"
    assert "TrainingRunSpec" in manifest.schemas
