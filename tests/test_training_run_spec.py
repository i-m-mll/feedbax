from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from feedbax.contracts.manifest import (
    TrainingRunManifest,
    spec_payload,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.training import (
    TRAINING_RUN_SPEC_SCHEMA_ID,
    TRAINING_RUN_SPEC_SCHEMA_VERSION,
    TrainingConfig,
    TrainingRunSpec,
    resolve_training_run_spec,
)
from tests.support.managed_train_fixture import managed_train_spec
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD


def _training_run_payload() -> dict[str, object]:
    spec, _ = managed_train_spec()
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
    assert spec.method_ref.key == STUDIO_GRAPH_METHOD.key
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


def test_training_run_spec_rejects_unregistered_old_version() -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        default_spec_registry.migrate(
            "TrainingRunSpec",
            {"schema_version": "feedbax.spec.training_run.v0"},
        )

    message = str(excinfo.value)
    assert "family='TrainingRunSpec'" in message
    assert "schema_id='feedbax.spec.training_run'" in message
    assert "No Feedbax structured spec migration path registered" in message


def test_training_run_spec_unknown_method_ref_fails_with_available_registry_keys() -> None:
    payload = _training_run_payload()
    payload["method_ref"]["name"] = "unknown"
    payload["worker_execution"]["method_contract"]["method_ref"] = "feedbax/unknown/v1"
    payload["worker_execution"]["effective_phase"]["method_ref"] = "feedbax/unknown/v1"

    spec = TrainingRunSpec.model_validate(payload)
    with pytest.raises(ValueError) as excinfo:
        resolve_training_run_spec(spec, managed_train_spec()[1].training_programs)

    message = str(excinfo.value)
    assert "/method_ref" in message
    assert "unknown method_ref 'feedbax/unknown/v1'" in message
    assert STUDIO_GRAPH_METHOD.key in message


def test_training_run_spec_unsupported_method_payload_version_fails_closed() -> None:
    payload = _training_run_payload()
    unsupported_version = "feedbax.spec.training_method.studio_graph_payload.v0"
    payload["method_payload"]["schema_version"] = unsupported_version
    payload["worker_execution"]["method_contract"]["method_payload_schema_version"] = (
        unsupported_version
    )

    spec = TrainingRunSpec.model_validate(payload)
    with pytest.raises(ValueError) as excinfo:
        resolve_training_run_spec(spec, managed_train_spec()[1].training_programs)

    message = str(excinfo.value)
    assert "/method_payload/schema_version" in message
    assert "no method payload migration path registered" in message
    assert STUDIO_GRAPH_METHOD.key in message


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
    del payload["trial_documents"]

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    assert "trial_documents" in str(excinfo.value)


def test_training_run_spec_worker_method_ref_must_match_request_method_ref() -> None:
    payload = _training_run_payload()
    payload = copy.deepcopy(payload)
    payload["worker_execution"]["method_contract"]["method_ref"] = "feedbax/other/v1"

    with pytest.raises(ValidationError) as excinfo:
        TrainingRunSpec.model_validate(payload)

    message = str(excinfo.value)
    assert "/worker_execution/method_contract/method_ref" in message
    assert STUDIO_GRAPH_METHOD.key in message


def test_descriptor_rejects_embedded_contract_or_effective_phase_drift() -> None:
    contract_payload = _training_run_payload()
    contract_payload["worker_execution"]["method_contract"]["axes"][0]["role"] = "environment"
    contract_spec = TrainingRunSpec.model_validate(contract_payload)

    with pytest.raises(ValueError, match="must exactly match.*payload-compiled"):
        resolve_training_run_spec(contract_spec, managed_train_spec()[1].training_programs)

    phase_payload = _training_run_payload()
    phase_payload["worker_execution"]["effective_phase"]["axes"][0]["role"] = "environment"
    phase_spec = TrainingRunSpec.model_validate(phase_payload)

    with pytest.raises(ValueError, match="effective_phase must exactly match"):
        resolve_training_run_spec(phase_spec, managed_train_spec()[1].training_programs)


def test_explicit_method_resolution_revalidates_model_copy_updates() -> None:
    registry = managed_train_spec()[1].training_programs
    spec = TrainingRunSpec.model_validate(_training_run_payload())
    original_resolution = resolve_training_run_spec(spec, registry)

    payload = spec.method_payload.model_copy(
        update={
            "payload": {
                **spec.method_payload.payload,
                "seed": 7,
            }
        }
    )
    changed = spec.model_copy(update={"method_payload": payload})

    changed_resolution = resolve_training_run_spec(changed, registry)
    assert changed_resolution is not original_resolution
    assert changed_resolution.payload.seed == 7

    invalid_contract = spec.worker_execution.method_contract.model_copy(
        update={"method_ref": "feedbax/other/v1"}
    )
    invalid_worker = spec.worker_execution.model_copy(update={"method_contract": invalid_contract})
    invalid = spec.model_copy(update={"worker_execution": invalid_worker})

    with pytest.raises(ValueError, match="method_contract/method_ref must match"):
        resolve_training_run_spec(invalid, registry)
