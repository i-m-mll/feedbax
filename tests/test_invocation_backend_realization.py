"""Invocation, backend-plan, and attempt boundary contracts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from feedbax.execution.records import (
    INVOCATION_SCHEMA_ID,
    InvocationExecutionPolicy,
    UnsupportedInvocationVersionError,
    invocation_for_operation,
    invocation_from_document,
)
from feedbax.declarations.science import BackendProtocol
from feedbax.orchestration.drivers.local import local_driver_registration
from feedbax.orchestration.drivers.runpod import runpod_driver_registration
from feedbax.orchestration.realization import (
    ATTEMPT_SCHEMA_ID,
    BACKEND_PLAN_SCHEMA_ID,
    Attempt,
    BackendRealizationRequest,
    ExpectedCost,
    MachineShape,
    OrchestrationBackend,
    UnsupportedBackendRecordVersionError,
    attempt_from_document,
    backend_plan_from_document,
)
from feedbax.workflow.plan import LogicalKey, Operation, PlanEdge, PlanNode, build_workflow_plan


def _sisu_invocation():
    operation_key = LogicalKey("campaign", "sisu-continuous-conditioning")
    node = PlanNode(
        key=operation_key,
        source_ref="campaign/sisu-continuous-conditioning.lock.json",
        operation=Operation(
            type_id="feedbax.operation.train",
            parameters={
                "compiled_schema_id": "rlrmp2.sisu.training",
                "semantic_hash": "a" * 64,
            },
            input_types={"experiment": "rlrmp2.sisu.experiment"},
            output_types={"training_run": "feedbax.training_run"},
            determinism="seeded",
            cache_policy="never",
            effect="external",
            capabilities=("training",),
        ),
        content_hash="b" * 64,
        execution_identity="sisu-tier-a",
    )
    edge = PlanEdge(
        consumer=operation_key,
        role_path=("experiment",),
        status="required",
        basis="authored",
        input_type="rlrmp2.sisu.experiment",
        external={"artifact_id": "sisu-experiment", "sha256": "c" * 64},
        external_type="rlrmp2.sisu.experiment",
    )
    plan = build_workflow_plan(operation_key, (node,), (edge,))
    return invocation_for_operation(
        plan,
        operation_key,
        bound_inputs={
            ("experiment",): {"artifact_id": "sisu-experiment", "sha256": "c" * 64}
        },
        scientific_seeds={"controller": 17, "trial": 23},
        execution_policy=InvocationExecutionPolicy(timeout_seconds=120, max_attempts=2),
    )


def _request(
    variant: str,
    *,
    machine: MachineShape | None = None,
    expected_cost: ExpectedCost | None = None,
    confirmation: str | None = None,
) -> BackendRealizationRequest:
    return BackendRealizationRequest(
        adapter_id=f"feedbax.orchestration.{variant}",
        adapter_version="1",
        capability_variant=variant,
        code_bundle_id="git:feedbax@d8fa266a",
        environment_bundle_id="uv-lock:" + "d" * 64,
        command=("feedbax", "execute-training-run-spec", "sisu-tier-a.json"),
        machine=machine or MachineShape(),
        network_requirements=() if variant.startswith("local") else ("egress:https",),
        secret_names=() if variant.startswith("local") else ("runpod_api_key",),
        timeout_seconds=120,
        retry_classification="same-plan",
        expected_cost=expected_cost,
        billable_confirmation_class=confirmation,
        external_effect_key=f"sisu-{variant}-effect",
        configuration={"sisu_profile": "tier-a"},
    )


def test_invocation_is_provider_neutral_versioned_and_identity_stable() -> None:
    invocation = _sisu_invocation()
    document = invocation.model_dump(mode="json")

    assert document["schema_id"] == INVOCATION_SCHEMA_ID
    assert invocation_from_document(document) == invocation
    assert "provider" not in str(document).lower()
    assert "pod" not in str(document).lower()
    assert "backend" not in str(document).lower()

    changed = {**document, "schema_version": "feedbax.spec.invocation.v0"}
    with pytest.raises(UnsupportedInvocationVersionError, match="no migration"):
        invocation_from_document(changed)


def test_invocation_refuses_provider_and_physical_input_coordinates() -> None:
    invocation = _sisu_invocation()
    document = invocation.model_dump(mode="json")
    document["inputs"][0]["reference"]["provider_id"] = "runpod"
    document["invocation_id"] = "0" * 64

    with pytest.raises(ValidationError, match="BackendPlan or Attempt"):
        invocation_from_document(document)


def test_local_sisu_backend_plan_reuses_driver_capability_seam_without_effect() -> None:
    invocation = _sisu_invocation()
    registration = local_driver_registration()
    backend = OrchestrationBackend(
        backend_id=registration.name,
        supported_scientific_capabilities=frozenset({"training"}),
        driver_capabilities=registration.supported_capabilities,
    )

    assert isinstance(backend, BackendProtocol)
    plan = backend.realize("training", (invocation, _request("local-stop")))

    assert plan.schema_id == BACKEND_PLAN_SCHEMA_ID
    assert plan.invocation_id == invocation.invocation_id
    assert plan.backend_id == "local"
    assert plan.driver_capability_variant == "local-stop"
    assert plan.expected_cost is None
    assert backend_plan_from_document(plan.model_dump(mode="json")) == plan


def test_paid_capable_sisu_plan_is_inert_and_reservation_bound() -> None:
    invocation = _sisu_invocation()
    registration = runpod_driver_registration()
    backend = OrchestrationBackend(
        backend_id=registration.name,
        supported_scientific_capabilities=frozenset({"training"}),
        driver_capabilities=registration.supported_capabilities,
    )
    request = _request(
        "engine-acquired",
        machine=MachineShape(
            accelerator_type="NVIDIA GeForce RTX 4090",
            accelerator_count=1,
            regions=("CA-MTL-1",),
        ),
        expected_cost=ExpectedCost(maximum=2.5, basis="120 second Tier A ceiling"),
        confirmation="authenticated-effect-reservation",
    )

    plan = backend.realize("training", (invocation, request))

    assert plan.backend_id == "runpod"
    assert plan.expected_cost.maximum == 2.5
    assert plan.billable_confirmation_class == "authenticated-effect-reservation"
    assert "pod_id" not in plan.model_dump(mode="json")
    assert "provider_resource_handle" not in plan.model_dump(mode="json")

    with pytest.raises(ValueError, match="expected cost"):
        backend.realize("training", (invocation, _request("engine-acquired")))


def test_attempt_is_a_separate_versioned_observation() -> None:
    invocation = _sisu_invocation()
    registration = local_driver_registration()
    backend = OrchestrationBackend(
        backend_id="local",
        supported_scientific_capabilities=frozenset({"training"}),
        driver_capabilities=registration.supported_capabilities,
    )
    plan = backend.realize("training", (invocation, _request("local-stop")))
    completed = datetime.now(timezone.utc)
    attempt = Attempt(
        attempt_id="sisu-local-attempt-1",
        invocation_id=invocation.invocation_id,
        backend_plan_id=plan.backend_plan_id,
        worker_identity="local-worker",
        status="succeeded",
        started_at=completed,
        terminal_at=completed,
        exit_classification="completed",
        event_refs=("event:sisu-local-attempt-1:completed",),
    )

    assert attempt.schema_id == ATTEMPT_SCHEMA_ID
    assert attempt_from_document(attempt.model_dump(mode="json")) == attempt

    bad = attempt.model_dump(mode="json")
    bad["schema_version"] = "feedbax.manifest.attempt.v0"
    with pytest.raises(UnsupportedBackendRecordVersionError, match="no migration"):
        attempt_from_document(bad)


def test_backend_capability_mismatch_fails_without_fallback() -> None:
    invocation = _sisu_invocation()
    registration = local_driver_registration()
    backend = OrchestrationBackend(
        backend_id="local",
        supported_scientific_capabilities=frozenset({"evaluation"}),
        driver_capabilities=registration.supported_capabilities,
    )

    with pytest.raises(ValueError, match="does not support capability 'training'"):
        backend.realize("training", (invocation, _request("local-stop")))


def test_backend_plan_refuses_observed_handles_and_secret_values() -> None:
    document = _request("local-stop").model_dump(mode="json")
    document["configuration"] = {"pod_id": "pod-1"}
    with pytest.raises(ValidationError, match="handles belong only in Attempt"):
        BackendRealizationRequest.model_validate(document)

    document["configuration"] = {"api_token": "secret-value"}
    with pytest.raises(ValidationError, match="secret names only"):
        BackendRealizationRequest.model_validate(document)
