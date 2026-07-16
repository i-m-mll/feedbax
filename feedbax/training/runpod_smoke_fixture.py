"""Self-contained native method fixture for RunPod lifecycle validation."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TrainingMethodDescriptor,
    standard_supervised_method_contract,
    standard_supervised_update_kernels,
)
from feedbax.contracts.worker import MetricGuardSpec, PhaseTransitionSpec
from feedbax.training.preparation import ExecutionPreparationResult


METHOD_REF = "feedbax.validation/native_smoke/v1"
GUARD_REF = "feedbax.validation.native_smoke.continue_training"


def method_contract(payload: StandardSupervisedMethodPayload):
    """Build the bounded 100-update smoke contract."""
    total = int(payload.metadata["total_batches"])
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(update={"legal_next": ["train_batch"]})
    transition = PhaseTransitionSpec(
        source="train_batch",
        target="train_batch",
        barrier="after_train_batch",
        guard=MetricGuardSpec(
            predicate_ref=GUARD_REF,
            metric_slots=[],
            metadata={"total_batches": total},
        ),
    )
    program = program.model_copy(update={"phases": [phase], "transitions": [transition]})
    return contract.model_copy(update={"method_ref": METHOD_REF, "phase_program": program})


def register_feedbax_training_methods(registry: Any) -> None:
    """Register the smoke method through the normal executor plugin hook."""
    def continue_training(
        slots: Mapping[str, Any], coordinate: Any, context: Mapping[str, Any]
    ) -> bool:
        del slots, context
        return coordinate.program_step < 100

    def prepare(_request: Any) -> ExecutionPreparationResult:
        return ExecutionPreparationResult(
            initial_slots={
                "model": jnp.array([0.0]),
                "optimizer": {"count": jnp.array([1.0])},
                "prng": jnp.array([0, 1], dtype=jnp.uint32),
                "batch_counter": jnp.array(0, dtype=jnp.int32),
            }
        )

    registry.register_descriptor(
        TrainingMethodDescriptor(
            method_ref=METHOD_REF,
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_compiler=method_contract,
            update_kernels_factory=standard_supervised_update_kernels,
            guard_predicates_factory=lambda _payload: {GUARD_REF: continue_training},
            preparation_provider=prepare,
            owner="feedbax.validation.native_smoke",
            package="feedbax",
        )
    )
