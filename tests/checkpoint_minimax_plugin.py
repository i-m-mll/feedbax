"""Typed training-method plugin for checkpoint CLI contract tests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts.training import (
    MethodPayloadEnvelope,
    MethodRefSpec,
    OptimizerSpec,
    TrainingMethodDescriptor,
)
from feedbax.contracts.worker import (
    EffectivePhaseSpec,
    MethodContractSpec,
    derive_consistency_predicate,
    toy_minimax_method_contract,
)
from feedbax.plugins import (
    TRAINING_METHODS,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)

PLUGIN_MODULE = "tests.checkpoint_minimax_plugin"
METHOD_REF = "tests/checkpoint_minimax/v1"
PAYLOAD_SCHEMA_ID = "tests.spec.training_method.checkpoint_minimax_payload"
PAYLOAD_SCHEMA_VERSION = f"{PAYLOAD_SCHEMA_ID}.v1"


class CheckpointMinimaxPayload(BaseModel):
    """Test-owned payload selecting the checkpoint minimax contract."""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["toy_minimax"] = "toy_minimax"
    optimizer: OptimizerSpec = Field(default_factory=lambda: OptimizerSpec(type="adamw"))


def checkpoint_minimax_method_contract(
    _payload: CheckpointMinimaxPayload | None = None,
) -> MethodContractSpec:
    """Return the exact contract advertised by the test method identity."""
    contract = toy_minimax_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    program.checkpoint_barriers[0].metadata["consistency_mode"] = "population-barrier"
    return contract.model_copy(
        update={
            "method_ref": METHOD_REF,
            "method_payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "phase_program": program,
        }
    )


def checkpoint_minimax_effective_phase() -> EffectivePhaseSpec:
    """Return the effective phase derived from the typed test contract."""
    contract = checkpoint_minimax_method_contract()
    return EffectivePhaseSpec(
        method_ref=METHOD_REF,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=derive_consistency_predicate(contract.phase_program),
    )


def checkpoint_minimax_method_ref() -> MethodRefSpec:
    return MethodRefSpec(package="tests", name="checkpoint_minimax", version="v1")


def checkpoint_minimax_method_payload() -> MethodPayloadEnvelope:
    return MethodPayloadEnvelope(
        schema_id=PAYLOAD_SCHEMA_ID,
        schema_version=PAYLOAD_SCHEMA_VERSION,
        payload=CheckpointMinimaxPayload().model_dump(mode="json"),
    )


def _update_kernels(_payload: CheckpointMinimaxPayload):
    def update(slots, coordinate, context):
        del coordinate, context
        return slots

    return {
        "toy_minimax.warmup_update": update,
        "toy_minimax.adversary_update": update,
    }


def checkpoint_minimax_method_descriptor() -> TrainingMethodDescriptor[CheckpointMinimaxPayload]:
    """Return the typed descriptor for the checkpoint minimax test method."""
    return TrainingMethodDescriptor(
        method_ref=METHOD_REF,
        payload_schema_id=PAYLOAD_SCHEMA_ID,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        payload_model=CheckpointMinimaxPayload,
        contract_compiler=checkpoint_minimax_method_contract,
        update_kernels_factory=_update_kernels,
        optimizer_spec_projector=lambda payload: payload.optimizer,
        owner="tests",
        package="tests",
    )


def _register(context) -> None:
    context.registry(TRAINING_METHODS).register_descriptor(
        checkpoint_minimax_method_descriptor()
    )


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.checkpoint_minimax",
        "1.0",
        1,
        families=(FamilyRequirement("training_methods"),),
    ),
    _register,
)
