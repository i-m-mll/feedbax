"""Focused contracts for declaration-owned training program extensions."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from feedbax.contracts.training import (
    DeclaredTrainingProgram,
    MethodPayloadEnvelope,
    MethodRefSpec,
    TrainingProgramRegistry,
    declare_training_program,
    standard_supervised_method_contract,
    standard_supervised_update_kernels,
)
from feedbax.plugins import (
    BootstrapError,
    BootstrapErrorCode,
    TRAINING_PROGRAMS,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    bootstrap_application,
    new_registration_context,
)
from feedbax.plugins.bootstrap import RegistryKey
from feedbax.training.preparation import ExecutionPreparationResult


METHOD_REF = "tests/program/v1"
SCHEMA_ID = "tests.spec.training_program"
SCHEMA_VERSION = "tests.spec.training_program.v1"


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gain: int = 1


def _program(*, prepare=None) -> DeclaredTrainingProgram[Payload]:
    return declare_training_program(
        method_ref=METHOD_REF,
        payload_schema_id=SCHEMA_ID,
        payload_schema_version=SCHEMA_VERSION,
        payload_model=Payload,
        contract_compiler=lambda _payload: standard_supervised_method_contract().model_copy(
            update={
                "method_ref": METHOD_REF,
                "method_payload_schema_version": SCHEMA_VERSION,
            }
        ),
        update_kernels_factory=standard_supervised_update_kernels,
        preparation_provider=prepare,
        owner="tests",
        package="tests",
    )


def test_minimal_program_pays_only_for_runtime_facet() -> None:
    program = _program()
    assert program.runtime is not None
    assert program.authoring is None
    assert program.preparation is None
    assert program.projection is None


def test_registry_resolves_declared_payload_and_rejects_unknown_version() -> None:
    registry = TrainingProgramRegistry()
    registry.register_program(_program())
    method_ref = MethodRefSpec(package="tests", name="program", version="v1")
    resolved = registry.resolve_execution(
        method_ref,
        MethodPayloadEnvelope(
            schema_id=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            payload={"gain": 3},
        ),
    )
    assert resolved.payload.gain == 3
    assert resolved.program is not None

    with pytest.raises(ValueError, match="no method payload migration path"):
        registry.validate_payload(
            method_ref,
            MethodPayloadEnvelope(
                schema_id=SCHEMA_ID,
                schema_version="tests.spec.training_program.v0",
                payload={"gain": 3},
            ),
            path="/method_payload",
        )


def test_duplicate_program_identity_fails_before_mutation() -> None:
    registry = TrainingProgramRegistry()
    registry.register_program(_program())
    with pytest.raises(ValueError, match="training program already registered"):
        registry.register_program(_program())
    assert registry.program_keys() == (METHOD_REF,)


def test_application_composition_derives_preparation_registry() -> None:
    def prepare(_request):
        return ExecutionPreparationResult(initial_slots={})

    registration = PluginRegistration(
        PluginDeclaration(
            "tests.training_program",
            "1",
            1,
            families=(FamilyRequirement("training_programs"),),
        ),
        lambda context: context.registry(TRAINING_PROGRAMS).register_program(
            _program(prepare=prepare)
        ),
    )
    state = asyncio.run(
        bootstrap_application(
            new_registration_context(local_component_source=None),
            registrations=(registration,),
        )
    )
    assert state.bundle.training_programs.program(METHOD_REF) is not None
    assert state.bundle.execution_preparations.get(METHOD_REF) is not None


def test_predecessor_training_families_are_not_composable() -> None:
    context = new_registration_context(local_component_source=None)
    for family in ("training_methods", "row_lowerers", "execution_preparations"):
        obsolete = RegistryKey(family, family, object, registered_keys=lambda _value: ())
        with pytest.raises(BootstrapError, match="is unavailable") as exc_info:
            context.registry(obsolete)
        assert exc_info.value.code is BootstrapErrorCode.MISSING_FAMILY
