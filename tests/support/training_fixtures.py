"""Explicit synthetic training registrations for lifecycle and contract tests."""

from functools import partial
from typing import Any, Callable, Mapping

import jax.numpy as jnp
from pydantic import BaseModel

from feedbax.contracts.training import (
    DeclaredTrainingProgram,
    StandardSupervisedMethodPayload,
    TrainingProgramRegistry,
    evolve_training_program,
    standard_supervised_method_contract,
    standard_supervised_training_program,
)
from feedbax.contracts.worker import MethodContractSpec, ProgressCoordinate, StateSlotSpec
from feedbax.plugins import (
    TRAINING_PROGRAMS,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
    RegistrationContext,
)
from feedbax.training.preparation import ExecutionPreparationResult


def synthetic_supervised_update_kernels(
    _payload: BaseModel | None = None,
) -> Mapping[str, Callable[..., Mapping[str, Any]]]:
    """Return synthetic slot updates for lifecycle fixtures, without numerical training."""

    def gradient_update(
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        updates: dict[str, Any] = {
            "model": slots["model"],
            "optimizer": slots["optimizer"],
            "prng": slots["prng"],
            "train_loss": float(coordinate.program_step + 1),
            "batch_counter": slots["batch_counter"] + 1,
        }
        model = slots["model"]
        optimizer = slots["optimizer"]
        if isinstance(optimizer, Mapping) and "count" in optimizer:
            updates["optimizer"] = {**dict(optimizer), "count": optimizer["count"] + 1}
            try:
                updates["model"] = model + optimizer["count"]
            except TypeError:
                updates["model"] = model
        return updates

    return {"feedbax.training.standard_supervised.gradient_update": gradient_update}


def synthetic_supervised_training_program() -> DeclaredTrainingProgram[
    StandardSupervisedMethodPayload
]:
    return evolve_training_program(
        standard_supervised_training_program(),
        update_kernels_factory=synthetic_supervised_update_kernels,
        owner="tests.support.training_fixtures",
    )


def synthetic_training_program_registry() -> TrainingProgramRegistry:
    registry = TrainingProgramRegistry()
    registry.register_program(synthetic_supervised_training_program())
    return registry


CLI_METHOD_REF = "tests/synthetic_supervised/v1"
CLI_LR_METHOD_REF = "tests/synthetic_supervised_lr/v1"
CLI_TWO_STEP_METHOD_REF = "tests/synthetic_supervised_two_step/v1"


def cli_fixture_method_contract(method_ref: str) -> MethodContractSpec:
    contract = standard_supervised_method_contract().model_copy(update={"method_ref": method_ref})
    if method_ref == CLI_TWO_STEP_METHOD_REF:
        contract.phase_program.phases[0].max_steps = 2
    if method_ref == CLI_LR_METHOD_REF:
        contract.state_slots.append(
            StateSlotSpec(name="learning_rate", role="metric", required=False)
        )
        contract.phase_program.phases[0].writes.append("learning_rate")
        contract.phase_program.update_steps[0].writes.append("learning_rate")
    return contract


def initial_training_slots() -> dict[str, Any]:
    return {
        "model": jnp.array([0.0]),
        "optimizer": {"count": jnp.array([1.0])},
        "prng": jnp.array([0, 1], dtype=jnp.uint32),
        "batch_counter": jnp.array(0, dtype=jnp.int32),
    }


def _cli_lr_update_kernels(
    payload: StandardSupervisedMethodPayload,
) -> Mapping[str, Callable[..., Mapping[str, Any]]]:
    """Emit the constant schedule value via the normal worker progress metrics."""
    schedule = payload.optimizer.lr_schedule
    assert schedule is not None and schedule.kind == "constant"
    kernels = synthetic_supervised_update_kernels(payload)
    ((kernel_ref, gradient_update),) = kernels.items()

    def update(
        slots: Mapping[str, Any], coordinate: ProgressCoordinate, context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return {
            **gradient_update(slots, coordinate, context),
            "learning_rate": schedule.learning_rate_0,
        }

    return {kernel_ref: update}


def _register_cli_fixture(context: RegistrationContext) -> None:
    def compile_contract(method_ref: str, _payload: BaseModel) -> MethodContractSpec:
        return cli_fixture_method_contract(method_ref)

    for method_ref in (CLI_METHOD_REF, CLI_TWO_STEP_METHOD_REF, CLI_LR_METHOD_REF):
        program = synthetic_supervised_training_program()
        if method_ref == CLI_LR_METHOD_REF:
            program = evolve_training_program(program, update_kernels_factory=_cli_lr_update_kernels)
        context.registry(TRAINING_PROGRAMS).register_program(
            evolve_training_program(
                program,
                method_ref=method_ref,
                contract_compiler=partial(compile_contract, method_ref),
                preparation_provider=lambda _request: ExecutionPreparationResult(
                    initial_slots=initial_training_slots()
                ),
                package="tests",
            )
        )


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.synthetic_training", "1", 1, families=(FamilyRequirement("training_programs"),)
    ),
    _register_cli_fixture,
)
