"""Explicit synthetic training registrations for lifecycle and contract tests."""

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
from feedbax.contracts.worker import ProgressCoordinate
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


def initial_training_slots() -> dict[str, Any]:
    return {
        "model": jnp.array([0.0]),
        "optimizer": {"count": jnp.array([1.0])},
        "prng": jnp.array([0, 1], dtype=jnp.uint32),
        "batch_counter": jnp.array(0, dtype=jnp.int32),
    }


def _register_cli_fixture(context: RegistrationContext) -> None:
    context.registry(TRAINING_PROGRAMS).register_program(
        evolve_training_program(
            synthetic_supervised_training_program(),
            method_ref=CLI_METHOD_REF,
            contract_compiler=lambda _payload: standard_supervised_method_contract().model_copy(
                update={"method_ref": CLI_METHOD_REF}
            ),
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
