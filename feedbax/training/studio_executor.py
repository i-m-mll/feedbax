"""Studio graph training kernels for the contract executor."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TrainingMethodRegistration,
    TrainingProgramRegistry,
    standard_supervised_method_contract,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import MetricGuardSpec, PhaseTransitionSpec, StateSlotSpec
from feedbax.runtime.parameter_constraints import apply_parameter_constraints


def executor_method_contract(*, total_batches: int):
    """Return the worker contract used by Studio graph training."""
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(update={"legal_next": ["train_batch"], "max_steps": 1})
    transition = PhaseTransitionSpec(
        source="train_batch",
        target="train_batch",
        barrier="after_train_batch",
        guard=MetricGuardSpec(
            predicate_ref="feedbax.training.studio_executor.continue_training",
            metric_slots=["train_loss"],
            metadata={"total_batches": total_batches},
        ),
    )
    update_step = program.update_steps[0].model_copy(
        update={
            "writes": [
                *program.update_steps[0].writes,
                "loss_terms",
                "grad_norm",
                "step_time_ms",
            ]
        }
    )
    state_slots = [
        *contract.state_slots,
        StateSlotSpec(name="loss_terms", role="metric", required=False),
        StateSlotSpec(name="grad_norm", role="metric", required=False),
        StateSlotSpec(name="step_time_ms", role="metric", required=False),
    ]
    program = program.model_copy(
        update={
            "phases": [phase],
            "transitions": [transition],
            "update_steps": [update_step],
        }
    )
    return contract.model_copy(update={"phase_program": program, "state_slots": state_slots})


class StudioUpdateKernel:
    """AOT numerical update with fixed graph structure and device-resident operands."""

    def __init__(
        self,
        *,
        compiled: Any,
        optimizer: optax.GradientTransformationExtraArgs,
        rollout_fn: Callable[[Any, Any, Any], dict[str, Any]],
        evaluate_loss_fn: Callable[[Any, dict[str, Any]], tuple[jax.Array, dict[str, jax.Array]]],
    ) -> None:
        trainable, frozen = eqx.partition(compiled.graph, compiled.trainable_filter)
        frozen_arrays, frozen_structure = eqx.partition(frozen, eqx.is_array)
        self._frozen_arrays = jax.device_put(frozen_arrays)
        self._frozen_structure = frozen_structure
        trainable_filter = compiled.trainable_filter
        self._filter = trainable_filter

        def numerical_update(trainable, optimizer_state, key, batch_counter, frozen_arrays):
            frozen = eqx.combine(frozen_arrays, frozen_structure)
            key, step_key = jr.split(key)

            def loss_from_trainable(candidate):
                graph = eqx.combine(candidate, frozen)
                return evaluate_loss_fn(compiled, rollout_fn(graph, compiled, step_key))

            (loss, terms), grads = eqx.filter_value_and_grad(loss_from_trainable, has_aux=True)(
                trainable
            )
            updates, optimizer_state = optimizer.update(grads, optimizer_state, trainable)
            graph = apply_parameter_constraints(
                eqx.combine(eqx.apply_updates(trainable, updates), frozen)
            )
            trainable, _ = eqx.partition(graph, trainable_filter)
            return (
                trainable,
                optimizer_state,
                key,
                batch_counter + 1,
                {"train_loss": loss, "loss_terms": terms, "grad_norm": optax.global_norm(grads)},
            )

        # Equinox's filtered lowering separates static graph structure from array operands.
        compile_t0 = time.perf_counter()
        self._update = (
            eqx.filter_jit(numerical_update)
            .lower(
                trainable,
                optimizer.init(trainable),
                jr.PRNGKey(0),
                jnp.array(0, dtype=jnp.int32),
                self._frozen_arrays,
            )
            .compile()
        )
        self.compile_time_ms = (time.perf_counter() - compile_t0) * 1000.0

    def __call__(self, slots: Mapping[str, Any]) -> Mapping[str, Any]:
        trainable, _ = eqx.partition(slots["model"], self._filter)
        trainable, optimizer_state, key, count, metrics = self._update(
            trainable,
            slots["optimizer"],
            slots["prng"],
            slots["batch_counter"],
            self._frozen_arrays,
        )
        graph = eqx.combine(trainable, self._frozen_arrays, self._frozen_structure)
        return {
            "model": graph,
            "optimizer": optimizer_state,
            "prng": key,
            "batch_counter": count,
            **metrics,
        }


def studio_training_registry(
    *,
    compiled: Any,
    optimizer: optax.GradientTransformationExtraArgs,
    total_batches: int,
    stop_event: Any,
    rollout_fn: Callable[[Any, Any, Any], dict[str, Any]],
    evaluate_loss_fn: Callable[[Any, dict[str, Any]], tuple[jax.Array, dict[str, jax.Array]]],
    observe_model: Callable[[Any], None] | None = None,
) -> TrainingProgramRegistry:
    """Build an executor registry whose kernel performs one Studio graph update."""
    method_contract = executor_method_contract(total_batches=total_batches)
    kernel = StudioUpdateKernel(
        compiled=compiled,
        optimizer=optimizer,
        rollout_fn=rollout_fn,
        evaluate_loss_fn=evaluate_loss_fn,
    )
    compiled.metadata["numerical_compile_time_ms"] = kernel.compile_time_ms

    def gradient_update(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context, coordinate
        step_t0 = time.perf_counter()
        updated = dict(kernel(slots))
        jax.block_until_ready(updated)
        loss, terms, grad_norm = jax.device_get(
            (updated["train_loss"], updated["loss_terms"], updated["grad_norm"])
        )
        updated.update(
            train_loss=float(loss),
            loss_terms={name: float(value) for name, value in terms.items()},
            grad_norm=float(grad_norm),
        )
        updated["step_time_ms"] = (time.perf_counter() - step_t0) * 1000.0
        if observe_model is not None:
            observe_model(updated["model"])
        return updated

    def continue_training(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del slots, context
        return coordinate.program_step < total_batches and not stop_event.is_set()

    registry = TrainingProgramRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref=standard_supervised_method_ref().key,
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=lambda: method_contract,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": gradient_update
            },
            guard_predicates_factory=lambda _payload: {
                "feedbax.training.studio_executor.continue_training": continue_training
            },
            owner="feedbax.training.studio_executor",
            package="feedbax",
        )
    )
    return registry
