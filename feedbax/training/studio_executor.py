"""Studio graph training kernels for the contract executor."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.random as jr
import optax

from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TrainingMethodRegistration,
    TrainingMethodRegistry,
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


def studio_training_registry(
    *,
    compiled: Any,
    optimizer: optax.GradientTransformationExtraArgs,
    total_batches: int,
    stop_event: Any,
    rollout_fn: Callable[[Any, Any, Any], dict[str, Any]],
    evaluate_loss_fn: Callable[[Any, dict[str, Any]], tuple[jax.Array, dict[str, jax.Array]]],
) -> TrainingMethodRegistry:
    """Build an executor registry whose kernel performs one Studio graph update."""
    method_contract = executor_method_contract(total_batches=total_batches)

    def gradient_update(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context, coordinate
        current_graph = slots["model"]
        optimizer_state = slots["optimizer"]
        rng_key, step_key = jr.split(slots["prng"])
        step_t0 = time.perf_counter()
        trainable, static = eqx.partition(current_graph, compiled.trainable_filter)

        def loss_from_trainable(trainable_graph):
            candidate = eqx.combine(static, trainable_graph)
            rollout = rollout_fn(candidate, compiled, step_key)
            return evaluate_loss_fn(compiled, rollout)

        (loss_value, loss_terms), grads = eqx.filter_value_and_grad(
            loss_from_trainable,
            has_aux=True,
        )(trainable)
        grad_norm = optax.global_norm(grads)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        updated_graph = eqx.combine(static, trainable)
        updated_graph = apply_parameter_constraints(updated_graph)
        compiled.graph = updated_graph
        loss = float(jax.block_until_ready(loss_value))
        return {
            "model": updated_graph,
            "optimizer": optimizer_state,
            "prng": rng_key,
            "train_loss": loss,
            "loss_terms": {
                key: float(jax.block_until_ready(value)) for key, value in loss_terms.items()
            },
            "grad_norm": float(jax.block_until_ready(grad_norm)),
            "step_time_ms": (time.perf_counter() - step_t0) * 1000.0,
        }

    def continue_training(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del slots, context
        return coordinate.global_step < total_batches and not stop_event.is_set()

    registry = TrainingMethodRegistry()
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
