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

from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    STUDIO_GRAPH_METHOD_PAYLOAD_ID,
    STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
    StudioGraphMethodPayload,
)
from feedbax.contracts.training import (
    DeclaredTrainingProgram,
    ScheduleProjection,
    TrainingMethodScheduleProjector,
    declare_training_program,
    standard_supervised_method_contract,
)
from feedbax.contracts.worker import MetricGuardSpec, PhaseTransitionSpec, StateSlotSpec
from feedbax.runtime._graph_key_schedule import derive_key, sample_training_trials
from feedbax.runtime.parameter_constraints import apply_parameter_constraints
from feedbax.training.optimizers import build_optimizer
from feedbax.training.preparation import ExecutionPreparationProvider, RestoredScheduleContextBinder


def studio_training_key(seed: int, stream: str) -> jax.Array:
    """Derive an independent named stream from the authored seed."""
    return derive_key(jr.PRNGKey(seed), "studio_training", stream)


def standard_train_optimizer(
    payload: StudioGraphMethodPayload, *, gradient_clip: float | None,
) -> optax.GradientTransformationExtraArgs:
    """Build the optimizer with its authored schedule clock, independent of its count."""
    policy = payload.continuation
    origin = (
        policy.progress.source_completed_batches
        if policy is not None and policy.schedule == "restart" else 0
    )
    count = origin if policy is not None and policy.optimizer == "preserve" else 0
    return build_optimizer(
        payload.optimizer, schedule_origin_step=origin, current_step=origin,
        optimizer_count_at_current_step=count,
        gradient_clip=gradient_clip,
    )


def standard_train_optimizer_step(payload: StudioGraphMethodPayload, slots: Mapping[str, Any]) -> int:
    optimizer_state = slots["optimizer"]
    if payload.optimizer.lr_schedule is not None:
        return int(optimizer_state[-1].count)
    adam = jax.tree.leaves(optimizer_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
    counts = [int(state.count) for state in adam if isinstance(state, optax.ScaleByAdamState)]
    return counts[0] if counts else int(slots["batch_counter"])


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
        batch_size: int | None = None,
        optimizer: optax.GradientTransformationExtraArgs,
        score_batch: Callable[[Any, jax.Array, int], tuple[jax.Array, dict[str, jax.Array]]] | None = None,
        rollout_fn: Callable[[Any, Any, Any], dict[str, Any]] | None = None,
        evaluate_loss_fn: Callable[[Any, dict[str, Any]], tuple[jax.Array, dict[str, jax.Array]]] | None = None,
        snapshot: Callable[[Any, jax.Array], dict[str, Any]] | None = None,
    ) -> None:
        if score_batch is None:
            if rollout_fn is None or evaluate_loss_fn is None:
                raise TypeError("Supply a batch score or the scenario worker's rollout and loss.")
            score_batch = _score_legacy_batch(compiled, rollout_fn, evaluate_loss_fn)
        elif rollout_fn is not None or evaluate_loss_fn is not None:
            raise TypeError("Supply exactly one scoring interface.")
        self._snapshot = snapshot
        trainable_filter = compiled.trainable_filter
        if batch_size is None:
            batch_size = compiled.training_spec.batch_size

        @eqx.filter_jit
        def numerical_update(model, optimizer_state, key, batch_counter):
            trainable, frozen = eqx.partition(model, trainable_filter)
            key, step_key = jr.split(key)

            def loss_from_trainable(candidate):
                return score_batch(eqx.combine(candidate, frozen), step_key, batch_size)

            (loss, terms), grads = eqx.filter_value_and_grad(loss_from_trainable, has_aux=True)(
                trainable
            )
            updates, optimizer_state = optimizer.update(grads, optimizer_state, trainable)
            graph = apply_parameter_constraints(
                eqx.combine(eqx.apply_updates(trainable, updates), frozen)
            )
            return (
                graph,
                optimizer_state,
                key,
                batch_counter + 1,
                {"train_loss": loss, "loss_terms": terms, "grad_norm": optax.global_norm(grads)},
            )

        self._numerical_update = numerical_update
        self._static_model = eqx.filter(compiled.graph, eqx.is_array, inverse=True)
        trainable = eqx.filter(compiled.graph, trainable_filter)
        compile_t0 = time.perf_counter()
        self._update = numerical_update.lower(
            compiled.graph,
            optimizer.init(trainable),
            jr.PRNGKey(0),
            jnp.array(0, dtype=jnp.int32),
        ).compile()
        self.compile_time_ms = (time.perf_counter() - compile_t0) * 1000.0

    def __call__(self, slots: Mapping[str, Any]) -> Mapping[str, Any]:
        model = slots["model"]
        operands = model, slots["optimizer"], slots["prng"], slots["batch_counter"]
        static_model = eqx.filter(model, eqx.is_array, inverse=True)
        # The restored model can carry non-array leaves that Equinox specializes on.
        if not eqx.tree_equal(static_model, self._static_model):
            self._update = self._numerical_update.lower(*operands).compile()
            self._static_model = static_model
        graph, optimizer_state, key, count, metrics = self._update(*operands)
        return {
            "model": graph,
            "optimizer": optimizer_state,
            "prng": key,
            "batch_counter": count,
            **metrics,
        }

    def reported_update(self, slots: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        updated = dict(self(slots))
        jax.block_until_ready(updated)
        loss, terms, grad_norm = jax.device_get(
            (updated["train_loss"], updated["loss_terms"], updated["grad_norm"])
        )
        updated.update(
            train_loss=float(loss),
            loss_terms={name: float(value) for name, value in terms.items()},
            grad_norm=float(grad_norm),
            step_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return updated

    def trajectory_snapshot(self, graph: Any, key: jax.Array) -> dict[str, Any]:
        return self._snapshot(graph, key) if self._snapshot is not None else {}


def _score_legacy_batch(compiled, rollout_fn, evaluate_loss_fn):
    """Preserve the protected scenario worker's established batch reduction."""
    def score(graph, key, count):
        def one(trial_key):
            return evaluate_loss_fn(compiled, rollout_fn(graph, compiled, trial_key))

        if count == 1:
            return one(key)
        trials = sample_training_trials(one, key, count)
        return jax.tree.map(lambda values: jnp.mean(values, axis=0), trials)

    return score


def studio_graph_method_contract(payload: StudioGraphMethodPayload):
    return executor_method_contract(total_batches=payload.target_total).model_copy(update={
        "method_ref": STUDIO_GRAPH_METHOD.key,
        "method_payload_schema_version": STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
    })


def studio_graph_training_program(
    *, preparation_provider: ExecutionPreparationProvider | None = None,
) -> DeclaredTrainingProgram[StudioGraphMethodPayload]:
    """Declare the checkpointable Studio update shared by both entry points."""
    def gradient_update(slots: Mapping[str, Any], coordinate: Any, context: Mapping[str, Any]):
        del coordinate
        return context["studio_graph_update"].reported_update(slots)

    def guard(payload: StudioGraphMethodPayload):
        def continue_training(slots: Mapping[str, Any], coordinate: Any, context: Mapping[str, Any]):
            del coordinate, context
            return int(slots["batch_counter"]) < payload.target_total

        return {"feedbax.training.studio_executor.continue_training": continue_training}

    return declare_training_program(
        method_ref=STUDIO_GRAPH_METHOD.key,
        payload_schema_id=STUDIO_GRAPH_METHOD_PAYLOAD_ID,
        payload_schema_version=STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
        payload_model=StudioGraphMethodPayload,
        contract_compiler=studio_graph_method_contract,
        update_kernels_factory=lambda _payload: {
            "feedbax.training.standard_supervised.gradient_update": gradient_update,
        },
        guard_predicates_factory=guard,
        preparation_provider=preparation_provider,
        schedule_projector=TrainingMethodScheduleProjector(
            projector_id="feedbax.training.studio_graph.schedule_projection",
            projector_version="v1",
            projector=lambda _payload, _coordinates: ScheduleProjection(),
        ),
        optimizer_spec_projector=lambda payload: payload.optimizer,
        optimizer_step_extractor=standard_train_optimizer_step,
        owner="feedbax.training.studio_executor",
        package="feedbax",
    )


def studio_schedule_context_binder(
    payload: StudioGraphMethodPayload,
) -> RestoredScheduleContextBinder | None:
    policy = payload.continuation
    if policy is None:
        return None
    authored = policy.progress.source_completed_batches if policy.schedule == "restart" else 0

    def bind_schedule(_slots, origin, current, count):
        if origin != authored or current != count:
            raise ValueError("/slots/optimizer: restored schedule clock differs from the prepared clock")
        return {}

    return bind_schedule
