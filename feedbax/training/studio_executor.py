"""Studio graph training kernels for the contract executor."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from feedbax.compiler.trainability import compile_trainability
from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    STUDIO_GRAPH_METHOD_PAYLOAD_ID,
    STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
    StudioGraphMethodPayload,
)
from feedbax.contracts.training import (
    DeclaredTrainingProgram,
    ScheduleProjection,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TrainingMethodRegistration,
    TrainingMethodScheduleProjector,
    TrainingProgramRegistry,
    TrainingSpec,
    declare_training_program,
    standard_supervised_method_contract,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import MetricGuardSpec, PhaseTransitionSpec, StateSlotSpec
from feedbax.objectives.document import ObjectiveExecutor
from feedbax.runtime.graph import Graph
from feedbax.runtime.trial_execution import TrialExecution
from feedbax.tasks.document import TaskParameters
from feedbax.runtime._graph_key_schedule import derive_key, sample_training_trials
from feedbax.runtime.parameter_constraints import apply_parameter_constraints
from feedbax.training.optimizers import build_optimizer
from feedbax.training.preparation import ExecutionPreparationRequest, ExecutionPreparationResult


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
        batch_size = compiled.batch_size if isinstance(compiled, PairedUpdateInputs) else compiled.training_spec.batch_size

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


def _legacy_snapshot(compiled):
    from feedbax.web.worker.execution import training_trajectory_snapshot

    return lambda graph, key: training_trajectory_snapshot(graph, compiled, key)


def studio_training_registry(
    *,
    compiled: Any,
    optimizer: optax.GradientTransformationExtraArgs,
    total_batches: int,
    rollout_fn: Callable[[Any, Any, Any], dict[str, Any]],
    evaluate_loss_fn: Callable[[Any, dict[str, Any]], tuple[jax.Array, dict[str, jax.Array]]],
    observe_model: Callable[[Any], None] | None = None,
) -> TrainingProgramRegistry:
    """Build an executor registry whose kernel performs one Studio graph update."""
    method_contract = executor_method_contract(total_batches=total_batches)
    kernel = StudioUpdateKernel(
        compiled=compiled,
        optimizer=optimizer,
        score_batch=_score_legacy_batch(compiled, rollout_fn, evaluate_loss_fn),
        snapshot=_legacy_snapshot(compiled),
    )
    compiled.metadata["numerical_compile_time_ms"] = kernel.compile_time_ms

    def gradient_update(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context, coordinate
        updated = kernel.reported_update(slots)
        if observe_model is not None:
            observe_model(updated["model"])
        return updated

    def continue_training(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del coordinate, context
        return int(slots["batch_counter"]) < total_batches

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


def studio_graph_method_contract(payload: StudioGraphMethodPayload):
    return executor_method_contract(total_batches=payload.target_total).model_copy(update={
        "method_ref": STUDIO_GRAPH_METHOD.key,
        "method_payload_schema_version": STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
    })


def studio_graph_training_program(component_registry: Any) -> DeclaredTrainingProgram[StudioGraphMethodPayload]:
    """Declare the existing Studio graph update for the checkpointable native executor."""
    def prepare(request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
        from feedbax.web.worker.execution import (
            _StudioWorkerLossService,
            _evaluate_loss,
            compile_training_run,
            rollout_graph,
        )

        spec, payload = request.run_spec, request.method_payload
        if not isinstance(payload, StudioGraphMethodPayload):
            raise ValueError("Studio graph preparation requires its resolved method payload")
        payload.require_supported_launch(resume=request.resume)
        if spec.graph.inline is None or spec.task_binding_spec is None or spec.objective.loss is None:
            raise ValueError("Studio graph training requires an inline graph, typed task bindings and a loss")
        continuation = spec.checkpoint_progress.continuation
        offset = continuation.source_completed_batches if continuation is not None else 0
        if offset + spec.training_config.n_batches != payload.target_total:
            raise ValueError("Studio graph segment does not reach its original batch target")
        compiled = compile_training_run(
            graph_spec=spec.graph.inline,
            training_spec=TrainingSpec(
                optimizer=payload.optimizer, loss=spec.objective.loss,
                n_batches=spec.training_config.n_batches, batch_size=spec.training_config.batch_size,
            ),
            task_spec=spec.task.model_dump(mode="json", exclude_none=True),
            task_binding_spec=spec.task_binding_spec.model_dump(mode="json", exclude_none=True),
            cfg=spec.training_config,
            component_registry=component_registry,
            executable_graph=request.executable_graph,
        )
        optimizer = standard_train_optimizer(payload, gradient_clip=spec.training_config.grad_clip)
        kernel = StudioUpdateKernel(
            compiled=compiled, optimizer=optimizer,
            score_batch=_score_legacy_batch(
                compiled, lambda graph, plan, key: rollout_graph(graph, plan, key=key), _evaluate_loss,
            ),
            snapshot=_legacy_snapshot(compiled),
        )
        trainable, _ = eqx.partition(compiled.graph, compiled.trainable_filter)

        def bind_schedule(_slots, origin, current, count):
            policy = payload.continuation
            assert policy is not None
            authored = policy.progress.source_completed_batches if policy.schedule == "restart" else 0
            if origin != authored or current != count:
                raise ValueError("/slots/optimizer: restored schedule clock differs from the prepared clock")
            return {}

        return ExecutionPreparationResult(
            initial_slots={
                "model": compiled.graph,
                "optimizer": optimizer.init(trainable),
                "prng": studio_training_key(payload.seed, "update"),
                "batch_counter": jnp.array(0, dtype=jnp.int32),
            },
            kernel_context={"studio_graph_update": kernel},
            loss_service=_StudioWorkerLossService(compiled),
            restored_schedule_context_binder=bind_schedule if payload.continuation is not None else None,
        )

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
        preparation_provider=prepare,
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


@dataclass(frozen=True)
class PairedUpdateInputs:
    graph: Graph
    trainable_filter: Any
    batch_size: int


def prepare_trial_update(
    graph: Graph,
    task_execution: TrialExecution,
    objective: ObjectiveExecutor,
    *,
    parameters: TaskParameters,
    run_key: jax.Array,
    batch_size: int,
    trainable_filter: Any,
    optimizer: optax.GradientTransformationExtraArgs,
) -> StudioUpdateKernel:
    """Use the existing bounded optimizer step for a separate task and objective."""
    if batch_size < 1:
        raise ValueError("A training batch must contain at least one trial.")

    def score(model, step_key, count):
        result = task_execution.training_batch(
            model, objective, parameters, run_key=run_key, update_key=step_key, count=count,
        )
        return result.loss, {name: term.value for name, term in result.terms.items()}

    def snapshot(model, key):
        trial = task_execution.task.ahead(parameters, key, key)
        retained, _ = task_execution.rollout(model, trial, key)
        return {"n_steps": task_execution.task.n_steps,
                "outputs": {name: value.tolist() for name, value in retained.items()}}

    return StudioUpdateKernel(
        compiled=PairedUpdateInputs(graph, trainable_filter, batch_size),
        snapshot=snapshot,
        optimizer=optimizer,
        score_batch=score,
    )


def prepare_managed_training(
    spec: TrainingRunSpec, registries, *, model_assets: ArtifactBlobProvider | None = None,
) -> ExecutionPreparationResult:
    """Prepare the document pair through the existing bounded update kernel."""
    documents = spec.trial_documents
    if documents is None:
        raise ValueError("/trial_documents: Managed Train requires a document pair.")
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    paired = compile_trial_execution(
        documents.model_name, documents.model, documents.task,
        registries.components, registries.expression_functions,
        objective=documents.objective, input_overrides=documents.input_overrides,
        initialization_key=studio_training_key(method.seed, "model"),
        artifact_provider=model_assets,
    )
    if not isinstance(paired, PreparedTrialExecution):
        problem = paired[0]
        raise ValueError("/trial_documents/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
    assert paired.objective is not None
    graph = paired.executable.graph
    _, mask = compile_trainability(paired.executable.resolved.graph, graph, registries.components)
    optimizer = standard_train_optimizer(method, gradient_clip=spec.training_config.grad_clip)
    kernel = prepare_trial_update(
        graph, paired.execution, paired.objective, parameters=TaskParameters(),
        run_key=studio_training_key(method.seed, "task-run"),
        batch_size=spec.training_config.batch_size, trainable_filter=mask, optimizer=optimizer,
    )

    def bind_schedule(_slots, origin, current, count):
        policy = method.continuation
        assert policy is not None
        authored = policy.progress.source_completed_batches if policy.schedule == "restart" else 0
        if origin != authored or current != count:
            raise ValueError("/slots/optimizer: restored schedule clock differs from the prepared clock")
        return {}

    return ExecutionPreparationResult(
        initial_slots={
            "model": graph, "optimizer": optimizer.init(eqx.filter(graph, mask)),
            "prng": studio_training_key(method.seed, "update"),
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        },
        kernel_context={"studio_graph_update": kernel},
        restored_schedule_context_binder=bind_schedule if method.continuation is not None else None,
    )
