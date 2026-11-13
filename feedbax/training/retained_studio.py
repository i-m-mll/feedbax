"""Studio scenario execution retained by the envelope path."""

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import (
    DeclaredTrainingProgram,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TrainingMethodRegistration,
    TrainingProgramRegistry,
    TrainingSpec,
    standard_supervised_method_ref,
)
from feedbax.training.preparation import ExecutionPreparationRequest, ExecutionPreparationResult
from feedbax.training.studio_executor import (
    StudioUpdateKernel,
    _score_legacy_batch,
    executor_method_contract,
    standard_train_optimizer,
    studio_graph_training_program as declare_studio_graph_training_program,
    studio_schedule_context_binder,
    studio_training_key,
)


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


def studio_graph_training_program(
    component_registry: Any,
) -> DeclaredTrainingProgram[StudioGraphMethodPayload]:
    """Declare the Studio graph method with retained scenario preparation."""
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

        return ExecutionPreparationResult(
            initial_slots={
                "model": compiled.graph,
                "optimizer": optimizer.init(trainable),
                "prng": studio_training_key(payload.seed, "update"),
                "batch_counter": jnp.array(0, dtype=jnp.int32),
            },
            kernel_context={"studio_graph_update": kernel},
            loss_service=_StudioWorkerLossService(compiled),
            restored_schedule_context_binder=studio_schedule_context_binder(payload),
        )

    return declare_studio_graph_training_program(preparation_provider=prepare)
