"""Prepare a document task and objective for managed training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from feedbax.compiler.trainability import compile_trainability
from feedbax.compiler.trial_execution import ResolvedTrialExecution, resolve_trial_execution
from feedbax.contracts.artifact_custody import ArtifactBlobProvider
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import TrainingRunSpec
from feedbax.objectives.document import ObjectiveExecutor
from feedbax.runtime.graph import Graph
from feedbax.runtime.trial_execution import TrialExecution
from feedbax.tasks.document import TaskParameters
from feedbax.training.preparation import ExecutionPreparationResult
from feedbax.training.studio_executor import (
    StudioUpdateKernel,
    standard_train_optimizer,
    studio_schedule_context_binder,
    studio_training_key,
)


@dataclass(frozen=True)
class PairedUpdateInputs:
    graph: Graph
    trainable_filter: Any


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
        compiled=PairedUpdateInputs(graph, trainable_filter),
        batch_size=batch_size,
        snapshot=snapshot,
        optimizer=optimizer,
        score_batch=score,
    )


def prepare_managed_training(
    spec: TrainingRunSpec, registries, *, model_assets: ArtifactBlobProvider | None = None,
) -> ExecutionPreparationResult:
    """Prepare the document pair through the existing bounded update kernel."""
    documents = spec.trial_documents
    trial = resolve_trial_execution(
        documents.model_name, documents.model, documents.task_source,
        registries.components, registries.expression_functions,
        python_tasks=registries.python_tasks, objective=documents.objective, input_overrides=documents.input_overrides,
    )
    if isinstance(trial, tuple):
        problem = trial[0]
        raise ValueError("/trial_documents/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
    return prepare_resolved_training(spec, trial, model_assets=model_assets)


def prepare_resolved_training(
    spec: TrainingRunSpec,
    trial: ResolvedTrialExecution,
    *,
    model_assets: ArtifactBlobProvider | None = None,
) -> ExecutionPreparationResult:
    """Realize the validated trial with the training method's initialization key."""
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    paired = trial.realize(
        initialization_key=studio_training_key(method.seed, "model"),
        artifact_provider=model_assets,
    )
    if isinstance(paired, tuple):
        problem = paired[0]
        raise ValueError("/trial_documents/" + "/".join(map(str, problem.path)) + ": " + problem.reason)
    assert paired.objective is not None
    graph = paired.executable.graph
    _, mask = compile_trainability(paired.executable.resolved.graph, graph, trial.graph.component_registry)
    optimizer = standard_train_optimizer(method, gradient_clip=spec.training_config.grad_clip)
    kernel = prepare_trial_update(
        graph, paired.execution, paired.objective, parameters=TaskParameters(),
        run_key=studio_training_key(method.seed, "task-run"),
        batch_size=spec.training_config.batch_size, trainable_filter=mask, optimizer=optimizer,
    )

    return ExecutionPreparationResult(
        initial_slots={
            "model": graph, "optimizer": optimizer.init(eqx.filter(graph, mask)),
            "prng": studio_training_key(method.seed, "update"),
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        },
        kernel_context={"studio_graph_update": kernel},
        restored_schedule_context_binder=studio_schedule_context_binder(method),
    )
