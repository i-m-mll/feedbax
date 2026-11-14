"""Deliver retained Studio graph training through the HTTP worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import tempfile
import time
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from feedbax.runtime._graph_key_schedule import KEY_SCHEDULE_ID
from feedbax.runtime.graph import Graph
from feedbax.runtime.retained_observables import retention_plan_to_json
from feedbax.contracts.training import (
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.training.retained_executor import execute_training_run_spec
from feedbax.training.retained_graph import (
    CompiledTrainingRun,
    StudioGraphLossService,
    _jsonable_value,
    evaluate_graph_loss,
    rollout_graph,
    training_trajectory_snapshot,
)
from feedbax.training.retained_studio import studio_training_registry
from feedbax.training.studio_executor import executor_method_contract
from feedbax.web.worker.checkpoint import cleanup_worker_checkpoint


def _build_optimizer(
    learning_rate: float,
    grad_clip: float | None,
) -> optax.GradientTransformationExtraArgs:
    """Return the worker optimizer, optionally including global-norm clipping."""
    if grad_clip is None:
        return optax.chain(optax.adam(learning_rate))
    return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(learning_rate))


@dataclass
class TrainingGraphResult:
    """Terminal metadata produced by generic graph training."""

    graph: Graph
    checkpoint_path: str | None
    final_loss: float
    final_loss_terms: dict[str, float]
    final_batch: int
    final_program_step: int
    stopped: bool
    manifest_path: str | None
    manifest_payload: dict[str, Any] | None
    execution_metadata: dict[str, Any]
    retention_plan: dict[str, Any]
    retained_observables: dict[str, Any]


def run_training_graph(
    compiled: CompiledTrainingRun,
    *,
    job_id: str,
    total_batches: int,
    cfg: Any,
    stop_event: Any,
    emit: Callable[[dict[str, Any]], None],
) -> TrainingGraphResult:
    """Train an executable graph via the contract executor and stream worker events."""
    learning_rate = float(getattr(cfg, "learning_rate", 1e-3))
    raw_grad_clip = getattr(cfg, "grad_clip", 1.0)
    grad_clip = None if raw_grad_clip is None else float(raw_grad_clip)
    optimizer = _build_optimizer(learning_rate, grad_clip)
    trainable, _static = eqx.partition(compiled.graph, compiled.trainable_filter)
    opt_state = optimizer.init(trainable)
    started_at = time.perf_counter()
    run_spec = _training_run_spec_for_compiled(
        compiled,
        job_id=job_id,
        total_batches=total_batches,
        cfg=cfg,
        grad_clip=grad_clip,
    )
    snapshot_interval = run_spec.training_config.snapshot_interval
    current_graph = compiled.graph

    def observe_model(graph: Graph) -> None:
        nonlocal current_graph
        current_graph = graph

    registry = studio_training_registry(
        compiled=compiled,
        optimizer=optimizer,
        total_batches=total_batches,
        rollout_fn=lambda graph, plan, key: rollout_graph(graph, plan, key=key),
        evaluate_loss_fn=evaluate_graph_loss,
        observe_model=observe_model,
    )

    def _emit_executor_progress(event: Mapping[str, Any]) -> None:
        progress_event = _studio_progress_event(
            event,
            job_id=job_id,
            total_batches=total_batches,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        batch = progress_event["batch"]
        program_step = progress_event["program_step"]
        emit(progress_event)
        emit(
            {
                "type": "training_log",
                "job_id": job_id,
                "batch": batch,
                "program_step": program_step,
                "level": "info",
                "message": (
                    f"Program step {program_step} | "
                    f"loss={progress_event['loss']:.4f} | "
                    f"grad_norm={progress_event['grad_norm']:.3f} | "
                    f"{progress_event['step_time_ms']:.0f}ms"
                ),
                "execution": "contract_executor",
            }
        )
        if batch % snapshot_interval == 0 or batch >= total_batches:
            emit(
                {
                    "type": "training_trajectory",
                    "job_id": job_id,
                    "batch": batch,
                    "program_step": program_step,
                    "trajectory": training_trajectory_snapshot(
                        current_graph,
                        compiled,
                        key=jr.PRNGKey(program_step),
                    ),
                    "execution": "contract_executor",
                }
            )

    result = execute_training_run_spec(
        run_spec,
        run_id=job_id,
        initial_slots={
            "model": compiled.graph,
            "optimizer": opt_state,
            "prng": jr.PRNGKey(0),
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        },
        registry=registry,
        loss_service=StudioGraphLossService(compiled),
        training_spec_payload=compiled.training_spec.model_dump(mode="json"),
        training_spec_payload_kind="TrainingSpec",
        training_spec_payload_schema_id="feedbax.spec.training",
        training_spec_payload_schema_version="feedbax.spec.training.v1",
        task_binding_spec=compiled.task_binding_spec.model_dump(mode="json", exclude_none=True),
        manifest_conflict_policy="reuse-identical",
        progress_callback=_emit_executor_progress,
        retain_history=False,
        stop_before_update=stop_event.is_set,
    )

    graph = result.final_slots["model"]
    compiled.graph = graph
    final_terms = _float_mapping(result.final_slots.get("loss_terms", {}))
    final_loss = _float_metric(
        result.final_slots.get(
            "train_loss",
            result.final_coordinate.metrics.get("train_loss", 0.0),
        )
    )
    checkpoint_path: str | None = None
    try:
        checkpoint_path = _write_checkpoint(graph)
        final_rollout = rollout_graph(graph, compiled, key=result.final_slots["prng"])
        return TrainingGraphResult(
            graph=graph,
            checkpoint_path=checkpoint_path,
            final_loss=final_loss,
            final_loss_terms=final_terms,
            final_batch=result.end_completed_batches,
            final_program_step=result.final_coordinate.program_step,
            stopped=result.status == "cancelled",
            manifest_path=str(result.manifest_path),
            manifest_payload=result.manifest.model_dump(mode="json", exclude_none=True),
            execution_metadata=dict(compiled.metadata),
            retention_plan=retention_plan_to_json(compiled.retention_plan),
            retained_observables=_retained_observables_payload(final_rollout),
        )
    except BaseException as exc:
        if checkpoint_path is not None:
            try:
                cleanup_worker_checkpoint(
                    checkpoint_path,
                    context="worker result finalization failed",
                )
            except BaseException as cleanup_exc:
                raise cleanup_exc from exc
        raise


def _training_run_spec_for_compiled(
    compiled: CompiledTrainingRun,
    *,
    job_id: str,
    total_batches: int,
    cfg: Any,
    grad_clip: float | None,
) -> TrainingRunSpec:
    method_contract = executor_method_contract(total_batches=total_batches)
    snapshot_interval = max(1, int(getattr(cfg, "snapshot_interval", 100)))
    effective_phase = standard_supervised_effective_phase_spec().model_copy(
        update={
            "phase_program": method_contract.phase_program,
            "state_slots": method_contract.state_slots,
        }
    )
    return TrainingRunSpec(
        graph={
            "kind": "GraphSpec",
            "inline": compiled.graph_spec.model_dump(mode="json", exclude_none=True),
        },
        task=TaskSpec.model_validate(compiled.metadata.get("task_spec", {}))
        if isinstance(compiled.metadata.get("task_spec"), Mapping)
        else TaskSpec(type="StudioTask", params={}),
        training_config=TrainingConfig(
            n_batches=total_batches,
            batch_size=compiled.training_spec.batch_size,
            learning_rate=float(getattr(cfg, "learning_rate", 1e-3)),
            grad_clip=grad_clip,
            n_reach_steps=compiled.n_steps,
            snapshot_interval=snapshot_interval,
        ),
        objective={"loss": compiled.training_spec.loss.model_dump(mode="json", exclude_none=True)},
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=method_contract,
            effective_phase=effective_phase,
        ),
        artifacts={},
        checkpoint_progress={
            "checkpoint_interval": snapshot_interval,
            "progress_interval": 1,
        },
        metadata={
            "job_id": job_id,
            "execution": "studio_contract_executor",
            "key_schedule_id": KEY_SCHEDULE_ID,
        },
    )


def _studio_progress_event(
    event: Mapping[str, Any],
    *,
    job_id: str,
    total_batches: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    coordinate = event["coordinate"]
    metrics = event.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    loss = _float_metric(metrics.get("train_loss", 0.0))
    loss_terms = _float_mapping(metrics.get("loss_terms", {}))
    scalar_metrics = {
        key: value for key, value in _float_mapping(metrics).items() if key not in {"loss_terms"}
    }
    scalar_metrics["elapsed_seconds"] = float(elapsed_seconds)
    return {
        "type": "training_progress",
        "job_id": job_id,
        "batch": coordinate["completed_batches"],
        "program_step": coordinate["program_step"],
        "total_batches": total_batches,
        "loss": loss,
        "loss_terms": loss_terms,
        "grad_norm": _float_metric(metrics.get("grad_norm", 0.0)),
        "step_time_ms": _float_metric(metrics.get("step_time_ms", 0.0)),
        "metrics": scalar_metrics,
        "status": "running",
        "execution": "contract_executor",
    }


def _float_metric(value: Any) -> float:
    try:
        return float(jax.block_until_ready(value))
    except (TypeError, ValueError):
        return 0.0


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(jax.block_until_ready(item))
        except (TypeError, ValueError):
            continue
    return result


def _retained_observables_payload(rollout: dict[str, Any]) -> dict[str, Any]:
    return {
        "observables": {key: _jsonable_value(value) for key, value in rollout["trace"].items()},
        "outputs": {
            f"graph_output:{key}": _jsonable_value(value)
            for key, value in rollout["outputs"].items()
        },
        "task_data": {
            f"task_data:{key}": _jsonable_value(value)
            for key, value in rollout["task_data"].items()
        },
    }


def _write_checkpoint(graph: Graph) -> str:
    ckpt_dir = tempfile.mkdtemp(prefix="feedbax_ckpt_")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.eqx")
    try:
        ready_graph = jax.block_until_ready(graph)
        eqx.tree_serialise_leaves(ckpt_path, ready_graph)
    except BaseException as exc:
        try:
            cleanup_worker_checkpoint(
                ckpt_path,
                context="worker checkpoint serialization failed",
            )
        except BaseException as cleanup_exc:
            raise cleanup_exc from exc
        raise
    return ckpt_path
