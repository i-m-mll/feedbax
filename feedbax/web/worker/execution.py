"""Generic graph execution substrate for Studio worker training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import os
import tempfile
import time
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax

from feedbax.runtime.graph import Graph, GraphTraceRequest, init_state_from_component
from feedbax.runtime.task_bindings import TaskInputPlan, expose_task_inputs
from feedbax.contracts.migrations import migrate_studio_task_binding_spec
from feedbax.runtime.retained_observables import (
    LossTermPlan,
    RetentionPlan,
    RetentionPlanError,
    SelectorRef,
    evaluate_loss_plan,
    lower_retention_plan,
    retention_plan_to_json,
)
from feedbax.studio.schema import validate_graph_connection_schema, validate_task_binding_schema
from feedbax.contracts.graph import (
    GraphSpec,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
)
from feedbax.contracts.studio_api import (
    TRAINING_TRAJECTORY_SCHEMA_ID,
    TRAINING_TRAJECTORY_SCHEMA_VERSION,
)
from feedbax.contracts.domain import DomainDiagnostic
from feedbax.contracts.training import (
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    TrainingSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.graphs.serialization import prototypes_from_task_bindings, spec_to_graph
from feedbax.objectives import ObjectiveExecutionRequirements
from feedbax.objectives.service import LossService, LoweredObjective
from feedbax.training.executor import execute_training_run_spec
from feedbax.training.studio_executor import (
    executor_method_contract,
    studio_training_registry,
)
from feedbax.web.worker.diagnostics import (
    GraphCompilationError,
    exception_diagnostic,
    graph_compilation_error,
)


def _build_optimizer(
    learning_rate: float,
    grad_clip: float | None,
) -> optax.GradientTransformationExtraArgs:
    """Return the worker optimizer, optionally including global-norm clipping."""
    if grad_clip is None:
        return optax.chain(optax.adam(learning_rate))
    return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(learning_rate))


@dataclass
class CompiledTrainingRun:
    """Preflighted generic worker execution plan."""

    graph: Graph
    graph_spec: GraphSpec
    training_spec: TrainingSpec
    task_binding_spec: StudioTaskBindingSpec
    task_inputs: tuple[TaskInputPlan, ...]
    retention_plan: RetentionPlan
    trace_requests: tuple[GraphTraceRequest, ...]
    loss_terms: tuple[LossTermPlan, ...]
    trainable_nodes: tuple[str, ...]
    trainable_filter: Any
    task_data: dict[str, jax.Array]
    n_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingGraphResult:
    """Terminal metadata produced by generic graph training."""

    graph: Graph
    checkpoint_path: str | None
    final_loss: float
    final_loss_terms: dict[str, float]
    final_batch: int
    manifest_path: str | None
    manifest_payload: dict[str, Any] | None
    execution_metadata: dict[str, Any]
    retention_plan: dict[str, Any]
    retained_observables: dict[str, Any]


def compile_training_run(
    *,
    graph_spec: dict[str, Any] | GraphSpec,
    training_spec: dict[str, Any] | TrainingSpec,
    task_spec: dict[str, Any],
    task_binding_spec: dict[str, Any] | StudioTaskBindingSpec,
    cfg: Any,
    component_registry: Any,
) -> CompiledTrainingRun:
    """Compile Studio specs into a generic executable graph training plan."""
    graph_model = (
        graph_spec if isinstance(graph_spec, GraphSpec) else GraphSpec.model_validate(graph_spec)
    )
    training_model = (
        training_spec
        if isinstance(training_spec, TrainingSpec)
        else TrainingSpec.model_validate(training_spec)
    )
    if training_model.batch_size != 1:
        raise GraphCompilationError(
            "Unsupported training batch size",
            [
                DomainDiagnostic(
                    severity="error",
                    code="worker.unsupported_batch_size",
                    message=(
                        "Generic graph worker currently supports batch_size=1; "
                        f"got batch_size={training_model.batch_size}"
                    ),
                    location={"path": "/training_spec/batch_size"},
                    details={"batch_size": training_model.batch_size},
                )
            ],
        )
    if isinstance(task_binding_spec, StudioTaskBindingSpec):
        binding_model = task_binding_spec
    else:
        binding_payload = migrate_studio_task_binding_spec(task_binding_spec).payload
        _reject_unsupported_task_data_value_spec_modes(binding_payload)
        binding_model = StudioTaskBindingSpec.model_validate(binding_payload)
    binding_errors = [
        issue
        for issue in validate_task_binding_schema(
            binding_model,
            graph_model,
            "/task_binding_spec",
            component_registry=component_registry,
        )
        if issue.severity == "error"
    ]
    if binding_errors:
        raise graph_compilation_error(
            "Invalid task_binding_spec for graph execution",
            binding_errors,
        )
    graph_errors = [
        issue
        for issue in validate_graph_connection_schema(
            graph_model,
            "/graph_spec",
            binding_model,
            component_registry=component_registry,
        )
        if issue.severity == "error"
    ]
    if graph_errors:
        raise graph_compilation_error(
            "Invalid graph_spec for graph execution",
            graph_errors,
        )

    try:
        graph = spec_to_graph(
            graph_model,
            component_registry,
            input_prototypes=prototypes_from_task_bindings(binding_model),
        )
    except NotImplementedError as exc:
        raise GraphCompilationError(
            "GraphSpec contains an unsupported executable component",
            [
                exception_diagnostic(
                    exc,
                    code="graph.unsupported_component",
                    message=f"GraphSpec contains unsupported executable component: {exc}",
                )
            ],
        ) from exc
    except Exception as exc:
        raise GraphCompilationError(
            "GraphSpec could not be instantiated for worker execution",
            [
                exception_diagnostic(
                    exc,
                    code="graph.instantiation_failed",
                    message=f"GraphSpec could not be instantiated for worker execution: {exc}",
                )
            ],
        ) from exc

    graph, task_inputs = expose_task_inputs(graph, binding_model)
    n_steps = int(getattr(cfg, "n_reach_steps", None) or training_model.n_batches or 1)
    task_data = _materialize_task_data(binding_model, task_spec, n_steps)
    try:
        retention_plan = lower_retention_plan(graph_model, training_model, task_spec=task_spec)
    except RetentionPlanError as exc:
        raise GraphCompilationError(
            "Invalid retention plan for graph execution",
            [
                exception_diagnostic(
                    exc,
                    code="graph.retention_plan_invalid",
                    message=f"Invalid retention plan for graph execution at {exc.path}: {exc}",
                    details={"path": exc.path, "selector": exc.selector},
                )
            ],
        ) from exc
    loss_terms = retention_plan.loss_terms
    trace_requests = _compile_trace_requests(graph_model, retention_plan)
    trainable_nodes = _derive_trainable_nodes(graph_model, component_registry)
    trainable_filter = _trainable_filter(graph, trainable_nodes)

    compiled = CompiledTrainingRun(
        graph=graph,
        graph_spec=graph_model,
        training_spec=training_model,
        task_binding_spec=binding_model,
        task_inputs=task_inputs,
        retention_plan=retention_plan,
        trace_requests=trace_requests,
        loss_terms=loss_terms,
        trainable_nodes=trainable_nodes,
        trainable_filter=trainable_filter,
        task_data=task_data,
        n_steps=n_steps,
        metadata={
            "execution": "generic_graph",
            "task_spec": dict(task_spec),
            "task_input_count": len(task_inputs),
            "trace_request_count": len(trace_requests),
            "loss_term_count": len(loss_terms),
            "trainable_nodes": list(trainable_nodes),
        },
    )
    _dry_run(compiled)
    return compiled


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
    snapshot_interval = max(1, int(getattr(cfg, "snapshot_interval", 100)))
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
    registry = studio_training_registry(
        compiled=compiled,
        optimizer=optimizer,
        total_batches=total_batches,
        stop_event=stop_event,
        rollout_fn=lambda graph, plan, key: rollout_graph(graph, plan, key=key),
        evaluate_loss_fn=_evaluate_loss,
    )

    def _emit_executor_progress(event: Mapping[str, Any]) -> None:
        progress_event = _studio_progress_event(
            event,
            job_id=job_id,
            total_batches=total_batches,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        emit(progress_event)
        emit(
            {
                "type": "training_log",
                "job_id": job_id,
                "program_step": progress_event["program_step"],
                "level": "info",
                "message": (
                    f"Program step {progress_event['program_step']} | "
                    f"loss={progress_event['loss']:.4f} | "
                    f"grad_norm={progress_event['grad_norm']:.3f} | "
                    f"{progress_event['step_time_ms']:.0f}ms"
                ),
                "execution": "contract_executor",
            }
        )
        if progress_event["program_step"] % snapshot_interval == 0 or (
            progress_event["program_step"] >= total_batches
        ):
            emit(
                {
                    "type": "training_trajectory",
                    "job_id": job_id,
                    "program_step": progress_event["program_step"],
                    "trajectory": _trajectory_snapshot(
                        compiled.graph,
                        compiled,
                        key=jr.PRNGKey(progress_event["program_step"]),
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
        loss_service=_StudioWorkerLossService(compiled),
        training_spec_payload=compiled.training_spec.model_dump(mode="json"),
        training_spec_payload_kind="TrainingSpec",
        training_spec_payload_schema_id="feedbax.spec.training",
        training_spec_payload_schema_version="feedbax.spec.training.v1",
        task_binding_spec=compiled.task_binding_spec.model_dump(mode="json", exclude_none=True),
        manifest_conflict_policy="reuse-identical",
        progress_callback=_emit_executor_progress,
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
    checkpoint_path = _write_checkpoint(job_id, graph)
    final_rollout = rollout_graph(graph, compiled, key=result.final_slots["prng"])
    return TrainingGraphResult(
        graph=graph,
        checkpoint_path=checkpoint_path,
        final_loss=final_loss,
        final_loss_terms=final_terms,
        final_batch=result.final_coordinate.program_step,
        manifest_path=str(result.manifest_path),
        manifest_payload=result.manifest.model_dump(mode="json", exclude_none=True),
        execution_metadata=dict(compiled.metadata),
        retention_plan=retention_plan_to_json(compiled.retention_plan),
        retained_observables=_retained_observables_payload(final_rollout),
    )


class _StudioWorkerLossService(LossService):
    """Adapter for Studio retained-observable losses already lowered in preflight."""

    def __init__(self, compiled: CompiledTrainingRun) -> None:
        super().__init__()
        self._compiled = compiled

    def lower_objective_slot(self, *args: Any, **kwargs: Any) -> LoweredObjective:
        del args, kwargs
        return LoweredObjective(
            loss=self._compiled.loss_terms,
            requirements=ObjectiveExecutionRequirements(),
            source_kind="loss_term",
        )


def _training_run_spec_for_compiled(
    compiled: CompiledTrainingRun,
    *,
    job_id: str,
    total_batches: int,
    cfg: Any,
    grad_clip: float | None,
) -> TrainingRunSpec:
    method_contract = executor_method_contract(total_batches=total_batches)
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
            snapshot_interval=max(1, int(getattr(cfg, "snapshot_interval", 100))),
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
            "checkpoint_interval": 1,
            "progress_interval": 1,
        },
        metadata={
            "job_id": job_id,
            "execution": "studio_contract_executor",
        },
    )


def _studio_progress_event(
    event: Mapping[str, Any],
    *,
    job_id: str,
    total_batches: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    coordinate = event.get("coordinate", {})
    coordinate = coordinate if isinstance(coordinate, Mapping) else {}
    metrics = event.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    program_step = int(coordinate.get("program_step") or 0)
    loss = _float_metric(metrics.get("train_loss", 0.0))
    loss_terms = _float_mapping(metrics.get("loss_terms", {}))
    scalar_metrics = {
        key: value for key, value in _float_mapping(metrics).items() if key not in {"loss_terms"}
    }
    scalar_metrics["elapsed_seconds"] = float(elapsed_seconds)
    return {
        "type": "training_progress",
        "job_id": job_id,
        "program_step": program_step,
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


def rollout_graph(
    graph: Graph,
    compiled: CompiledTrainingRun,
    *,
    key: jax.Array,
) -> dict[str, Any]:
    """Roll out one trial through the executable graph boundary."""
    state = init_state_from_component(graph)
    cycle_values = graph.initial_cycle_port_values(state)
    keys = jr.split(key, compiled.n_steps)
    input_sequences = {
        plan.graph_input: compiled.task_data[plan.data_id] for plan in compiled.task_inputs
    }

    def _step_inputs_at(i):
        return {name: value[i] for name, value in input_sequences.items()}

    step_inputs_seq = jax.vmap(_step_inputs_at)(jnp.arange(compiled.n_steps))

    def _step(carry, args):
        state, cycle_values = carry
        step_inputs, step_key = args
        outputs, state, cycle_values, trace = graph.step_with_trace(
            step_inputs,
            state,
            cycle_values,
            key=step_key,
            trace=compiled.trace_requests,
        )
        return (state, cycle_values), {"outputs": outputs, "trace": trace}

    (final_state, _), seq = jax.lax.scan(_step, (state, cycle_values), (step_inputs_seq, keys))
    return {
        "outputs": seq["outputs"],
        "trace": seq["trace"],
        "task_data": compiled.task_data,
        "final_state": final_state,
    }


def _materialize_task_data(
    task_binding_spec: StudioTaskBindingSpec,
    task_spec: dict[str, Any],
    n_steps: int,
) -> dict[str, jax.Array]:
    data: dict[str, jax.Array] = {}
    for item in task_binding_spec.exposed_data:
        value = _materialize_one_task_data(item, task_spec, n_steps)
        data[item.id] = value
        data[item.path] = value
    return data


def _reject_unsupported_task_data_value_spec_modes(task_binding_spec: Mapping[str, Any]) -> None:
    for item in task_binding_spec.get("exposed_data", []):
        if not isinstance(item, Mapping):
            continue
        value_spec = item.get("value_spec")
        if not isinstance(value_spec, Mapping):
            continue
        mode = value_spec.get("mode")
        if mode is not None and mode not in {"constant", "function"}:
            raise ValueError(
                f"Task data {item.get('id')!r} at {item.get('path')!r} uses unsupported "
                f"value_spec mode={mode!r}"
            )


def _materialize_one_task_data(
    item: StudioTaskDataSpec,
    task_spec: dict[str, Any],
    n_steps: int,
) -> jax.Array:
    value_spec = item.value_spec
    shape = _runtime_shape(item.expected_shape, n_steps)
    if value_spec is None:
        return jnp.zeros(shape, dtype=jnp.float32)
    if value_spec.mode == "constant":
        if value_spec.value is None:
            raise ValueError(
                f"Task data {item.id!r} at {item.path!r} has constant value_spec without value"
            )
        value = value_spec.value
        if isinstance(value, dict):
            fill = float(value.get("inactive", value.get("value", 0.0)))
            return jnp.full(shape, fill, dtype=jnp.float32)
        arr = jnp.asarray(value, dtype=jnp.float32)
        return jnp.broadcast_to(arr, shape)
    if value_spec.mode == "function":
        if value_spec.function_id == "delayed_reach_target_position":
            return _delayed_reach_target_position(task_spec, n_steps, shape)
        if value_spec.function_id == "delayed_reach_movement_target":
            target = _delayed_reach_target_position(task_spec, n_steps, (n_steps, 4))[..., :2]
            return jnp.broadcast_to(target, shape)
        raise ValueError(
            f"Task data {item.id!r} at {item.path!r} uses unsupported value_spec "
            f"function_id={value_spec.function_id!r}"
        )
    raise ValueError(
        f"Task data {item.id!r} at {item.path!r} uses unsupported value_spec "
        f"mode={value_spec.mode!r}"
    )


def _runtime_shape(expected_shape: list[Any] | None, n_steps: int) -> tuple[int, ...]:
    if not expected_shape:
        return (n_steps, 1)
    dims: list[int] = []
    if expected_shape[0] != "time":
        dims.append(n_steps)
    for dim in expected_shape:
        if dim == "time":
            dims.append(n_steps)
        elif isinstance(dim, int):
            dims.append(int(dim))
        elif isinstance(dim, float) and dim.is_integer():
            dims.append(int(dim))
        else:
            dims.append(1)
    return tuple(dims)


def _delayed_reach_target_position(
    task_spec: dict[str, Any],
    n_steps: int,
    shape: tuple[int, ...],
) -> jax.Array:
    params = task_spec.get("params", {}) if isinstance(task_spec, dict) else {}
    workspace = params.get("workspace") or [[-0.25, -0.25], [0.25, 0.25]]
    start = jnp.asarray(workspace[0], dtype=jnp.float32)
    end = jnp.asarray(workspace[1], dtype=jnp.float32)
    progress = jnp.linspace(0.0, 1.0, n_steps, dtype=jnp.float32)[:, None]
    pos = start + progress * (end - start)
    vel = jnp.gradient(pos, axis=0)
    value = jnp.concatenate([pos, vel], axis=-1)
    return jnp.broadcast_to(value[..., : shape[-1]], shape)


def _compile_trace_requests(
    graph_spec: GraphSpec,
    retention_plan: RetentionPlan,
) -> tuple[GraphTraceRequest, ...]:
    requests: dict[str, GraphTraceRequest] = {}
    for observable in retention_plan.observables:
        request = _trace_request_from_selector_ref(observable.selector)
        if request is not None:
            requests[request.selector] = request
    if not requests and graph_spec.output_ports:
        selector = f"graph_output:{graph_spec.output_ports[0]}"
        requests[selector] = GraphTraceRequest(
            kind="graph_output",
            selector=selector,
            port=graph_spec.output_ports[0],
        )
    return tuple(requests.values())


def _trace_request_from_selector_ref(selector: SelectorRef) -> GraphTraceRequest | None:
    if selector.kind == "task_data":
        return None
    if selector.kind == "graph_output":
        return GraphTraceRequest(
            kind="graph_output",
            selector=selector.selector,
            port=selector.path or selector.selector.removeprefix("graph_output:"),
        )
    if selector.kind == "port":
        return GraphTraceRequest(
            kind="port",
            selector=selector.selector,
            node=selector.node_id,
            port=selector.port,
        )
    if selector.kind == "edge":
        source_node, source_port, target_node, target_port = _parse_edge(
            selector.edge_id or selector.selector
        )
        return GraphTraceRequest(
            kind="edge",
            selector=selector.selector,
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port,
        )
    if selector.kind == "recurrent_carry":
        _source_node, _source_port, target_node, target_port = _parse_edge(
            selector.edge_id or selector.selector
        )
        return GraphTraceRequest(
            kind="recurrent_carry",
            selector=selector.selector,
            node=target_node,
            port=target_port,
        )
    if selector.kind == "state_path":
        return GraphTraceRequest(
            kind="state_path",
            selector=selector.selector,
            path=_state_path_from_selector(selector.path or selector.selector),
        )
    return None


def _parse_node_port(value: str) -> tuple[str, str]:
    node, sep, port = value.rpartition(".")
    if not sep or not node or not port:
        raise ValueError(f"Selector {value!r} must be formatted as node.port")
    return node, port


def _parse_edge(value: str) -> tuple[str, str, str | None, str | None]:
    value = value.removeprefix("edge:").removeprefix("recurrent_carry:")
    source, sep, target = value.partition("->")
    source_node, source_port = _parse_node_port(source)
    if not sep:
        return source_node, source_port, None, None
    target_node, target_port = _parse_node_port(target)
    return source_node, source_port, target_node, target_port


def _state_path_from_selector(selector: str) -> str:
    path = selector.removeprefix("state_path:").removeprefix("path:")
    if path.startswith("states."):
        path = path.removeprefix("states.")
    return path


def _derive_trainable_nodes(graph_spec: GraphSpec, registry: Any) -> tuple[str, ...]:
    trainable: list[str] = []
    for node_id, node in graph_spec.nodes.items():
        raw = node.params.get("trainable")
        if raw is False:
            continue
        meta = registry.get(node.type)
        trainable_by_default = bool(
            getattr(meta, "trainable_by_default", False) if meta is not None else False
        )
        if raw is True or trainable_by_default:
            trainable.append(node_id)
    return tuple(trainable)


def _trainable_filter(graph: Graph, trainable_nodes: tuple[str, ...]):
    filter_spec = jt.map(lambda _: False, graph)
    for node_name in trainable_nodes:
        if node_name not in graph.nodes:
            continue
        node_filter = jt.map(eqx.is_inexact_array, graph.nodes[node_name])
        filter_spec = eqx.tree_at(
            lambda g, name=node_name: g.nodes[name],
            filter_spec,
            node_filter,
        )
    return filter_spec


def _dry_run(compiled: CompiledTrainingRun) -> None:
    try:
        rollout = rollout_graph(compiled.graph, compiled, key=jr.PRNGKey(0))
        _evaluate_loss(compiled, rollout)
    except Exception as exc:
        raise ValueError(f"Generic graph worker preflight failed: {exc}") from exc


def _evaluate_loss(
    compiled: CompiledTrainingRun,
    rollout: dict[str, Any],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if not compiled.loss_terms:
        output_name = next(iter(rollout["outputs"]))
        value = rollout["outputs"][output_name]
        loss = jnp.mean(jnp.square(value))
        return loss, {"default_output": loss}
    return evaluate_loss_plan(compiled.loss_terms, _rollout_trace_map(rollout))


def _rollout_trace_map(rollout: dict[str, Any]) -> dict[str, Any]:
    trace = dict(rollout["trace"])
    for name, value in rollout["outputs"].items():
        trace[f"graph_output:{name}"] = value
    for name, value in rollout["task_data"].items():
        trace[f"task_data:{name}"] = value
    return trace


def _trajectory_snapshot(
    graph: Graph,
    compiled: CompiledTrainingRun,
    key: jax.Array,
) -> dict[str, Any]:
    rollout = rollout_graph(graph, compiled, key=key)
    observables = {key: _jsonable_value(value) for key, value in rollout["trace"].items()}
    outputs = {key: _jsonable_value(value) for key, value in rollout["outputs"].items()}
    tracks: dict[str, Any] = {}
    time_length = compiled.n_steps

    def add_track(
        *,
        namespace: str,
        name: str,
        anchor_id: str,
        role: str,
        label: str,
        value: Any,
    ) -> None:
        nonlocal time_length
        samples = _live_xy_samples(value, length=time_length)
        if samples is None:
            return
        if not tracks:
            time_length = len(samples)
        elif len(samples) != time_length:
            return
        selector = {
            "namespace": namespace,
            "compact": f"{namespace}:{name}",
            "target_id": name,
            "role": role,
        }
        tracks[selector["compact"]] = {
            "anchor_id": anchor_id,
            "selector": selector,
            "samples": samples,
            "dim": 2,
            "dtype": "float32",
            "units": None,
            "frame": "world",
            "label": label,
            "metadata": {"source": "live_training_snapshot"},
        }

    for name, value in rollout["outputs"].items():
        add_track(
            namespace="graph_output",
            name=str(name),
            anchor_id=str(name),
            role="observed",
            label=str(name).replace("_", " ").title(),
            value=_legacy_xy_trajectory(value),
        )
    target = _legacy_target_trajectory(rollout["task_data"])
    if target is not None:
        target_name, target_value = target
        add_track(
            namespace="task_data",
            name=target_name,
            anchor_id=target_name,
            role="target",
            label=target_name.replace("_", " ").title(),
            value=target_value,
        )

    trajectory: dict[str, Any] = {
        "schema_id": TRAINING_TRAJECTORY_SCHEMA_ID,
        "schema_version": TRAINING_TRAJECTORY_SCHEMA_VERSION,
        "source_kind": "live_snapshot",
        "fidelity": "lower_fidelity_live_snapshot",
        "time": {
            "length": time_length,
            "units": "step",
            "values": list(range(time_length)),
            "metadata": {"source": "live_training_snapshot"},
        },
        "tracks": tracks,
        "n_steps": compiled.n_steps,
        "observables": observables,
        "outputs": outputs,
    }
    return trajectory


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


def _jsonable_value(value: Any) -> Any:
    try:
        value = jax.device_get(value)
    except TypeError:
        pass

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable_value(getattr(value, item.name))
            for item in fields(value)
            if hasattr(value, item.name)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {str(key): _jsonable_value(getattr(value, key)) for key in getattr(value, "_fields")}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_jsonable_value(item) for item in value]
    return repr(value)


def _legacy_xy_trajectory(value: Any) -> Any | None:
    if value is None:
        return None
    value = jax.device_get(value)
    for attr in ("pos", "position", "effector_pos"):
        if hasattr(value, attr):
            return _legacy_xy_trajectory(getattr(value, attr))
    if isinstance(value, Mapping):
        for key in ("pos", "position", "effector_pos"):
            if key in value:
                return _legacy_xy_trajectory(value[key])
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for key in ("pos", "position", "effector_pos"):
            if hasattr(value, key):
                return _legacy_xy_trajectory(getattr(value, key))
    if hasattr(value, "ndim") and value.ndim >= 1 and value.shape[-1] >= 2:
        return value[..., :2]
    return None


def _legacy_target_trajectory(task_data: dict[str, Any]) -> tuple[str, Any] | None:
    for key, value in task_data.items():
        if not (str(key).endswith("target") or "target" in str(key)):
            continue
        target = _legacy_xy_trajectory(value)
        if target is not None:
            return str(key), target
    return None


def _live_xy_samples(value: Any, *, length: int) -> list[list[float]] | None:
    if value is None:
        return None
    array = jnp.asarray(jax.device_get(value))
    if array.ndim == 0 or array.shape[-1] < 2:
        return None
    array = array[..., :2]
    if array.ndim == 1:
        array = array.reshape((1, 2))
    elif array.ndim > 2:
        array = array.reshape((-1, 2))
    samples = [[float(point[0]), float(point[1])] for point in jax.device_get(array)]
    if len(samples) == 1 and length > 1:
        samples = samples * length
    return samples


def _write_checkpoint(job_id: str, graph: Graph) -> str | None:
    ckpt_dir = tempfile.mkdtemp(prefix="feedbax_ckpt_")
    ckpt_path = os.path.join(ckpt_dir, f"{job_id}.eqx")
    ready_graph = jax.block_until_ready(graph)
    eqx.tree_serialise_leaves(ckpt_path, ready_graph)
    return ckpt_path
