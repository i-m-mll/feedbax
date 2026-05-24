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

from feedbax.graph import Graph, GraphTraceRequest, init_state_from_component
from feedbax.retained_observables import (
    LossTermPlan,
    RetentionPlan,
    RetentionPlanError,
    SelectorRef,
    evaluate_loss_plan,
    lower_retention_plan,
    retention_plan_to_json,
)
from feedbax.studio_schema import validate_task_binding_schema
from feedbax.web.models.graph import (
    GraphSpec,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
)
from feedbax.web.models.training import TrainingSpec
from feedbax.web.serialization import spec_to_graph


_DEFAULT_TRAINABLE_COMPONENT_TYPES = {
    "Linear",
    "MLP",
    "GRU",
    "LSTM",
    "Network",
    "Network Template",
    "Simple Feedback Loop",
}


@dataclass(frozen=True)
class TaskInputPlan:
    """A task-data stream bound to one graph input port."""

    data_id: str
    data_path: str
    graph_input: str
    target_node: str
    target_port: str
    role: str


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
        raise ValueError(
            "Generic graph worker currently supports batch_size=1; "
            f"got batch_size={training_model.batch_size}"
        )
    binding_model = (
        task_binding_spec
        if isinstance(task_binding_spec, StudioTaskBindingSpec)
        else StudioTaskBindingSpec.model_validate(task_binding_spec)
    )
    binding_errors = [
        issue
        for issue in validate_task_binding_schema(
            binding_model,
            graph_model,
            "/task_binding_spec",
        )
        if issue.severity == "error"
    ]
    if binding_errors:
        summary = "; ".join(f"{issue.type}: {issue.message}" for issue in binding_errors)
        raise ValueError(f"Invalid task_binding_spec for graph execution: {summary}")

    try:
        graph = spec_to_graph(graph_model, {})
    except NotImplementedError as exc:
        raise ValueError(f"GraphSpec contains unsupported executable component: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"GraphSpec could not be instantiated for worker execution: {exc}") from exc

    graph, task_inputs = _expose_task_inputs(graph, binding_model)
    n_steps = int(getattr(cfg, "n_reach_steps", None) or training_model.n_batches or 1)
    task_data = _materialize_task_data(binding_model, task_spec, n_steps)
    try:
        retention_plan = lower_retention_plan(graph_model, training_model)
    except RetentionPlanError as exc:
        raise ValueError(
            f"Invalid retention plan for graph execution at {exc.path}: {exc}"
        ) from exc
    loss_terms = retention_plan.loss_terms
    trace_requests = _compile_trace_requests(graph_model, retention_plan)
    trainable_nodes = _derive_trainable_nodes(graph_model)
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
    """Train an executable graph and stream Studio-compatible worker events."""
    graph = compiled.graph
    learning_rate = float(getattr(cfg, "learning_rate", 1e-3))
    grad_clip = float(getattr(cfg, "grad_clip", 1.0))
    snapshot_interval = max(1, int(getattr(cfg, "snapshot_interval", 100)))
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(learning_rate))

    trainable, static = eqx.partition(graph, compiled.trainable_filter)
    opt_state = optimizer.init(trainable)
    rng_key = jr.PRNGKey(0)
    final_terms: dict[str, float] = {}
    final_loss = 0.0

    def _loss_from_trainable(trainable_graph, static_graph, step_key):
        current_graph = eqx.combine(static_graph, trainable_graph)
        rollout = rollout_graph(current_graph, compiled, key=step_key)
        return _evaluate_loss(compiled, rollout)

    for batch in range(total_batches):
        if stop_event.is_set():
            break

        rng_key, step_key = jr.split(rng_key)
        step_t0 = time.perf_counter()
        (loss_value, loss_terms), grads = eqx.filter_value_and_grad(
            _loss_from_trainable,
            has_aux=True,
        )(trainable, static, step_key)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        graph = eqx.combine(static, trainable)
        compiled.graph = graph

        final_loss = float(jax.block_until_ready(loss_value))
        final_terms = {
            key: float(jax.block_until_ready(value)) for key, value in loss_terms.items()
        }
        grad_norm_value = float(jax.block_until_ready(grad_norm))
        step_time_ms = (time.perf_counter() - step_t0) * 1000.0

        emit(
            {
                "type": "training_progress",
                "job_id": job_id,
                "batch": batch + 1,
                "total_batches": total_batches,
                "loss": final_loss,
                "loss_terms": final_terms,
                "grad_norm": grad_norm_value,
                "step_time_ms": step_time_ms,
                "status": "running",
                "execution": "generic_graph",
            }
        )
        emit(
            {
                "type": "training_log",
                "job_id": job_id,
                "batch": batch + 1,
                "level": "info",
                "message": (
                    f"Step {batch + 1} | loss={final_loss:.4f} | "
                    f"grad_norm={grad_norm_value:.3f} | {step_time_ms:.0f}ms"
                ),
            }
        )

        if (batch + 1) % snapshot_interval == 0 or batch + 1 == total_batches:
            snapshot = _trajectory_snapshot(graph, compiled, step_key)
            emit(
                {
                    "type": "training_trajectory",
                    "job_id": job_id,
                    "batch": batch + 1,
                    "trajectory": snapshot,
                    "execution": "generic_graph",
                }
            )

    checkpoint_path = _write_checkpoint(job_id, graph)
    final_rollout = rollout_graph(graph, compiled, key=rng_key)
    return TrainingGraphResult(
        graph=graph,
        checkpoint_path=checkpoint_path,
        final_loss=final_loss,
        final_loss_terms=final_terms,
        execution_metadata=dict(compiled.metadata),
        retention_plan=retention_plan_to_json(compiled.retention_plan),
        retained_observables=_retained_observables_payload(final_rollout),
    )


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


def _expose_task_inputs(
    graph: Graph,
    task_binding_spec: StudioTaskBindingSpec,
) -> tuple[Graph, tuple[TaskInputPlan, ...]]:
    input_ports = list(graph.input_ports)
    input_bindings = dict(graph.input_bindings)
    plans: list[TaskInputPlan] = []
    for binding in task_binding_spec.bindings:
        target_key = (binding.target_node_id, binding.target_port)
        if binding.target_node_id not in graph.nodes:
            raise ValueError(
                f"Task binding {binding.id!r} targets missing node "
                f"{binding.target_node_id!r}"
            )
        if binding.target_port not in graph.nodes[binding.target_node_id].input_ports:
            raise ValueError(
                f"Task binding {binding.id!r} targets missing port "
                f"{binding.target_node_id}.{binding.target_port}"
            )
        graph_input = next(
            (name for name, bound in input_bindings.items() if tuple(bound) == target_key),
            f"task:{binding.source_data_id}->{binding.target_node_id}.{binding.target_port}",
        )
        if graph_input not in input_ports:
            input_ports.append(graph_input)
        input_bindings[graph_input] = target_key
        data = _task_data_by_id(task_binding_spec)[binding.source_data_id]
        plans.append(
            TaskInputPlan(
                data_id=binding.source_data_id,
                data_path=data.path,
                graph_input=graph_input,
                target_node=binding.target_node_id,
                target_port=binding.target_port,
                role=binding.role,
            )
        )

    graph = eqx.tree_at(
        lambda g: (g.input_ports, g.input_bindings),
        graph,
        (tuple(input_ports), input_bindings),
    )
    return graph, tuple(plans)


def _task_data_by_id(spec: StudioTaskBindingSpec) -> dict[str, StudioTaskDataSpec]:
    data = {item.id: item for item in spec.exposed_data}
    missing = [binding.source_data_id for binding in spec.bindings if binding.source_data_id not in data]
    if missing:
        raise ValueError(f"Task bindings reference unknown task data ids: {sorted(set(missing))}")
    return data


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


def _derive_trainable_nodes(graph_spec: GraphSpec) -> tuple[str, ...]:
    trainable: list[str] = []
    for node_id, node in graph_spec.nodes.items():
        raw = node.params.get("trainable")
        if raw is False:
            continue
        if raw is True or node.type in _DEFAULT_TRAINABLE_COMPONENT_TYPES:
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
    observables = {
        key: _jsonable_value(value) for key, value in rollout["trace"].items()
    }
    outputs = {
        key: _jsonable_value(value) for key, value in rollout["outputs"].items()
    }
    trajectory: dict[str, Any] = {
        "n_steps": compiled.n_steps,
        "t": list(range(compiled.n_steps)),
        "observables": observables,
        "outputs": outputs,
    }
    if "effector" in rollout["outputs"]:
        effector = _legacy_xy_trajectory(rollout["outputs"]["effector"])
        if effector is not None:
            trajectory["effector"] = _jsonable_value(effector)
    target = _legacy_target_trajectory(rollout["task_data"])
    if target is not None:
        trajectory["target"] = _jsonable_value(target)
    return trajectory


def _retained_observables_payload(rollout: dict[str, Any]) -> dict[str, Any]:
    return {
        "observables": {
            key: _jsonable_value(value) for key, value in rollout["trace"].items()
        },
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
        return {
            str(key): _jsonable_value(getattr(value, key))
            for key in getattr(value, "_fields")
        }
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_jsonable_value(item) for item in value]
    return repr(value)


def _legacy_xy_trajectory(value: Any) -> Any | None:
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


def _legacy_target_trajectory(task_data: dict[str, Any]) -> Any | None:
    for key, value in task_data.items():
        if not (str(key).endswith("target") or "target" in str(key)):
            continue
        target = _legacy_xy_trajectory(value)
        if target is not None:
            return target
    return None


def _write_checkpoint(job_id: str, graph: Graph) -> str | None:
    ckpt_dir = tempfile.mkdtemp(prefix="feedbax_ckpt_")
    ckpt_path = os.path.join(ckpt_dir, f"{job_id}.eqx")
    ready_graph = jax.block_until_ready(graph)
    eqx.tree_serialise_leaves(ckpt_path, ready_graph)
    return ckpt_path
