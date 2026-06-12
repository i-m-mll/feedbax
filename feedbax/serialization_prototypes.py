from __future__ import annotations

from math import prod
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import jax.tree as jt

from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.state import CartesianState


STATEFUL_PROTOTYPE_TYPES = {"Channel", "DelayLine", "FirstOrderFilter"}


def array_proto_from_shape(shape: Any) -> jax.Array | None:
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return jnp.zeros(tuple(int(dim) for dim in shape))
    except (TypeError, ValueError):
        return None


def shape_from_proto(proto: Any) -> list[int] | None:
    leaves = jt.leaves(proto)
    if len(leaves) != 1 or not hasattr(leaves[0], "shape"):
        return None
    return [int(dim) for dim in leaves[0].shape]


def proto_from_value(value: Any) -> Any:
    return jt.map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)


def _task_sample_shape(shape: Any) -> list[int] | None:
    if not isinstance(shape, (list, tuple)):
        return None
    dims = list(shape)
    if dims and dims[0] == "time":
        dims = dims[1:]
    if not dims:
        return []
    sample: list[int] = []
    for dim in dims:
        if isinstance(dim, int):
            sample.append(dim)
        elif isinstance(dim, float) and dim.is_integer():
            sample.append(int(dim))
        else:
            return None
    return sample


def prototypes_from_task_bindings(task_binding_spec: Any) -> dict[tuple[str, str], Any]:
    """Return node-port input prototypes derivable from Studio task bindings."""

    exposed = getattr(task_binding_spec, "exposed_data", None)
    bindings = getattr(task_binding_spec, "bindings", None)
    if exposed is None and isinstance(task_binding_spec, Mapping):
        exposed = task_binding_spec.get("exposed_data", [])
    if bindings is None and isinstance(task_binding_spec, Mapping):
        bindings = task_binding_spec.get("bindings", [])

    data_by_id: dict[str, Any] = {}
    for item in exposed or []:
        item_id = getattr(item, "id", None) if not isinstance(item, Mapping) else item.get("id")
        if item_id is not None:
            data_by_id[str(item_id)] = item

    prototypes: dict[tuple[str, str], Any] = {}
    for binding in bindings or []:
        source_id = (
            getattr(binding, "source_data_id", None)
            if not isinstance(binding, Mapping)
            else binding.get("source_data_id")
        )
        target_node = (
            getattr(binding, "target_node_id", None)
            if not isinstance(binding, Mapping)
            else binding.get("target_node_id")
        )
        target_port = (
            getattr(binding, "target_port", None)
            if not isinstance(binding, Mapping)
            else binding.get("target_port")
        )
        item = data_by_id.get(str(source_id))
        if item is None or target_node is None or target_port is None:
            continue
        expected_shape = (
            getattr(item, "expected_shape", None)
            if not isinstance(item, Mapping)
            else item.get("expected_shape")
        )
        sample_shape = _task_sample_shape(expected_shape)
        if sample_shape is not None:
            prototypes[(str(target_node), str(target_port))] = jnp.zeros(tuple(sample_shape))
    return prototypes


def explicit_proto(
    params: Mapping[str, Any],
    *,
    node_name: str,
    node_type: str,
) -> Any | None:
    if "input_shape" not in params:
        return None
    proto = array_proto_from_shape(params.get("input_shape"))
    if proto is None:
        raise ValueError(
            f"{node_type} node {node_name!r} has invalid input_shape "
            f"{params.get('input_shape')!r}; expected a list of integer dimensions"
        )
    return proto


def validate_or_add_input_shape(
    params: Mapping[str, Any],
    inferred_proto: Any | None,
    *,
    node_name: str,
    node_type: str,
) -> dict[str, Any]:
    next_params = dict(params)
    explicit = explicit_proto(next_params, node_name=node_name, node_type=node_type)
    if inferred_proto is not None and explicit is not None:
        inferred_shape = shape_from_proto(inferred_proto)
        explicit_shape = shape_from_proto(explicit)
        if inferred_shape is not None and explicit_shape != inferred_shape:
            raise ValueError(
                f"{node_type} node {node_name!r} input_shape {explicit_shape!r} "
                f"does not match inferred input prototype shape {inferred_shape!r} "
                "for port 'input'"
            )
    proto = inferred_proto if inferred_proto is not None else explicit
    if proto is None:
        raise ValueError(
            f"{node_type} node {node_name!r} port 'input' requires an input prototype. "
            "Connect it to a source with a known output shape or provide input_shape."
        )
    shape = shape_from_proto(proto)
    if shape is None:
        raise ValueError(
            f"{node_type} node {node_name!r} port 'input' prototype is not serializable "
            "as input_shape; only single-array prototypes are currently supported"
        )
    next_params["input_shape"] = shape
    return next_params


def _defer_or_raise(reason: str, *, strict: bool) -> dict[str, Any]:
    if strict:
        raise ValueError(reason)
    return {}


def _required_input(
    input_prototypes: Mapping[tuple[str, str], Any],
    node_name: str,
    node_type: str,
    port: str,
    *,
    strict: bool,
) -> Any | None:
    proto = input_prototypes.get((node_name, port))
    if proto is None:
        _defer_or_raise(
            f"{node_type} node {node_name!r} port {port!r} requires an input prototype",
            strict=strict,
        )
    return proto


def _lookup_registry_meta(component_registry: Any, name: str) -> Any | None:
    if component_registry is None:
        return None
    if isinstance(component_registry, Mapping):
        meta = component_registry.get(name)
    elif hasattr(component_registry, "get"):
        meta = component_registry.get(name)
    elif isinstance(component_registry, (list, tuple)):
        meta = next((item for item in component_registry if getattr(item, "name", None) == name), None)
    else:
        meta = None
    return meta


def _lookup_output_prototype_fn(component_registry: Any, name: str) -> Any | None:
    meta = _lookup_registry_meta(component_registry, name)
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        return meta.get("output_prototype_fn")
    return getattr(meta, "output_prototype_fn", None)


def _lookup_default_params(component_registry: Any, name: str) -> dict[str, Any]:
    meta = _lookup_registry_meta(component_registry, name)
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta.get("default_params", {}))
    if hasattr(meta, "default_params"):
        return dict(getattr(meta, "default_params"))
    return {}


def _registered_output_prototypes(
    node_name: str,
    node_spec: ComponentSpec,
    input_prototypes: Mapping[tuple[str, str], Any],
    component_registry: Any,
) -> dict[str, Any] | None:
    output_prototype_fn = _lookup_output_prototype_fn(component_registry, node_spec.type)
    if output_prototype_fn is None:
        return None
    node_inputs = {
        port: input_prototypes[(node_name, port)]
        for port in node_spec.input_ports
        if (node_name, port) in input_prototypes
    }
    params = _lookup_default_params(component_registry, node_spec.type)
    params.update(node_spec.params)
    outputs = output_prototype_fn(params, node_inputs)
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"Output prototype function for {node_spec.type!r} node {node_name!r} "
            "must return a mapping of output port names to prototypes"
        )
    return dict(outputs)


def output_prototypes_for_node(
    node_name: str,
    node_spec: ComponentSpec,
    input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSpec],
    component_registry: Any = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    params = node_spec.params
    node_type = node_spec.type

    if node_type == "Subgraph" or node_name in subgraphs or node_type == "Network":
        subgraph = subgraphs.get(node_name)
        if subgraph is None:
            raise ValueError(
                f"{node_type} node {node_name!r} requires a subgraph, but none was provided"
            )
        nested_inputs = {
            (node, port): input_prototypes[(node_name, graph_port)]
            for graph_port, (node, port) in subgraph.input_bindings.items()
            if (node_name, graph_port) in input_prototypes
        }
        normalized = normalize_stateful_prototypes(
            subgraph,
            nested_inputs,
            component_registry=component_registry,
        )
        return bound_output_prototypes(
            normalized,
            subgraphs=dict(normalized.subgraphs or {}),
            input_prototypes=nested_inputs,
            component_registry=component_registry,
        )

    registered_outputs = _registered_output_prototypes(
        node_name,
        node_spec,
        input_prototypes,
        component_registry,
    )
    if registered_outputs is not None:
        return registered_outputs

    if node_type == "Constant":
        return {"output": proto_from_value(params.get("value", 0.0))}
    if node_type in {"Ramp", "Sine", "Pulse"}:
        value = params.get("amplitude", params.get("slope", 1.0))
        return {"output": proto_from_value(value)}
    if node_type == "Noise":
        return {"output": jnp.zeros(tuple(int(dim) for dim in params.get("shape", [1])))}
    if node_type in {"Gain", "Saturation"}:
        proto = _required_input(input_prototypes, node_name, node_type, "input", strict=strict)
        return {"output": proto} if proto is not None else {}
    if node_type in {"Sum", "Multiply"}:
        proto = input_prototypes.get((node_name, "a"))
        if proto is None:
            proto = input_prototypes.get((node_name, "b"))
        if proto is None:
            return _defer_or_raise(
                f"{node_type} node {node_name!r} requires an input prototype for port 'a' or 'b'",
                strict=strict,
            )
        return {"output": proto}
    if node_type == "Ravel":
        proto = _required_input(input_prototypes, node_name, node_type, "input", strict=strict)
        if proto is None:
            return {}
        leaves = jt.leaves(proto)
        if not leaves or any(not hasattr(leaf, "shape") for leaf in leaves):
            return _defer_or_raise(
                f"Ravel node {node_name!r} port 'input' prototype is not array-shaped",
                strict=strict,
            )
        return {"output": jnp.zeros((sum(prod(leaf.shape) for leaf in leaves),))}
    if node_type in STATEFUL_PROTOTYPE_TYPES:
        proto = input_prototypes.get((node_name, "input"))
        if proto is None:
            proto = explicit_proto(params, node_name=node_name, node_type=node_type)
        if proto is None:
            return _defer_or_raise(
                f"{node_type} node {node_name!r} port 'input' requires an input prototype",
                strict=strict,
            )
        return {"output": proto}
    if node_type == "Linear":
        return {"output": jnp.zeros((int(params.get("output_size", 1)),))}
    if node_type == "MLP":
        return {"output": jnp.zeros((int(params.get("output_size", 1)),))}
    if node_type == "GRU":
        hidden = jnp.zeros((int(params.get("hidden_size", 1)),))
        return {"output": hidden, "hidden": hidden}
    if node_type == "LSTM":
        hidden = jnp.zeros((int(params.get("hidden_size", 1)),))
        return {"output": hidden, "hidden": hidden, "cell": hidden}
    if node_type == "Mux":
        available_ports = [
            port for port in node_spec.input_ports if (node_name, port) in input_prototypes
        ]
        if len(available_ports) < 2:
            return _defer_or_raise(
                f"Mux {node_name!r} needs at least two connected inputs",
                strict=strict,
            )
        missing_ports = [
            port for port in node_spec.input_ports if (node_name, port) not in input_prototypes
        ]
        if missing_ports:
            return _defer_or_raise(
                f"Mux node {node_name!r} is missing input prototypes for ports "
                + ", ".join(repr(port) for port in missing_ports),
                strict=strict,
            )
        parts = [input_prototypes[(node_name, port)] for port in node_spec.input_ports]
        if not parts:
            return _defer_or_raise(
                f"Mux node {node_name!r} requires at least one input port",
                strict=strict,
            )
        shapes = [shape_from_proto(part) for part in parts]
        if any(shape is None or len(shape) != 1 for shape in shapes):
            return _defer_or_raise(
                f"Mux node {node_name!r} input prototypes must be one-dimensional arrays",
                strict=strict,
            )
        return {"output": jnp.zeros((sum(shape[0] for shape in shapes if shape),))}
    if node_type in {"PointMass", "TwoLinkArm", "Arm6MuscleRigidTendon"}:
        effector = CartesianState()
        return {"effector": effector}
    raise ValueError(
        f"Node {node_name!r} has unsupported component type {node_type!r} for prototype inference"
    )


def bound_output_prototypes(
    spec: GraphSpec,
    *,
    subgraphs: Mapping[str, GraphSpec],
    input_prototypes: Mapping[tuple[str, str], Any],
    component_registry: Any = None,
) -> dict[str, Any]:
    node_inputs = infer_node_input_prototypes(
        spec,
        input_prototypes,
        subgraphs,
        component_registry=component_registry,
    )
    outputs: dict[str, Any] = {}
    for graph_port, (node_name, node_port) in spec.output_bindings.items():
        node_spec = spec.nodes.get(node_name)
        if node_spec is None:
            raise ValueError(
                f"Graph output binding {graph_port!r} references missing node {node_name!r}"
            )
        node_outputs = output_prototypes_for_node(
            node_name,
            node_spec,
            node_inputs,
            subgraphs,
            component_registry=component_registry,
        )
        if node_port not in node_outputs:
            raise ValueError(
                f"Graph output binding {graph_port!r} references missing output port "
                f"{node_name}.{node_port}"
            )
        outputs[graph_port] = node_outputs[node_port]
    return outputs


def infer_node_input_prototypes(
    spec: GraphSpec,
    external_input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSpec],
    component_registry: Any = None,
) -> dict[tuple[str, str], Any]:
    input_prototypes: dict[tuple[str, str], Any] = dict(external_input_prototypes)
    for graph_port, (node_name, node_port) in spec.input_bindings.items():
        graph_proto = external_input_prototypes.get(("__graph__", graph_port))
        if graph_proto is not None:
            input_prototypes[(node_name, node_port)] = graph_proto

    for wire in spec.wires:
        initializer = wire.recurrent_initializer
        if initializer is None:
            continue
        init_proto = None
        if initializer.get("kind") == "zeros":
            init_proto = array_proto_from_shape(initializer.get("shape"))
        elif initializer.get("kind") == "constant" and "value" in initializer:
            init_proto = proto_from_value(initializer["value"])
        if init_proto is not None:
            input_prototypes[(wire.target_node, wire.target_port)] = init_proto

    for _ in range(max(1, len(spec.nodes) + len(spec.wires) + 1)):
        changed = False
        for wire in spec.wires:
            source_spec = spec.nodes.get(wire.source_node)
            if source_spec is None:
                continue
            outputs = output_prototypes_for_node(
                wire.source_node,
                source_spec,
                input_prototypes,
                subgraphs,
                component_registry=component_registry,
                strict=False,
            )
            proto = outputs.get(wire.source_port)
            if proto is None:
                continue
            key = (wire.target_node, wire.target_port)
            current_shape = (
                shape_from_proto(input_prototypes[key]) if key in input_prototypes else None
            )
            next_shape = shape_from_proto(proto)
            if key not in input_prototypes:
                input_prototypes[key] = proto
                changed = True
            elif (
                current_shape is not None and next_shape is not None and current_shape != next_shape
            ):
                raise ValueError(
                    f"Graph wiring shape mismatch at {wire.target_node}.{wire.target_port}: "
                    f"existing prototype shape {current_shape!r}, "
                    f"{wire.source_node}.{wire.source_port} provides {next_shape!r}"
                )
        if not changed:
            break

    for wire in spec.wires:
        if (wire.target_node, wire.target_port) in input_prototypes:
            continue
        target_spec = spec.nodes.get(wire.target_node)
        if target_spec is None or target_spec.type not in STATEFUL_PROTOTYPE_TYPES:
            continue
        source_spec = spec.nodes.get(wire.source_node)
        if source_spec is None:
            continue
        outputs = output_prototypes_for_node(
            wire.source_node,
            source_spec,
            input_prototypes,
            subgraphs,
            component_registry=component_registry,
            strict=True,
        )
        if wire.source_port not in outputs:
            raise ValueError(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} references a source port "
                "with no inferable output prototype"
            )

    return input_prototypes


def normalize_stateful_prototypes(
    spec: GraphSpec,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
    component_registry: Any = None,
) -> GraphSpec:
    subgraphs = dict(spec.subgraphs or {})
    node_inputs = infer_node_input_prototypes(
        spec,
        input_prototypes or {},
        subgraphs,
        component_registry=component_registry,
    )
    nodes: dict[str, ComponentSpec] = {}
    normalized_subgraphs: dict[str, GraphSpec] = {}

    for node_name, node_spec in spec.nodes.items():
        params = dict(node_spec.params)
        if node_spec.type in STATEFUL_PROTOTYPE_TYPES:
            params = validate_or_add_input_shape(
                params,
                node_inputs.get((node_name, "input")),
                node_name=node_name,
                node_type=node_spec.type,
            )
        nodes[node_name] = node_spec.model_copy(update={"params": params})

        if node_name in subgraphs:
            nested_inputs = {
                (node, port): node_inputs[(node_name, graph_port)]
                for graph_port, (node, port) in subgraphs[node_name].input_bindings.items()
                if (node_name, graph_port) in node_inputs
            }
            normalized_subgraphs[node_name] = normalize_stateful_prototypes(
                subgraphs[node_name],
                nested_inputs,
                component_registry=component_registry,
            )

    return spec.model_copy(
        update={
            "nodes": nodes,
            "subgraphs": normalized_subgraphs or None,
        }
    )
