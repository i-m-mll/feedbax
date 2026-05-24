from __future__ import annotations

from math import prod
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from feedbax.channel import Channel
from feedbax.components import (
    Gain,
    Sum,
    Multiply,
    Constant,
    Ravel,
    Ramp,
    Sine,
    Pulse,
    Saturation,
    DelayLine,
    Noise,
    Linear,
    MLP,
    Mux,
    GRU,
    LSTM,
    Spring,
    Damper,
)
from feedbax.graph_templates import (
    standard_network_subgraph,
)
from feedbax.filters import FirstOrderFilter
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen
from feedbax.graph import Component, Graph, Wire
from feedbax.intervene.intervene import (
    AddNoise,
    AddNoiseParams,
    ConstantInput,
    ConstantInputParams,
    Copy,
    CurlField,
    CurlFieldParams,
    FixedField,
    FixedFieldParams,
    NetworkClamp,
    NetworkConstantInput,
    NetworkIntervenorParams,
)
from feedbax.loss import CompositeLoss
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.nn import SimpleStagedNetwork
from feedbax.noise import Normal
from feedbax.penzai_component import (
    PENZAI_AVAILABLE,
    PenzaiSubgraph,
    build_penzai_subgraph,
)
from feedbax.state import CartesianState
from feedbax.task import DelayedReaches, SimpleReaches, Stabilization, TaskComponent
from feedbax.web.models.graph import (
    ComponentSpec,
    GraphSpec,
    WireSpec,
)


_HIDDEN_TYPES: dict[str, Callable[..., eqx.Module]] = {
    "GRUCell": eqx.nn.GRUCell,
    "LSTMCell": eqx.nn.LSTMCell,
    "Linear": eqx.nn.Linear,
    "GRU": eqx.nn.GRUCell,
    "LSTM": eqx.nn.LSTMCell,
}
_NONLINEARITIES: dict[str, Callable[[jax.Array], jax.Array]] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "softmax": jax.nn.softmax,
    "identity": lambda x: x,
}


def _resolve_nonlinearity(name: str | None) -> Callable[[jax.Array], jax.Array]:
    if not name:
        return _NONLINEARITIES["identity"]
    if name not in _NONLINEARITIES:
        raise ValueError(f"Unknown nonlinearity: {name!r}. Valid options: {list(_NONLINEARITIES)}")
    return _NONLINEARITIES[name]


def _nonlinearity_name(fn: Callable[[jax.Array], jax.Array]) -> str:
    for name, func in _NONLINEARITIES.items():
        if fn is func:
            return name
    name = getattr(fn, "__name__", "")
    return name if name in _NONLINEARITIES else "identity"


def _merge_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(params)
    return merged


def _lookup_defaults(component_registry: Any, name: str) -> dict[str, Any]:
    if component_registry is None:
        return {}
    if isinstance(component_registry, dict):
        meta = component_registry.get(name)
        if meta is None:
            return {}
        if hasattr(meta, "default_params"):
            return dict(getattr(meta, "default_params"))
        if isinstance(meta, Mapping):
            return dict(meta.get("default_params", {}))
        return {}
    if isinstance(component_registry, (list, tuple)):
        for meta in component_registry:
            if getattr(meta, "name", None) == name:
                return dict(getattr(meta, "default_params", {}))
    return {}


_STATEFUL_PROTOTYPE_TYPES = {"Channel", "DelayLine", "FirstOrderFilter"}


def _array_proto_from_shape(shape: Any) -> jax.Array | None:
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return jnp.zeros(tuple(int(dim) for dim in shape))
    except (TypeError, ValueError):
        return None


def _shape_from_proto(proto: Any) -> list[int] | None:
    leaves = jt.leaves(proto)
    if len(leaves) != 1 or not hasattr(leaves[0], "shape"):
        return None
    return [int(dim) for dim in leaves[0].shape]


def _proto_from_value(value: Any) -> Any:
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


def _explicit_proto(
    params: Mapping[str, Any],
    *,
    node_name: str,
    node_type: str,
) -> Any | None:
    if "input_shape" not in params:
        return None
    proto = _array_proto_from_shape(params.get("input_shape"))
    if proto is None:
        raise ValueError(
            f"{node_type} node {node_name!r} has invalid input_shape "
            f"{params.get('input_shape')!r}; expected a list of integer dimensions"
        )
    return proto


def _validate_or_add_input_shape(
    params: Mapping[str, Any],
    inferred_proto: Any | None,
    *,
    node_name: str,
    node_type: str,
) -> dict[str, Any]:
    next_params = dict(params)
    explicit_proto = _explicit_proto(next_params, node_name=node_name, node_type=node_type)
    if inferred_proto is not None and explicit_proto is not None:
        inferred_shape = _shape_from_proto(inferred_proto)
        explicit_shape = _shape_from_proto(explicit_proto)
        if inferred_shape is not None and explicit_shape != inferred_shape:
            raise ValueError(
                f"{node_type} node {node_name!r} input_shape {explicit_shape!r} "
                f"does not match inferred input prototype shape {inferred_shape!r} "
                "for port 'input'"
            )
    proto = inferred_proto if inferred_proto is not None else explicit_proto
    if proto is None:
        raise ValueError(
            f"{node_type} node {node_name!r} port 'input' requires an input prototype. "
            "Connect it to a source with a known output shape or provide input_shape."
        )
    shape = _shape_from_proto(proto)
    if shape is None:
        raise ValueError(
            f"{node_type} node {node_name!r} port 'input' prototype is not serializable "
            "as input_shape; only single-array prototypes are currently supported"
        )
    next_params["input_shape"] = shape
    return next_params


def _output_prototypes_for_node(
    node_name: str,
    node_spec: ComponentSpec,
    input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSpec],
) -> dict[str, Any]:
    params = node_spec.params
    node_type = node_spec.type

    if node_type == "Subgraph" or node_name in subgraphs or node_type == "Network":
        subgraph = subgraphs.get(node_name)
        if subgraph is None:
            return {}
        nested_inputs = {
            (node, port): input_prototypes[(node_name, graph_port)]
            for graph_port, (node, port) in subgraph.input_bindings.items()
            if (node_name, graph_port) in input_prototypes
        }
        normalized = _normalize_stateful_prototypes(subgraph, nested_inputs)
        return _bound_output_prototypes(
            normalized,
            subgraphs=dict(normalized.subgraphs or {}),
            input_prototypes=nested_inputs,
        )

    if node_type == "Constant":
        return {"output": _proto_from_value(params.get("value", 0.0))}
    if node_type in {"Ramp", "Sine", "Pulse"}:
        value = params.get("amplitude", params.get("slope", 1.0))
        return {"output": _proto_from_value(value)}
    if node_type == "Noise":
        return {"output": jnp.zeros(tuple(int(dim) for dim in params.get("shape", [1])))}
    if node_type in {"Gain", "Saturation"}:
        proto = input_prototypes.get((node_name, "input"))
        return {"output": proto} if proto is not None else {}
    if node_type in {"Sum", "Multiply"}:
        proto = input_prototypes.get((node_name, "a"))
        if proto is None:
            proto = input_prototypes.get((node_name, "b"))
        return {"output": proto} if proto is not None else {}
    if node_type == "Ravel":
        proto = input_prototypes.get((node_name, "input"))
        if proto is None:
            return {}
        leaves = jt.leaves(proto)
        if not leaves or any(not hasattr(leaf, "shape") for leaf in leaves):
            return {}
        return {"output": jnp.zeros((sum(prod(leaf.shape) for leaf in leaves),))}
    if node_type in _STATEFUL_PROTOTYPE_TYPES:
        proto = input_prototypes.get((node_name, "input"))
        if proto is None:
            proto = _explicit_proto(params, node_name=node_name, node_type=node_type)
        return {"output": proto} if proto is not None else {}
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
        parts = [
            input_prototypes[(node_name, port)]
            for port in node_spec.input_ports
            if (node_name, port) in input_prototypes
        ]
        if not parts:
            return {}
        shapes = [_shape_from_proto(part) for part in parts]
        if any(shape is None or len(shape) != 1 for shape in shapes):
            return {}
        return {"output": jnp.zeros((sum(shape[0] for shape in shapes if shape),))}
    if node_type in {"PointMass", "TwoLinkArm", "Arm6MuscleRigidTendon"}:
        effector = CartesianState()
        return {"effector": effector}
    return {}


def _bound_output_prototypes(
    spec: GraphSpec,
    *,
    subgraphs: Mapping[str, GraphSpec],
    input_prototypes: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    node_inputs = _infer_node_input_prototypes(spec, input_prototypes, subgraphs)
    outputs: dict[str, Any] = {}
    for graph_port, (node_name, node_port) in spec.output_bindings.items():
        node_spec = spec.nodes.get(node_name)
        if node_spec is None:
            continue
        node_outputs = _output_prototypes_for_node(node_name, node_spec, node_inputs, subgraphs)
        if node_port in node_outputs:
            outputs[graph_port] = node_outputs[node_port]
    return outputs


def _infer_node_input_prototypes(
    spec: GraphSpec,
    external_input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSpec],
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
            init_proto = _array_proto_from_shape(initializer.get("shape"))
        elif initializer.get("kind") == "constant" and "value" in initializer:
            init_proto = _proto_from_value(initializer["value"])
        if init_proto is not None:
            input_prototypes[(wire.target_node, wire.target_port)] = init_proto

    for _ in range(max(1, len(spec.nodes) + len(spec.wires) + 1)):
        changed = False
        for wire in spec.wires:
            source_spec = spec.nodes.get(wire.source_node)
            if source_spec is None:
                continue
            outputs = _output_prototypes_for_node(
                wire.source_node,
                source_spec,
                input_prototypes,
                subgraphs,
            )
            proto = outputs.get(wire.source_port)
            if proto is None:
                continue
            key = (wire.target_node, wire.target_port)
            current_shape = _shape_from_proto(input_prototypes[key]) if key in input_prototypes else None
            next_shape = _shape_from_proto(proto)
            if key not in input_prototypes:
                input_prototypes[key] = proto
                changed = True
            elif current_shape is not None and next_shape is not None and current_shape != next_shape:
                raise ValueError(
                    f"Graph wiring shape mismatch at {wire.target_node}.{wire.target_port}: "
                    f"existing prototype shape {current_shape!r}, "
                    f"{wire.source_node}.{wire.source_port} provides {next_shape!r}"
                )
        if not changed:
            break
    return input_prototypes


def _normalize_stateful_prototypes(
    spec: GraphSpec,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> GraphSpec:
    subgraphs = dict(spec.subgraphs or {})
    node_inputs = _infer_node_input_prototypes(spec, input_prototypes or {}, subgraphs)
    nodes: dict[str, ComponentSpec] = {}
    normalized_subgraphs: dict[str, GraphSpec] = {}

    for node_name, node_spec in spec.nodes.items():
        params = dict(node_spec.params)
        if node_spec.type in _STATEFUL_PROTOTYPE_TYPES:
            params = _validate_or_add_input_shape(
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
            normalized_subgraphs[node_name] = _normalize_stateful_prototypes(
                subgraphs[node_name],
                nested_inputs,
            )

    return spec.model_copy(
        update={
            "nodes": nodes,
            "subgraphs": normalized_subgraphs or None,
        }
    )


def _standard_network_subgraph_from_params(params: Mapping[str, Any]) -> GraphSpec | None:
    try:
        input_size = int(params["input_size"])
        hidden_size = int(params["hidden_size"])
        out_size_raw = params.get("out_size", params.get("output_size"))
        if out_size_raw is None:
            return None
        out_size = int(out_size_raw)
    except (KeyError, TypeError, ValueError):
        return None
    hidden_type = str(params.get("hidden_type", "GRUCell"))
    cell_type = "LSTM" if hidden_type in {"LSTM", "LSTMCell"} else "GRU"
    return standard_network_subgraph(
        input_size=input_size,
        hidden_size=hidden_size,
        out_size=out_size,
        cell_type=cell_type,
        out_nonlinearity=str(params.get("out_nonlinearity", "identity")),
        description="Migrated legacy Network subgraph",
    )


def _migrate_spec(spec: GraphSpec) -> GraphSpec:
    nodes: dict[str, ComponentSpec] = {}
    for node_id, node_spec in spec.nodes.items():
        next_type = node_spec.type
        if next_type == "SimpleStagedNetwork":
            next_type = "Network"
        if next_type == "FeedbackChannel":
            next_type = "Channel"
        if next_type == "PenzaiSubgraph":
            next_type = "PenzaiAdapter"
        params = dict(node_spec.params)
        if next_type == "Network" and "output_size" in params and "out_size" not in params:
            params["out_size"] = params.get("output_size")
        input_ports = list(node_spec.input_ports)
        if next_type == "Network":
            input_ports = ["input" if port == "target" else port for port in input_ports]
        nodes[node_id] = ComponentSpec(
            type=next_type,
            params=params,
            input_ports=input_ports,
            output_ports=list(node_spec.output_ports),
        )

    def _rename_port(node_name: str, port: str) -> str:
        node = nodes.get(node_name)
        if node and node.type == "Network" and port == "target":
            return "input"
        return port

    wires = [
        WireSpec(
            source_node=wire.source_node,
            source_port=_rename_port(wire.source_node, wire.source_port),
            target_node=wire.target_node,
            target_port=_rename_port(wire.target_node, wire.target_port),
            temporality=wire.temporality,
            recurrent_initializer=wire.recurrent_initializer,
        )
        for wire in spec.wires
    ]

    input_ports = ["input" if port == "target" else port for port in spec.input_ports]
    input_bindings = {
        ("input" if name == "target" else name): (
            node,
            _rename_port(node, port),
        )
        for name, (node, port) in spec.input_bindings.items()
    }

    subgraphs = (
        {node_id: _migrate_spec(subgraph) for node_id, subgraph in spec.subgraphs.items()}
        if spec.subgraphs
        else {}
    )
    for node_id, node_spec in nodes.items():
        if node_spec.type != "Network" or node_id in subgraphs:
            continue
        subgraph = _standard_network_subgraph_from_params(node_spec.params)
        if subgraph is not None:
            subgraphs[node_id] = subgraph
    user_ports = dict(spec.user_ports) if spec.user_ports else None
    taps = list(spec.taps) if spec.taps else None

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=input_ports,
        output_ports=list(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=dict(spec.output_bindings),
        subgraphs=subgraphs or None,
        barnacles=spec.barnacles,
        user_ports=user_ports,
        taps=taps,
        retained_observables=spec.retained_observables,
        metadata=spec.metadata,
    )


def graph_to_spec(graph: Any) -> GraphSpec:
    """Serialize a Graph-like object to GraphSpec."""
    if not isinstance(graph, Graph):
        raise TypeError("graph_to_spec requires feedbax.graph.Graph")

    nodes: dict[str, ComponentSpec] = {}
    subgraphs: dict[str, GraphSpec] = {}

    def _to_native(value: Any):
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    for name, component in graph.nodes.items():
        if isinstance(component, Graph):
            subgraphs[name] = graph_to_spec(component)
            nodes[name] = ComponentSpec(
                type="Subgraph",
                params={},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, SimpleStagedNetwork):
            hidden_type_name = type(component.hidden).__name__
            cell_type = "LSTM" if hidden_type_name == "LSTMCell" else "GRU"
            out_nonlinearity = _nonlinearity_name(component.out_nonlinearity)
            subgraphs[name] = standard_network_subgraph(
                input_size=component.input_size,
                hidden_size=component.hidden_size,
                out_size=component.out_size,
                cell_type=cell_type,
                out_nonlinearity=out_nonlinearity,
                name=f"{name} internals",
                description="Auto-generated Network subgraph",
            )

            params = {
                "input_size": component.input_size,
                "hidden_size": component.hidden_size,
                "out_size": component.out_size,
                "hidden_type": hidden_type_name,
                "hidden_nonlinearity": _nonlinearity_name(component.hidden_nonlinearity),
                "out_nonlinearity": out_nonlinearity,
                "hidden_noise_std": component.hidden_noise_std or 0.0,
                "encoding_size": component.encoding_size or 0,
            }
            nodes[name] = ComponentSpec(
                type="Network",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, Gain):
            nodes[name] = ComponentSpec(
                type="Gain",
                params={"gain": component.gain},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Sum):
            nodes[name] = ComponentSpec(
                type="Sum",
                params={},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Multiply):
            nodes[name] = ComponentSpec(
                type="Multiply",
                params={},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Constant):
            value = jt.map(_to_native, component.value)
            nodes[name] = ComponentSpec(
                type="Constant",
                params={"value": value},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Ravel):
            nodes[name] = ComponentSpec(
                type="Ravel",
                params={},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Ramp):
            slope = jt.map(_to_native, component.slope)
            intercept = jt.map(_to_native, component.intercept)
            nodes[name] = ComponentSpec(
                type="Ramp",
                params={"slope": slope, "intercept": intercept, "dt": component.dt},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Sine):
            amplitude = jt.map(_to_native, component.amplitude)
            offset = jt.map(_to_native, component.offset)
            nodes[name] = ComponentSpec(
                type="Sine",
                params={
                    "amplitude": amplitude,
                    "frequency": component.frequency,
                    "phase": component.phase,
                    "offset": offset,
                    "dt": component.dt,
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Pulse):
            amplitude = jt.map(_to_native, component.amplitude)
            offset = jt.map(_to_native, component.offset)
            nodes[name] = ComponentSpec(
                type="Pulse",
                params={
                    "amplitude": amplitude,
                    "period": component.period,
                    "duty_cycle": component.duty_cycle,
                    "offset": offset,
                    "dt": component.dt,
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Noise):
            nodes[name] = ComponentSpec(
                type="Noise",
                params={
                    "mean": component.mean,
                    "std": component.std,
                    "shape": list(component.shape),
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Saturation):
            nodes[name] = ComponentSpec(
                type="Saturation",
                params={"min_val": component.min_val, "max_val": component.max_val},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, DelayLine):
            input_shape = _shape_from_proto(component.input_proto)
            params = {"delay": component.delay, "init_value": component.init_value}
            if input_shape is not None:
                params["input_shape"] = input_shape
            nodes[name] = ComponentSpec(
                type="DelayLine",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Linear):
            nodes[name] = ComponentSpec(
                type="Linear",
                params={
                    "input_size": component.input_size,
                    "output_size": component.output_size,
                    "use_bias": component.use_bias,
                    "activation": component.activation_name,
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Mux):
            nodes[name] = ComponentSpec(
                type="Mux",
                params={"n_inputs": component.n_inputs},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, MLP):
            nodes[name] = ComponentSpec(
                type="MLP",
                params={
                    "input_size": component.input_size,
                    "output_size": component.output_size,
                    "hidden_sizes": list(component.hidden_sizes),
                    "activation": component.activation_name,
                    "final_activation": component.final_activation_name,
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, GRU):
            nodes[name] = ComponentSpec(
                type="GRU",
                params={"input_size": component.input_size, "hidden_size": component.hidden_size},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, LSTM):
            nodes[name] = ComponentSpec(
                type="LSTM",
                params={"input_size": component.input_size, "hidden_size": component.hidden_size},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Spring):
            nodes[name] = ComponentSpec(
                type="Spring",
                params={"stiffness": component.stiffness},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, Damper):
            nodes[name] = ComponentSpec(
                type="Damper",
                params={"damping": component.damping},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, ReluMuscle):
            nodes[name] = ComponentSpec(
                type="ReluMuscle",
                params={
                    "max_isometric_force": component.max_isometric_force,
                    "tau_activation": component.tau_activation,
                    "tau_deactivation": component.tau_deactivation,
                    "min_activation": component.min_activation,
                    "dt": component.dt,
                    "initial_activation": float(component._initial_state),
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue
        if isinstance(component, RigidTendonHillMuscleThelen):
            nodes[name] = ComponentSpec(
                type="RigidTendonHillMuscleThelen",
                params={
                    "max_isometric_force": component.max_isometric_force,
                    "optimal_muscle_length": component.optimal_muscle_length,
                    "tendon_slack_length": component.tendon_slack_length,
                    "vmax_factor": component.vmax_factor,
                    "min_activation": component.min_activation,
                    "tau_activation": component.tau_activation,
                    "tau_deactivation": component.tau_deactivation,
                    "dt": component.dt,
                    "initial_activation": float(component._initial_state),
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, Mechanics):
            plant_type = "Unknown"
            if isinstance(component.plant, DirectForceInput):
                skeleton = component.plant.skeleton
                if isinstance(skeleton, TwoLinkArm):
                    plant_type = "TwoLinkArm"
                elif isinstance(skeleton, PointMass):
                    plant_type = "PointMass"
                else:
                    plant_type = type(skeleton).__name__
            elif isinstance(component.plant, AnalyticalMusculoskeletalPlant):
                # Bug: 1005721 — AnalyticalMusculoskeletalPlant serialization type mapping
                plant_type = "Arm6MuscleRigidTendon"
            params = {
                "dt": component.dt,
            }
            nodes[name] = ComponentSpec(
                type=plant_type,
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, Channel):
            noise_std = 0.0
            if isinstance(component.noise_func, Normal):
                noise_std = float(component.noise_func.std)
            params = {
                "delay": component.delay,
                "noise_std": noise_std,
                "add_noise": component.add_noise,
            }
            input_proto_leaves = jt.leaves(component.input_proto)
            if len(input_proto_leaves) == 1 and hasattr(input_proto_leaves[0], "shape"):
                params["input_shape"] = [int(dim) for dim in input_proto_leaves[0].shape]
            nodes[name] = ComponentSpec(
                type="Channel",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, FirstOrderFilter):
            params = {
                "tau_rise": component.tau_rise,
                "tau_decay": component.tau_decay,
                "dt": component.dt,
                "init_value": component.init_value,
            }
            input_shape = _shape_from_proto(component.input_proto)
            if input_shape is not None:
                params["input_shape"] = input_shape
            nodes[name] = ComponentSpec(
                type="FirstOrderFilter",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, CurlField):
            params = {
                "scale": component._initial_state.scale,
                "amplitude": component._initial_state.amplitude,
                "active": component._initial_state.active,
            }
            nodes[name] = ComponentSpec(
                type="CurlField",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, FixedField):
            params = {
                "scale": component._initial_state.scale,
                "amplitude": component._initial_state.amplitude,
                "field": jnp.asarray(component._initial_state.field).tolist(),
                "active": component._initial_state.active,
            }
            nodes[name] = ComponentSpec(
                type="FixedField",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, AddNoise):
            params = {
                "scale": component._initial_state.scale,
                "active": component._initial_state.active,
            }
            nodes[name] = ComponentSpec(
                type="AddNoise",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, NetworkClamp):
            params = {
                "scale": component._initial_state.scale,
                "active": component._initial_state.active,
            }
            nodes[name] = ComponentSpec(
                type="NetworkClamp",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, NetworkConstantInput):
            params = {
                "scale": component._initial_state.scale,
                "active": component._initial_state.active,
            }
            nodes[name] = ComponentSpec(
                type="NetworkConstantInput",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, ConstantInput):
            params = {
                "scale": component._initial_state.scale,
                "active": component._initial_state.active,
            }
            nodes[name] = ComponentSpec(
                type="ConstantInput",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, Copy):
            nodes[name] = ComponentSpec(
                type="Copy",
                params={},
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, PenzaiSubgraph):
            # Persist builder_name for round-tripping through spec_to_graph.
            # Legacy objects without builder_name get "__unknown__" so deserialization
            # produces a clear error rather than a crash.  Bug: bc551f7
            builder_name = getattr(component, "builder_name", None) or "__unknown__"
            nodes[name] = ComponentSpec(
                type="PenzaiAdapter",
                params={
                    "builder_name": builder_name,
                    "input_port": component.input_ports[0] if component.input_ports else "input",
                    "output_port": component.output_ports[0]
                    if component.output_ports
                    else "output",
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, TaskComponent):
            task = component.task
            params: dict[str, Any] = {
                "n_steps": task.n_steps,
            }
            task_type = type(task).__name__
            if isinstance(task, SimpleReaches):
                params.update(
                    {
                        "workspace": jnp.asarray(task.workspace).tolist(),
                        "eval_n_directions": task.eval_n_directions,
                        "eval_reach_length": task.eval_reach_length,
                        "eval_grid_n": task.eval_grid_n,
                    }
                )
            elif isinstance(task, DelayedReaches):
                params.update(
                    {
                        "workspace": jnp.asarray(task.workspace).tolist(),
                        "train_endpoint_mode": task.train_endpoint_mode,
                        "epoch_len_ranges": [list(item) for item in task.epoch_len_ranges],
                        "target_on_epochs": jnp.asarray(task.target_on_epochs).tolist(),
                        "hold_epochs": jnp.asarray(task.hold_epochs).tolist(),
                        "move_epochs": jnp.asarray(task.move_epochs).tolist(),
                        "p_catch_trial": task.p_catch_trial,
                        "eval_n_directions": task.eval_n_directions,
                        "eval_reach_length": task.eval_reach_length,
                        "eval_grid_n": task.eval_grid_n,
                    }
                )
            elif isinstance(task, Stabilization):
                params.update({"workspace": jnp.asarray(task.workspace).tolist()})
            nodes[name] = ComponentSpec(
                type=task_type,
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        nodes[name] = ComponentSpec(
            type=type(component).__name__,
            params={},
            input_ports=list(component.input_ports),
            output_ports=list(component.output_ports),
        )

    return GraphSpec(
        nodes=nodes,
        wires=[
            WireSpec(
                source_node=wire.source_node,
                source_port=wire.source_port,
                target_node=wire.target_node,
                target_port=wire.target_port,
                temporality=wire.temporality,
                recurrent_initializer=wire.recurrent_initializer,
            )
            for wire in graph.wires
        ],
        input_ports=list(graph.input_ports),
        output_ports=list(graph.output_ports),
        input_bindings=dict(graph.input_bindings),
        output_bindings=dict(graph.output_bindings),
        subgraphs=subgraphs or None,
        retained_observables=getattr(graph, "retained_observables", None),
        metadata=None,
    )


def _build_network(params: Mapping[str, Any]) -> SimpleStagedNetwork:
    hidden_type = _HIDDEN_TYPES.get(str(params.get("hidden_type", "GRUCell"))) or eqx.nn.GRUCell
    hidden_nonlinearity = _resolve_nonlinearity(str(params.get("hidden_nonlinearity", "tanh")))
    out_nonlinearity = _resolve_nonlinearity(str(params.get("out_nonlinearity", "tanh")))
    encoding_size = int(params.get("encoding_size", 0) or 0)
    encoding_size = encoding_size if encoding_size > 0 else None
    out_size = params.get("out_size", params.get("output_size"))
    out_size = int(out_size) if out_size not in (None, "") else None
    if out_size is not None and out_size <= 0:
        out_size = None
    hidden_noise_std = params.get("hidden_noise_std", 0.0)
    if hidden_noise_std in (None, 0, 0.0):
        hidden_noise_std = None
    return SimpleStagedNetwork(
        input_size=int(params.get("input_size", 0)),
        hidden_size=int(params.get("hidden_size", 0)),
        out_size=out_size,
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        hidden_nonlinearity=hidden_nonlinearity,
        out_nonlinearity=out_nonlinearity,
        hidden_noise_std=hidden_noise_std,
        key=jr.PRNGKey(0),
    )


def _build_mechanics(params: Mapping[str, Any]) -> Mechanics:
    plant_type = params.get("plant_type", "TwoLinkArm")
    if plant_type == "TwoLinkArm":
        plant = DirectForceInput(TwoLinkArm())
    elif plant_type == "PointMass":
        plant = DirectForceInput(PointMass(mass=float(params.get("mass", 1.0))))
    else:
        raise ValueError(f"Unsupported plant_type '{plant_type}'")
    return Mechanics(plant=plant, dt=float(params.get("dt", 0.01)))


def _build_channel(params: Mapping[str, Any]) -> Channel:
    delay = int(params.get("delay", 0))
    add_noise = bool(params.get("add_noise", True))
    noise_std = params.get("noise_std", 0.0)
    noise_func = None
    if add_noise and noise_std not in (None, 0, 0.0):
        noise_func = Normal(std=float(noise_std))
    input_proto = None
    input_shape = params.get("input_shape")
    if isinstance(input_shape, (list, tuple)):
        input_proto = jnp.zeros(tuple(int(dim) for dim in input_shape))
    return Channel(
        delay=delay,
        noise_func=noise_func,
        add_noise=add_noise,
        input_proto=input_proto,
    )


def _build_filter(params: Mapping[str, Any]) -> FirstOrderFilter:
    input_proto = _array_proto_from_shape(params.get("input_shape"))
    return FirstOrderFilter(
        tau_rise=float(params.get("tau_rise", 0.05)),
        tau_decay=float(params.get("tau_decay", 0.05)),
        dt=float(params.get("dt", 0.001)),
        input_proto=input_proto,
        init_value=float(params.get("init_value", 0.0)),
    )


def _build_task_component(task_type: str, params: Mapping[str, Any]) -> TaskComponent:
    loss_func = CompositeLoss({})
    if task_type == "SimpleReaches":
        task = SimpleReaches(
            loss_func=loss_func,
            n_steps=int(params.get("n_steps", 200)),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
            eval_n_directions=int(params.get("eval_n_directions", 7)),
            eval_reach_length=float(params.get("eval_reach_length", 0.5)),
            eval_grid_n=int(params.get("eval_grid_n", 1)),
        )
    elif task_type == "DelayedReaches":
        task = DelayedReaches(
            loss_func=loss_func,
            n_steps=int(params.get("n_steps", 140)),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
            train_endpoint_mode=str(params.get("train_endpoint_mode", "workspace")),
            epoch_len_ranges=tuple(
                tuple(int(value) for value in item)
                for item in params.get("epoch_len_ranges", [[5, 15], [10, 20]])
            ),
            target_on_epochs=tuple(int(value) for value in params.get("target_on_epochs", [1, 2])),
            hold_epochs=tuple(int(value) for value in params.get("hold_epochs", [0, 1])),
            move_epochs=tuple(int(value) for value in params.get("move_epochs", [2])),
            p_catch_trial=float(params.get("p_catch_trial", 0.5)),
            eval_n_directions=int(params.get("eval_n_directions", 7)),
            eval_reach_length=float(params.get("eval_reach_length", 0.5)),
            eval_grid_n=int(params.get("eval_grid_n", 1)),
        )
    elif task_type == "Stabilization":
        task = Stabilization(
            loss_func=loss_func,
            n_steps=int(params.get("n_steps", 200)),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
        )
    else:
        raise ValueError(f"Unsupported task type '{task_type}'")

    mode = params.get("mode", "open_loop")
    if mode != "open_loop":
        raise ValueError("Only open_loop TaskComponent is supported from GraphSpec")

    trial_spec = task.get_train_trial_with_intervenor_params(key=jr.PRNGKey(0))
    return TaskComponent(task=task, trial_spec=trial_spec, mode="open_loop")


def _build_gain(params: Mapping[str, Any]) -> Gain:
    return Gain(gain=float(params.get("gain", 1.0)))


def _build_sum(params: Mapping[str, Any]) -> Sum:
    return Sum()


def _build_multiply(params: Mapping[str, Any]) -> Multiply:
    return Multiply()


def _build_constant(params: Mapping[str, Any]) -> Constant:
    return Constant(value=params.get("value", 0.0))


def _build_ramp(params: Mapping[str, Any]) -> Ramp:
    return Ramp(
        slope=params.get("slope", 1.0),
        intercept=params.get("intercept", 0.0),
        dt=float(params.get("dt", 0.01)),
    )


def _build_sine(params: Mapping[str, Any]) -> Sine:
    return Sine(
        amplitude=params.get("amplitude", 1.0),
        frequency=float(params.get("frequency", 1.0)),
        phase=float(params.get("phase", 0.0)),
        offset=params.get("offset", 0.0),
        dt=float(params.get("dt", 0.01)),
    )


def _build_pulse(params: Mapping[str, Any]) -> Pulse:
    return Pulse(
        amplitude=params.get("amplitude", 1.0),
        period=float(params.get("period", 1.0)),
        duty_cycle=float(params.get("duty_cycle", 0.5)),
        offset=params.get("offset", 0.0),
        dt=float(params.get("dt", 0.01)),
    )


def _build_noise(params: Mapping[str, Any]) -> Noise:
    shape = params.get("shape", [1])
    if not isinstance(shape, (list, tuple)):
        shape = [int(shape)]
    return Noise(
        mean=float(params.get("mean", 0.0)),
        std=float(params.get("std", 1.0)),
        shape=shape,
    )


def _build_saturation(params: Mapping[str, Any]) -> Saturation:
    return Saturation(
        min_val=float(params.get("min_val", -1.0)),
        max_val=float(params.get("max_val", 1.0)),
    )


def _build_delay_line(params: Mapping[str, Any]) -> DelayLine:
    input_proto = _array_proto_from_shape(params.get("input_shape"))
    return DelayLine(
        delay=int(params.get("delay", 1)),
        init_value=float(params.get("init_value", 0.0)),
        input_proto=input_proto,
    )


def _build_mlp(params: Mapping[str, Any]) -> MLP:
    hidden_sizes = params.get("hidden_sizes", [64])
    if not isinstance(hidden_sizes, (list, tuple)):
        hidden_sizes = [int(hidden_sizes)]
    return MLP(
        input_size=int(params.get("input_size", 1)),
        output_size=int(params.get("output_size", 1)),
        hidden_sizes=hidden_sizes,
        activation=str(params.get("activation", "relu")),
        final_activation=str(params.get("final_activation", "identity")),
        key=jr.PRNGKey(0),
    )


def _build_linear(params: Mapping[str, Any]) -> Linear:
    return Linear(
        input_size=int(params.get("input_size", 1)),
        output_size=int(params.get("output_size", 1)),
        use_bias=bool(params.get("use_bias", True)),
        activation=str(params.get("activation", "identity")),
        key=jr.PRNGKey(0),
    )


def _build_mux(params: Mapping[str, Any]) -> Mux:
    return Mux(n_inputs=int(params.get("n_inputs", 2)))


def _build_gru(params: Mapping[str, Any]) -> GRU:
    return GRU(
        input_size=int(params.get("input_size", 1)),
        hidden_size=int(params.get("hidden_size", 1)),
        key=jr.PRNGKey(0),
    )


def _build_lstm(params: Mapping[str, Any]) -> LSTM:
    return LSTM(
        input_size=int(params.get("input_size", 1)),
        hidden_size=int(params.get("hidden_size", 1)),
        key=jr.PRNGKey(0),
    )


def _build_spring(params: Mapping[str, Any]) -> Spring:
    return Spring(stiffness=float(params.get("stiffness", 1.0)))


def _build_damper(params: Mapping[str, Any]) -> Damper:
    return Damper(damping=float(params.get("damping", 1.0)))


def _build_relu_muscle(params: Mapping[str, Any]) -> ReluMuscle:
    return ReluMuscle(
        max_isometric_force=float(params.get("max_isometric_force", 500.0)),
        tau_activation=float(params.get("tau_activation", 0.015)),
        tau_deactivation=float(params.get("tau_deactivation", 0.05)),
        min_activation=float(params.get("min_activation", 0.0)),
        dt=float(params.get("dt", 0.01)),
        initial_activation=float(params.get("initial_activation", 0.0)),
    )


def _build_rigid_tendon_hill_muscle_thelen(
    params: Mapping[str, Any],
) -> RigidTendonHillMuscleThelen:
    return RigidTendonHillMuscleThelen(
        max_isometric_force=float(params.get("max_isometric_force", 500.0)),
        optimal_muscle_length=float(params.get("optimal_muscle_length", 0.1)),
        tendon_slack_length=float(params.get("tendon_slack_length", 0.1)),
        vmax_factor=float(params.get("vmax_factor", 10.0)),
        min_activation=float(params.get("min_activation", 0.001)),
        tau_activation=float(params.get("tau_activation", 0.015)),
        tau_deactivation=float(params.get("tau_deactivation", 0.05)),
        dt=float(params.get("dt", 0.01)),
        initial_activation=float(params.get("initial_activation", 0.001)),
    )


def _params_from_network_subgraph(subgraph: GraphSpec, outer_params: dict) -> dict:
    cell_node_name = next(
        (name for name, node in subgraph.nodes.items() if node.type in ("GRU", "LSTM")),
        None,
    )
    if cell_node_name is None:
        raise ValueError(
            "Network subgraph is malformed: missing cell node. "
            "Re-open the Network node in Studio to regenerate it."
        )
    cell_node = subgraph.nodes[cell_node_name]

    # Use output_bindings["output"] if present to find readout node
    readout_node_name = None
    if subgraph.output_bindings and "output" in subgraph.output_bindings:
        readout_node_name = subgraph.output_bindings["output"][0]

    if readout_node_name is None or readout_node_name not in subgraph.nodes:
        # Fallback to scanning for Linear node
        readout_node_name = next(
            (name for name, node in subgraph.nodes.items() if node.type == "Linear"),
            None,
        )

    if readout_node_name is None:
        raise ValueError(
            "Network subgraph is malformed: missing readout node. "
            "Re-open the Network node in Studio to regenerate it."
        )
    readout_node = subgraph.nodes[readout_node_name]

    params = dict(outer_params)
    params.update(
        {
            "hidden_type": cell_node.type,
            "hidden_size": cell_node.params.get("hidden_size"),
            "out_size": readout_node.params.get("output_size"),
            "out_nonlinearity": readout_node.params.get("activation"),
        }
    )
    return params


def spec_to_graph(
    spec: GraphSpec,
    component_registry: dict,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> Graph:
    """Instantiate a Graph-like object from GraphSpec."""
    spec = _migrate_spec(spec)
    spec = _normalize_stateful_prototypes(spec, input_prototypes)

    nodes: dict[str, Component] = {}
    for node_name, node_spec in spec.nodes.items():
        defaults = _lookup_defaults(component_registry, node_spec.type)
        params = _merge_params(node_spec.params, defaults)

        if node_spec.type == "Subgraph":
            if not spec.subgraphs or node_name not in spec.subgraphs:
                raise ValueError(f"Missing subgraph spec for '{node_name}'")
            nodes[node_name] = spec_to_graph(spec.subgraphs[node_name], component_registry)
            continue
        if node_spec.type == "Network":
            subgraph = (spec.subgraphs or {}).get(node_name)
            if subgraph is None:
                raise ValueError(
                    f"Network node {node_name!r} has no subgraph. "
                    "Open it in Studio to generate the internal architecture, then save again."
                )
            nodes[node_name] = spec_to_graph(subgraph, component_registry)
            continue
        if spec.subgraphs and node_name in spec.subgraphs:
            nodes[node_name] = spec_to_graph(spec.subgraphs[node_name], component_registry)
            continue
        if node_spec.type == "Gain":
            nodes[node_name] = _build_gain(params)
            continue
        if node_spec.type == "Sum":
            nodes[node_name] = _build_sum(params)
            continue
        if node_spec.type == "Multiply":
            nodes[node_name] = _build_multiply(params)
            continue
        if node_spec.type == "Constant":
            nodes[node_name] = _build_constant(params)
            continue
        if node_spec.type == "Ravel":
            nodes[node_name] = Ravel()
            continue
        if node_spec.type == "Ramp":
            nodes[node_name] = _build_ramp(params)
            continue
        if node_spec.type == "Sine":
            nodes[node_name] = _build_sine(params)
            continue
        if node_spec.type == "Pulse":
            nodes[node_name] = _build_pulse(params)
            continue
        if node_spec.type == "Noise":
            nodes[node_name] = _build_noise(params)
            continue
        if node_spec.type == "Saturation":
            nodes[node_name] = _build_saturation(params)
            continue
        if node_spec.type == "DelayLine":
            nodes[node_name] = _build_delay_line(params)
            continue
        if node_spec.type == "Linear":
            nodes[node_name] = _build_linear(params)
            continue
        if node_spec.type == "MLP":
            nodes[node_name] = _build_mlp(params)
            continue
        if node_spec.type == "Mux":
            nodes[node_name] = _build_mux(params)
            continue
        if node_spec.type == "GRU":
            nodes[node_name] = _build_gru(params)
            continue
        if node_spec.type == "LSTM":
            nodes[node_name] = _build_lstm(params)
            continue
        if node_spec.type == "Spring":
            nodes[node_name] = _build_spring(params)
            continue
        if node_spec.type == "Damper":
            nodes[node_name] = _build_damper(params)
            continue
        if node_spec.type == "ReluMuscle":
            nodes[node_name] = _build_relu_muscle(params)
            continue
        if node_spec.type == "RigidTendonHillMuscleThelen":
            nodes[node_name] = _build_rigid_tendon_hill_muscle_thelen(params)
            continue
        if node_spec.type == "MomentArmProjection":
            # Bug: 1005721 — MomentArmProjection is a display-only node type
            # used in composite template graphs. No Python Component class
            # exists yet; raise a clear error instead of the generic
            # "Unsupported component type" crash.
            raise NotImplementedError(
                f"MomentArmProjection node '{node_name}' has no Python builder yet. "
                "It is a display-only abstraction used in composite subgraph templates."
            )
        if node_spec.type == "RadialForceProjection":
            # Bug: 1005721 — same as MomentArmProjection above.
            raise NotImplementedError(
                f"RadialForceProjection node '{node_name}' has no Python builder yet. "
                "It is a display-only abstraction used in composite subgraph templates."
            )
        if node_spec.type == "Arm6MuscleRigidTendon":
            # Bug: 1005721 — AnalyticalMusculoskeletalPlant deserialization.
            # This plant type is used in the worker for musculoskeletal dynamics,
            # but cannot be fully instantiated from spec alone (requires body
            # presets, muscle topology, etc.). The studio uses this for display
            # and worker dispatch, not for local graph instantiation.
            raise NotImplementedError(
                f"Arm6MuscleRigidTendon node '{node_name}' requires musculoskeletal "
                "plant data (body presets, muscle topology) not stored in GraphSpec. "
                "This node type is supported in the training worker but not in "
                "local graph instantiation. For testing, use TwoLinkArm instead."
            )
        if node_spec.type in {"TwoLinkArm", "PointMass"}:
            next_params = dict(params)
            next_params["plant_type"] = node_spec.type
            nodes[node_name] = _build_mechanics(next_params)
            continue
        if node_spec.type == "Channel":
            nodes[node_name] = _build_channel(params)
            continue
        if node_spec.type == "FirstOrderFilter":
            nodes[node_name] = _build_filter(params)
            continue
        if node_spec.type == "CurlField":
            nodes[node_name] = CurlField(
                params=CurlFieldParams(
                    scale=float(params.get("scale", 1.0)),
                    amplitude=float(params.get("amplitude", 1.0)),
                    active=bool(params.get("active", False)),
                )
            )
            continue
        if node_spec.type == "FixedField":
            nodes[node_name] = FixedField(
                params=FixedFieldParams(
                    scale=float(params.get("scale", 1.0)),
                    amplitude=float(params.get("amplitude", 1.0)),
                    field=jnp.asarray(params.get("field", [0.0, 0.0])),
                    active=bool(params.get("active", False)),
                )
            )
            continue
        if node_spec.type == "AddNoise":
            nodes[node_name] = AddNoise(
                params=AddNoiseParams(
                    scale=float(params.get("scale", 1.0)),
                    active=bool(params.get("active", False)),
                )
            )
            continue
        if node_spec.type == "NetworkClamp":
            nodes[node_name] = NetworkClamp(
                params=NetworkIntervenorParams(
                    scale=float(params.get("scale", 1.0)),
                    active=bool(params.get("active", False)),
                )
            )
            continue
        if node_spec.type == "NetworkConstantInput":
            nodes[node_name] = NetworkConstantInput(
                params=NetworkIntervenorParams(
                    scale=float(params.get("scale", 1.0)),
                    active=bool(params.get("active", False)),
                )
            )
            continue
        if node_spec.type == "ConstantInput":
            nodes[node_name] = ConstantInput(
                params=ConstantInputParams(
                    scale=float(params.get("scale", 1.0)),
                    active=bool(params.get("active", False)),
                )
            )
            continue
        if node_spec.type == "Copy":
            nodes[node_name] = Copy()
            continue
        if node_spec.type in {"SimpleReaches", "DelayedReaches", "Stabilization"}:
            nodes[node_name] = _build_task_component(node_spec.type, params)
            continue
        if node_spec.type == "PenzaiAdapter":
            builder_name = str(params.get("builder_name", ""))
            if not builder_name:
                raise ValueError(
                    f"PenzaiAdapter node '{node_name}' requires 'builder_name' parameter"
                )
            if not PENZAI_AVAILABLE:
                raise ImportError(
                    "penzai is required to instantiate PenzaiAdapter. "
                    "Install with: pip install penzai"
                )
            # Build the PenzaiSubgraph using the registered builder
            builder_params = {
                k: v
                for k, v in params.items()
                if k not in ("builder_name", "input_port", "output_port")
            }
            nodes[node_name] = build_penzai_subgraph(
                builder_name=builder_name,
                params=builder_params,
                input_port=str(params.get("input_port", "input")),
                output_port=str(params.get("output_port", "output")),
            )
            continue

        raise ValueError(f"Unsupported component type '{node_spec.type}'")

    wires = tuple(
        Wire(
            wire.source_node,
            wire.source_port,
            wire.target_node,
            wire.target_port,
            wire.temporality,
            wire.recurrent_initializer,
        )
        for wire in spec.wires
        if wire.source_node in nodes and wire.target_node in nodes
    )

    input_bindings = {name: tuple(binding) for name, binding in spec.input_bindings.items()}
    output_bindings = {name: tuple(binding) for name, binding in spec.output_bindings.items()}

    return Graph(
        nodes=nodes,
        wires=wires,
        input_ports=tuple(spec.input_ports),
        output_ports=tuple(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
