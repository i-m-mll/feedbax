from __future__ import annotations

from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from feedbax.channel import Channel
from feedbax.filters import FirstOrderFilter
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
from feedbax.nn import SimpleStagedNetwork
from feedbax.noise import Normal
from feedbax.task import DelayedReaches, SimpleReaches, Stabilization, TaskComponent
from feedbax.web.models.graph import ComponentSpec, GraphSpec, WireSpec


_HIDDEN_TYPES: dict[str, Callable[..., eqx.Module]] = {
    "GRUCell": eqx.nn.GRUCell,
    "LSTMCell": eqx.nn.LSTMCell,
    "Linear": eqx.nn.Linear,
}
_NONLINEARITIES: dict[str, Callable[[jax.Array], jax.Array]] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "identity": lambda x: x,
}


def _resolve_nonlinearity(name: str | None) -> Callable[[jax.Array], jax.Array]:
    if not name:
        return _NONLINEARITIES["identity"]
    return _NONLINEARITIES.get(name, _NONLINEARITIES["identity"])


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


def _migrate_spec(spec: GraphSpec) -> GraphSpec:
    nodes: dict[str, ComponentSpec] = {}
    for node_id, node_spec in spec.nodes.items():
        next_type = node_spec.type
        if next_type == "SimpleStagedNetwork":
            next_type = "Network"
        if next_type == "FeedbackChannel":
            next_type = "Channel"
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
        else None
    )

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=input_ports,
        output_ports=list(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=dict(spec.output_bindings),
        subgraphs=subgraphs,
        metadata=spec.metadata,
    )


def graph_to_spec(graph: Any) -> GraphSpec:
    """Serialize a Graph-like object to GraphSpec."""
    if not isinstance(graph, Graph):
        raise TypeError("graph_to_spec requires feedbax.graph.Graph")

    nodes: dict[str, ComponentSpec] = {}
    subgraphs: dict[str, GraphSpec] = {}

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
            params = {
                "input_size": component.input_size,
                "hidden_size": component.hidden_size,
                "out_size": component.out_size,
                "hidden_type": type(component.hidden).__name__,
                "hidden_nonlinearity": _nonlinearity_name(component.hidden_nonlinearity),
                "out_nonlinearity": _nonlinearity_name(component.out_nonlinearity),
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
            params = {
                "plant_type": plant_type,
                "dt": component.dt,
            }
            nodes[name] = ComponentSpec(
                type="Mechanics",
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
                        "delay_steps": task.delay_steps,
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
            )
            for wire in graph.wires
        ],
        input_ports=list(graph.input_ports),
        output_ports=list(graph.output_ports),
        input_bindings=dict(graph.input_bindings),
        output_bindings=dict(graph.output_bindings),
        subgraphs=subgraphs or None,
        metadata=None,
    )


def _build_network(params: Mapping[str, Any]) -> SimpleStagedNetwork:
    hidden_type = _HIDDEN_TYPES.get(str(params.get("hidden_type", "GRUCell")), eqx.nn.GRUCell)
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
        plant = DirectForceInput(PointMass())
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
    return Channel(delay=delay, noise_func=noise_func, add_noise=add_noise)


def _build_filter(params: Mapping[str, Any]) -> FirstOrderFilter:
    return FirstOrderFilter(
        tau_rise=float(params.get("tau_rise", 0.05)),
        tau_decay=float(params.get("tau_decay", 0.05)),
        dt=float(params.get("dt", 0.001)),
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
            n_steps=int(params.get("n_steps", 240)),
            workspace=jnp.asarray(params.get("workspace", [[-1.0, -1.0], [1.0, 1.0]])),
            delay_steps=int(params.get("delay_steps", 40)),
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


def spec_to_graph(spec: GraphSpec, component_registry: dict) -> Graph:
    """Instantiate a Graph-like object from GraphSpec."""
    spec = _migrate_spec(spec)

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
            nodes[node_name] = _build_network(params)
            continue
        if node_spec.type == "Mechanics":
            nodes[node_name] = _build_mechanics(params)
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

        raise ValueError(f"Unsupported component type '{node_spec.type}'")

    wires = tuple(
        Wire(
            wire.source_node,
            wire.source_port,
            wire.target_node,
            wire.target_port,
        )
        for wire in spec.wires
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
