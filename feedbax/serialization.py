from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import jax.tree as jt

from feedbax.channel import Channel
from feedbax.components import (
    Constant,
    DelayLine,
    Damper,
    GRU,
    Gain,
    LSTM,
    Linear,
    MLP,
    Multiply,
    Mux,
    Noise,
    Pulse,
    Ramp,
    Ravel,
    Saturation,
    Sine,
    Spring,
    Sum,
)
from feedbax.filters import FirstOrderFilter
from feedbax.graph import Component, Graph, Wire
from feedbax.graph_templates import standard_network_subgraph
from feedbax.intervene.intervene import (
    AddNoise,
    ConstantInput,
    Copy,
    CurlField,
    FixedField,
    NetworkClamp,
    NetworkConstantInput,
)
from feedbax.mechanics.linear_state_space import LinearStateSpace
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.analytical_plant import AnalyticalMusculoskeletalPlant
from feedbax.nn import SimpleStagedNetwork
from feedbax.noise import Normal
from feedbax.penzai_component import PenzaiSubgraph
from feedbax.task import DelayedReaches, SimpleReaches, Stabilization, TaskComponent
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    WireSpec,
)
from feedbax.graph_channel_adapters import materialize_additive_channel_adapters
from feedbax.serialization_builders import build_component, nonlinearity_name
from feedbax.serialization_prototypes import (
    normalize_stateful_prototypes,
    prototypes_from_task_bindings,
    shape_from_proto,
)


SUPPORTED_GRAPH_SPEC_VERSIONS = frozenset({"1.0.0"})

__all__ = ["graph_to_spec", "prototypes_from_task_bindings", "spec_to_graph"]


def _merge_params(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    required_params: set[str] | None = None,
    node_name: str = "",
    node_type: str = "",
) -> dict[str, Any]:
    missing_required = sorted((required_params or set()) - set(params))
    if missing_required:
        raise ValueError(
            f"{node_type} node {node_name!r} is missing required parameter(s): "
            + ", ".join(repr(name) for name in missing_required)
        )
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
    if hasattr(component_registry, "get"):
        meta = component_registry.get(name)
        if meta is None:
            return {}
        if hasattr(meta, "default_params"):
            return dict(getattr(meta, "default_params"))
        return {}
    if isinstance(component_registry, (list, tuple)):
        for meta in component_registry:
            if getattr(meta, "name", None) == name:
                return dict(getattr(meta, "default_params", {}))
    return {}


def _lookup_required_params(component_registry: Any, name: str) -> set[str]:
    if component_registry is None:
        return set()

    def _from_meta(meta: Any) -> set[str]:
        schema = getattr(meta, "param_schema", None)
        if schema is None and isinstance(meta, Mapping):
            schema = meta.get("param_schema", [])
        required: set[str] = set()
        for item in schema or []:
            item_name = (
                getattr(item, "name", None) if not isinstance(item, Mapping) else item.get("name")
            )
            item_required = (
                getattr(item, "required", None)
                if not isinstance(item, Mapping)
                else item.get("required")
            )
            if item_name is not None and bool(item_required):
                required.add(str(item_name))
        return required

    if isinstance(component_registry, dict):
        meta = component_registry.get(name)
        return _from_meta(meta) if meta is not None else set()
    if hasattr(component_registry, "get"):
        meta = component_registry.get(name)
        return _from_meta(meta) if meta is not None else set()
    if isinstance(component_registry, (list, tuple)):
        for meta in component_registry:
            if getattr(meta, "name", None) == name:
                return _from_meta(meta)
    return set()


def _validate_supported_spec_versions(spec: GraphSpec, *, path: str = "graph") -> None:
    version = spec.metadata.version if spec.metadata is not None else None
    if version is not None and version not in SUPPORTED_GRAPH_SPEC_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_GRAPH_SPEC_VERSIONS))
        raise ValueError(
            f"Unsupported GraphSpec version {version!r} at {path}; supported versions: {supported}"
        )
    for node_name, subgraph in (spec.subgraphs or {}).items():
        _validate_supported_spec_versions(subgraph, path=f"{path}.subgraphs[{node_name!r}]")


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

    input_bindings = {
        name: (
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
    user_ports = dict(spec.user_ports) if spec.user_ports else None
    taps = list(spec.taps) if spec.taps else None

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=list(spec.input_ports),
        output_ports=list(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=dict(spec.output_bindings),
        subgraphs=subgraphs or None,
        barnacles=spec.barnacles,
        user_ports=user_ports,
        taps=taps,
        retained_observables=spec.retained_observables,
        additive_channel_adapters=list(spec.additive_channel_adapters),
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
            out_nonlinearity = nonlinearity_name(component.out_nonlinearity)
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
                "hidden_nonlinearity": nonlinearity_name(component.hidden_nonlinearity),
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
            input_shape = shape_from_proto(component.input_proto)
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

        if isinstance(component, LinearStateSpace):
            params = {
                "A": component.A.tolist(),
                "B": component.B.tolist(),
                "B_w": component.B_w.tolist(),
                "dt": component.dt,
                "initial_state": list(component.initial_state),
                "pos_slice": list(component.pos_slice),
                "vel_slice": list(component.vel_slice),
            }
            nodes[name] = ComponentSpec(
                type="LinearStateSpace",
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
            input_shape = shape_from_proto(component.input_proto)
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


def spec_to_graph(
    spec: GraphSpec,
    component_registry: Any | None = None,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> Graph:
    """Instantiate a Graph-like object from GraphSpec."""
    from feedbax.component_registry import get_component_registry

    execution_registry = (
        component_registry if hasattr(component_registry, "names") else get_component_registry()
    )
    metadata_registry = component_registry if component_registry is not None else execution_registry
    _validate_supported_spec_versions(spec)
    spec = _migrate_spec(spec)
    spec = materialize_additive_channel_adapters(spec)
    spec = normalize_stateful_prototypes(
        spec,
        input_prototypes,
        component_registry=metadata_registry,
    )

    nodes: dict[str, Component] = {}
    for node_name, node_spec in spec.nodes.items():
        defaults = _lookup_defaults(metadata_registry, node_spec.type)
        required_params = _lookup_required_params(metadata_registry, node_spec.type)
        params = _merge_params(
            node_spec.params,
            defaults,
            required_params=required_params,
            node_name=node_name,
            node_type=node_spec.type,
        )

        if node_spec.type == "Subgraph":
            if not spec.subgraphs or node_name not in spec.subgraphs:
                raise ValueError(f"Missing subgraph spec for '{node_name}'")
            nodes[node_name] = spec_to_graph(spec.subgraphs[node_name], metadata_registry)
            continue
        if node_spec.type == "Network":
            subgraph = (spec.subgraphs or {}).get(node_name)
            if subgraph is None:
                raise ValueError(
                    f"Network node {node_name!r} has no subgraph. "
                    "Open it in Studio to generate the internal architecture, then save again."
                )
            nodes[node_name] = spec_to_graph(subgraph, metadata_registry)
            continue
        if spec.subgraphs and node_name in spec.subgraphs:
            nodes[node_name] = spec_to_graph(spec.subgraphs[node_name], metadata_registry)
            continue
        nodes[node_name] = build_component(
            node_name,
            node_spec.type,
            params,
            component_registry=execution_registry,
        )

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
