from __future__ import annotations

from typing import Any, Literal, Mapping, cast

import jax.numpy as jnp
import jax.tree as jt

from feedbax.bodies import FeedbackChannels
from feedbax.channel import Channel
from feedbax.components import (
    Constant,
    DelayLine,
    Damper,
    Demux,
    ElementwiseAffineModulator,
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
from feedbax.control.affine import AffineFeedbackController
from feedbax.filters import FirstOrderFilter
from feedbax.graph import Component, Graph, Wire
from feedbax.graph_templates import recurrent_controller_template_graph
from feedbax.intervene.intervene import (
    AddNoise,
    ConstantInput,
    Copy,
    CurlField,
    DynamicsMatrixPerturb,
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
from feedbax.noise import CompositeNoise, Multiplicative, Normal
from feedbax.penzai_component import PenzaiSubgraph
from feedbax.tasks import DelayedReaches, SimpleReaches, Stabilization, TaskComponent
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    WireSpec,
)
from feedbax.graph_channel_adapters import materialize_additive_channel_adapters
from feedbax.migrations import migrate_graph_spec
from feedbax.parameter_constraints import apply_parameter_constraints, normalize_parameter_constraints
from feedbax.serialization_builders import build_component, nonlinearity_name
from feedbax.serialization_prototypes import (
    normalize_stateful_prototypes,
    prototypes_from_task_bindings,
    shape_from_proto,
)
from feedbax.state_feedback import StateFeedbackSelector


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


def graph_to_spec(graph: Any) -> GraphSpec:
    """Serialize a Graph-like object to GraphSpec."""
    if not isinstance(graph, Graph):
        raise TypeError("graph_to_spec requires feedbax.graph.Graph")

    nodes: dict[str, ComponentSpec] = {}
    subgraphs: dict[str, GraphSpec] = {}
    expanded_input_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    expanded_output_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    expanded_wires: list[WireSpec] = []
    expanded_parameter_constraints = []
    expanded_nodes: set[str] = set()

    def _to_native(value: Any):
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _channel_noise_params(component: Channel) -> dict[str, Any]:
        params: dict[str, Any] = {
            "noise_model": getattr(component, "noise_model", "additive_gaussian"),
            "add_noise": component.add_noise,
        }
        additive_std = 0.0
        signal_dependent_std = 0.0
        noise_func = component.noise_func
        terms = noise_func.terms if isinstance(noise_func, CompositeNoise) else (noise_func,)
        for term in terms:
            if isinstance(term, Normal):
                additive_std = float(term.std)
            elif isinstance(term, Multiplicative) and isinstance(term.noise_func, Normal):
                signal_dependent_std = float(term.noise_func.std)

        if params["noise_model"] == "signal_dependent_plus_additive":
            params["additive_noise_std"] = additive_std
            params["signal_dependent_noise_std"] = signal_dependent_std
        elif params["noise_model"] == "signal_dependent_gaussian":
            params["signal_dependent_noise_std"] = signal_dependent_std
        else:
            params["noise_std"] = additive_std
        noise_role = getattr(component, "noise_role", None)
        noise_timing = getattr(component, "noise_timing", None)
        if noise_role is not None:
            params["noise_role"] = noise_role
        if noise_timing is not None:
            params["noise_timing"] = noise_timing
        return params

    def _prefixed_node(component_name: str, node_name: str) -> str:
        return f"{component_name}_{node_name}"

    def _remap_template_binding(
        component_name: str,
        binding: tuple[str, str],
    ) -> tuple[str, str]:
        return (_prefixed_node(component_name, binding[0]), binding[1])

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
            template = recurrent_controller_template_graph(
                input_size=component.input_size,
                hidden_size=component.hidden_size,
                out_size=component.out_size,
                cell_type=cell_type,
                out_nonlinearity=out_nonlinearity,
                population_structure=component.population_structure,
                name=f"{name} recurrent controller",
                description="Serialized SimpleStagedNetwork as explicit graph nodes",
            )
            for template_node_name, node_spec in template.nodes.items():
                node_name = _prefixed_node(name, template_node_name)
                if node_name in nodes:
                    raise ValueError(
                        f"Cannot inline SimpleStagedNetwork {name!r}: generated node "
                        f"{node_name!r} already exists"
                    )
                nodes[node_name] = node_spec
            expanded_nodes.add(name)
            expanded_wires.extend(
                WireSpec(
                    source_node=_prefixed_node(name, wire.source_node),
                    source_port=wire.source_port,
                    target_node=_prefixed_node(name, wire.target_node),
                    target_port=wire.target_port,
                    temporality=wire.temporality,
                    recurrent_initializer=wire.recurrent_initializer,
                )
                for wire in template.wires
            )
            expanded_input_bindings.update(
                {
                    (name, port): _remap_template_binding(name, binding)
                    for port, binding in template.input_bindings.items()
                }
            )
            expanded_output_bindings.update(
                {
                    (name, port): _remap_template_binding(name, binding)
                    for port, binding in template.output_bindings.items()
                }
            )
            expanded_parameter_constraints.extend(
                constraint.model_copy(
                    update={"node": _prefixed_node(name, constraint.node)}
                )
                for constraint in template.parameter_constraints
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
        if isinstance(component, ElementwiseAffineModulator):
            nodes[name] = ComponentSpec(
                type="ElementwiseAffineModulator",
                params={
                    "signal_shape": list(component.signal_shape),
                    "baseline": component.baseline.tolist(),
                    "gain_init": component.gain.tolist(),
                    "bias_init": component.bias.tolist(),
                },
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
        if isinstance(component, Demux):
            nodes[name] = ComponentSpec(
                type="Demux",
                params={"sizes": list(component.sizes)},
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
            if (
                plant_type == "PointMass"
                and isinstance(component.plant, DirectForceInput)
                and isinstance(component.plant.skeleton, PointMass)
            ):
                skeleton = component.plant.skeleton
                nodes[name] = nodes[name].model_copy(
                    update={
                        "params": {
                            **params,
                            "mass": float(skeleton.mass),
                            "damping": float(skeleton.damping),
                        }
                    }
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

        if isinstance(component, AffineFeedbackController):
            nodes[name] = ComponentSpec(
                type="AffineFeedbackController",
                params=component.to_params(),
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, StateFeedbackSelector):
            nodes[name] = ComponentSpec(
                type="StateFeedbackSelector",
                params=component.to_params(),
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, Channel):
            params = {
                "delay": component.delay,
                **_channel_noise_params(component),
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

        if isinstance(component, FeedbackChannels):
            channel_leaves = jt.leaves(
                component.channels,
                is_leaf=lambda item: isinstance(item, Channel),
            )
            spec_leaves = jt.leaves(
                component.specs,
                is_leaf=lambda item: hasattr(item, "where"),
            )
            if len(channel_leaves) != 1 or len(spec_leaves) != 1:
                raise ValueError(
                    "graph_to_spec only supports single-channel FeedbackChannels built-ins"
                )
            channel = channel_leaves[0]
            feedback_spec = spec_leaves[0]
            selector = getattr(feedback_spec.where, "_feedbax_feedback_selector", None)
            paths = getattr(feedback_spec.where, "_feedbax_feedback_paths", None)
            if selector is None:
                raise ValueError(
                    "FeedbackChannels component cannot be serialized without a declarative "
                    "selector; build it through GraphSpec with 'selector' or 'paths'"
                )
            params = {
                "delay": channel.delay,
                **_channel_noise_params(channel),
                "selector": selector,
            }
            if paths is not None:
                params["paths"] = list(paths)
            input_proto_leaves = jt.leaves(channel.input_proto)
            if input_proto_leaves and all(hasattr(leaf, "shape") for leaf in input_proto_leaves):
                params["input_shape"] = [
                    [int(dim) for dim in leaf.shape] for leaf in input_proto_leaves
                ]
            nodes[name] = ComponentSpec(
                type="FeedbackChannels",
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
                "label": component.label,
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
                "label": component.label,
            }
            nodes[name] = ComponentSpec(
                type="FixedField",
                params=params,
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
            continue

        if isinstance(component, DynamicsMatrixPerturb):
            params = {
                "scale": component._initial_state.scale,
                "delta_A": jnp.asarray(component._initial_state.delta_A).tolist(),
                "active": component._initial_state.active,
                "label": component.label,
                "mass": component.mass,
            }
            nodes[name] = ComponentSpec(
                type="DynamicsMatrixPerturb",
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
                        "preset": task.preset,
                        "train_endpoint_mode": task.train_endpoint_mode,
                        "epoch_len_ranges": [list(item) for item in task.epoch_len_ranges],
                        "epoch_names": list(task.epoch_names),
                        "target_on_epochs": jnp.asarray(task.target_on_epochs).tolist(),
                        "hold_epochs": jnp.asarray(task.hold_epochs).tolist(),
                        "move_epochs": jnp.asarray(task.move_epochs).tolist(),
                        "p_catch_trial": task.p_catch_trial,
                        "target_visible_from_start": task.target_visible_from_start,
                        "go_cue_event_name": task.go_cue_event_name,
                        "catch_metadata_policy": task.catch_metadata_policy,
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

    def _remap_wire_endpoint(node: str, port: str, *, direction: str) -> tuple[str, str]:
        bindings = expanded_output_bindings if direction == "source" else expanded_input_bindings
        if (node, port) in bindings:
            return bindings[(node, port)]
        if node in expanded_nodes:
            raise ValueError(
                f"Cannot serialize SimpleStagedNetwork endpoint {node!r}.{port!r}: "
                f"no explicit {direction} binding exists for that port"
            )
        return node, port

    def _remap_graph_binding(binding: tuple[str, str], *, direction: str) -> tuple[str, str]:
        return _remap_wire_endpoint(binding[0], binding[1], direction=direction)

    wires = [
        *expanded_wires,
        *(
            WireSpec(
                source_node=_remap_wire_endpoint(
                    wire.source_node,
                    wire.source_port,
                    direction="source",
                )[0],
                source_port=_remap_wire_endpoint(
                    wire.source_node,
                    wire.source_port,
                    direction="source",
                )[1],
                target_node=_remap_wire_endpoint(
                    wire.target_node,
                    wire.target_port,
                    direction="target",
                )[0],
                target_port=_remap_wire_endpoint(
                    wire.target_node,
                    wire.target_port,
                    direction="target",
                )[1],
                temporality=cast(Literal["instant", "recurrent"], wire.temporality),
                recurrent_initializer=wire.recurrent_initializer,
            )
            for wire in graph.wires
        ),
    ]

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=list(graph.input_ports),
        output_ports=list(graph.output_ports),
        input_bindings={
            name: _remap_graph_binding(binding, direction="target")
            for name, binding in graph.input_bindings.items()
        },
        output_bindings={
            name: _remap_graph_binding(binding, direction="source")
            for name, binding in graph.output_bindings.items()
        },
        subgraphs=subgraphs or None,
        retained_observables=getattr(graph, "retained_observables", None),
        parameter_constraints=[
            *list(getattr(graph, "parameter_constraints", ())),
            *expanded_parameter_constraints,
        ],
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
    migration = migrate_graph_spec(spec)
    spec = GraphSpec.model_validate(migration.payload)
    spec = materialize_additive_channel_adapters(spec)
    spec = normalize_stateful_prototypes(
        spec,
        input_prototypes,
        component_registry=metadata_registry,
    )

    nodes: dict[str, Component] = {}
    for node_name, node_spec in spec.nodes.items():
        node_type = node_spec.type
        node_params = dict(node_spec.params)
        resolve_component_spec = getattr(metadata_registry, "resolve_component_spec", None)
        should_resolve_component_spec = getattr(
            metadata_registry,
            "should_resolve_component_spec",
            None,
        )
        if (
            callable(resolve_component_spec)
            and callable(should_resolve_component_spec)
            and should_resolve_component_spec(
                node_type,
                param_schema_version=node_spec.param_schema_version,
            )
        ):
            resolution = resolve_component_spec(
                node_type,
                node_params,
                param_schema_version=node_spec.param_schema_version,
            )
            node_type = resolution.type_id
            node_params = resolution.params

        defaults = _lookup_defaults(metadata_registry, node_type)
        required_params = _lookup_required_params(metadata_registry, node_type)
        params = _merge_params(
            node_params,
            defaults,
            required_params=required_params,
            node_name=node_name,
            node_type=node_type,
        )

        if node_type == "Subgraph":
            if not spec.subgraphs or node_name not in spec.subgraphs:
                raise ValueError(f"Missing subgraph spec for '{node_name}'")
            nodes[node_name] = spec_to_graph(spec.subgraphs[node_name], metadata_registry)
            continue
        if node_type == "Network":
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
            node_type,
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

    input_bindings = {
        name: (binding[0], binding[1]) for name, binding in spec.input_bindings.items()
    }
    output_bindings = {
        name: (binding[0], binding[1]) for name, binding in spec.output_bindings.items()
    }

    graph = Graph(
        nodes=nodes,
        wires=wires,
        input_ports=tuple(spec.input_ports),
        output_ports=tuple(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        parameter_constraints=normalize_parameter_constraints(spec.parameter_constraints),
    )
    return apply_parameter_constraints(graph)
