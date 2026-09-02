from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from equinox.nn import State
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.models.feedback import FeedbackChannels
from feedbax.runtime.channel import Channel
from feedbax.component_registry import ComponentRegistry
from feedbax.runtime.affine_composer import AffineValueComposer
from feedbax.runtime.components import Demux, ElementwiseAffineModulator
from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioTaskBindingSpec, WireSpec
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.task_bindings import (
    apply_task_parameter_state_inits,
    expose_task_bindings,
    expose_task_inputs,
)
from feedbax.studio.schema import validate_task_binding_schema
from feedbax.intervene import (
    CurlField,
    CurlFieldParams,
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
    FixedField,
    FixedFieldParams,
)
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.runtime.noise import CompositeNoise, Multiplicative, Normal
from feedbax.compiler.serialization import (
    graph_to_spec,
    prototypes_from_task_bindings,
)
from tests.graph_compiler_test_support import spec_to_graph
from feedbax.compiler.prototypes import output_prototypes_for_node
from feedbax.compiler.prototypes import infer_node_input_prototypes
from feedbax.runtime.state import CartesianState


def _components() -> ComponentRegistry:
    return ComponentRegistry(load_user_components=False)


def _single_node_spec(
    component_type: str,
    params: Mapping[str, Any],
    *,
    input_ports: list[str],
    output_ports: list[str],
) -> GraphSpec:
    return GraphSpec(
        nodes={
            "field": ComponentSpec(
                type=component_type,
                params=dict(params),
                input_ports=input_ports,
                output_ports=output_ports,
            )
        },
        input_ports=input_ports,
        output_ports=output_ports,
        input_bindings={port: ("field", port) for port in input_ports},
        output_bindings={port: ("field", port) for port in output_ports},
    )


def _call_component(component, inputs: dict[str, Any]) -> dict[str, Any]:
    state = State(component)
    outputs, _ = component(inputs, state, key=jr.PRNGKey(0))
    return outputs


def _force_field_cases():
    effector = CartesianState(pos=jnp.array([0.5, -0.25]), vel=jnp.array([1.5, -2.0]))
    force = jnp.array([0.25, -0.5], dtype=jnp.float32)
    delta_A = [[0.1, 0.0, 0.5, 0.0], [0.0, -0.2, 0.0, 0.25]]
    return (
        (
            "FixedField",
            FixedField(
                params=FixedFieldParams(
                    scale=2.0,
                    amplitude=1.5,
                    field=jnp.array([0.4, -0.6]),
                    active=True,
                ),
                label="fixed_parity",
            ),
            {
                "scale": 2.0,
                "amplitude": 1.5,
                "field": [0.4, -0.6],
                "active": True,
                "label": "fixed_parity",
            },
            {"force": force},
            ["force", "params_override"],
        ),
        (
            "CurlField",
            CurlField(
                params=CurlFieldParams(scale=1.25, amplitude=0.75, active=True),
                label="curl_parity",
            ),
            {
                "scale": 1.25,
                "amplitude": 0.75,
                "active": True,
                "label": "curl_parity",
            },
            {"effector": effector, "force": force},
            ["effector", "force", "params_override"],
        ),
        (
            "DynamicsMatrixPerturb",
            DynamicsMatrixPerturb(
                params=DynamicsMatrixPerturbParams(
                    scale=0.5,
                    delta_A=jnp.asarray(delta_A),
                    active=True,
                ),
                label="dynamics_parity",
                mass=1.75,
            ),
            {
                "scale": 0.5,
                "delta_A": delta_A,
                "active": True,
                "label": "dynamics_parity",
                "mass": 1.75,
            },
            {"effector": effector, "force": force},
            ["effector", "force", "params_override"],
        ),
    )


def test_elementwise_affine_modulator_graphspec_round_trips_and_runs() -> None:
    spec = GraphSpec(
        nodes={
            "modulate": ComponentSpec(
                type="ElementwiseAffineModulator",
                params={
                    "signal_shape": [3],
                    "baseline": 1.0,
                    "gain_init": [0.5, -1.0, 2.0],
                    "bias_init": [0.1, 0.0, -0.2],
                    "trainable": True,
                },
                input_ports=["signal", "modulator", "scale", "bias"],
                output_ports=["output"],
            )
        },
        input_ports=["signal", "modulator"],
        output_ports=["output"],
        input_bindings={
            "signal": ("modulate", "signal"),
            "modulator": ("modulate", "modulator"),
        },
        output_bindings={"output": ("modulate", "output")},
    )

    graph = spec_to_graph(spec, component_registry=_components())
    component = graph.nodes["modulate"]
    assert isinstance(component, ElementwiseAffineModulator)
    assert component.signal_shape == (3,)
    assert jnp.allclose(component.gain, jnp.array([0.5, -1.0, 2.0]))
    assert jnp.allclose(component.bias, jnp.array([0.1, 0.0, -0.2]))
    assert len(eqx.filter(component, eqx.is_inexact_array).gain) == 3

    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"signal": jnp.array([2.0, 4.0, 8.0]), "modulator": jnp.array(0.5)},
        state,
        key=jax.random.PRNGKey(0),
    )
    expected = (
        jnp.array([2.0, 4.0, 8.0]) * (1.0 + jnp.array([0.5, -1.0, 2.0]) * 0.5)
        + jnp.array([0.1, 0.0, -0.2]) * 0.5
    )
    assert jnp.allclose(outputs["output"], expected)

    roundtrip = graph_to_spec(graph)
    params = roundtrip.nodes["modulate"].params
    assert roundtrip.nodes["modulate"].type == "ElementwiseAffineModulator"
    assert roundtrip.nodes["modulate"].input_ports == ["signal", "modulator", "scale", "bias"]
    assert params["signal_shape"] == [3]
    assert params["baseline"] == [1.0, 1.0, 1.0]
    assert params["gain_init"] == [0.5, -1.0, 2.0]
    assert jnp.allclose(jnp.asarray(params["bias_init"]), jnp.array([0.1, 0.0, -0.2]))


def test_task_data_reference_vector_composes_into_affine_controller_reference() -> None:
    task_binding_spec = {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "target_position",
                "label": "Target position",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.target_position",
                "bindable": True,
                "expected_shape": ["time", 2],
                "dtype": "vector",
                "metadata": {"temporal_support": "trajectory"},
            }
        ],
        "bindings": [
            {
                "id": "task:target_position->reference_mux:in_0",
                "source_data_id": "target_position",
                "target_node_id": "reference_mux",
                "target_port": "in_0",
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }
    spec = GraphSpec(
        nodes={
            "zero_velocity": ComponentSpec(
                type="Constant",
                params={"value": [0.0, 0.0]},
                input_ports=[],
                output_ports=["output"],
            ),
            "zero_feedback": ComponentSpec(
                type="Constant",
                params={"value": [0.0, 0.0, 0.0, 0.0]},
                input_ports=[],
                output_ports=["output"],
            ),
            "reference_mux": ComponentSpec(
                type="Mux",
                params={"n_inputs": 2},
                input_ports=["in_0", "in_1"],
                output_ports=["output"],
            ),
            "controller": ComponentSpec(
                type="AffineFeedbackController",
                params={
                    "gain": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                    "schedule_policy": "hold",
                },
                input_ports=["feedback", "reference", "feedforward"],
                output_ports=["command"],
            ),
        },
        wires=[
            WireSpec(
                source_node="zero_velocity",
                source_port="output",
                target_node="reference_mux",
                target_port="in_1",
            ),
            WireSpec(
                source_node="zero_feedback",
                source_port="output",
                target_node="controller",
                target_port="feedback",
            ),
            WireSpec(
                source_node="reference_mux",
                source_port="output",
                target_node="controller",
                target_port="reference",
            ),
        ],
        output_ports=["command"],
        output_bindings={"command": ("controller", "command")},
    )

    graph = spec_to_graph(
        spec,
        component_registry=_components(),
        input_prototypes=prototypes_from_task_bindings(task_binding_spec),
    )
    graph, plans = expose_task_inputs(
        graph,
        StudioTaskBindingSpec.model_validate(task_binding_spec),
    )
    assert plans[0].graph_input == "task:target_position->reference_mux.in_0"

    state = init_state_from_component(graph)
    outputs, _ = graph(
        {plans[0].graph_input: jnp.asarray([1.5, -2.0], dtype=jnp.float32)},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["command"], jnp.asarray([1.5, -2.0]))
    roundtrip = graph_to_spec(graph)
    assert roundtrip.nodes["zero_velocity"].type == "Constant"
    assert roundtrip.nodes["controller"].type == "AffineFeedbackController"
    assert roundtrip.input_bindings[plans[0].graph_input] == ("reference_mux", "in_0")


def _derived_mux_gru_spec(declared_input_size: Any = "__missing__") -> GraphSpec:
    cell_params: dict[str, Any] = {"hidden_size": 5}
    if declared_input_size != "__missing__":
        cell_params["input_size"] = declared_input_size
    return GraphSpec(
        nodes={
            "input_mux": ComponentSpec(
                type="Mux",
                params={"n_inputs": 2},
                input_ports=["in_0", "in_1"],
                output_ports=["output"],
            ),
            "cell": ComponentSpec(
                type="GRU",
                params=cell_params,
                input_ports=["input", "hidden"],
                output_ports=["output", "hidden"],
            ),
        },
        wires=[
            WireSpec(
                source_node="input_mux",
                source_port="output",
                target_node="cell",
                target_port="input",
            )
        ],
        output_ports=["hidden"],
        output_bindings={"hidden": ("cell", "hidden")},
        derived_dimensions=[
            {
                "node": "cell",
                "param": "input_size",
                "port": "input",
                "metadata": {"dimension_source": "mux_concat_inputs"},
            }
        ],
    )


def _derived_mux_task_bindings(
    *,
    second_role: str = "model_input",
    first_shape: list[Any] | None = None,
    second_shape: list[Any] | None = None,
) -> StudioTaskBindingSpec:
    first_shape = first_shape or ["time", 2]
    second_shape = second_shape or ["time", 1]
    return StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "kinematics",
                    "label": "Kinematics",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.kinematics",
                    "bindable": True,
                    "expected_shape": first_shape,
                    "dtype": "float32",
                    "metadata": {"temporal_support": "trajectory"},
                },
                {
                    "id": "gain_schedule",
                    "label": "Gain schedule",
                    "kind": "signal",
                    "role": second_role,
                    "path": "inputs.gain_schedule",
                    "bindable": True,
                    "expected_shape": second_shape,
                    "dtype": "float32",
                    "metadata": {"temporal_support": "trajectory"},
                },
            ],
            "bindings": [
                {
                    "id": "task:kinematics->input_mux:in_0",
                    "source_data_id": "kinematics",
                    "target_node_id": "input_mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                },
                {
                    "id": "task:gain_schedule->input_mux:in_1",
                    "source_data_id": "gain_schedule",
                    "target_node_id": "input_mux",
                    "target_port": "in_1",
                    "role": second_role,
                    "metadata": {},
                },
            ],
            "metadata": {},
        }
    )


def test_derived_dimension_rule_sets_gru_input_size_from_task_data_mux_fan_in() -> None:
    spec = _derived_mux_gru_spec()
    task_binding_spec = _derived_mux_task_bindings()

    graph = spec_to_graph(
        spec,
        component_registry=_components(),
        input_prototypes=prototypes_from_task_bindings(task_binding_spec),
    )

    assert graph.nodes["cell"].input_size == 3
    spec_payload = spec.model_dump(mode="json", exclude_none=True)
    assert "input_size" not in spec_payload["nodes"]["cell"]["params"]
    assert "input_size_source" not in spec_payload["nodes"]["cell"]["params"]


def test_derived_dimension_rule_counts_component_parameter_mux_fan_in() -> None:
    spec = _derived_mux_gru_spec()
    task_binding_spec = _derived_mux_task_bindings(
        second_role="component_parameter",
        second_shape=["time", 4],
    )

    graph = spec_to_graph(
        spec,
        component_registry=_components(),
        input_prototypes=prototypes_from_task_bindings(task_binding_spec),
    )

    assert graph.nodes["cell"].input_size == 6


def test_derived_dimension_rule_rejects_declared_conflict() -> None:
    spec = _derived_mux_gru_spec(declared_input_size=7)
    task_binding_spec = _derived_mux_task_bindings()

    with pytest.raises(ValueError, match="Derived dimension conflict"):
        spec_to_graph(
            spec,
            component_registry=_components(),
            input_prototypes=prototypes_from_task_bindings(task_binding_spec),
        )


def test_derived_dimension_rule_rejects_null_declaration() -> None:
    spec = _derived_mux_gru_spec(declared_input_size=None)
    task_binding_spec = _derived_mux_task_bindings()

    with pytest.raises(ValueError, match="declared null"):
        spec_to_graph(
            spec,
            component_registry=_components(),
            input_prototypes=prototypes_from_task_bindings(task_binding_spec),
        )


def test_point_mass_graphspec_preserves_mass_damping_and_dt() -> None:
    spec = GraphSpec(
        nodes={
            "mechanics": ComponentSpec(
                type="PointMass",
                params={"dt": 0.02, "mass": 0.75, "damping": 4.5},
                input_ports=["force"],
                output_ports=["effector", "state"],
            )
        },
        input_ports=["force"],
        output_ports=["effector"],
        input_bindings={"force": ("mechanics", "force")},
        output_bindings={"effector": ("mechanics", "effector")},
    )

    graph = spec_to_graph(spec, component_registry=_components())

    mechanics = graph.nodes["mechanics"]
    assert isinstance(mechanics, Mechanics)
    assert isinstance(mechanics.plant, DirectForceInput)
    assert isinstance(mechanics.plant.skeleton, PointMass)
    assert mechanics.dt == 0.02
    assert mechanics.plant.skeleton.mass == 0.75
    assert mechanics.plant.skeleton.damping == 4.5

    roundtrip = graph_to_spec(graph)
    params = roundtrip.nodes["mechanics"].params
    assert params["dt"] == 0.02
    assert params["mass"] == 0.75
    assert params["damping"] == 4.5


def test_feedback_channels_materialize_point_mass_selector() -> None:
    spec = GraphSpec(
        nodes={
            "mechanics": ComponentSpec(
                type="PointMass",
                params={"dt": 0.01, "mass": 1.0, "damping": 2.0},
                input_ports=["force"],
                output_ports=["effector", "state"],
            ),
            "feedback": ComponentSpec(
                type="FeedbackChannels",
                params={
                    "selector": "point_mass_pos_vel",
                    "delay": 2,
                    "noise_std": 0.05,
                    "add_noise": True,
                    "noise_role": "sensory_feedback",
                    "noise_timing": "pre_controller",
                },
                input_ports=["mechanics"],
                output_ports=["feedback"],
            ),
        },
        wires=[
            WireSpec(
                source_node="mechanics",
                source_port="state",
                target_node="feedback",
                target_port="mechanics",
                temporality="recurrent",
                recurrent_initializer={"kind": "state_output", "state_slot": "mechanics"},
            )
        ],
        output_ports=["feedback"],
        output_bindings={"feedback": ("feedback", "feedback")},
    )

    graph = spec_to_graph(spec, component_registry=_components())

    mechanics = graph.nodes["mechanics"]
    feedback = graph.nodes["feedback"]
    assert isinstance(mechanics, Mechanics)
    assert isinstance(feedback, FeedbackChannels)
    assert isinstance(feedback.channels, Channel)
    assert feedback.channels.delay == 2
    assert feedback.channels.noise_role == "sensory_feedback"
    assert feedback.channels.noise_timing == "pre_controller"
    assert isinstance(feedback.channels.noise_func, Normal)
    assert feedback.channels.noise_func.std == 0.05

    selected = feedback.specs.where(mechanics._initial_state)
    assert len(selected) == 2
    assert selected[0].shape == (2,)
    assert selected[1].shape == (2,)

    roundtrip = graph_to_spec(graph)
    params = roundtrip.nodes["feedback"].params
    assert params["selector"] == "point_mass_pos_vel"
    assert params["paths"] == ["plant.skeleton.pos", "plant.skeleton.vel"]
    assert params["noise_role"] == "sensory_feedback"


def test_channel_structured_motor_noise_round_trips() -> None:
    spec = GraphSpec(
        nodes={
            "efferent": ComponentSpec(
                type="Channel",
                params={
                    "delay": 0,
                    "noise_model": "signal_dependent_plus_additive",
                    "additive_noise_std": 0.18,
                    "signal_dependent_noise_std": 0.1,
                    "add_noise": True,
                    "noise_role": "motor_command",
                    "noise_timing": "pre_force_filter",
                    "input_shape": [2],
                },
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("efferent", "input")},
        output_bindings={"output": ("efferent", "output")},
    )

    graph = spec_to_graph(spec, component_registry=_components())

    channel = graph.nodes["efferent"]
    assert isinstance(channel, Channel)
    assert channel.noise_model == "signal_dependent_plus_additive"
    assert channel.noise_role == "motor_command"
    assert channel.noise_timing == "pre_force_filter"
    assert isinstance(channel.noise_func, CompositeNoise)
    assert isinstance(channel.noise_func[0], Multiplicative)
    assert isinstance(channel.noise_func[0].noise_func, Normal)
    assert channel.noise_func[0].noise_func.std == 0.1
    assert isinstance(channel.noise_func[1], Normal)
    assert channel.noise_func[1].std == 0.18

    roundtrip = graph_to_spec(graph)
    params = roundtrip.nodes["efferent"].params
    assert params["noise_model"] == "signal_dependent_plus_additive"
    assert params["additive_noise_std"] == 0.18
    assert params["signal_dependent_noise_std"] == 0.1
    assert params["noise_role"] == "motor_command"
    assert params["noise_timing"] == "pre_force_filter"
    assert params["input_shape"] == [2]


def test_channel_can_express_plant_process_force_noise_role() -> None:
    graph = spec_to_graph(
        GraphSpec(
            nodes={
                "plant_noise": ComponentSpec(
                    type="Channel",
                    params={
                        "delay": 0,
                        "noise_model": "additive_gaussian",
                        "noise_std": 0.03,
                        "add_noise": True,
                        "noise_role": "plant_process_load",
                        "noise_timing": "post_force_filter_pre_mechanics",
                        "input_shape": [2],
                    },
                    input_ports=["input"],
                    output_ports=["output"],
                )
            },
            input_ports=["force"],
            output_ports=["force"],
            input_bindings={"force": ("plant_noise", "input")},
            output_bindings={"force": ("plant_noise", "output")},
        ),
        component_registry=_components(),
    )

    channel = graph.nodes["plant_noise"]
    assert isinstance(channel, Channel)
    assert channel.noise_role == "plant_process_load"
    assert channel.noise_timing == "post_force_filter_pre_mechanics"
    assert isinstance(channel.noise_func, Normal)
    assert channel.noise_func.std == 0.03


@pytest.mark.parametrize(
    ("component_type", "direct_component", "params", "inputs", "input_ports"),
    _force_field_cases(),
)
def test_force_field_graphspec_builders_preserve_contract_and_runtime(
    component_type: str,
    direct_component,
    params: dict[str, Any],
    inputs: dict[str, Any],
    input_ports: list[str],
) -> None:
    spec = _single_node_spec(
        component_type,
        params,
        input_ports=input_ports,
        output_ports=["force"],
    )

    graph = spec_to_graph(spec, component_registry=_components())
    materialized = graph.nodes["field"]

    assert isinstance(materialized, type(direct_component))
    assert materialized.label == params["label"]
    assert tuple(input_ports) == materialized.input_ports
    assert materialized.output_ports == ("force",)
    assert set(materialized.task_parameter_state_indices()) == {params["label"]}
    with pytest.deprecated_call(match="intervention_state_indices"):
        assert set(materialized.intervention_state_indices()) == {params["label"]}

    direct_out = _call_component(direct_component, inputs)
    materialized_out = _call_component(materialized, inputs)
    assert jnp.allclose(materialized_out["force"], direct_out["force"])

    roundtrip = graph_to_spec(graph)
    node = roundtrip.nodes["field"]
    assert node.type == component_type
    assert node.input_ports == input_ports
    assert node.output_ports == ["force"]
    assert node.params["label"] == params["label"]
    assert bool(node.params["active"]) is True
    if component_type == "FixedField":
        assert node.params["field"] == pytest.approx(params["field"])
        assert node.params["amplitude"] == pytest.approx(params["amplitude"])
    if component_type == "CurlField":
        assert node.params["amplitude"] == pytest.approx(params["amplitude"])
    if component_type == "DynamicsMatrixPerturb":
        assert jnp.allclose(jnp.asarray(node.params["delta_A"]), jnp.asarray(params["delta_A"]))
        assert node.params["mass"] == pytest.approx(params["mass"])


def test_force_field_registry_exposes_params_override_ports_and_builders() -> None:
    registry = ComponentRegistry(load_user_components=False)

    expected = {
        "FixedField": {"force", "params_override"},
        "CurlField": {"effector", "force", "params_override"},
        "DynamicsMatrixPerturb": {"effector", "force", "params_override"},
    }
    for component_type, input_ports in expected.items():
        meta = registry.get(component_type)
        assert meta is not None
        assert meta.builder is not None
        assert set(meta.input_ports) == input_ports
        assert meta.output_ports == ["force"]
        assert "label" in meta.default_params
        if component_type == "DynamicsMatrixPerturb":
            assert "delta_A" in meta.default_params
            assert "mass" in meta.default_params


@pytest.mark.parametrize(
    ("component_type", "input_ports"),
    [
        ("FixedField", ["force", "params_override"]),
        ("CurlField", ["effector", "force", "params_override"]),
        ("DynamicsMatrixPerturb", ["effector", "force", "params_override"]),
    ],
)
def test_force_field_output_prototypes_passthrough_force(
    component_type: str,
    input_ports: list[str],
) -> None:
    registry = ComponentRegistry(load_user_components=False)
    force_proto = jnp.zeros((2,), dtype=jnp.float32)
    node_spec = ComponentSpec(
        type=component_type,
        params={},
        input_ports=input_ports,
        output_ports=["force"],
    )

    outputs = output_prototypes_for_node(
        "field",
        node_spec,
        {("field", "force"): force_proto},
        {},
        registry,
    )

    assert outputs["force"] is force_proto


@pytest.mark.parametrize(
    ("component_type", "input_ports"),
    [
        ("FixedField", ["force", "params_override"]),
        ("CurlField", ["effector", "force", "params_override"]),
        ("DynamicsMatrixPerturb", ["effector", "force", "params_override"]),
    ],
)
def test_force_field_output_prototypes_require_force_input(
    component_type: str,
    input_ports: list[str],
) -> None:
    registry = ComponentRegistry(load_user_components=False)
    node_spec = ComponentSpec(
        type=component_type,
        params={},
        input_ports=input_ports,
        output_ports=["force"],
    )

    with pytest.raises(ValueError, match="input prototype 'force'"):
        output_prototypes_for_node("field", node_spec, {}, {}, registry)


def _tree_shapes(proto: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(dim) for dim in leaf.shape) for leaf in jax.tree.leaves(proto))


def _registered_outputs(
    component_type: str,
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    input_ports: list[str],
    output_ports: list[str],
) -> dict[str, Any]:
    registry = ComponentRegistry(load_user_components=False)
    node_spec = ComponentSpec(
        type=component_type,
        params=dict(params),
        input_ports=input_ports,
        output_ports=output_ports,
    )
    return output_prototypes_for_node(
        "node",
        node_spec,
        {("node", port): proto for port, proto in inputs.items()},
        {},
        registry,
    )


@pytest.mark.parametrize(
    "case",
    [
        {
            "type": "Gain",
            "inputs": {"input": jnp.zeros((3,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "Sum",
            "inputs": {"a": jnp.zeros((2,))},
            "input_ports": ["a", "b"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
        {
            "type": "Multiply",
            "inputs": {"b": jnp.zeros((4,))},
            "input_ports": ["a", "b"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "ElementwiseAffineModulator",
            "params": {"signal_shape": [5]},
            "inputs": {},
            "input_ports": ["signal", "modulator", "scale", "bias"],
            "output_ports": ["output"],
            "shapes": {"output": ((5,),)},
        },
        {
            "type": "Constant",
            "params": {"value": 1.0},
            "input_ports": [],
            "output_ports": ["output"],
            "shapes": {"output": ((),)},
        },
        {
            "type": "Ramp",
            "params": {"slope": 1.0},
            "input_ports": [],
            "output_ports": ["output"],
            "shapes": {"output": ((),)},
        },
        {
            "type": "Sine",
            "params": {"amplitude": 1.0},
            "input_ports": [],
            "output_ports": ["output"],
            "shapes": {"output": ((),)},
        },
        {
            "type": "Pulse",
            "params": {"amplitude": 1.0},
            "input_ports": [],
            "output_ports": ["output"],
            "shapes": {"output": ((),)},
        },
        {
            "type": "Noise",
            "params": {"shape": [2, 3]},
            "input_ports": [],
            "output_ports": ["output"],
            "shapes": {"output": ((2, 3),)},
        },
        {
            "type": "Saturation",
            "inputs": {"input": jnp.zeros((3,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "DelayLine",
            "params": {"input_shape": [2]},
            "inputs": {},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
        {
            "type": "MLP",
            "params": {"output_size": 7},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((7,),)},
        },
        {
            "type": "Linear",
            "params": {"output_size": 6},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((6,),)},
        },
        {
            "type": "GRU",
            "params": {"hidden_size": 4},
            "input_ports": ["input", "hidden"],
            "output_ports": ["output", "hidden"],
            "shapes": {"output": ((4,),), "hidden": ((4,),)},
        },
        {
            "type": "VanillaRNN",
            "params": {"hidden_size": 4},
            "input_ports": ["input", "hidden"],
            "output_ports": ["output", "hidden"],
            "shapes": {"output": ((4,),), "hidden": ((4,),)},
        },
        {
            "type": "LSTM",
            "params": {"hidden_size": 4},
            "input_ports": ["input", "hidden", "cell"],
            "output_ports": ["output", "hidden", "cell"],
            "shapes": {"output": ((4,),), "hidden": ((4,),), "cell": ((4,),)},
        },
        {
            "type": "Spring",
            "inputs": {"displacement": jnp.zeros((2,))},
            "input_ports": ["displacement"],
            "output_ports": ["force"],
            "shapes": {"force": ((2,),)},
        },
        {
            "type": "Damper",
            "inputs": {"velocity": jnp.zeros((2,))},
            "input_ports": ["velocity"],
            "output_ports": ["force"],
            "shapes": {"force": ((2,),)},
        },
        {
            "type": "Channel",
            "params": {"input_shape": [3]},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "FeedbackChannels",
            "params": {"input_shape": [[2], [3]]},
            "input_ports": ["mechanics"],
            "output_ports": ["feedback"],
            "shapes": {"feedback": ((2,), (3,))},
        },
        {
            "type": "FirstOrderFilter",
            "inputs": {"input": jnp.zeros((4,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "AddNoise",
            "inputs": {"input": jnp.zeros((4,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "NetworkClamp",
            "inputs": {"input": jnp.zeros((4,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "ReluMuscle",
            "input_ports": ["excitation"],
            "output_ports": ["force", "activation"],
            "shapes": {"force": ((),), "activation": ((),)},
        },
        {
            "type": "NetworkConstantInput",
            "inputs": {"input": jnp.zeros((4,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "ConstantInput",
            "inputs": {"input": jnp.zeros((4,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "Integrator",
            "params": {"n_dims": 3},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "Derivative",
            "inputs": {"input": jnp.zeros((2,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
        {
            "type": "PID",
            "params": {"n_dims": 2},
            "input_ports": ["error"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
        {
            "type": "PIDDiscrete",
            "inputs": {"error": jnp.zeros((5,))},
            "input_ports": ["error"],
            "output_ports": ["output"],
            "shapes": {"output": ((5,),)},
        },
        {
            "type": "IntegratorDiscrete",
            "params": {"n_dims": 2},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
        {
            "type": "UnitDelay",
            "inputs": {"input": jnp.zeros((3,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "ZeroOrderHold",
            "params": {"n_dims": 4},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((4,),)},
        },
        {
            "type": "Mux",
            "params": {"n_inputs": 3},
            "inputs": {
                "in_0": jnp.zeros((2,)),
                "in_1": {"x": jnp.zeros((2, 2))},
                "in_2": jnp.zeros(()),
            },
            "input_ports": ["in_0", "in_1", "in_2"],
            "output_ports": ["output"],
            "shapes": {"output": ((7,),)},
        },
        {
            "type": "Ravel",
            "inputs": {"input": {"x": jnp.zeros((2, 3)), "y": jnp.zeros((1,))}},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((7,),)},
        },
        {
            "type": "Switch",
            "inputs": {
                "true_input": jnp.zeros((3,)),
                "false_input": jnp.zeros((3,)),
            },
            "input_ports": ["condition", "true_input", "false_input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "DeadZone",
            "inputs": {"input": jnp.zeros((3,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "RateLimiter",
            "params": {"n_dims": 2},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
        {
            "type": "HighPassFilter",
            "inputs": {"input": jnp.zeros((3,))},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((3,),)},
        },
        {
            "type": "BandPassFilter",
            "params": {"n_dims": 2},
            "input_ports": ["input"],
            "output_ports": ["output"],
            "shapes": {"output": ((2,),)},
        },
    ],
)
def test_registered_builtin_output_prototypes(case: dict[str, Any]) -> None:
    outputs = _registered_outputs(
        case["type"],
        case.get("params", {}),
        case.get("inputs", {}),
        input_ports=case["input_ports"],
        output_ports=case["output_ports"],
    )

    for port, expected_shapes in case["shapes"].items():
        assert _tree_shapes(outputs[port]) == expected_shapes


@pytest.mark.parametrize("component_type", ["TwoLinkArm", "PointMass"])
def test_registered_mechanics_state_output_prototypes(component_type: str) -> None:
    outputs = _registered_outputs(
        component_type,
        {},
        {},
        input_ports=["force"],
        output_ports=["effector", "state"],
    )

    assert isinstance(outputs["effector"], CartesianState)
    assert outputs["state"] is outputs["effector"]


def test_registered_linear_state_space_output_prototype() -> None:
    outputs = _registered_outputs(
        "LinearStateSpace",
        {
            "A": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "B": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            "pos_slice": [0, 1],
            "vel_slice": [1, 3],
        },
        {},
        input_ports=["force", "epsilon"],
        output_ports=["effector", "state"],
    )

    assert isinstance(outputs["effector"], CartesianState)
    assert outputs["effector"].pos.shape == (1,)
    assert outputs["effector"].vel.shape == (2,)
    assert outputs["effector"].force.shape == (2,)
    assert outputs["state"].shape == (3,)


def test_demux_output_prototype_splits_input_final_dimension() -> None:
    outputs = _registered_outputs(
        "Demux",
        {"sizes": [2, 1, 3]},
        {"input": jnp.zeros((4, 6))},
        input_ports=["input"],
        output_ports=["out_0", "out_1", "out_2"],
    )

    assert outputs["out_0"].shape == (4, 2)
    assert outputs["out_1"].shape == (4, 1)
    assert outputs["out_2"].shape == (4, 3)


def test_demux_output_prototype_requires_input_prototype() -> None:
    with pytest.raises(ValueError, match="Demux output prototype requires input prototype 'input'"):
        _registered_outputs(
            "Demux",
            {"sizes": [2, 1]},
            {},
            input_ports=["input"],
            output_ports=["out_0", "out_1"],
        )


def test_demux_output_prototype_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="Demux input final dimension 4.*sum\\(sizes\\) 5"):
        _registered_outputs(
            "Demux",
            {"sizes": [2, 3]},
            {"input": jnp.zeros((4,))},
            input_ports=["input"],
            output_ports=["out_0", "out_1"],
        )


def test_demux_output_prototype_infers_through_graph_with_deferred_input() -> None:
    registry = ComponentRegistry(load_user_components=False)
    spec = GraphSpec(
        nodes={
            "join": ComponentSpec(
                type="Mux",
                params={"n_inputs": 2},
                input_ports=["in_0", "in_1"],
                output_ports=["output"],
            ),
            "split": ComponentSpec(
                type="Demux",
                params={"sizes": [2, 3]},
                input_ports=["input"],
                output_ports=["out_0", "out_1"],
            ),
            "left": ComponentSpec(
                type="Gain",
                params={"gain": 1.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "right": ComponentSpec(
                type="Gain",
                params={"gain": 1.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="split",
                source_port="out_0",
                target_node="left",
                target_port="input",
            ),
            WireSpec(
                source_node="split",
                source_port="out_1",
                target_node="right",
                target_port="input",
            ),
            WireSpec(
                source_node="join",
                source_port="output",
                target_node="split",
                target_port="input",
            ),
        ],
        input_ports=["a", "b"],
        output_ports=["left", "right"],
        input_bindings={"a": ("join", "in_0"), "b": ("join", "in_1")},
        output_bindings={"left": ("left", "output"), "right": ("right", "output")},
    )

    inputs = infer_node_input_prototypes(
        spec,
        {("__graph__", "a"): jnp.zeros((2,)), ("__graph__", "b"): jnp.zeros((3,))},
        {},
        component_registry=registry,
    )

    assert inputs[("split", "input")].shape == (5,)
    assert inputs[("left", "input")].shape == (2,)
    assert inputs[("right", "input")].shape == (3,)


def test_fixed_field_graphspec_preserves_params_override_semantics() -> None:
    graph = spec_to_graph(
        _single_node_spec(
            "FixedField",
            {
                "scale": 1.0,
                "amplitude": 1.0,
                "field": [0.0, 0.0],
                "active": False,
                "label": "override_field",
            },
            input_ports=["force", "params_override"],
            output_ports=["force"],
        ),
        component_registry=_components(),
    )
    field = graph.nodes["field"]

    outputs = _call_component(
        field,
        {
            "force": jnp.array([0.5, -0.5], dtype=jnp.float32),
            "params_override": FixedFieldParams(
                scale=2.0,
                amplitude=3.0,
                field=jnp.array([1.0, -2.0], dtype=jnp.float32),
                active=True,
            ),
        },
    )

    assert jnp.allclose(outputs["force"], jnp.array([6.5, -12.5], dtype=jnp.float32))


def test_dynamics_matrix_perturb_graphspec_rejects_bad_delta_shape() -> None:
    with pytest.raises(ValueError, match="delta_A must have shape"):
        spec_to_graph(
            _single_node_spec(
                "DynamicsMatrixPerturb",
                {
                    "scale": 1.0,
                    "delta_A": [[1.0, 2.0, 3.0]],
                    "active": True,
                    "mass": 1.0,
                },
                input_ports=["effector", "force", "params_override"],
                output_ports=["force"],
            ),
            component_registry=_components(),
        )


def test_dynamics_matrix_perturb_graphspec_requires_explicit_mass() -> None:
    with pytest.raises(ValueError, match="missing required parameter.*'mass'"):
        spec_to_graph(
            _single_node_spec(
                "DynamicsMatrixPerturb",
                {"scale": 1.0, "delta_A": [[1.0, 0.0], [0.0, 1.0]], "active": True},
                input_ports=["effector", "force", "params_override"],
                output_ports=["force"],
            ),
            component_registry=_components(),
        )


def _affine_composer_params() -> dict[str, Any]:
    return {
        "schema_version": "feedbax.component.affine_value_composer.v1",
        "output_block_size": 2,
        "feature_rules": [
            {
                "kind": "target_relative_difference",
                "state_slice": [0, 2],
                "target_slice": [0, 2],
            },
            {"kind": "identity", "state_slice": [2, 4]},
        ],
        "gain_init": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
        "bias_init": [0.5, -0.5],
        "use_bias": True,
        "label": "composer_binding",
    }


def _affine_composer_inputs() -> dict[str, Any]:
    return {
        "base": jnp.array([10.0, 20.0], dtype=jnp.float32),
        "state": jnp.array([3.0, 4.0, 5.0, 6.0], dtype=jnp.float32),
        "target": jnp.array([1.0, 2.0], dtype=jnp.float32),
    }


def test_affine_value_composer_runtime_and_trainable_defaults() -> None:
    params = _affine_composer_params()
    component = AffineValueComposer(
        output_block_size=2,
        feature_rules=params["feature_rules"],
        gain_init=params["gain_init"],
        bias_init=params["bias_init"],
        use_bias=True,
        label="composer_binding",
    )

    outputs = _call_component(component, _affine_composer_inputs())

    assert jnp.allclose(outputs["value"], jnp.array([12.5, 26.5], dtype=jnp.float32))

    updated = eqx.tree_at(
        lambda node: (node.gain, node.bias),
        component,
        (
            jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32),
            jnp.array([1.0, 2.0], dtype=jnp.float32),
        ),
    )
    updated_outputs = _call_component(updated, _affine_composer_inputs())

    assert jnp.allclose(updated_outputs["value"], jnp.array([16.0, 28.0], dtype=jnp.float32))


def test_affine_value_composer_graphspec_materializes_round_trips_and_overrides() -> None:
    spec = _single_node_spec(
        "AffineValueComposer",
        _affine_composer_params(),
        input_ports=["base", "state", "target", "gain", "bias"],
        output_ports=["value"],
    )

    graph = spec_to_graph(spec, component_registry=_components())
    component = graph.nodes["field"]

    assert isinstance(component, AffineValueComposer)
    assert component.output_block_size == 2
    assert component.feature_dim == 4
    assert set(component.task_parameter_state_indices()) == {
        "composer_binding.gain",
        "composer_binding.bias",
    }

    inputs = {
        **_affine_composer_inputs(),
        "gain": jnp.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        "bias": jnp.array([2.0, 3.0], dtype=jnp.float32),
    }
    outputs = _call_component(component, inputs)
    assert jnp.allclose(outputs["value"], jnp.array([14.0, 25.0], dtype=jnp.float32))

    roundtrip = graph_to_spec(graph)
    node = roundtrip.nodes["field"]
    assert node.type == "AffineValueComposer"
    assert node.param_schema_version == "feedbax.component.affine_value_composer.v1"
    assert node.params["output_block_size"] == 2
    assert node.params["feature_rules"] == _affine_composer_params()["feature_rules"]
    assert jnp.allclose(
        jnp.asarray(node.params["gain_init"]),
        jnp.asarray(_affine_composer_params()["gain_init"]),
    )
    assert jnp.allclose(
        jnp.asarray(node.params["bias_init"]),
        jnp.asarray(_affine_composer_params()["bias_init"]),
    )
    assert node.params["label"] == "composer_binding"


def test_affine_value_composer_component_parameter_binding_updates_state(
    application_registry_bundle,
) -> None:
    spec = GraphSpec(
        nodes={
            "field": ComponentSpec(
                type="AffineValueComposer",
                params=_affine_composer_params(),
                input_ports=["base", "state", "target", "gain", "bias"],
                output_ports=["value"],
            )
        },
        input_ports=["base", "state", "target"],
        output_ports=["value"],
        input_bindings={
            "base": ("field", "base"),
            "state": ("field", "state"),
            "target": ("field", "target"),
        },
        output_bindings={"value": ("field", "value")},
    )
    graph = spec_to_graph(
        spec,
        component_registry=application_registry_bundle.components,
    )
    binding_spec = StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "composer_gain",
                    "label": "Composer gain",
                    "kind": "intervention",
                    "role": "component_parameter",
                    "path": "intervene.composer_gain",
                    "bindable": True,
                    "dtype": "array",
                    "value_spec": {
                        "mode": "constant",
                        "value": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    },
                    "metadata": {"temporal_support": "constant"},
                },
                {
                    "id": "composer_bias",
                    "label": "Composer bias",
                    "kind": "intervention",
                    "role": "component_parameter",
                    "path": "intervene.composer_bias",
                    "bindable": True,
                    "dtype": "vector",
                    "value_spec": {
                        "mode": "constant",
                        "value": [1.0, 2.0],
                    },
                    "metadata": {"temporal_support": "constant"},
                },
            ],
            "bindings": [
                {
                    "id": "task:composer_gain->field:gain",
                    "source_data_id": "composer_gain",
                    "target_node_id": "field",
                    "target_port": "gain",
                    "role": "component_parameter",
                    "metadata": {"task_parameter_label": "composer_binding.gain"},
                },
                {
                    "id": "task:composer_bias->field:bias",
                    "source_data_id": "composer_bias",
                    "target_node_id": "field",
                    "target_port": "bias",
                    "role": "component_parameter",
                    "metadata": {"task_parameter_label": "composer_binding.bias"},
                },
            ],
            "metadata": {},
        }
    )

    issues = validate_task_binding_schema(
        binding_spec,
        spec,
        "/task_binding_spec",
        component_registry=application_registry_bundle.components,
    )
    assert not [issue for issue in issues if issue.severity == "error"]
    exposure = expose_task_bindings(graph, binding_spec)
    assert exposure.input_plans == ()
    assert {plan.target_label for plan in exposure.state_init_plans} == {
        "composer_binding.gain",
        "composer_binding.bias",
    }

    state = init_state_from_component(exposure.graph)
    updated_state = apply_task_parameter_state_inits(
        state,
        exposure.graph,
        exposure.state_init_plans,
        {
            "composer_gain": jnp.array(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=jnp.float32,
            ),
            "composer_bias": jnp.array([1.0, 2.0], dtype=jnp.float32),
        },
    )
    component = exposure.graph.nodes["field"]
    outputs, _ = component(_affine_composer_inputs(), updated_state, key=jr.PRNGKey(0))

    assert jnp.allclose(outputs["value"], jnp.array([16.0, 28.0], dtype=jnp.float32))


def test_affine_value_composer_output_prototypes_and_negative_validation() -> None:
    registry = ComponentRegistry(load_user_components=False)
    node_spec = ComponentSpec(
        type="AffineValueComposer",
        params=_affine_composer_params(),
        input_ports=["base", "state", "target", "gain", "bias"],
        output_ports=["value"],
    )
    inputs = {
        ("field", "base"): jnp.zeros((2,), dtype=jnp.float32),
        ("field", "state"): jnp.zeros((4,), dtype=jnp.float32),
        ("field", "target"): jnp.zeros((2,), dtype=jnp.float32),
    }

    outputs = output_prototypes_for_node("field", node_spec, inputs, {}, registry)

    assert outputs["value"] is inputs[("field", "base")]

    with pytest.raises(ValueError, match="input prototype 'base'"):
        output_prototypes_for_node(
            "field",
            node_spec,
            {("field", "state"): jnp.zeros((4,), dtype=jnp.float32)},
            {},
            registry,
        )
    with pytest.raises(ValueError, match="input prototype 'state'"):
        output_prototypes_for_node(
            "field",
            node_spec,
            {("field", "base"): jnp.zeros((2,), dtype=jnp.float32)},
            {},
            registry,
        )
    with pytest.raises(ValueError, match="state_slice .* exceeds state size"):
        output_prototypes_for_node(
            "field",
            node_spec.model_copy(
                update={
                    "params": {
                        **_affine_composer_params(),
                        "feature_rules": [{"kind": "identity", "state_slice": [3, 5]}],
                    }
                }
            ),
            inputs,
            {},
            registry,
        )
    with pytest.raises(ValueError, match="base prototype trailing dimension"):
        output_prototypes_for_node(
            "field",
            node_spec.model_copy(
                update={"params": {**_affine_composer_params(), "output_block_size": 3}}
            ),
            inputs,
            {},
            registry,
        )


def test_affine_value_composer_registry_metadata() -> None:
    registry = ComponentRegistry(load_user_components=False)
    meta = registry.get("AffineValueComposer")

    assert meta is not None
    assert meta.output_prototype_fn is not None
    assert meta.param_schema_version == "feedbax.component.affine_value_composer.v1"
    assert set(meta.input_ports) == {"base", "state", "target", "gain", "bias"}
    assert meta.output_ports == ["value"]
    assert meta.default_params["label"] == "affine_value_composer"
    assert meta.default_params["schema_version"] == "feedbax.component.affine_value_composer.v1"


def test_channel_rejects_unknown_noise_model() -> None:
    with pytest.raises(ValueError, match="Unsupported Channel noise_model"):
        spec_to_graph(
            GraphSpec(
                nodes={
                    "channel": ComponentSpec(
                        type="Channel",
                        params={"delay": 0, "noise_model": "unsupported", "input_shape": [2]},
                        input_ports=["input"],
                        output_ports=["output"],
                    )
                },
                input_ports=["input"],
                output_ports=["output"],
                input_bindings={"input": ("channel", "input")},
                output_bindings={"output": ("channel", "output")},
            ),
            component_registry=_components(),
        )


def test_demux_graphspec_materializes_and_round_trips_dynamic_ports() -> None:
    spec = GraphSpec(
        nodes={
            "split": ComponentSpec(
                type="Demux",
                params={"sizes": [2, 1, 3]},
                input_ports=["input"],
                output_ports=["out_0", "out_1", "out_2"],
            )
        },
        input_ports=["input"],
        output_ports=["first", "middle", "last"],
        input_bindings={"input": ("split", "input")},
        output_bindings={
            "first": ("split", "out_0"),
            "middle": ("split", "out_1"),
            "last": ("split", "out_2"),
        },
    )

    graph = spec_to_graph(spec, component_registry=_components())

    split = graph.nodes["split"]
    assert isinstance(split, Demux)
    assert split.sizes == (2, 1, 3)
    assert split.output_ports == ("out_0", "out_1", "out_2")

    state = init_state_from_component(graph)
    outputs, _ = graph({"input": jnp.arange(6.0)}, state, key=jax.random.PRNGKey(0))

    assert jnp.allclose(outputs["first"], jnp.array([0.0, 1.0]))
    assert jnp.allclose(outputs["middle"], jnp.array([2.0]))
    assert jnp.allclose(outputs["last"], jnp.array([3.0, 4.0, 5.0]))

    roundtrip = graph_to_spec(graph)
    node = roundtrip.nodes["split"]
    assert node.type == "Demux"
    assert node.params["sizes"] == [2, 1, 3]
    assert node.input_ports == ["input"]
    assert node.output_ports == ["out_0", "out_1", "out_2"]
    assert roundtrip.output_bindings == spec.output_bindings


def test_graphspec_build_materializes_omitted_dynamic_ports_from_policy() -> None:
    spec = GraphSpec(
        nodes={
            "join": ComponentSpec(type="Mux", params={"n_inputs": 2}),
            "split": ComponentSpec(type="Demux", params={"sizes": [2, 1]}),
        },
        wires=[
            WireSpec(
                source_node="join",
                source_port="output",
                target_node="split",
                target_port="input",
            )
        ],
        input_ports=["left", "right"],
        output_ports=["left", "right"],
        input_bindings={"left": ("join", "in_0"), "right": ("join", "in_1")},
        output_bindings={"left": ("split", "out_0"), "right": ("split", "out_1")},
    )

    graph = spec_to_graph(spec, component_registry=_components())
    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"left": jnp.array([1.0, 2.0]), "right": jnp.array([3.0])},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert tuple(graph.nodes["join"].input_ports) == ("in_0", "in_1")
    assert tuple(graph.nodes["split"].output_ports) == ("out_0", "out_1")
    assert jnp.allclose(outputs["left"], jnp.array([1.0, 2.0]))
    assert jnp.allclose(outputs["right"], jnp.array([3.0]))


def test_mux_graphspec_rejects_n_inputs_port_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Mux node 'join'.*n_inputs=2"):
        spec_to_graph(
            GraphSpec(
                nodes={
                    "join": ComponentSpec(
                        type="Mux",
                        params={"n_inputs": 2},
                        input_ports=["in_0", "in_1", "in_2"],
                        output_ports=["output"],
                    )
                },
                output_ports=["output"],
                output_bindings={"output": ("join", "output")},
            ),
            component_registry=_components(),
        )


def test_demux_graphspec_rejects_sizes_output_port_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Demux node 'split'.*sizes=\\[2, 1, 3\\]"):
        spec_to_graph(
            GraphSpec(
                nodes={
                    "split": ComponentSpec(
                        type="Demux",
                        params={"sizes": [2, 1, 3]},
                        input_ports=["input"],
                        output_ports=["out_0", "out_1"],
                    )
                },
                input_ports=["input"],
                output_ports=["tail"],
                input_bindings={"input": ("split", "input")},
                output_bindings={"tail": ("split", "out_1")},
            ),
            component_registry=_components(),
        )


def test_demux_graphspec_executes_as_internal_node() -> None:
    spec = GraphSpec(
        nodes={
            "join": ComponentSpec(
                type="Mux",
                params={"n_inputs": 2},
                input_ports=["in_0", "in_1"],
                output_ports=["output"],
            ),
            "split": ComponentSpec(
                type="Demux",
                params={"sizes": [2, 1]},
                input_ports=["input"],
                output_ports=["out_0", "out_1"],
            ),
        },
        wires=[
            WireSpec(
                source_node="join",
                source_port="output",
                target_node="split",
                target_port="input",
            )
        ],
        input_ports=["left", "right"],
        output_ports=["left", "right"],
        input_bindings={"left": ("join", "in_0"), "right": ("join", "in_1")},
        output_bindings={"left": ("split", "out_0"), "right": ("split", "out_1")},
    )

    graph = spec_to_graph(spec, component_registry=_components())
    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"left": jnp.array([1.0, 2.0]), "right": jnp.array([3.0])},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["left"], jnp.array([1.0, 2.0]))
    assert jnp.allclose(outputs["right"], jnp.array([3.0]))


def test_demux_graphspec_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="Demux sizes must be positive"):
        spec_to_graph(
            GraphSpec(
                nodes={
                    "split": ComponentSpec(
                        type="Demux",
                        params={"sizes": [2, 0]},
                        input_ports=["input"],
                        output_ports=["out_0", "out_1"],
                    )
                },
                input_ports=["input"],
                output_ports=["output"],
                input_bindings={"input": ("split", "input")},
                output_bindings={"output": ("split", "out_0")},
            ),
            component_registry=_components(),
        )


def test_demux_rejects_mismatched_input_width() -> None:
    graph = spec_to_graph(
        GraphSpec(
            nodes={
                "split": ComponentSpec(
                    type="Demux",
                    params={"sizes": [2, 2]},
                    input_ports=["input"],
                    output_ports=["out_0", "out_1"],
                )
            },
            input_ports=["input"],
            output_ports=["output"],
            input_bindings={"input": ("split", "input")},
            output_bindings={"output": ("split", "out_0")},
        ),
        component_registry=_components(),
    )
    state = init_state_from_component(graph)

    with pytest.raises(ValueError, match="Demux input final dimension"):
        graph({"input": jnp.arange(3.0)}, state, key=jax.random.PRNGKey(0))
