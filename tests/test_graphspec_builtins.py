from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from equinox.nn import State
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.bodies import FeedbackChannels
from feedbax.channel import Channel
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
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
from feedbax.noise import CompositeNoise, Multiplicative, Normal
from feedbax.serialization import graph_to_spec, spec_to_graph
from feedbax.state import CartesianState


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

    graph = spec_to_graph(spec)

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

    graph = spec_to_graph(spec)

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

    graph = spec_to_graph(spec)

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
        )
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

    graph = spec_to_graph(spec)
    materialized = graph.nodes["field"]

    assert isinstance(materialized, type(direct_component))
    assert materialized.label == params["label"]
    assert tuple(input_ports) == materialized.input_ports
    assert materialized.output_ports == ("force",)
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
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)

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
        )
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
                {"scale": 1.0, "delta_A": [[1.0, 2.0, 3.0]], "active": True},
                input_ports=["effector", "force", "params_override"],
                output_ports=["force"],
            )
        )


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
            )
        )
