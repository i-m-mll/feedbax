from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from feedbax.bodies import FeedbackChannels
from feedbax.channel import Channel
from feedbax.components import Demux
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.graph import init_state_from_component
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.noise import CompositeNoise, Multiplicative, Normal
from feedbax.serialization import graph_to_spec, spec_to_graph


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

    graph = spec_to_graph(spec)

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

    graph = spec_to_graph(spec)
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
            )
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
        )
    )
    state = init_state_from_component(graph)

    with pytest.raises(ValueError, match="Demux input final dimension"):
        graph({"input": jnp.arange(3.0)}, state, key=jax.random.PRNGKey(0))
