from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.control.affine import AffineFeedbackController
from feedbax.runtime.graph import Graph, init_state_from_component
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph


def test_affine_feedback_controller_static_gain_and_bias() -> None:
    controller = AffineFeedbackController(
        gain=[[2.0, -1.0], [0.5, 3.0]],
        bias=[1.0, -2.0],
    )
    state = init_state_from_component(controller)

    outputs, _ = controller(
        {"feedback": jnp.array([3.0, 4.0])},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["command"], jnp.array([3.0, 11.5]))


def test_affine_feedback_controller_time_varying_reference_and_feedforward() -> None:
    controller = AffineFeedbackController(
        gain=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 3.0]],
        ],
        bias=[[0.0, 0.0], [1.0, -1.0]],
        feedforward=[[0.5, -0.5], [2.0, 4.0]],
    )
    state = init_state_from_component(controller)
    inputs = {
        "feedback": jnp.array([1.0, 2.0]),
        "reference": jnp.array([4.0, 6.0]),
    }

    first, state = controller(inputs, state, key=jax.random.PRNGKey(0))
    second, _ = controller(inputs, state, key=jax.random.PRNGKey(1))

    assert jnp.allclose(first["command"], jnp.array([3.5, 3.5]))
    assert jnp.allclose(second["command"], jnp.array([9.0, 15.0]))


def test_affine_feedback_controller_accepts_input_feedforward() -> None:
    controller = AffineFeedbackController(gain=jnp.eye(2))
    state = init_state_from_component(controller)

    outputs, _ = controller(
        {
            "feedback": jnp.array([1.0, 2.0]),
            "feedforward": jnp.array([3.0, 4.0]),
        },
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["command"], jnp.array([4.0, 6.0]))


def test_affine_feedback_controller_rejects_ambiguous_feedforward() -> None:
    controller = AffineFeedbackController(gain=jnp.eye(2), feedforward=[1.0, 1.0])
    state = init_state_from_component(controller)

    with pytest.raises(ValueError, match="feedforward is ambiguous"):
        controller(
            {
                "feedback": jnp.array([1.0, 2.0]),
                "feedforward": jnp.array([3.0, 4.0]),
            },
            state,
            key=jax.random.PRNGKey(0),
        )


def test_affine_feedback_controller_graph_spec_round_trips() -> None:
    spec = GraphSpec(
        nodes={
            "controller": ComponentSpec(
                type="AffineFeedbackController",
                params={
                    "gain": [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[2.0, 0.0], [0.0, 3.0]],
                    ],
                    "bias": [[0.0, 0.0], [1.0, -1.0]],
                    "schedule_policy": "hold",
                },
                input_ports=["feedback", "reference", "feedforward"],
                output_ports=["command"],
            )
        },
        input_ports=["feedback", "reference"],
        output_ports=["command"],
        input_bindings={
            "feedback": ("controller", "feedback"),
            "reference": ("controller", "reference"),
        },
        output_bindings={"command": ("controller", "command")},
    )

    graph = spec_to_graph(spec, ComponentRegistry(load_user_components=False))
    assert isinstance(graph.nodes["controller"], AffineFeedbackController)

    state = init_state_from_component(graph)
    inputs = {
        "feedback": jnp.array([1.0, 2.0]),
        "reference": jnp.array([4.0, 6.0]),
    }
    first, state, _ = graph.step(inputs, state, key=jax.random.PRNGKey(0))
    second, _, _ = graph.step(inputs, state, key=jax.random.PRNGKey(1))

    assert jnp.allclose(first["command"], jnp.array([3.0, 4.0]))
    assert jnp.allclose(second["command"], jnp.array([7.0, 11.0]))

    roundtrip = graph_to_spec(
        Graph(
            nodes={"controller": graph.nodes["controller"]},
            input_ports=("feedback", "reference"),
            output_ports=("command",),
            input_bindings={
                "feedback": ("controller", "feedback"),
                "reference": ("controller", "reference"),
            },
            output_bindings={"command": ("controller", "command")},
        )
    )
    node = roundtrip.nodes["controller"]
    assert node.type == "AffineFeedbackController"
    assert node.params["gain"] == spec.nodes["controller"].params["gain"]
    assert node.params["bias"] == spec.nodes["controller"].params["bias"]
    assert node.params["schedule_policy"] == "hold"


def test_affine_feedback_controller_is_registered_for_graph_specs() -> None:
    meta = ComponentRegistry().get("AffineFeedbackController")

    assert meta is not None
    assert meta.input_ports == ["feedback", "reference", "feedforward"]
    assert meta.output_ports == ["command"]
    assert {schema.name for schema in meta.param_schema} >= {
        "gain",
        "bias",
        "feedforward",
        "schedule_policy",
    }
