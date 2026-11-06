from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.serialization import graph_to_spec
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.control.affine import AffineFeedbackController
from feedbax.runtime.graph import Graph, init_state_from_component
from feedbax.runtime.iteration import run_component
from tests.support.graph_compiler_test_support import spec_to_graph


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


@pytest.mark.parametrize(
    ("feedback", "params", "optional_port", "optional_value", "nested", "expected"),
    [
        pytest.param(
            [[1.5, -2.5], [3.5, 4.5]],
            {},
            None,
            None,
            False,
            [[1.5, -2.5], [3.5, 4.5]],
            id="float-feedback",
        ),
        pytest.param(
            [[1, -2], [3, 4]],
            {},
            "reference",
            [5.5, 6.5],
            False,
            [[4.5, 8.5], [2.5, 2.5]],
            id="float-reference",
        ),
        pytest.param(
            [[1, -2], [3, 4]],
            {"bias": [[0.5, 1.5], [-1.5, 2.5]]},
            None,
            None,
            True,
            [[1.5, -0.5], [1.5, 6.5]],
            id="float-bias-with-unused-subgraph-reference",
        ),
        pytest.param(
            [[1, -2], [3, 4]],
            {"feedforward": [[1.5, -0.5], [0.5, 2.5]]},
            None,
            None,
            False,
            [[2.5, -2.5], [3.5, 6.5]],
            id="float-feedforward-schedule",
        ),
        pytest.param(
            [[[1, -2], [3, 4]], [[5, -6], [7, 8]]],
            {},
            "feedforward",
            [0.5, 1.5],
            False,
            [[[1.5, -0.5], [3.5, 5.5]], [[5.5, -4.5], [7.5, 9.5]]],
            id="float-feedforward-port-with-batched-feedback",
        ),
    ],
)
def test_affine_command_survives_a_delayed_channel(
    feedback, params, optional_port, optional_value, nested, expected
) -> None:
    nodes = {
        "controller": ComponentSpec(
            type="AffineFeedbackController",
            params={"gain": [[1, 0], [0, 1]], **params},
            input_ports=["feedback", "reference", "feedforward"],
            output_ports=["command"],
        ),
        "delay": ComponentSpec(
            type="Channel",
            params={"delay": 1, "add_noise": False},
            input_ports=["input"],
            output_ports=["output"],
        ),
    }
    wires = [
        WireSpec(
            source_node="controller",
            source_port="command",
            target_node="delay",
            target_port="input",
        )
    ]
    if optional_port is not None:
        nodes["source"] = ComponentSpec(
            type="Constant", params={"value": optional_value}, output_ports=["output"]
        )
        wires.append(
            WireSpec(
                source_node="source",
                source_port="output",
                target_node="controller",
                target_port=optional_port,
            )
        )
    spec = GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=["feedback"],
        output_ports=["command", "delayed_command"],
        input_bindings={"feedback": ("controller", "feedback")},
        output_bindings={
            "command": ("controller", "command"),
            "delayed_command": ("delay", "output"),
        },
    )
    feedback_target = "controller"
    if nested:
        # The child exposes reference, but the parent only supplies feedback.
        child = spec.model_copy(
            update={
                "input_ports": ["feedback", "reference"],
                "input_bindings": {**spec.input_bindings, "reference": ("controller", "reference")},
            }
        )
        feedback_target = "nested"
        spec = GraphSpec(
            nodes={
                "nested": ComponentSpec(
                    type="Subgraph",
                    input_ports=child.input_ports,
                    output_ports=child.output_ports,
                )
            },
            subgraphs={"nested": child},
            input_ports=["feedback"],
            input_bindings={"feedback": ("nested", "feedback")},
            output_ports=child.output_ports,
            output_bindings={port: ("nested", port) for port in child.output_ports},
        )
    feedback = jnp.asarray(feedback)
    if jnp.issubdtype(feedback.dtype, jnp.floating):
        feedback = feedback.astype(jnp.float32)
    expected = jnp.asarray(expected, dtype=jnp.float32)
    delayed = jnp.concatenate((jnp.zeros_like(expected[:1]), expected[:-1]))
    registry = ComponentRegistry(load_user_components=False)

    def compile_and_check(graph_spec: GraphSpec) -> Graph:
        graph = compile_graph(
            GraphDocument(graph=graph_spec),
            registry,
            input_prototypes={(feedback_target, "feedback"): jnp.zeros_like(feedback[0])},
        ).graph
        outputs, _, _ = run_component(
            graph,
            {"feedback": feedback},
            init_state_from_component(graph),
            key=jax.random.key(0),
            n_steps=len(feedback),
        )
        assert outputs["command"].dtype == outputs["delayed_command"].dtype == jnp.float32
        assert jnp.array_equal(outputs["command"], expected)
        assert jnp.array_equal(outputs["delayed_command"], delayed)
        return graph

    graph = compile_and_check(spec)
    roundtrip = GraphSpec.model_validate_json(graph_to_spec(graph).model_dump_json())
    compile_and_check(roundtrip)
