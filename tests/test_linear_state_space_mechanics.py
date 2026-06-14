from __future__ import annotations

import jax
import jax.numpy as jnp

from feedbax.runtime.graph import Graph, init_state_from_component
from feedbax.mechanics import LinearStateSpace
from feedbax.runtime.state_feedback import StateFeedbackSelector
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.component_registry import ComponentRegistry


def test_linear_state_space_steps_without_epsilon() -> None:
    component = LinearStateSpace(
        A=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        B=jnp.array([[0.0], [1.0]]),
        initial_state=jnp.array([1.0, 2.0]),
    )
    state = init_state_from_component(component)

    outputs, state = component({"force": jnp.array([3.0])}, state, key=jax.random.PRNGKey(0))

    expected = jnp.array([3.0, 5.0])
    assert jnp.allclose(outputs["state"], expected)
    assert jnp.allclose(state.get(component.state_index).vector, expected)


def test_linear_state_space_uses_basis_b_w_injection() -> None:
    component = LinearStateSpace(
        A=jnp.eye(3),
        B=jnp.zeros((3, 1)),
        B_w=jnp.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        initial_state=jnp.zeros(3),
    )
    state = init_state_from_component(component)

    outputs, _ = component(
        {"force": jnp.array([0.0]), "epsilon": jnp.array([3.0, 4.0])},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["state"], jnp.array([3.0, 8.0, 0.0]))


def test_linear_state_space_projects_effector_slices() -> None:
    component = LinearStateSpace(
        A=jnp.eye(5),
        B=jnp.zeros((5, 2)),
        initial_state=jnp.array([9.0, 1.0, 2.0, 3.0, 4.0]),
        pos_slice=(1, 3),
        vel_slice=(3, 5),
    )
    state = init_state_from_component(component)

    outputs, _ = component({"force": jnp.array([5.0, 6.0])}, state, key=jax.random.PRNGKey(0))

    assert jnp.allclose(outputs["effector"].pos, jnp.array([1.0, 2.0]))
    assert jnp.allclose(outputs["effector"].vel, jnp.array([3.0, 4.0]))
    assert jnp.allclose(outputs["effector"].force, jnp.array([5.0, 6.0]))


def test_linear_state_space_graph_spec_round_trips() -> None:
    spec = GraphSpec(
        nodes={
            "mechanics": ComponentSpec(
                type="LinearStateSpace",
                params={
                    "A": [[1.0, 1.0], [0.0, 1.0]],
                    "B": [[0.0], [1.0]],
                    "B_w": [[1.0], [0.5]],
                    "dt": 0.02,
                    "initial_state": [1.0, 2.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            )
        },
        input_ports=["force", "epsilon"],
        output_ports=["state"],
        input_bindings={
            "force": ("mechanics", "force"),
            "epsilon": ("mechanics", "epsilon"),
        },
        output_bindings={"state": ("mechanics", "state")},
    )

    graph = spec_to_graph(spec, {})
    assert isinstance(graph.nodes["mechanics"], LinearStateSpace)

    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"force": jnp.array([3.0]), "epsilon": jnp.array([4.0])},
        state,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(outputs["state"], jnp.array([7.0, 7.0]))

    restored = graph_to_spec(
        Graph(
            nodes={"mechanics": graph.nodes["mechanics"]},
            input_ports=("force", "epsilon"),
            output_ports=("state",),
            input_bindings={
                "force": ("mechanics", "force"),
                "epsilon": ("mechanics", "epsilon"),
            },
            output_bindings={"state": ("mechanics", "state")},
        )
    )

    assert restored.nodes["mechanics"].type == "LinearStateSpace"
    assert restored.nodes["mechanics"].params["A"] == [[1.0, 1.0], [0.0, 1.0]]
    assert restored.nodes["mechanics"].params["pos_slice"] == [0, 1]


def test_linear_state_space_is_registered_for_graph_specs() -> None:
    meta = ComponentRegistry().get("LinearStateSpace")

    assert meta is not None
    assert meta.input_ports == ["force", "epsilon"]
    assert meta.output_ports == ["effector", "state"]
    assert {schema.name for schema in meta.param_schema} >= {
        "A",
        "B",
        "B_w",
        "initial_state",
        "pos_slice",
        "vel_slice",
    }


def _delayed_feedback_graph_spec(*, target_relative: bool = False) -> GraphSpec:
    selector_channels = [
        {"slice": "position", "transform": "target_minus" if target_relative else "identity"},
        {"slice": "velocity", "transform": "negate" if target_relative else "identity"},
    ]
    input_ports = ["force", "target"] if target_relative else ["force"]
    input_bindings = {"force": ("mechanics", "force")}
    if target_relative:
        input_bindings["target"] = ("selector", "target")
    return GraphSpec(
        nodes={
            "mechanics": ComponentSpec(
                type="LinearStateSpace",
                params={
                    "A": jnp.eye(8).tolist(),
                    "B": jnp.zeros((8, 2)).tolist(),
                    "initial_state": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                    "pos_slice": [0, 2],
                    "vel_slice": [2, 4],
                },
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            ),
            "selector": ComponentSpec(
                type="StateFeedbackSelector",
                params={
                    "expected_state_dim": 8,
                    "state_slices": {
                        "position": {"start": 0, "stop": 2, "block_size": 4, "delay": 1},
                        "velocity": {"start": 2, "stop": 4, "block_size": 4, "delay": 1},
                    },
                    "channels": selector_channels,
                    "output_size": 4,
                },
                input_ports=["state", "target"],
                output_ports=["feedback"],
            ),
        },
        wires=[
            WireSpec(
                source_node="mechanics",
                source_port="state",
                target_node="selector",
                target_port="state",
            )
        ],
        input_ports=input_ports,
        output_ports=["feedback"],
        input_bindings=input_bindings,
        output_bindings={"feedback": ("selector", "feedback")},
    )


def test_state_feedback_selector_materializes_delayed_lss_state_slice() -> None:
    graph = spec_to_graph(_delayed_feedback_graph_spec(), {})

    assert isinstance(graph.nodes["selector"], StateFeedbackSelector)

    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"force": jnp.array([0.0, 0.0])},
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["feedback"], jnp.array([10.0, 20.0, 30.0, 40.0]))


def test_state_feedback_selector_materializes_target_relative_lss_feedback() -> None:
    graph = spec_to_graph(_delayed_feedback_graph_spec(target_relative=True), {})
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {
            "force": jnp.array([0.0, 0.0]),
            "target": jnp.array([12.0, 23.0]),
        },
        state,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(outputs["feedback"], jnp.array([2.0, 3.0, -30.0, -40.0]))


def test_state_feedback_selector_prototype_feeds_stateful_materialization() -> None:
    spec = _delayed_feedback_graph_spec()
    spec.nodes["delay"] = ComponentSpec(
        type="Channel",
        params={"delay": 1, "add_noise": False},
        input_ports=["input"],
        output_ports=["output"],
    )
    spec.wires.append(
        WireSpec(
            source_node="selector",
            source_port="feedback",
            target_node="delay",
            target_port="input",
        )
    )
    spec.output_bindings = {"feedback": ("delay", "output")}

    graph = spec_to_graph(spec, {})

    assert graph.nodes["delay"].input_proto.shape == (4,)


def test_state_feedback_selector_graph_spec_round_trips() -> None:
    graph = spec_to_graph(_delayed_feedback_graph_spec(target_relative=True), {})
    restored = graph_to_spec(graph)

    selector = restored.nodes["selector"]
    assert selector.type == "StateFeedbackSelector"
    assert selector.params["state_slices"]["position"]["delay"] == 1
    assert selector.params["channels"][0]["transform"] == "target_minus"
    assert selector.params["output_size"] == 4
