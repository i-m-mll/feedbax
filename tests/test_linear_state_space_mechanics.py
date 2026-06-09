from __future__ import annotations

import jax
import jax.numpy as jnp

from feedbax.graph import Graph, init_state_from_component
from feedbax.mechanics import LinearStateSpace
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.serialization import graph_to_spec, spec_to_graph
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
