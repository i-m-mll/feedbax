from __future__ import annotations

import jax
import jax.numpy as jnp

from feedbax.components import Linear
from feedbax.graph import Graph, init_state_from_component
from feedbax.graph_templates import network_template_graph, standard_network_subgraph
from feedbax.nn import SimpleStagedNetwork
from feedbax.web.models.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.web.serialization import spec_to_graph
from feedbax.web.services.component_registry import ComponentRegistry


def _task_binding_spec() -> dict:
    return {
        "schema_version": "feedbax.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "inputs",
                "label": "Inputs",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs",
                "bindable": True,
                "dtype": "vector",
                "expected_shape": [3],
                "metadata": {},
            }
        ],
        "bindings": [
            {
                "id": "task:inputs->network:input",
                "source_data_id": "inputs",
                "target_node_id": "network",
                "target_port": "input",
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _network_controller_graph_spec(*, node_type: str = "Network") -> GraphSpec:
    network = network_template_graph(
        {
            "input_size": 3,
            "hidden_size": 5,
            "out_size": 2,
            "hidden_type": "GRUCell",
            "out_nonlinearity": "tanh",
        }
    )
    network_node = network.nodes["network"].model_copy(update={"type": node_type})
    subgraphs = network.subgraphs if node_type == "Network" else None
    return GraphSpec(
        nodes={
            "network": network_node,
            "mechanics": ComponentSpec(
                type="PointMass",
                params={"dt": 0.02},
                input_ports=["force"],
                output_ports=["effector", "state"],
            ),
        },
        wires=[
            WireSpec(
                source_node="network",
                source_port="output",
                target_node="mechanics",
                target_port="force",
            )
        ],
        input_ports=["input", "feedback"],
        output_ports=["effector"],
        input_bindings={
            "input": ("network", "input"),
            "feedback": ("network", "feedback"),
        },
        output_bindings={"effector": ("mechanics", "effector")},
        subgraphs=subgraphs,
    )


def test_spec_to_graph_network_instantiates_serialized_subgraph() -> None:
    graph = spec_to_graph(_network_controller_graph_spec(node_type="Network"), {})

    assert isinstance(graph.nodes["network"], Graph)
    assert not isinstance(graph.nodes["network"], SimpleStagedNetwork)
    assert set(graph.nodes["network"].nodes) == {"input_mux", "cell", "readout"}


def test_gru_network_subgraph_runs_with_recurrent_zero_initializer() -> None:
    subgraph = standard_network_subgraph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="identity",
    )
    graph = spec_to_graph(subgraph, {})
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {
            "input": jnp.ones((3, 2)),
            "feedback": jnp.zeros((3, 1)),
        },
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert outputs["output"].shape == (3, 2)
    assert outputs["hidden"].shape == (3, 4)


def test_lstm_network_subgraph_runs_with_recurrent_zero_initializers() -> None:
    subgraph = standard_network_subgraph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="LSTM",
        out_nonlinearity="identity",
    )
    graph = spec_to_graph(subgraph, {})
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {
            "input": jnp.ones((3, 2)),
            "feedback": jnp.zeros((3, 1)),
        },
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert outputs["output"].shape == (3, 2)
    assert outputs["hidden"].shape == (3, 4)


def test_linear_activation_from_graph_spec_is_honored() -> None:
    spec = GraphSpec(
        nodes={
            "linear": ComponentSpec(
                type="Linear",
                params={
                    "input_size": 1,
                    "output_size": 1,
                    "activation": "sigmoid",
                },
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("linear", "input")},
        output_bindings={"output": ("linear", "output")},
    )
    graph = spec_to_graph(spec, {})
    state = init_state_from_component(graph)
    component = graph.nodes["linear"]

    assert isinstance(component, Linear)
    assert component.activation_name == "sigmoid"

    inputs = {"input": jnp.array([2.0])}
    outputs, _ = graph(inputs, state, key=jax.random.PRNGKey(0))
    raw_output = component.layer(inputs["input"])

    assert jnp.allclose(outputs["output"], jax.nn.sigmoid(raw_output))
    assert not jnp.allclose(outputs["output"], raw_output)


def test_builtin_executable_templates_are_exposed_in_component_registry() -> None:
    definitions = {component.name: component for component in ComponentRegistry().list_all()}

    network = definitions["Network Template"]
    feedback = definitions["Simple Feedback Loop"]

    assert network.template_id == "feedbax.templates.network"
    assert network.template_kind == "executable"
    assert network.template_graph is not None
    assert set(network.template_graph.nodes) == {"input_mux", "cell", "readout"}

    assert feedback.template_id == "feedbax.templates.simple_feedback"
    assert feedback.template_kind == "executable"
    assert feedback.template_graph is not None
    assert feedback.template_graph.subgraphs is not None
    assert "network" in feedback.template_graph.subgraphs


def test_generic_template_node_instantiates_persisted_subgraph() -> None:
    subgraph = standard_network_subgraph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="identity",
    )
    spec = GraphSpec(
        nodes={
            "controller": ComponentSpec(
                type="Network Template",
                params={},
                input_ports=["input", "feedback"],
                output_ports=["output", "hidden"],
            )
        },
        input_ports=["input", "feedback"],
        output_ports=["output"],
        input_bindings={
            "input": ("controller", "input"),
            "feedback": ("controller", "feedback"),
        },
        output_bindings={"output": ("controller", "output")},
        subgraphs={"controller": subgraph},
    )

    graph = spec_to_graph(spec, {})
    assert isinstance(graph.nodes["controller"], Graph)

    state = init_state_from_component(graph)
    outputs, _ = graph(
        {
            "input": jnp.ones((3, 2)),
            "feedback": jnp.zeros((3, 1)),
        },
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert outputs["output"].shape == (3, 2)
