from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from feedbax.channel import Channel
from feedbax.components import DelayLine, Linear
from feedbax.filters import FirstOrderFilter
from feedbax.graph import Graph, Wire, init_state_from_component
from feedbax.graph_templates import (
    network_template_graph,
    simple_feedback_template_graph,
    standard_network_subgraph,
)
from feedbax.nn import SimpleStagedNetwork
from feedbax.contracts.graph import ComponentSpec, GraphMetadata, GraphSpec, WireSpec
from feedbax.serialization import graph_to_spec, spec_to_graph
from feedbax.component_registry import ComponentRegistry


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


def test_delayed_channel_infers_vector_prototype_and_runs_under_scan() -> None:
    spec = GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Linear",
                params={"input_size": 3, "output_size": 4, "activation": "identity"},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "delay": ComponentSpec(
                type="Channel",
                params={"delay": 2, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="readout",
                source_port="output",
                target_node="delay",
                target_port="input",
            )
        ],
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("readout", "input")},
        output_bindings={"output": ("delay", "output")},
    )
    graph = spec_to_graph(spec, {})
    state = init_state_from_component(graph)

    assert isinstance(graph.nodes["delay"], Channel)
    assert graph.nodes["delay"].input_proto.shape == (4,)

    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    def step(carry, args):
        step_state = carry
        value, key = args
        step_outputs, step_state, _ = graph.step({"input": value}, step_state, key=key)
        return step_state, step_outputs

    _, outputs = jax.lax.scan(step, state, (jnp.ones((5, 3)), keys))

    assert outputs["output"].shape == (5, 4)
    assert jnp.allclose(outputs["output"][:2], 0.0)


def test_simple_feedback_template_infers_feedback_motor_and_force_filter_shapes() -> None:
    spec = simple_feedback_template_graph(
        {
            "feedback_delay": 2,
            "motor_delay": 2,
            "tau_rise": 0.05,
            "tau_decay": 0.06,
            "network": {
                "input_size": 8,
                "hidden_size": 5,
                "out_size": 2,
                "out_nonlinearity": "identity",
            },
        }
    )

    assert "input_shape" not in spec.nodes["feedback"].params
    assert "input_shape" not in spec.nodes["efferent"].params
    assert "input_shape" not in spec.nodes["force_filter"].params

    graph = spec_to_graph(spec, {})
    assert graph.nodes["feedback"].input_proto.shape == (6,)
    assert graph.nodes["efferent"].input_proto.shape == (2,)
    assert graph.nodes["force_filter"].input_proto.shape == (2,)

    state = init_state_from_component(graph)
    outputs, _ = graph(
        {"input": jnp.ones((4, 2))},
        state,
        key=jax.random.PRNGKey(1),
        n_steps=4,
    )

    assert outputs["effector"].pos.shape == (4, 2)


def test_stateful_input_shape_fields_round_trip() -> None:
    graph = Graph(
        nodes={
            "source": Linear(2, 3, key=jax.random.PRNGKey(0)),
            "channel": Channel(delay=1, add_noise=False, input_proto=jnp.zeros(3)),
            "delay": DelayLine(delay=1, input_proto=jnp.zeros(3)),
            "filter": FirstOrderFilter(input_proto=jnp.zeros(3)),
        },
        wires=(
            Wire("source", "output", "channel", "input"),
            Wire("channel", "output", "delay", "input"),
            Wire("delay", "output", "filter", "input"),
        ),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("source", "input")},
        output_bindings={"output": ("filter", "output")},
    )

    spec = graph_to_spec(graph)
    assert spec.nodes["channel"].params["input_shape"] == [3]
    assert spec.nodes["delay"].params["input_shape"] == [3]
    assert spec.nodes["filter"].params["input_shape"] == [3]

    restored = spec_to_graph(spec, {})
    assert restored.nodes["channel"].input_proto.shape == (3,)
    assert restored.nodes["delay"].input_proto.shape == (3,)
    assert restored.nodes["filter"].input_proto.shape == (3,)


def test_spec_to_graph_round_trips_external_boundary_verbatim() -> None:
    spec = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["target"],
        output_ports=["readout"],
        input_bindings={"target": ("gain", "input")},
        output_bindings={"readout": ("gain", "output")},
    )

    round_tripped = graph_to_spec(spec_to_graph(spec, {}))

    assert round_tripped.input_ports == spec.input_ports
    assert round_tripped.output_ports == spec.output_ports
    assert round_tripped.input_bindings == spec.input_bindings
    assert round_tripped.output_bindings == spec.output_bindings


def test_stateful_prototype_preflight_error_includes_node_and_port() -> None:
    spec = GraphSpec(
        nodes={
            "delay": ComponentSpec(
                type="DelayLine",
                params={"delay": 1},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    )

    with pytest.raises(ValueError, match="DelayLine node 'delay' port 'input'"):
        spec_to_graph(spec, {})


def test_spec_to_graph_rejects_unsupported_graph_spec_version() -> None:
    spec = GraphSpec(
        metadata=GraphMetadata(
            name="bad",
            created_at="2026-06-11T00:00:00Z",
            updated_at="2026-06-11T00:00:00Z",
            version="9.0.0",
        )
    )

    with pytest.raises(ValueError, match="Unsupported GraphSpec version '9.0.0'"):
        spec_to_graph(spec, {})


def test_spec_to_graph_rejects_missing_required_registry_param() -> None:
    registry = ComponentRegistry()
    spec = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("gain", "input")},
        output_bindings={"output": ("gain", "output")},
    )

    with pytest.raises(ValueError, match="Gain node 'gain'.*'gain'"):
        spec_to_graph(spec, {"Gain": registry.get("Gain")})


def test_spec_to_graph_rejects_missing_network_subgraph_during_prototype_inference() -> None:
    spec = GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                params={"input_size": 3, "hidden_size": 5, "out_size": 2},
                input_ports=["input", "feedback"],
                output_ports=["output", "hidden"],
            ),
            "mechanics": ComponentSpec(
                type="PointMass",
                params={"dt": 0.02},
                input_ports=["force"],
                output_ports=["effector"],
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
        input_ports=["input"],
        output_ports=["effector"],
        input_bindings={"input": ("network", "input")},
        output_bindings={"effector": ("mechanics", "effector")},
    )

    with pytest.raises(ValueError, match="Network node 'network' requires a subgraph"):
        spec_to_graph(spec, {})


def test_spec_to_graph_rejects_missing_source_output_prototype() -> None:
    spec = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "sink": ComponentSpec(
                type="Channel",
                params={"delay": 1, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="gain",
                source_port="output",
                target_node="sink",
                target_port="input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("sink", "output")},
    )

    with pytest.raises(ValueError, match="Gain node 'gain' port 'input'"):
        spec_to_graph(spec, {})
