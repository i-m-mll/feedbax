from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from feedbax.runtime.channel import Channel
from feedbax.runtime.components import DelayLine, Linear
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.graph import Graph, Wire, init_state_from_component
from feedbax.contracts.graphs.templates import (
    network_template_graph,
    recurrent_graph_input_initializer,
    recurrent_node_output_initializer,
    recurrent_controller_template_graph,
    simple_feedback_template_graph,
)
from feedbax.contracts.graphs.normalization import normalize_graph_for_studio_authoring
from feedbax.models.networks import LeakyRNNCell, SimpleStagedNetwork, VanillaRNN
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.graphs.prototypes import infer_node_input_prototypes
from feedbax.contracts.graphs.serialization import graph_to_spec, spec_to_graph
from feedbax.component_registry import ComponentRegistry


def _task_binding_spec() -> dict:
    return {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
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


def _network_controller_graph_spec() -> GraphSpec:
    network = network_template_graph(
        {
            "input_size": 3,
            "hidden_size": 5,
            "out_size": 2,
            "hidden_type": "GRUCell",
            "out_nonlinearity": "tanh",
        }
    )
    return GraphSpec(
        nodes={
            **network.nodes,
            "mechanics": ComponentSpec(
                type="PointMass",
                params={"dt": 0.02},
                input_ports=["force"],
                output_ports=["effector", "state"],
            ),
        },
        wires=[
            WireSpec(
                source_node=network.output_bindings["output"][0],
                source_port=network.output_bindings["output"][1],
                target_node="mechanics",
                target_port="force",
            )
        ],
        input_ports=["input", "feedback"],
        output_ports=["effector"],
        input_bindings=dict(network.input_bindings),
        output_bindings={"effector": ("mechanics", "effector")},
        parameter_constraints=network.parameter_constraints,
    )


def test_spec_to_graph_controller_template_instantiates_explicit_nodes() -> None:
    graph = spec_to_graph(_network_controller_graph_spec(), {})

    assert not any(isinstance(node, SimpleStagedNetwork) for node in graph.nodes.values())
    assert set(graph.nodes) == {"input_mux", "cell", "readout", "mechanics"}


def test_gru_network_subgraph_runs_with_recurrent_zero_initializer() -> None:
    subgraph = recurrent_controller_template_graph(
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
    subgraph = recurrent_controller_template_graph(
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


def test_vanilla_rnn_network_subgraph_runs_and_serializes_roundtrip() -> None:
    subgraph = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="VanillaRNN",
        out_nonlinearity="identity",
    )
    json_roundtrip = GraphSpec.model_validate_json(subgraph.model_dump_json())

    assert json_roundtrip.nodes["cell"].type == "VanillaRNN"
    assert json_roundtrip.nodes["cell"].input_ports == ["input", "hidden"]
    assert json_roundtrip.nodes["cell"].output_ports == ["output", "hidden"]

    graph = spec_to_graph(json_roundtrip, {})
    state = init_state_from_component(graph)

    assert isinstance(graph.nodes["cell"], VanillaRNN)
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

    serialized = graph_to_spec(graph)
    runtime_roundtrip = GraphSpec.model_validate_json(serialized.model_dump_json())
    assert runtime_roundtrip.nodes["cell"].type == "VanillaRNN"
    assert runtime_roundtrip.nodes["cell"].params["activation"] == "tanh"


def test_network_hidden_type_normalization_preserves_vanilla_and_rejects_unknown() -> None:
    subgraph = network_template_graph(
        {
            "input_size": 3,
            "hidden_size": 4,
            "out_size": 2,
            "hidden_type": "VanillaRNNCell",
        }
    )

    assert subgraph.nodes["cell"].type == "VanillaRNN"

    with pytest.raises(
        ValueError,
        match="Network\\.params\\.hidden_type.*BogusCell.*VanillaRNN",
    ):
        network_template_graph(
            {
                "input_size": 3,
                "hidden_size": 4,
                "out_size": 2,
                "hidden_type": "BogusCell",
            }
        )

    with pytest.raises(
        ValueError,
        match="Recurrent Controller\\.params\\.cell_type.*Linear.*VanillaRNN",
    ):
        recurrent_controller_template_graph(
            input_size=3,
            hidden_size=4,
            out_size=2,
            cell_type="Linear",
        )


@pytest.mark.parametrize(
    ("nonlinearity", "expected_fn"),
    [(jnp.tanh, jnp.tanh), (jax.nn.relu, jax.nn.relu)],
)
def test_leaky_rnn_cell_plain_recurrence_math(nonlinearity, expected_fn) -> None:
    cell = LeakyRNNCell(
        2,
        2,
        use_bias=True,
        nonlinearity=nonlinearity,
        dt=1.0,
        tau=1.0,
        key=jax.random.PRNGKey(0),
    )
    weight_ih = jnp.array([[0.5, -0.25], [0.75, 0.1]])
    weight_hh = jnp.array([[0.2, 0.3], [-0.4, 0.6]])
    bias = jnp.array([0.1, -0.2])
    cell = eqx.tree_at(lambda c: c.weight_ih, cell, weight_ih)
    cell = eqx.tree_at(lambda c: c.weight_hh, cell, weight_hh)
    cell = eqx.tree_at(lambda c: c.bias, cell, bias)

    inputs = jnp.array([1.0, -2.0])
    state = jnp.array([0.25, -0.5])
    expected = expected_fn(weight_ih @ inputs + weight_hh @ state + bias)

    assert jnp.allclose(cell(inputs, state), expected)


def _graph_input_recurrent_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 0.5},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        wires=[
            WireSpec(
                source_node="gain",
                source_port="output",
                target_node="gain",
                target_port="input",
                temporality="recurrent",
                recurrent_initializer=recurrent_graph_input_initializer(
                    "seed",
                    state_slot="input",
                ),
            )
        ],
        input_ports=["seed"],
        output_ports=["out"],
        output_bindings={"out": ("gain", "output")},
    )


def _node_output_recurrent_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "encoder": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 0.5},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="gain",
                source_port="output",
                target_node="gain",
                target_port="input",
                temporality="recurrent",
                recurrent_initializer=recurrent_node_output_initializer(
                    "encoder",
                    "output",
                    state_slot="input",
                ),
            )
        ],
        input_ports=["context"],
        output_ports=["out"],
        input_bindings={"context": ("encoder", "input")},
        output_bindings={"out": ("gain", "output")},
    )


def test_graph_input_recurrent_initializer_graphspec_runs_from_external_input() -> None:
    graph = spec_to_graph(
        _graph_input_recurrent_spec(),
        {},
        input_prototypes={("__graph__", "seed"): jnp.zeros((2,))},
    )
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {"seed": jnp.array([8.0, 10.0])},
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert jnp.allclose(
        outputs["out"],
        jnp.array([[4.0, 5.0], [2.0, 2.5], [1.0, 1.25]]),
    )


def test_graph_input_recurrent_initializer_serializes_verbatim() -> None:
    spec = _graph_input_recurrent_spec()

    json_roundtrip = GraphSpec.model_validate_json(spec.model_dump_json())
    runtime_roundtrip = graph_to_spec(
        spec_to_graph(
            spec,
            {},
            input_prototypes={("__graph__", "seed"): jnp.zeros((2,))},
        )
    )

    assert json_roundtrip.wires[0].recurrent_initializer == {
        "kind": "graph-input",
        "scope": "trial",
        "source": "seed",
        "state_slot": "input",
    }
    assert runtime_roundtrip.wires[0].recurrent_initializer == spec.wires[0].recurrent_initializer


def test_graph_input_recurrent_initializer_feeds_prototype_inference() -> None:
    spec = _graph_input_recurrent_spec()

    input_prototypes = infer_node_input_prototypes(
        spec,
        {("__graph__", "seed"): jnp.zeros((3,))},
        {},
        component_registry={},
    )

    assert input_prototypes[("gain", "input")].shape == (3,)


def test_node_output_recurrent_initializer_graphspec_runs_from_source_node() -> None:
    graph = spec_to_graph(
        _node_output_recurrent_spec(),
        {},
        input_prototypes={("__graph__", "context"): jnp.zeros((2,))},
    )
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {"context": jnp.array([3.0, 5.0])},
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert jnp.allclose(
        outputs["out"],
        jnp.array([[3.0, 5.0], [1.5, 2.5], [0.75, 1.25]]),
    )


def test_node_output_recurrent_initializer_serializes_verbatim() -> None:
    spec = _node_output_recurrent_spec()

    json_roundtrip = GraphSpec.model_validate_json(spec.model_dump_json())
    runtime_roundtrip = graph_to_spec(
        spec_to_graph(
            spec,
            {},
            input_prototypes={("__graph__", "context"): jnp.zeros((2,))},
        )
    )

    assert json_roundtrip.wires[0].recurrent_initializer == {
        "kind": "node-output",
        "scope": "trial",
        "source_node": "encoder",
        "source_port": "output",
        "state_slot": "input",
    }
    assert runtime_roundtrip.wires[0].recurrent_initializer == spec.wires[0].recurrent_initializer


def _constant_recurrent_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 0.5},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        wires=[
            WireSpec(
                source_node="gain",
                source_port="output",
                target_node="gain",
                target_port="input",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "constant",
                    "scope": "trial",
                    "value": [8.0, 10.0],
                    "dtype": "float32",
                    "state_slot": "input",
                },
            )
        ],
        input_ports=[],
        output_ports=["out"],
        output_bindings={"out": ("gain", "output")},
    )


def test_constant_recurrent_initializer_serializes_dtype_verbatim() -> None:
    spec = _constant_recurrent_spec()

    json_roundtrip = GraphSpec.model_validate_json(spec.model_dump_json())
    runtime_roundtrip = graph_to_spec(spec_to_graph(spec, {}))

    assert json_roundtrip.wires[0].recurrent_initializer == {
        "kind": "constant",
        "scope": "trial",
        "value": [8.0, 10.0],
        "dtype": "float32",
        "state_slot": "input",
    }
    assert runtime_roundtrip.wires[0].recurrent_initializer == spec.wires[0].recurrent_initializer


def test_constant_recurrent_initializer_dtype_preserved_end_to_end(enable_jax_x64) -> None:
    spec = _constant_recurrent_spec()
    graph = spec_to_graph(GraphSpec.model_validate_json(spec.model_dump_json()), {})

    wire = next(w for w in graph.wires if w.temporality == "recurrent")
    value = graph._initial_value_from_recurrent_initializer(wire)

    assert value.dtype == jnp.float32
    assert jnp.array_equal(value, jnp.asarray([8.0, 10.0], dtype=jnp.float32))


def test_node_output_recurrent_initializer_feeds_prototype_inference() -> None:
    spec = _node_output_recurrent_spec()

    input_prototypes = infer_node_input_prototypes(
        spec,
        {("__graph__", "context"): jnp.zeros((3,))},
        {},
        component_registry={},
    )

    assert input_prototypes[("gain", "input")].shape == (3,)


def test_recurrent_controller_template_is_explicit_plain_graph() -> None:
    subgraph = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="identity",
    )

    assert set(subgraph.nodes) == {"input_mux", "cell", "readout"}
    assert subgraph.input_ports == ["input", "feedback"]
    assert subgraph.input_bindings == {
        "input": ("input_mux", "in_0"),
        "feedback": ("input_mux", "in_1"),
    }
    assert subgraph.output_bindings["hidden"] == ("cell", "hidden")
    assert any(
        wire.source_node == "cell"
        and wire.source_port == "hidden"
        and wire.target_node == "cell"
        and wire.target_port == "hidden"
        and wire.temporality == "recurrent"
        for wire in subgraph.wires
    )


def test_authoring_normalization_does_not_turn_unsupported_runtime_params_into_ports() -> None:
    spec = GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="SimpleStagedNetwork",
                params={
                    "input_size": 4,
                    "hidden_size": 3,
                    "output_size": 2,
                    "unsupported_gating": "multiplicative",
                    "unsupported_gain": [0.1, -0.2, 0.3],
                },
                input_ports=["target", "feedback"],
                output_ports=["output", "hidden"],
            )
        },
        input_ports=["target", "feedback"],
        output_ports=["output", "hidden"],
        input_bindings={
            "target": ("network", "target"),
            "feedback": ("network", "feedback"),
        },
        output_bindings={
            "output": ("network", "output"),
            "hidden": ("network", "hidden"),
        },
    )

    normalized = normalize_graph_for_studio_authoring(spec)

    assert normalized.subgraphs is None
    assert normalized.nodes["network"].type == "SimpleStagedNetwork"
    assert "modulator_input" not in normalized.nodes["network"].params


def test_legacy_runtime_network_serialization_inlines_explicit_controller_nodes() -> None:
    network = SimpleStagedNetwork(
        input_size=4,
        hidden_size=3,
        out_size=2,
        key=jax.random.PRNGKey(5),
    )
    spec = graph_to_spec(
        Graph(
            nodes={"network": network},
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
            input_bindings={"input": ("network", "input"), "feedback": ("network", "feedback")},
            output_bindings={"output": ("network", "output"), "hidden": ("network", "hidden")},
        )
    )
    assert spec.subgraphs is None
    assert "network" not in spec.nodes
    assert set(spec.nodes) == {"network_input_mux", "network_cell", "network_readout"}
    assert spec.input_bindings == {
        "input": ("network_input_mux", "in_0"),
        "feedback": ("network_input_mux", "in_1"),
    }
    assert spec.output_bindings == {
        "output": ("network_readout", "output"),
        "hidden": ("network_cell", "hidden"),
    }
    assert spec.nodes["network_cell"].params["input_size"] == 4


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

    network = definitions["Recurrent Controller"]
    feedback = definitions["Simple Feedback Loop"]

    assert "Network" not in definitions

    assert network.template_id == "feedbax.templates.recurrent_controller"
    assert network.template_kind == "executable"
    assert network.template_graph is not None
    assert set(network.template_graph.nodes) == {"input_mux", "cell", "readout"}

    assert feedback.template_id == "feedbax.templates.simple_feedback"
    assert feedback.template_kind == "executable"
    assert feedback.template_graph is not None
    assert feedback.template_graph.subgraphs is None
    assert {"input_mux", "cell", "readout"} <= set(feedback.template_graph.nodes)


def test_generic_template_node_instantiates_persisted_subgraph() -> None:
    subgraph = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="identity",
    )
    spec = GraphSpec(
        nodes={
            "controller": ComponentSpec(
                type="Recurrent Controller",
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


def test_delayed_center_out_task_params_round_trip_through_graph_spec() -> None:
    spec = GraphSpec(
        nodes={
            "task": ComponentSpec(
                type="DelayedReaches",
                params={
                    "preset": "delayed_center_out",
                    "n_control_stages": 8,
                    "workspace": [[-1.0, -1.0], [1.0, 1.0]],
                    "epoch_len_ranges": [[2, 2]],
                    "p_catch_trial": 0.25,
                },
                output_ports=["inputs", "targets", "inits", "intervene"],
            )
        },
        output_ports=["inputs"],
        output_bindings={"inputs": ("task", "inputs")},
    )

    round_tripped = graph_to_spec(spec_to_graph(spec, {}))
    params = round_tripped.nodes["task"].params

    assert params["preset"] == "delayed_center_out"
    assert params["n_steps"] == 9
    assert params["epoch_names"] == ["prep", "movement"]
    assert params["target_on_epochs"] == [0, 1]
    assert params["hold_epochs"] == [0]
    assert params["move_epochs"] == [1]
    assert params["target_visible_from_start"] is True
    assert params["go_cue_event_name"] == "go_cue"
    assert params["catch_metadata_policy"] == "flag"
    assert params["p_catch_trial"] == 0.25


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
    spec = GraphSpec(schema_version="feedbax.spec.graph.v99")

    with pytest.raises(ValueError, match="source_version='feedbax.spec.graph.v99'"):
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
