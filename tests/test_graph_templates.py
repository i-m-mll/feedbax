from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from feedbax.runtime.channel import Channel, ChannelSpec
from feedbax.runtime.components import DelayLine, Linear
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.graph import Graph, GraphTraceRequest, Wire, init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.compiler.templates import (
    network_template_graph,
    recurrent_graph_input_initializer,
    recurrent_node_output_initializer,
    recurrent_controller_template_graph,
    simple_feedback_template_graph,
)
from feedbax.compiler import GraphCompilationError, GraphDocument, compile_graph
from feedbax.compiler.normalization import normalize_graph_for_studio_authoring
from feedbax.models.feedback import SimpleFeedback
from feedbax.models.networks import LeakyRNNCell, VanillaRNN, recurrent_controller
from feedbax.objectives import CompositeLoss
from feedbax.tasks import SimpleReaches, prepare_trial
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.compiler.prototypes import infer_node_input_prototypes
from feedbax.compiler.serialization import graph_to_spec
from tests.support.graph_compiler_test_support import spec_to_graph
from feedbax.component_registry import ComponentParamValidationError, ComponentRegistry


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
                type="feedbax.PointMass",
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
    graph = spec_to_graph(_network_controller_graph_spec(), ComponentRegistry())

    assert set(graph.nodes) == {"input_mux", "cell", "readout", "mechanics"}


def test_gru_network_subgraph_runs_with_recurrent_zero_initializer() -> None:
    subgraph = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="GRU",
        out_nonlinearity="identity",
    )
    graph = spec_to_graph(subgraph, ComponentRegistry())
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
    graph = spec_to_graph(subgraph, ComponentRegistry())
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

    assert json_roundtrip.nodes["cell"].type == "feedbax.VanillaRNN"
    assert json_roundtrip.nodes["cell"].input_ports == ["input", "hidden"]
    assert json_roundtrip.nodes["cell"].output_ports == ["output", "hidden"]

    graph = spec_to_graph(json_roundtrip, ComponentRegistry())
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
    assert runtime_roundtrip.nodes["cell"].type == "feedbax.VanillaRNN"
    assert runtime_roundtrip.nodes["cell"].params["activation"] == "tanh"


def test_noisy_vanilla_rnn_rollout_preserves_float32_under_x64(enable_jax_x64) -> None:
    spec = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="VanillaRNN",
        cell_params={"use_noise": True, "noise_strength": 0.2, "dt": 0.25, "tau": 0.5},
        dtype="float32",
    )
    graph = spec_to_graph(spec, ComponentRegistry())
    state = init_state_from_component(graph)
    inputs = {
        "input": jnp.ones((3, 2), dtype=jnp.float32),
        "feedback": jnp.zeros((3, 1), dtype=jnp.float32),
    }
    rollout = eqx.filter_jit(graph)
    key = jax.random.key(0)
    outputs, _ = rollout(inputs, state, key=key, n_steps=3)
    repeated, _ = rollout(inputs, state, key=key, n_steps=3)
    independent, _ = rollout(inputs, state, key=jax.random.key(1), n_steps=3)

    assert outputs["hidden"].dtype == jnp.float32
    assert jnp.array_equal(outputs["hidden"], repeated["hidden"])
    assert not jnp.array_equal(outputs["hidden"], independent["hidden"])


@pytest.mark.parametrize(
    ("cell_use_bias", "readout_use_bias"),
    [(False, False), (True, True)],
)
def test_vanilla_rnn_bias_configuration_survives_runtime_roundtrip(
    cell_use_bias: bool,
    readout_use_bias: bool,
) -> None:
    authored = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="VanillaRNN",
        out_nonlinearity="identity",
    )
    authored.nodes["cell"].params["use_bias"] = cell_use_bias
    authored.nodes["readout"].params["use_bias"] = readout_use_bias

    graph = spec_to_graph(authored, ComponentRegistry())
    serialized = graph_to_spec(graph)
    restored_spec = GraphSpec.model_validate_json(serialized.model_dump_json())
    restored_graph = spec_to_graph(restored_spec, ComponentRegistry())
    canonical = graph_to_spec(restored_graph)

    assert serialized.nodes["cell"].params["use_bias"] is cell_use_bias
    assert serialized.nodes["readout"].params["use_bias"] is readout_use_bias
    assert restored_graph.nodes["cell"].cell.use_bias is cell_use_bias
    assert restored_graph.nodes["readout"].use_bias is readout_use_bias
    assert (restored_graph.nodes["cell"].cell.bias is not None) is cell_use_bias
    assert (restored_graph.nodes["readout"].layer.bias is not None) is readout_use_bias
    assert canonical.model_dump(mode="json") == serialized.model_dump(mode="json")


def test_vanilla_rnn_without_bias_executes_as_homogeneous_linear_recurrence() -> None:
    authored = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="VanillaRNN",
        out_nonlinearity="identity",
    )
    authored.nodes["cell"].params.update({"activation": "identity", "use_bias": False})
    authored.nodes["readout"].params["use_bias"] = False
    graph = spec_to_graph(
        GraphSpec.model_validate_json(
            graph_to_spec(spec_to_graph(authored, ComponentRegistry())).model_dump_json()
        ),
        ComponentRegistry(),
    )
    state = init_state_from_component(graph)

    outputs, _ = graph(
        {
            "input": jnp.zeros((3, 2)),
            "feedback": jnp.zeros((3, 1)),
        },
        state,
        key=jax.random.PRNGKey(0),
        n_steps=3,
    )

    assert jnp.array_equal(outputs["hidden"], jnp.zeros((3, 4)))
    assert jnp.array_equal(outputs["output"], jnp.zeros((3, 2)))


def test_vanilla_rnn_omitted_bias_configuration_keeps_enabled_default() -> None:
    authored = recurrent_controller_template_graph(
        input_size=3,
        hidden_size=4,
        out_size=2,
        cell_type="VanillaRNN",
        out_nonlinearity="identity",
    )
    authored.nodes["cell"].params.pop("use_bias", None)
    authored.nodes["readout"].params.pop("use_bias", None)

    graph = spec_to_graph(authored, ComponentRegistry())
    serialized = graph_to_spec(graph)

    assert graph.nodes["cell"].cell.use_bias is True
    assert graph.nodes["readout"].use_bias is True
    assert serialized.nodes["cell"].params["use_bias"] is True
    assert serialized.nodes["readout"].params["use_bias"] is True


def test_network_hidden_type_normalization_preserves_vanilla_and_rejects_unknown() -> None:
    subgraph = network_template_graph(
        {
            "input_size": 3,
            "hidden_size": 4,
            "out_size": 2,
            "hidden_type": "VanillaRNNCell",
        }
    )

    assert subgraph.nodes["cell"].type == "feedbax.VanillaRNN"

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
                type="feedbax.Gain",
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
                type="feedbax.Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "gain": ComponentSpec(
                type="feedbax.Gain",
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
        ComponentRegistry(),
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


def test_compiled_nested_recurrent_rollout_seeds_from_parent_wire() -> None:
    child = GraphSpec(
        nodes={
            "linear": ComponentSpec(
                type="feedbax.Linear",
                params={"input_size": 1, "output_size": 1, "use_bias": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "delay": ComponentSpec(
                type="feedbax.Channel",
                params={"delay": 1, "noise_std": 0.25},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="linear",
                source_port="output",
                target_node="linear",
                target_port="input",
                temporality="recurrent",
                recurrent_initializer=recurrent_graph_input_initializer("seed", state_slot="input"),
            ),
            WireSpec(
                source_node="linear",
                source_port="output",
                target_node="delay",
                target_port="input",
            ),
        ],
        input_ports=["seed"],
        output_ports=["out"],
        output_bindings={"out": ("delay", "output")},
    )
    registry = ComponentRegistry()

    def compile_parent(*, wired: bool) -> Graph:
        wires = []
        bindings = {"raw": ("encoder", "input")}
        if wired:
            wires.append(
                WireSpec(
                    source_node="encoder",
                    source_port="output",
                    target_node="nested",
                    target_port="seed",
                )
            )
        else:
            bindings["seed"] = ("nested", "seed")
        spec = GraphSpec(
            nodes={
                "encoder": ComponentSpec(
                    type="feedbax.Gain",
                    params={"gain": 2.0},
                    input_ports=["input"],
                    output_ports=["output"],
                ),
                "nested": ComponentSpec(
                    type="feedbax.Subgraph",
                    input_ports=["seed"],
                    output_ports=["out"],
                ),
            },
            subgraphs={"nested": child},
            wires=wires,
            input_ports=list(bindings),
            input_bindings=bindings,
            output_ports=["out"],
            output_bindings={"out": ("nested", "out")},
        )
        return spec_to_graph(
            spec,
            registry,
            input_prototypes={("__graph__", name): jnp.zeros(1) for name in bindings},
        )

    key = jax.random.PRNGKey(17)
    raw = jnp.array([8.0])
    direct = compile_parent(wired=False)
    expected, _, expected_history = direct(
        {"raw": jnp.broadcast_to(raw, (4, 1)), "seed": 2.0 * raw},
        init_state_from_component(direct),
        key=key,
        n_steps=4,
        return_state_history=True,
    )

    wired = compile_parent(wired=True)
    inputs = {"raw": raw}
    outputs, final_state, history = wired(
        inputs,
        init_state_from_component(wired),
        key=key,
        n_steps=4,
        return_state_history=True,
    )
    state_selector = "state_path:nested.delay.output"
    traced, traced_state, trace = wired.rollout_with_trace(
        inputs,
        init_state_from_component(wired),
        key=key,
        n_steps=4,
        trace=(GraphTraceRequest(kind="state_path", selector=state_selector),),
    )
    assert jnp.array_equal(outputs["out"], expected["out"])
    assert jnp.array_equal(history.nested.delay.output, expected_history.nested.delay.output)
    assert jnp.array_equal(history.nested.delay.output[0], jnp.zeros(1))
    assert jnp.array_equal(traced["out"], outputs["out"])
    assert eqx.tree_equal(traced_state, final_state)
    assert jnp.array_equal(trace[state_selector], history.nested.delay.output[1:])


def test_graph_input_recurrent_initializer_serializes_verbatim() -> None:
    spec = _graph_input_recurrent_spec()

    json_roundtrip = GraphSpec.model_validate_json(spec.model_dump_json())
    runtime_roundtrip = graph_to_spec(
        spec_to_graph(
            spec,
            ComponentRegistry(),
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
    )

    assert input_prototypes[("gain", "input")].shape == (3,)


def test_node_output_recurrent_initializer_graphspec_runs_from_source_node() -> None:
    graph = spec_to_graph(
        _node_output_recurrent_spec(),
        ComponentRegistry(),
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
            ComponentRegistry(),
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
                type="feedbax.Gain",
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
    runtime_roundtrip = graph_to_spec(spec_to_graph(spec, ComponentRegistry()))

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
    graph = spec_to_graph(
        GraphSpec.model_validate_json(spec.model_dump_json()),
        ComponentRegistry(),
    )

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


def test_nested_recurrent_graph_serialization_requires_authored_subgraph_identity() -> None:
    network = recurrent_controller(
        input_size=4,
        hidden_size=3,
        out_size=2,
        key=jax.random.PRNGKey(5),
    )
    with pytest.raises(
        ValueError,
        match="Programmatic composite node.*has no declared component binding",
    ):
        graph_to_spec(
            Graph(
                nodes={"network": network},
                input_ports=["input", "feedback"],
                output_ports=["output", "hidden"],
                input_bindings={
                    "input": ("network", "input"),
                    "feedback": ("network", "feedback"),
                },
                output_bindings={
                    "output": ("network", "output"),
                    "hidden": ("network", "hidden"),
                },
            )
        )


def test_linear_activation_from_graph_spec_is_honored() -> None:
    spec = GraphSpec(
        nodes={
            "linear": ComponentSpec(
                type="feedbax.Linear",
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
    graph = spec_to_graph(spec, ComponentRegistry())
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

    network = definitions["feedbax.RecurrentController"]
    feedback = definitions["feedbax.SimpleFeedbackLoop"]

    assert "Network" not in definitions

    assert network.template_id == "feedbax.templates.recurrent_controller"
    assert network.template_kind == "executable"
    assert network.template_graph is not None
    assert set(network.template_graph.nodes) == {"input_mux", "cell", "readout"}

    assert feedback.template_id == "feedbax.templates.simple_feedback"
    assert feedback.template_kind == "executable"
    assert feedback.template_graph is not None
    assert set(feedback.template_graph.subgraphs["net"].nodes) == {"input_mux", "cell", "readout"}
    assert {"feedback_select", "feedback", "feedback_values", "net"} <= set(
        feedback.template_graph.nodes
    )


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
                type="feedbax.RecurrentController",
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

    cell_key = jax.random.key(42)
    graph = compile_graph(
        GraphDocument(graph=spec),
        ComponentRegistry(),
        initialization_keys={("controller", "cell"): cell_key},
    ).graph
    controller = graph.nodes["controller"]
    assert isinstance(controller, Graph)
    assert jnp.array_equal(
        controller.nodes["cell"].cell.weight_ih,
        eqx.nn.GRUCell(3, 4, key=cell_key).weight_ih,
    )

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
                type="feedbax.Linear",
                params={"input_size": 3, "output_size": 4, "activation": "identity"},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "delay": ComponentSpec(
                type="feedbax.Channel",
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
    graph = spec_to_graph(spec, ComponentRegistry())
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

    graph = spec_to_graph(spec, ComponentRegistry())
    assert graph.nodes["feedback"].input_proto.pos.shape == (2,)
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


@pytest.mark.parametrize("start", [[0.0, 0.0], [0.4, -0.3]])
def test_delayed_feedback_template_matches_direct_construction_at_reset_and_continuation(
    start,
) -> None:
    key = jax.random.key(0)
    task = SimpleReaches(
        n_steps=4,
        loss_func=CompositeLoss(()),
        workspace=[[-1, -1], [1, 1]],
        fixed_endpoints=[start, [0.6, 0.2]],
    )
    trial = task.get_train_trial(key)
    registry = ComponentRegistry()
    spec = simple_feedback_template_graph(
        {"feedback_delay": 2, "network": {"input_size": 10, "hidden_size": 5, "out_size": 2}}
    )
    compiled = spec_to_graph(spec, registry)
    graph = spec_to_graph(graph_to_spec(compiled, registry), registry)
    from feedbax.runtime.feedback import StateSelector

    direct = SimpleFeedback(
        graph.nodes["net"],
        graph.nodes["mechanics"],
        feedback_spec=ChannelSpec(where=StateSelector(("effector",)), delay=2, noise_func=None),
        motor_noise_func=None,
    ).prepare()
    assert direct.wires == graph.wires
    assert set(graph_to_spec(direct).nodes) == set(spec.nodes)

    prepared = prepare_trial(graph, trial)
    direct_prepared = prepare_trial(direct, trial)
    initial = graph.state_view(prepared.init_state)
    assert jnp.allclose(initial.mechanics.effector.pos, jnp.asarray(start))
    assert jnp.allclose(initial.mechanics.plant.skeleton.pos, initial.mechanics.effector.pos)
    initial_feedback = initial.mechanics.effector
    assert eqx.tree_equal(initial.feedback.output, initial_feedback)
    for sample in initial.feedback.queue:
        assert eqx.tree_equal(sample, initial_feedback)

    outputs, _, history = run_component(
        graph, prepared.inputs, prepared.init_state, key=key, n_steps=prepared.n_steps
    )
    expected, _, direct_history = run_component(
        direct,
        direct_prepared.inputs,
        direct_prepared.init_state,
        key=key,
        n_steps=direct_prepared.n_steps,
    )
    assert jnp.allclose(outputs["effector"].pos[0], jnp.asarray(start))
    assert eqx.tree_equal(outputs, expected, rtol=1e-6, atol=1e-7)
    assert eqx.tree_equal(history.feedback, direct_history.feedback, rtol=1e-6, atol=1e-7)

    if any(start):
        state = prepare_trial(graph, trial).init_state
        cycle = None
        for t in range(2):
            step_inputs = jax.tree.map(lambda value: value[t], prepared.inputs)
            _, state, cycle = graph.step(step_inputs, state, cycle, key=key)
        remaining = jax.tree.map(lambda value: value[2:], prepared.inputs)
        continued, _, continued_history = graph(
            remaining, state, key=key, n_steps=1, cycle_init=cycle, return_state_history=True
        )
        assert eqx.tree_equal(
            continued, jax.tree.map(lambda value: value[2:], outputs), rtol=1e-6, atol=1e-7
        )
        assert eqx.tree_equal(
            continued_history.feedback.output,
            jax.tree.map(lambda value: value[2:], history.feedback.output),
            rtol=1e-6,
            atol=1e-7,
        )


def test_multi_channel_feedback_round_trip_preserves_noise_and_state_tree() -> None:
    from feedbax.runtime.feedback import StateSelector
    from feedbax.runtime.noise import Multiplicative, Normal

    registry = ComponentRegistry()
    spec = simple_feedback_template_graph(
        {"network": {"input_size": 10, "hidden_size": 5, "out_size": 2}}
    )
    compiled = spec_to_graph(spec, registry)
    direct = SimpleFeedback(
        compiled.nodes["net"],
        compiled.nodes["mechanics"],
        feedback_spec={
            "pose": ChannelSpec(StateSelector(("effector.pos", "effector.vel")), 2, Normal(0.03)),
            "load": [ChannelSpec(StateSelector(("effector.force",)), 1)],
        },
        motor_delay=1,
        motor_noise_func=Multiplicative(Normal(0.01)) + Normal(0.02),
        tau_rise=0.05,
        tau_decay=0.06,
    ).prepare()
    serialized = graph_to_spec(direct, registry)
    restored = spec_to_graph(serialized, registry)
    key = jax.random.key(9)
    task = SimpleReaches(
        n_steps=4,
        loss_func=CompositeLoss(()),
        workspace=[[-1, -1], [1, 1]],
        fixed_endpoints=[[0.4, -0.3], [0.6, 0.2]],
    )
    trial = task.get_train_trial(key)

    def rollout(model):
        prepared = prepare_trial(model, trial)
        return run_component(
            model, prepared.inputs, prepared.init_state, key=key, n_steps=prepared.n_steps
        )

    expected, _, direct_history = rollout(direct)
    actual, _, restored_history = rollout(restored)
    assert eqx.tree_equal(actual, expected)
    assert eqx.tree_equal(restored_history, direct_history)
    assert jnp.any(restored_history.feedback["pose"].noise[0][1:] != 0)
    assert jnp.any(restored_history.efferent.noise[1:] != 0)
    assert restored_history.feedback["load"][0].noise is None


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

    restored = spec_to_graph(spec, ComponentRegistry())
    assert restored.nodes["channel"].input_proto.shape == (3,)
    assert restored.nodes["delay"].input_proto.shape == (3,)
    assert restored.nodes["filter"].input_proto.shape == (3,)


def test_spec_to_graph_round_trips_external_boundary_verbatim() -> None:
    spec = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="feedbax.Gain",
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

    round_tripped = graph_to_spec(spec_to_graph(spec, ComponentRegistry()))

    assert round_tripped.input_ports == spec.input_ports
    assert round_tripped.output_ports == spec.output_ports
    assert round_tripped.input_bindings == spec.input_bindings
    assert round_tripped.output_bindings == spec.output_bindings


def test_delayed_center_out_task_params_round_trip_through_graph_spec() -> None:
    spec = GraphSpec(
        nodes={
            "task": ComponentSpec(
                type="feedbax.DelayedReaches",
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

    round_tripped = graph_to_spec(spec_to_graph(spec, ComponentRegistry()))
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
                type="feedbax.DelayLine",
                params={"delay": 1},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    )

    with pytest.raises(ValueError, match="DelayLine node 'delay' port 'input'"):
        spec_to_graph(spec, ComponentRegistry())


def test_spec_to_graph_rejects_unsupported_graph_spec_version() -> None:
    spec = GraphSpec.model_construct(schema_version="feedbax.spec.graph.v99")

    with pytest.raises(ValueError, match="source_version='feedbax.spec.graph.v99'"):
        spec_to_graph(spec, ComponentRegistry())


def test_spec_to_graph_rejects_missing_required_registry_param() -> None:
    registry = ComponentRegistry()
    spec = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="feedbax.TransferFunction",
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

    with pytest.raises(GraphCompilationError) as caught:
        spec_to_graph(spec, registry)

    diagnostic = caught.value.record.diagnostics[0]
    assert diagnostic.severity == "error"
    assert diagnostic.source_anchor.authored_path == "/graph/nodes/gain"
    assert diagnostic.actionable_context["action"]
    cause = caught.value.__cause__
    assert isinstance(cause, ComponentParamValidationError)
    assert {(failure.path, failure.code) for failure in cause.failures} == {
        (("den",), "missing"),
        (("num",), "missing"),
    }


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
                type="feedbax.PointMass",
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
        spec_to_graph(spec, ComponentRegistry())


def test_spec_to_graph_rejects_missing_source_output_prototype() -> None:
    spec = GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="feedbax.Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "sink": ComponentSpec(
                type="feedbax.Channel",
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

    with pytest.raises(ValueError, match="Gain node 'gain'.*input prototype 'input'"):
        spec_to_graph(spec, ComponentRegistry())
