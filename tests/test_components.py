import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax
import pytest

import feedbax.components.equinox as graph_eqx
import feedbax.runtime.components as runtime_components
import feedbax._vendor.jax_cookbook.tree as jtree
from feedbax._vendor.jax_cookbook import load, save
from feedbax._vendor.jax_cookbook.tree import filter_spec_leaves
from feedbax.runtime.channel import Channel, ChannelSpec
from feedbax.runtime.components import ElementwiseAffineModulator
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.runtime.state_indices import align_state_indices_like
from feedbax.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.runtime.where_selectors import (
    attr_str_tree_to_where_func,
    where_func_to_attr_str_tree,
)
from feedbax.models.networks import LeakyRNNCell, recurrent_controller
from feedbax.models.feedback import SimpleFeedback


def test_generated_convolution_restores_checkpoint_from_exported_graph(tmp_path):
    from pathlib import Path

    from feedbax.compiler import GraphDocument, compile_graph
    from feedbax.compiler.serialization import graph_to_spec
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec

    registry = ComponentRegistry(load_user_components=False)
    registry.ingest(registry.discover_modules(graph_eqx), provenance="feedbax")
    type_id = "feedbax.equinox.Conv2d"
    repo_root = Path(__file__).resolve().parents[1]
    assert registry.component_source_paths(repo_root)[type_id] == (
        "feedbax/components/equinox.py",
        "scripts/generate_eqx_components.py",
    )
    authored = GraphSpec(
        nodes={
            "conv": ComponentSpec(
                type=type_id,
                params={
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_size": [2, 2],
                    "padding": [[1, 0], [0, 1]],
                    "use_bias": False,
                },
            ),
            "gain": ComponentSpec(type="Gain", params={"gain": 2.0}),
        },
        wires=[
            WireSpec(
                source_node="conv", source_port="output", target_node="gain", target_port="input"
            )
        ],
        input_ports=["image"],
        output_ports=["result"],
        input_bindings={"image": ("conv", "input")},
        output_bindings={"result": ("gain", "output")},
    )
    image = jnp.ones((1, 2, 2))

    def build(spec):
        return compile_graph(
            GraphDocument(graph=spec),
            registry,
            input_prototypes={("conv", "input"): image},
        ).graph

    graph = build(authored)
    trained = eqx.tree_at(
        lambda model: model.nodes["conv"].layer.weight,
        graph,
        jnp.ones((1, 1, 2, 2)),
    )
    exported = graph_to_spec(trained, registry)
    assert exported.nodes["conv"].type == type_id
    assert exported.nodes["conv"].params["padding"] == [[1, 0], [0, 1]]
    assert "weight" not in exported.nodes["conv"].params
    checkpoint = tmp_path / "learned.eqx"
    eqx.tree_serialise_leaves(checkpoint, trained)
    restored = eqx.tree_deserialise_leaves(
        checkpoint, build(GraphSpec.model_validate_json(exported.model_dump_json()))
    )
    outputs, _ = restored(
        {"image": image}, init_state_from_component(restored), key=jax.random.key(0)
    )
    assert jnp.array_equal(outputs["result"], jnp.asarray([[[4.0, 2.0], [8.0, 4.0]]]))


def test_generated_equinox_components_default_trainable_leaves_to_float32():
    layer = graph_eqx.Linear(2, 3, key=jax.random.PRNGKey(0))
    gru = graph_eqx.GRUCell(2, 3, key=jax.random.PRNGKey(1))

    assert layer.layer.weight.dtype == jnp.float32
    assert layer.layer.bias.dtype == jnp.float32
    assert gru.layer.weight_ih.dtype == jnp.float32
    assert gru.layer.weight_hh.dtype == jnp.float32


def test_generated_equinox_components_preserve_explicit_float64():
    with jax.experimental.enable_x64():
        layer = graph_eqx.Linear(2, 3, dtype=jnp.float64, key=jax.random.PRNGKey(0))
        gru = graph_eqx.GRUCell(2, 3, dtype=jnp.float64, key=jax.random.PRNGKey(1))

        assert layer.layer.weight.dtype == jnp.float64
        assert layer.layer.bias.dtype == jnp.float64
        assert gru.layer.weight_ih.dtype == jnp.float64
        assert gru.layer.weight_hh.dtype == jnp.float64


def test_generated_equinox_wrapper_contract_metadata_marks_state_and_key_layers():
    assert graph_eqx.EQUINOX_WRAPPER_SCHEMA_VERSION == "feedbax.equinox_wrappers.v3"
    assert graph_eqx.EQUINOX_WRAPPER_EQUINOX_VERSION == eqx.__version__

    wrapper_names = {
        name
        for name in graph_eqx.__all__
        if name
        not in {
            "EQUINOX_WRAPPER_SCHEMA_VERSION",
            "EQUINOX_WRAPPER_GENERATOR",
            "EQUINOX_WRAPPER_EQUINOX_VERSION",
            "EQUINOX_WRAPPER_CONTRACTS",
        }
    }
    assert set(graph_eqx.EQUINOX_WRAPPER_CONTRACTS) == wrapper_names

    assert graph_eqx.BatchNorm.wrapper_contract == {
        "call_kind": "batch_norm",
        "state_handling": "threads_eqx_state",
        "key_handling": "optional_forwarded",
    }
    assert graph_eqx.Dropout.wrapper_contract["key_handling"] == "forwarded"
    assert graph_eqx.MultiheadAttention.wrapper_contract["key_handling"] == "forwarded"


@pytest.mark.parametrize("explicit_weight", [False, True])
def test_generated_embedding_weight_is_the_live_table(explicit_weight):
    key = jax.random.key(0)
    weight = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) if explicit_weight else None
    component = graph_eqx.Embedding(3, 2, weight=weight, key=key)

    def forward(model):
        return model({"input": jnp.asarray(1)}, None, key=key)[0]["output"]

    if explicit_weight:
        assert jnp.array_equal(forward(component), weight[1])
    assert len(jax.tree.leaves(eqx.filter(component, eqx.is_inexact_array))) == 1
    replacement = jnp.full((3, 2), 7.0)
    edited = eqx.tree_at(lambda model: model.weight, component, replacement)
    assert jnp.array_equal(forward(edited), replacement[1])

    trained = _adam_step(component, lambda model: jnp.sum(forward(model)))
    assert not jnp.array_equal(trained.weight[1], component.weight[1])
    assert jnp.array_equal(forward(trained), trained.weight[1])


@pytest.mark.parametrize("init_alpha", [0.25, jnp.asarray(0.25)])
def test_generated_prelu_init_alpha_exposes_live_slope(init_alpha):
    key = jax.random.key(0)
    component = graph_eqx.PReLU(init_alpha=init_alpha)

    def forward(model):
        return model({"input": jnp.asarray(-2.0)}, None, key=key)[0]["output"]

    assert forward(component) == -0.5
    assert len(jax.tree.leaves(eqx.filter(component, eqx.is_inexact_array))) == 1
    edited = eqx.tree_at(lambda model: model.init_alpha, component, jnp.asarray(0.75))
    assert forward(edited) == -1.5

    trained = _adam_step(component, forward)
    assert trained.init_alpha > component.init_alpha
    assert forward(trained) == -2 * trained.init_alpha


def test_generated_mlp_exposes_live_per_neuron_activations():
    key = jax.random.key(0)
    component = graph_eqx.MLP(
        1,
        1,
        2,
        1,
        activation=eqx.nn.PReLU(0.25),
        final_activation=eqx.nn.PReLU(0.5),
        use_bias=False,
        use_final_bias=False,
        key=key,
    )
    component = eqx.tree_at(
        lambda model: tuple(linear.weight for linear in model.layer.layers),
        component,
        replace_fn=jnp.ones_like,
    )

    def forward(model):
        return model({"input": jnp.asarray([-1.0])}, None, key=key)[0]["output"][0]

    assert component.activation.negative_slope.shape == (1, 2)
    assert component.final_activation.negative_slope.shape == (1,)
    assert len(jax.tree.leaves(eqx.filter(component, eqx.is_inexact_array))) == 4
    assert forward(component) == -0.25

    edited_hidden = eqx.tree_at(
        lambda model: model.activation.negative_slope,
        component,
        jnp.asarray([[0.5, 0.75]]),
    )
    assert forward(edited_hidden) == -0.625
    edited_final = eqx.tree_at(
        lambda model: model.final_activation.negative_slope, component, jnp.asarray([0.75])
    )
    assert forward(edited_final) == -0.375

    trained = _adam_step(component, forward)
    assert jnp.all(trained.activation.negative_slope > component.activation.negative_slope)
    assert jnp.all(trained.final_activation.negative_slope > component.final_activation.negative_slope)
    assert forward(trained) < forward(component)


def _adam_step(model, loss):
    optimizer = optax.adam(0.01)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    updates, _ = optimizer.update(eqx.filter_grad(loss)(model), optimizer_state)
    return eqx.apply_updates(model, updates)


def test_positional_loader_rejects_old_generated_prelu_layout(tmp_path):
    live_slope = jnp.array([0.5, 0.75])
    old_shadow = jnp.array([0.1, 0.2])
    downstream_leaf = jnp.array([1.0, 2.0])
    checkpoint = tmp_path / "parameters.eqx"

    def setup(*, key):
        return graph_eqx.PReLU(jnp.zeros_like(live_slope)), jnp.zeros_like(downstream_leaf)

    # The v2 stream placed the shadow slope between the live slope and the next leaf.
    save(checkpoint, (live_slope, old_shadow, downstream_leaf), hyperparameters={})
    with pytest.raises(ValueError, match="unread data.*PyTree leaf layout"):
        load(checkpoint, setup)

    save(checkpoint, (graph_eqx.PReLU(live_slope), downstream_leaf), hyperparameters={})
    restored, restored_downstream = load(checkpoint, setup)
    assert jnp.array_equal(restored.init_alpha, live_slope)
    assert jnp.array_equal(restored_downstream, downstream_leaf)


def test_generated_batchnorm_threads_equinox_state():
    component = graph_eqx.BatchNorm(3, axis_name="batch")
    state = init_state_from_component(component)
    inputs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)

    def run_one(x, current_state):
        return component({"input": x}, current_state, key=jax.random.PRNGKey(0))

    outputs, new_state = jax.vmap(
        run_one,
        in_axes=(0, None),
        out_axes=(0, None),
        axis_name="batch",
    )(inputs, state)

    assert outputs["output"].shape == (4, 3)
    old_leaves = jtu.tree_leaves(state)
    new_leaves = jtu.tree_leaves(new_state)
    assert any(
        hasattr(old, "shape") and not jnp.allclose(old, new)
        for old, new in zip(old_leaves, new_leaves)
    )


def test_generated_dropout_honors_training_and_inference_modes():
    inputs = {"input": jnp.ones((16,), dtype=jnp.float32)}

    training = graph_eqx.Dropout(p=0.5, inference=False)
    training_state = init_state_from_component(training)
    training_outputs, _ = training(inputs, training_state, key=jax.random.PRNGKey(1))
    assert not jnp.allclose(training_outputs["output"], inputs["input"])

    inference = graph_eqx.Dropout(p=0.5, inference=True)
    inference_state = init_state_from_component(inference)
    inference_outputs, _ = inference(inputs, inference_state, key=jax.random.PRNGKey(1))
    assert jnp.allclose(inference_outputs["output"], inputs["input"])


def test_generated_multihead_attention_forwards_dropout_key():
    component = graph_eqx.MultiheadAttention(
        num_heads=1,
        query_size=2,
        dropout_p=0.5,
        inference=False,
        key=jax.random.PRNGKey(0),
    )
    state = init_state_from_component(component)
    value = jnp.ones((2, 2), dtype=jnp.float32)

    outputs, new_state = component(
        {"query": value, "key_": value, "value": value},
        state,
        key=jax.random.PRNGKey(1),
    )

    assert outputs["output"].shape == (2, 2)
    assert new_state is state


def test_runtime_neural_components_default_trainable_leaves_to_float32():
    linear = runtime_components.Linear(2, 3, key=jax.random.PRNGKey(0))
    mlp = runtime_components.MLP(2, 3, hidden_sizes=(4,), key=jax.random.PRNGKey(1))
    gru = runtime_components.GRU(2, 3, key=jax.random.PRNGKey(2))

    assert linear.layer.weight.dtype == jnp.float32
    assert mlp.linears[0].weight.dtype == jnp.float32
    assert gru.cell.weight_ih.dtype == jnp.float32
    assert gru.state_index.init.hidden.dtype == jnp.float32


def test_runtime_named_activations_use_shared_static_callables():
    first = runtime_components.Linear(2, 3, key=jax.random.PRNGKey(0))
    second = runtime_components.Linear(2, 3, key=jax.random.PRNGKey(1))
    first_mlp = runtime_components.MLP(2, 3, hidden_sizes=(4,), key=jax.random.PRNGKey(2))
    second_mlp = runtime_components.MLP(2, 3, hidden_sizes=(4,), key=jax.random.PRNGKey(3))

    assert first.activation_name == "identity"
    assert first.activation is second.activation
    assert first_mlp.activation is second_mlp.activation
    assert first_mlp.final_activation is second_mlp.final_activation
    assert jtu.tree_structure(first) == jtu.tree_structure(second)


def test_runtime_linear_named_activation_ensemble_aligns_state_indices():
    ensemble = jtree.get_ensemble(
        lambda *, key: runtime_components.Linear(2, 3, key=key),
        n=2,
        key=jax.random.PRNGKey(0),
    )

    aligned = align_state_indices_like(ensemble, ensemble)

    assert aligned.layer.weight.shape == (2, 3, 2)


def test_runtime_linear_accepts_custom_activation_callable():
    def square(x):
        return x * x

    linear = runtime_components.Linear(2, 2, activation=square, key=jax.random.PRNGKey(0))
    state = init_state_from_component(linear)
    raw = linear.layer(jnp.array([2.0, 3.0], dtype=jnp.float32))

    outputs, _ = linear(
        {"input": jnp.array([2.0, 3.0], dtype=jnp.float32)},
        state,
        key=jax.random.PRNGKey(1),
    )

    assert linear.activation is square
    assert linear.activation_name == "square"
    assert jnp.allclose(outputs["output"], raw * raw)


@pytest.mark.parametrize("out_size", [None, 1])
def test_recurrent_controller_activates_output(out_size):
    net = recurrent_controller(
        2, 3, out_size=out_size, out_nonlinearity="sigmoid", key=jax.random.key(0)
    )
    outputs, _ = net(
        {"input": jnp.array([[1.0, -2.0]])},
        init_state_from_component(net),
        key=jax.random.key(1),
    )
    hidden = outputs["hidden"][0]
    raw = hidden if out_size is None else net.nodes["readout"].layer(hidden)

    assert outputs["output"].shape == (1, 3 if out_size is None else out_size)
    assert jnp.allclose(outputs["output"][0], jax.nn.sigmoid(raw))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_recurrent_controller_preserves_dtype_with_noisy_carry(dtype):
    with jax.experimental.enable_x64():
        net = recurrent_controller(
            2,
            3,
            out_size=1,
            hidden_noise_std=0.1,
            hidden_nonlinearity="relu",
            dtype=dtype,
            key=jax.random.key(0),
        )
        hidden = jnp.arange(1, 4, dtype=dtype)
        state = net.nodes["cell"].replace_state_value(
            init_state_from_component(net), ("hidden",), hidden
        )
        assert jnp.array_equal(net.initial_cycle_port_values(state)["cell", "hidden"], hidden)
        outputs, _ = net(
            {"input": jnp.ones((2, 2), dtype=jnp.float64)},
            state,
            key=jax.random.key(1),
        )

        assert net.nodes["cell"].cell.weight_ih.dtype == dtype
        assert net.nodes["readout"].layer.weight.dtype == dtype
        assert net.nodes["cell"].state_index.init.hidden.dtype == dtype
        assert outputs["hidden"].dtype == outputs["output"].dtype == dtype


def test_recurrent_controller_rejects_unregistered_hidden_factory():
    with pytest.raises(TypeError, match="built-in recurrent cell"):
        recurrent_controller(2, 3, hidden_type=eqx.nn.Lambda, key=jax.random.key(0))


def test_recurrent_controller_threads_key_to_stochastic_hidden_cell():
    net = recurrent_controller(
        2,
        3,
        out_size=1,
        hidden_type=LeakyRNNCell,
        cell_params={"use_noise": True, "noise_strength": 0.1},
        key=jax.random.key(0),
    )
    state = init_state_from_component(net)
    inputs = {"input": jnp.ones((2, 2), dtype=jnp.float32)}
    first, _ = net(inputs, state, key=jax.random.key(1))
    repeated, _ = net(inputs, state, key=jax.random.key(1))
    second, _ = net(inputs, state, key=jax.random.key(2))

    assert jnp.array_equal(first["hidden"], repeated["hidden"])
    assert not jnp.array_equal(first["hidden"], second["hidden"])


def test_recurrent_controller_lstm_matches_seeded_cell_encoder_and_readout():
    key = jax.random.key(0)
    key_cell, key_encoder, key_readout = jax.random.split(key, 3)
    net = recurrent_controller(
        2, 3, out_size=1, encoding_size=4, hidden_type=eqx.nn.LSTMCell,
        use_bias=False, key=key,
    )
    cell = eqx.nn.LSTMCell(4, 3, use_bias=False, key=key_cell)
    encoder = eqx.nn.Linear(2, 4, key=key_encoder)
    readout = eqx.nn.Linear(3, 1, key=key_readout)
    signal = jnp.array([[1.0, -2.0], [0.5, 0.0]])
    outputs, state = net(
        {"input": {"task": signal[:, :1]}, "feedback": signal[:, 1:]},
        init_state_from_component(net),
        key=jax.random.key(1),
    )

    hidden, carry = jnp.zeros(3), jnp.zeros(3)
    expected_hidden = []
    for x in signal:
        hidden, carry = cell(encoder(x), (hidden, carry))
        expected_hidden.append(hidden)

    expected_hidden = jnp.stack(expected_hidden)
    assert jnp.allclose(outputs["hidden"], expected_hidden)
    assert jnp.allclose(outputs["output"], expected_hidden @ readout.weight.T)
    assert jnp.allclose(net.state_view(state).cell.cell, carry)


def test_recurrent_controller_steps_with_post_activation_carry():
    graph = recurrent_controller(2, 3, hidden_nonlinearity="relu", key=jax.random.key(0))
    initial = jnp.arange(1, 4, dtype=jnp.float32)
    state = graph.nodes["cell"].replace_state_value(
        init_state_from_component(graph), ("hidden",), initial
    )
    inputs = {"input": jnp.array([1.0, 2.0])}
    cell = graph.nodes["cell"].cell

    first, state, carry = graph.step(inputs, state, key=jax.random.key(1))
    second, _, carry = graph.step(inputs, state, carry, key=jax.random.key(2))

    assert jnp.array_equal(first["hidden"], jax.nn.relu(cell(inputs["input"], initial)))
    expected = jax.nn.relu(cell(inputs["input"], first["hidden"]))
    assert jnp.array_equal(second["hidden"], expected)
    assert jnp.array_equal(carry["cell", "hidden"], expected)


def _call_modulator(component, signal, modulator):
    state = init_state_from_component(component)
    outputs, _ = component(
        {"signal": signal, "modulator": modulator},
        state,
        key=jax.random.PRNGKey(0),
    )
    return outputs["output"]


def test_elementwise_affine_modulator_defaults_to_identity():
    component = ElementwiseAffineModulator(signal_shape=(3,))

    output = _call_modulator(
        component,
        jnp.array([1.0, -2.0, 3.0]),
        jnp.array(10.0),
    )

    assert jnp.allclose(output, jnp.array([1.0, -2.0, 3.0]))


def test_elementwise_affine_modulator_multiplicative_scalar_modulator():
    component = ElementwiseAffineModulator(
        signal_shape=(3,),
        baseline=1.0,
        gain_init=jnp.array([0.5, -1.0, 2.0]),
        bias_init=0.0,
    )
    signal = jnp.array([2.0, 4.0, 8.0])

    output = _call_modulator(component, signal, jnp.array(0.25))

    assert jnp.allclose(output, signal * (1.0 + component.gain * 0.25))


def test_elementwise_affine_modulator_additive_vector_modulator():
    component = ElementwiseAffineModulator(
        signal_shape=(3,),
        baseline=1.0,
        gain_init=0.0,
        bias_init=jnp.array([1.0, 2.0, -1.0]),
    )
    signal = jnp.array([2.0, 4.0, 8.0])
    modulator = jnp.array([0.5, 1.5, 2.0])

    output = _call_modulator(component, signal, modulator)

    assert jnp.allclose(output, signal + component.bias * modulator)


def test_elementwise_affine_modulator_accepts_explicit_scale_and_bias_inputs():
    component = ElementwiseAffineModulator(signal_shape=(2,), gain_init=0.0, bias_init=0.0)
    state = init_state_from_component(component)

    outputs, _ = component(
        {
            "signal": jnp.array([2.0, 4.0]),
            "modulator": jnp.array(0.5),
            "scale": jnp.array([1.0, -0.5]),
            "bias": jnp.array([0.2, 0.4]),
        },
        state,
        key=jax.random.PRNGKey(0),
    )

    expected = jnp.array([2.0, 4.0]) * (1.0 + jnp.array([1.0, -0.5]) * 0.5)
    expected = expected + jnp.array([0.2, 0.4]) * 0.5
    assert jnp.allclose(outputs["output"], expected)


def test_elementwise_affine_modulator_rejects_bad_shapes():
    with pytest.raises(ValueError, match="gain_init shape"):
        ElementwiseAffineModulator(signal_shape=(3,), gain_init=jnp.ones((2,)))

    component = ElementwiseAffineModulator(signal_shape=(3,))
    with pytest.raises(ValueError, match="input shapes"):
        _call_modulator(component, jnp.ones((3,)), jnp.ones((2,)))


def test_channel_delay():
    channel = Channel(
        delay=2,
        noise_func=None,
        add_noise=False,
        input_proto=jnp.zeros(2),
        init_value=0.0,
    )
    state = init_state_from_component(channel)

    out1, state = channel({"input": jnp.array([1.0, 1.0])}, state, key=jax.random.PRNGKey(0))
    out2, state = channel({"input": jnp.array([2.0, 2.0])}, state, key=jax.random.PRNGKey(1))
    out3, state = channel({"input": jnp.array([3.0, 3.0])}, state, key=jax.random.PRNGKey(2))

    assert (out1["output"] == jnp.array([0.0, 0.0])).all()
    assert (out2["output"] == jnp.array([0.0, 0.0])).all()
    assert (out3["output"] == jnp.array([1.0, 1.0])).all()


def test_simplefeedback_runs():
    key = jax.random.PRNGKey(0)
    model = _make_simplefeedback(key)

    n_steps = 5
    inputs = {"input": jnp.zeros((n_steps, 1))}
    state = init_state_from_component(model)

    outputs, _, history = run_component(
        model,
        inputs,
        state,
        key=key,
        n_steps=n_steps,
    )

    assert outputs["effector"].pos.shape == (n_steps, 2)
    assert history.mechanics.effector.pos.shape == (n_steps + 1, 2)


def _make_simplefeedback(key):
    skeleton = PointMass(mass=1.0, damping=0.0)
    plant = DirectForceInput(skeleton)
    mechanics = Mechanics(plant, dt=0.1)

    feedback_spec = ChannelSpec(
        where=lambda state: state.effector.pos,
        delay=0,
        noise_func=None,
    )

    net = recurrent_controller(
        input_size=3,  # 1 task input + 2 feedback
        hidden_size=4,
        out_size=2,
        key=key,
    )

    model = SimpleFeedback(
        net,
        mechanics,
        feedback_spec=feedback_spec,
        motor_delay=0,
        tau_rise=0.0,
        tau_decay=0.0,
    )

    return model


def test_simplefeedback_component_accessors_are_graph_nodes():
    model = _make_simplefeedback(jax.random.PRNGKey(0))

    assert model.net is model.nodes["net"]
    assert model.mechanics is model.nodes["mechanics"]
    assert isinstance(model.nodes["feedback"], Channel)
    assert "feedback_select" in model.nodes
    assert model.efferent_channel is model.nodes["efferent"]
    assert model.force_lp is None


def test_simplefeedback_training_selector_targets_executable_graph_node():
    model = _make_simplefeedback(jax.random.PRNGKey(0))
    where = attr_str_tree_to_where_func("net")
    filter_spec = filter_spec_leaves(model, where)

    assert where(model) is model.nodes["net"]
    assert any(jax.tree.leaves(filter_spec.nodes["net"]))


def test_simplefeedback_nodes_selector_round_trips_to_executable_graph_node():
    model = _make_simplefeedback(jax.random.PRNGKey(0))
    where_str = where_func_to_attr_str_tree(lambda model: model.nodes["net"])
    where = attr_str_tree_to_where_func(where_str)

    assert where_str == "nodes['net']"
    assert where(model) is model.nodes["net"]


def test_simplefeedback_model_net_update_replaces_executable_graph_node():
    model = _make_simplefeedback(jax.random.PRNGKey(0))
    replacement = recurrent_controller(
        input_size=3,
        hidden_size=4,
        out_size=2,
        key=jax.random.PRNGKey(1),
    )

    updated = eqx.tree_at(lambda item: item.net, model, replacement)

    updated_leaves = [leaf for leaf in jax.tree.leaves(updated.net) if hasattr(leaf, "shape")]
    replacement_leaves = [leaf for leaf in jax.tree.leaves(replacement) if hasattr(leaf, "shape")]

    assert updated.net is updated.nodes["net"]
    assert all(
        jnp.array_equal(updated_leaf, replacement_leaf)
        for updated_leaf, replacement_leaf in zip(updated_leaves, replacement_leaves)
    )
