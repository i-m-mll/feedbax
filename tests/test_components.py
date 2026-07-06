import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import pytest

import feedbax.components.equinox as graph_eqx
import feedbax.runtime.components as runtime_components
import jax_cookbook.tree as jtree
from jax_cookbook.tree import filter_spec_leaves
from feedbax.runtime.channel import Channel, ChannelSpec
from feedbax.runtime.components import ElementwiseAffineModulator
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.runtime.state_indices import align_state_indices_like
from feedbax.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.config.selectors import attr_str_tree_to_where_func, where_func_to_attr_str_tree
from feedbax.models.networks import LeakyRNNCell, SimpleStagedNetwork
from feedbax.models.feedback import SimpleFeedback


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
    assert graph_eqx.EQUINOX_WRAPPER_SCHEMA_VERSION == "feedbax.equinox_wrappers.v2"
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
    assert gru._initial_state.hidden.dtype == jnp.float32


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


def test_simplestagednetwork_defaults_trainable_leaves_to_float32_with_float64_inputs():
    with jax.experimental.enable_x64():
        net = SimpleStagedNetwork(
            input_size=2,
            hidden_size=3,
            out_size=1,
            key=jax.random.PRNGKey(0),
        )
        state = init_state_from_component(net)

        outputs, state = net(
            {
                "input": jnp.ones((2,), dtype=jnp.float64),
            },
            state,
            key=jax.random.PRNGKey(1),
        )

        assert net.hidden.weight_ih.dtype == jnp.float32
        assert net.hidden.weight_hh.dtype == jnp.float32
        assert net.readout.weight.dtype == jnp.float32
        assert net._initial_state.hidden.dtype == jnp.float32
        assert outputs["output"].dtype == jnp.float32
        assert outputs["output"].shape == (1,)


def test_simplestagednetwork_preserves_explicit_float64_trainable_leaves():
    with jax.experimental.enable_x64():
        net = SimpleStagedNetwork(
            input_size=2,
            hidden_size=3,
            out_size=1,
            dtype=jnp.float64,
            key=jax.random.PRNGKey(0),
        )

        assert net.hidden.weight_ih.dtype == jnp.float64
        assert net.hidden.weight_hh.dtype == jnp.float64
        assert net.readout.weight.dtype == jnp.float64
        assert net._initial_state.hidden.dtype == jnp.float64


def test_simplestagednetwork_threads_key_to_stochastic_hidden_cell() -> None:
    net = SimpleStagedNetwork(
        input_size=2,
        hidden_size=3,
        out_size=1,
        hidden_type=lambda *args, **kwargs: LeakyRNNCell(
            *args,
            **kwargs,
            use_noise=True,
            noise_strength=0.1,
        ),
        key=jax.random.PRNGKey(0),
    )
    state = init_state_from_component(net)

    outputs, _ = net(
        {"input": jnp.ones((2,), dtype=jnp.float32)},
        state,
        key=jax.random.PRNGKey(1),
    )

    assert outputs["output"].shape == (1,)


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


def test_elementwise_affine_modulator_matches_multiplicative_sisu_formula():
    net = SimpleStagedNetwork(
        input_size=3,
        hidden_size=3,
        out_size=2,
        sisu_gating="multiplicative",
        key=jax.random.PRNGKey(0),
    )
    alpha = jnp.array([0.2, -0.3, 0.5])
    net = eqx.tree_at(lambda item: item.sisu_alpha, net, alpha)
    component = ElementwiseAffineModulator(
        signal_shape=(3,),
        baseline=1.0,
        gain_init=net.sisu_alpha,
        bias_init=0.0,
    )
    hidden = jnp.array([1.0, 2.0, 3.0])

    for sisu_value in (0.0, 0.5, 1.0):
        output = _call_modulator(component, hidden, jnp.array(sisu_value))
        expected = hidden * (1.0 + net.sisu_alpha * sisu_value)
        assert jnp.allclose(output, expected)


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

    net = SimpleStagedNetwork(
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
    assert model.feedback_channels is model.nodes["feedback"]
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
    replacement = SimpleStagedNetwork(
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
