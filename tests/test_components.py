import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from feedbax._tree import filter_spec_leaves
from feedbax.channel import Channel, ChannelSpec
from feedbax.components import ElementwiseAffineModulator
from feedbax.graph import init_state_from_component
from feedbax.iterate import run_component
from feedbax.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.misc import attr_str_tree_to_where_func, where_func_to_attr_str_tree
from feedbax.nn import SimpleStagedNetwork
from feedbax.bodies import SimpleFeedback


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

    updated_leaves = [
        leaf for leaf in jax.tree.leaves(updated.net) if hasattr(leaf, "shape")
    ]
    replacement_leaves = [
        leaf for leaf in jax.tree.leaves(replacement) if hasattr(leaf, "shape")
    ]

    assert updated.net is updated.nodes["net"]
    assert all(
        jnp.array_equal(updated_leaf, replacement_leaf)
        for updated_leaf, replacement_leaf in zip(updated_leaves, replacement_leaves)
    )
