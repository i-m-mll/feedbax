import jax
import jax.numpy as jnp

from feedbax.channel import Channel, ChannelSpec
from feedbax.graph import init_state_from_component
from feedbax.iterate import run_component
from feedbax.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.nn import SimpleStagedNetwork
from feedbax.bodies import SimpleFeedback


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
