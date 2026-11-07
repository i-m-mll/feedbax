import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.runtime import Channel, Graph, toggle_channel_noise
from feedbax.runtime.noise import Normal


@pytest.mark.parametrize("legacy_carry", [False, True], ids=["fresh-carry", "legacy-carry"])
def test_direct_channel_can_toggle_noise_in_a_jitted_scan(legacy_carry: bool):
    channel = eqx.filter_jit(
        lambda std, initial: Channel(delay=0, noise_func=Normal(std=std), init_value=initial)
    )(jnp.asarray(0.7), jnp.asarray(0.0))
    disabled = toggle_channel_noise(channel, False)
    restored = toggle_channel_noise(disabled, True)
    signal = jnp.arange(6, dtype=jnp.float32).reshape(6, 1)
    key = jr.key(1)

    @eqx.filter_jit
    def rollout(model, carry):
        def step(carry, step_input):
            value, step_key = step_input
            outputs, carry = model({"input": value}, carry, key=step_key)
            return carry, outputs["output"]

        return jax.lax.scan(step, carry, (signal, jr.split(key, len(signal))))[1]

    def initial_carry(model):
        if legacy_carry:
            model = eqx.tree_at(lambda ch: ch.state_index.init.noise, model, replace=None)
        return model.init_state(key=key)

    noisy = rollout(channel, initial_carry(channel))
    assert not jnp.array_equal(noisy, signal)
    assert jnp.array_equal(rollout(disabled, initial_carry(disabled)), signal)
    assert jnp.array_equal(rollout(restored, initial_carry(restored)), noisy)


def test_nested_channel_toggles_reuse_carry_with_the_delay_queue_intact():
    delayed = Channel(delay=2, noise_func=Normal(std=0.7))
    nested = Graph(
        nodes={"delay": delayed},
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("delay", "input")},
        output_bindings={"output": ("delay", "output")},
    )
    model = Graph(
        nodes={"nested": nested, "instant": Channel(delay=0, add_noise=False)},
        input_ports=("input", "instant"),
        output_ports=("output", "instant"),
        input_bindings={"input": ("nested", "input"), "instant": ("instant", "input")},
        output_bindings={"output": ("nested", "output"), "instant": ("instant", "output")},
    ).prepare()
    flipped = toggle_channel_noise(model)
    restored = toggle_channel_noise(flipped)
    key = jr.key(0)
    prefix = jnp.arange(1, 5, dtype=jnp.float32).reshape(4, 1)
    suffix = prefix + 4

    @eqx.filter_jit
    def rollout(graph, signal, carry):
        outputs, carry, _ = graph.rollout_with_trace(
            {"input": signal, "instant": signal}, carry, key=key, n_steps=len(signal), trace=()
        )
        return outputs, carry

    _, warm = rollout(model, prefix, model.init_state(key=key))
    flipped_outputs, flipped_carry = rollout(flipped, suffix, warm)
    assert jnp.array_equal(flipped_outputs["output"], jnp.concatenate((prefix[-2:], suffix[:-2])))
    assert not jnp.array_equal(flipped_outputs["instant"], suffix)
    assert jnp.array_equal(
        flipped.nodes["nested"].nodes["delay"].state_view(flipped_carry).noise, jnp.zeros(1)
    )

    expected, _ = rollout(model, prefix, flipped_carry)
    actual, _ = rollout(restored, prefix, flipped_carry)
    assert jnp.array_equal(actual["output"], expected["output"])
    assert jnp.array_equal(actual["instant"], prefix)
