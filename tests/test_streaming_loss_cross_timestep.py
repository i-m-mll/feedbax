"""Streaming-loss tests for cross-timestep finite-difference terms."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest
from equinox.nn import State, StateIndex

from feedbax.runtime.streaming import init_streaming_state_window, update_streaming_state_window
from feedbax.runtime._rollout import scan_rollout
from feedbax.objectives.loss import (
    CompositeLoss, NthDifferenceLoss, StateDerivativeLoss, TargetSpec, TargetStateLoss,
)
from feedbax.objectives.streaming import make_streaming_loss_fn


class _FakeState(eqx.Module):
    x: jnp.ndarray
    hidden: jnp.ndarray


def _full_state(values: jnp.ndarray) -> _FakeState:
    return _FakeState(
        x=values[None, :, :],
        hidden=values[None, :, :],
    )


def _step_state(values: jnp.ndarray, t: int) -> _FakeState:
    return _FakeState(
        x=values[t],
        hidden=values[t],
    )


def _stream_loss(loss, values: jnp.ndarray):
    streaming_fn = make_streaming_loss_fn(
        loss,
        trial_specs=None,
        model=None,
        n_steps=values.shape[0],
    )
    order = getattr(streaming_fn, "streaming_order", 0)
    window = init_streaming_state_window(_step_state(values, 0), order)
    total = jnp.asarray(0.0, dtype=values.dtype)

    for t in range(values.shape[0]):
        state_t = _step_state(values, t)
        loss_input = state_t
        if order > 0:
            window = update_streaming_state_window(window, state_t)
            loss_input = window
        total = total + streaming_fn(loss_input, jnp.asarray(t))

    return total


def test_streaming_order_one_matches_full_trajectory_loss():
    t = jnp.arange(7, dtype=jnp.float32)[:, None]
    values = jnp.concatenate([0.2 * t, -0.1 * t], axis=-1)
    loss = StateDerivativeLoss(label="state_derivative", where=lambda s: s.hidden)

    full = loss(_full_state(values), trial_specs=None, model=None).total
    streamed = _stream_loss(loss, values)

    assert jnp.allclose(streamed, full)


def test_streaming_order_two_matches_full_trajectory_loss():
    t = jnp.arange(9, dtype=jnp.float32)[:, None]
    values = jnp.concatenate([jnp.sin(0.4 * t), jnp.cos(0.3 * t)], axis=-1)
    loss = NthDifferenceLoss(label="second_difference", order=2, where=lambda s: s.x)

    full = loss(_full_state(values), trial_specs=None, model=None).total
    streamed = _stream_loss(loss, values)

    assert jnp.allclose(streamed, full, atol=1e-6)


def test_streaming_uses_max_order_for_composite_cross_timestep_terms():
    t = jnp.arange(10, dtype=jnp.float32)[:, None]
    values = jnp.concatenate([jnp.sin(0.2 * t), t**2], axis=-1)
    first = NthDifferenceLoss(label="first", order=1, where=lambda s: s.x)
    second = NthDifferenceLoss(label="second", order=2, where=lambda s: s.x)
    loss = CompositeLoss(
        terms={"first": first, "second": second},
        weights={"first": 0.5, "second": 2.0},
    )

    streaming_fn = make_streaming_loss_fn(
        loss,
        trial_specs=None,
        model=None,
        n_steps=values.shape[0],
    )
    full = loss(_full_state(values), trial_specs=None, model=None).total
    streamed = _stream_loss(loss, values)

    assert streaming_fn.streaming_order == 2
    assert jnp.allclose(streamed, full, atol=1e-6)


class _TrialSpecs(eqx.Module):
    targets: dict = eqx.field(default_factory=dict)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("loss_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_scan_streaming_loss_dtype_and_gradient_parity(
    order, loss_dtype, state_dtype, checkpoint,
):
    """Real scans preserve loss precision independently of the state precision."""
    with jax.experimental.enable_x64():
        n_steps = 5
        index = StateIndex(jnp.zeros(2, dtype=state_dtype))
        init_state = State(index)
        keys = jr.split(jr.PRNGKey(12), n_steps)
        inputs = jnp.linspace(-0.4, 0.8, n_steps * 2, dtype=state_dtype).reshape(-1, 2)
        specs = _TrialSpecs()

        def norm(x):
            return jnp.sum(x.astype(loss_dtype) ** 2, axis=-1)

        if order == 0:
            loss = TargetStateLoss(
                label="target", where=lambda s: s.x, norm=norm,
                spec=TargetSpec(value=jnp.asarray(0.2, dtype=state_dtype)),
            )
        else:
            loss = NthDifferenceLoss(order=order, where=lambda s: s.x, norm=norm)

        def run(scale, streaming):
            def step_fn(step_input, state, step_carry, key, t):
                x = jnp.tanh(
                    state.get(index) + scale * step_input
                    + 0.01 * jr.normal(key, (2,), dtype=state_dtype)
                    + 0.02 * (t.astype(state_dtype) + step_carry)
                )
                return x, state.set(index, x), step_carry + state_dtype(1)

            streaming_fn = (
                make_streaming_loss_fn(loss, specs, None, n_steps) if streaming else None
            )
            outputs, state, aux = scan_rollout(
                step_fn, init_state, inputs, keys, n_steps,
                state_view_fn=lambda s: _FakeState(s.get(index), s.get(index)),
                init_step_carry=state_dtype(0), checkpoint=checkpoint,
                streaming_loss_fn=streaming_fn,
            )
            if streaming:
                value = aux
            else:
                history = jt.map(lambda x: x[None, 1:], aux)
                value = loss(history, specs, None).total
            return value, (outputs, state.get(index))

        def streamed(scale):
            return run(scale, True)

        def full(scale):
            return run(scale, False)

        scale = state_dtype(0.7)
        for transform in (lambda fn: fn, jax.jit):
            actual = transform(jax.value_and_grad(streamed, has_aux=True))(scale)
            expected = transform(jax.value_and_grad(full, has_aux=True))(scale)
            assert actual[0][0].dtype == loss_dtype
            for a, b in zip(jt.leaves(actual), jt.leaves(expected)):
                assert jnp.allclose(a, b, rtol=2e-6, atol=1e-7)

        scales = jnp.array([0.4, 0.7], dtype=state_dtype)
        actual = jax.jit(jax.vmap(jax.value_and_grad(streamed, has_aux=True)))(scales)
        expected = jax.jit(jax.vmap(jax.value_and_grad(full, has_aux=True)))(scales)
        assert actual[0][0].dtype == loss_dtype
        for a, b in zip(jt.leaves(actual), jt.leaves(expected)):
            assert jnp.allclose(a, b, rtol=2e-6, atol=1e-7)
