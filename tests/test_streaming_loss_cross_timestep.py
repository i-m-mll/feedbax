"""Streaming-loss tests for cross-timestep finite-difference terms."""

import equinox as eqx
import jax.numpy as jnp

from feedbax.runtime.streaming import init_streaming_state_window, update_streaming_state_window
from feedbax.loss import CompositeLoss, NthDifferenceLoss, StateDerivativeLoss
from feedbax.streaming_loss import make_streaming_loss_fn


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
