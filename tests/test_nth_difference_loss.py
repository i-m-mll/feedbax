"""Unit tests for the unified finite-difference loss family."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from feedbax.loss import NthDifferenceLoss, OutputJerkLoss, StateDerivativeLoss


class _FakeNet(eqx.Module):
    hidden: jnp.ndarray


class _FakeEffector(eqx.Module):
    vel: jnp.ndarray


class _FakeMechanics(eqx.Module):
    effector: _FakeEffector


class _FakeStates(eqx.Module):
    net: _FakeNet
    mechanics: _FakeMechanics


def _states(hidden: jnp.ndarray, vel: jnp.ndarray) -> _FakeStates:
    return _FakeStates(
        net=_FakeNet(hidden=hidden),
        mechanics=_FakeMechanics(effector=_FakeEffector(vel=vel)),
    )


def test_nth_difference_order_one_matches_state_derivative_wrapper():
    t = jnp.arange(7, dtype=jnp.float32)[None, :, None]
    hidden = jnp.broadcast_to(0.25 * t, (3, 7, 4))
    states = _states(hidden=hidden, vel=jnp.zeros((3, 7, 2)))

    generic = NthDifferenceLoss(
        label="state_derivative",
        order=1,
        where=lambda s: s.net.hidden,
    )
    wrapper = StateDerivativeLoss(label="state_derivative")

    assert jnp.allclose(
        generic(states, trial_specs=None, model=None).total,
        wrapper(states, trial_specs=None, model=None).total,
    )


def test_nth_difference_order_two_matches_output_jerk_wrapper():
    t = jnp.arange(11, dtype=jnp.float32)[None, :, None]
    vel = jnp.broadcast_to(jnp.sin(0.4 * t), (2, 11, 3))
    states = _states(hidden=jnp.zeros((2, 11, 5)), vel=vel)

    generic = NthDifferenceLoss(
        label="output_jerk",
        order=2,
        where=lambda s: s.mechanics.effector.vel,
    )
    wrapper = OutputJerkLoss(label="output_jerk")

    assert jnp.allclose(
        generic(states, trial_specs=None, model=None).total,
        wrapper(states, trial_specs=None, model=None).total,
    )


def test_nth_difference_order_three_is_available():
    t = jnp.arange(8, dtype=jnp.float32)[None, :, None]
    x = jnp.broadcast_to(t**3, (2, 8, 1))
    states = {"x": x}

    loss = NthDifferenceLoss(label="third_difference", order=3, where=lambda s: s["x"])
    value = loss(states, trial_specs=None, model=None).total

    # Third finite difference of t**3 is constant 6.
    assert jnp.allclose(value, 36.0)


def test_nth_difference_rejects_non_positive_order():
    with pytest.raises(ValueError, match="order must be >= 1"):
        NthDifferenceLoss(order=0)
