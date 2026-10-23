"""Unit tests for `OutputJerkLoss`.

Bug: 7e1d257 (feedbax). Verifies that the compositional
``mean(||v_{t+1} - 2 v_t + v_{t-1}||²)`` output-jerk loss term:

  - returns 0 on a constant-velocity trajectory (zero jerk),
  - returns the analytically expected value on a sinusoidal velocity profile,
  - has the correct gradient sign (penalises larger second-differences),
  - works with both single Arrays and PyTree-of-Arrays substates,
  - composes cleanly into ``CompositeLoss`` so a zero weight is a no-op,
  - is invariant to a leading ensemble (vmap) axis on the state.

Mirrors ``tests/test_state_derivative_loss.py`` (e8f3738), which exercises
the analogous ``order=1`` smoothness term.

:copyright: Copyright 2025 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from feedbax.objectives.loss import CompositeLoss, OutputJerkLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEffector(eqx.Module):
    vel: jnp.ndarray


class _FakeMechanics(eqx.Module):
    effector: _FakeEffector


class _FakeStates(eqx.Module):
    """Minimal stand-in for the rollout state PyTree.

    Shapes follow feedbax's ``(trial, time, ...)`` convention. Provides
    ``state.mechanics.effector.vel`` so the default `where` selector works.
    """

    mechanics: _FakeMechanics


def _states_from_vel(vel: jnp.ndarray) -> _FakeStates:
    return _FakeStates(mechanics=_FakeMechanics(effector=_FakeEffector(vel=vel)))


def _const_vel_states(trial: int = 4, time: int = 9, units: int = 3) -> _FakeStates:
    """Constant-velocity trajectory — second-difference is exactly zero."""
    vel = jnp.ones((trial, time, units)) * 0.5
    return _states_from_vel(vel)


def _ramp_vel_states(trial: int = 4, time: int = 9, units: int = 3, slope: float = 0.7):
    """Linear-in-time velocity v_t = slope * t * 1_units. Acceleration is constant
    (= slope), so jerk (second difference) is zero."""
    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
    v = jnp.broadcast_to(slope * t, (trial, time, units))
    return _states_from_vel(v)


def _sin_vel_states(
    trial: int = 4, time: int = 9, units: int = 3, omega: float = 0.5, amp: float = 1.0
):
    """Sinusoidal velocity v_t = amp * sin(omega * t) * 1_units.

    Closed-form second-difference (per unit, per t):
        v_{t+1} - 2 v_t + v_{t-1}
            = amp [sin(ω(t+1)) - 2 sin(ωt) + sin(ω(t-1))]
            = amp · 2 sin(ωt) (cos(ω) - 1)
        ⇒ ||·||² (over `units` features) = units · amp² · 4 sin²(ωt) (cos(ω) - 1)²
    """
    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
    v = jnp.broadcast_to(amp * jnp.sin(omega * t), (trial, time, units))
    return _states_from_vel(v)


# ---------------------------------------------------------------------------
# Core analytical correctness
# ---------------------------------------------------------------------------


def test_constant_velocity_gives_zero_jerk():
    """Constant velocity ⇒ jerk = 0 ⇒ loss = 0."""
    states = _const_vel_states()
    loss = OutputJerkLoss(label="output_jerk")
    out = loss(states, trial_specs=None, model=None)
    val = jnp.mean(out.value)
    assert jnp.isclose(val, 0.0)


def test_linear_velocity_gives_zero_jerk():
    """Linear-in-time velocity (constant acceleration) ⇒ jerk = 0 ⇒ loss = 0."""
    states = _ramp_vel_states(slope=0.7)
    loss = OutputJerkLoss(label="output_jerk")
    out = loss(states, trial_specs=None, model=None)
    val = jnp.mean(out.value)
    assert jnp.isclose(val, 0.0, atol=1e-6)


def test_sinusoidal_matches_closed_form():
    """Sinusoidal velocity gives a known closed-form second-difference."""
    trial, time, units = 4, 11, 3
    omega, amp = 0.5, 1.0
    states = _sin_vel_states(trial=trial, time=time, units=units, omega=omega, amp=amp)
    loss = OutputJerkLoss(label="output_jerk")
    out = loss(states, trial_specs=None, model=None)

    # Per-(trial, t) squared-norm of the second-difference, for t in [1, T-2]:
    #     ||·||² = units · amp² · 4 sin²(ωt) (cos(ω) - 1)²
    t_inner = jnp.arange(1, time - 1, dtype=jnp.float32)
    coeff = units * (amp ** 2) * 4.0 * (jnp.cos(omega) - 1.0) ** 2
    per_t = coeff * jnp.sin(omega * t_inner) ** 2
    # OutputJerkLoss takes mean over the contracted time axis (length T-2),
    # then TermTree's default reduction is mean over the trial axis. Trials
    # are identical here so the trial-mean is the same as any single trial.
    expected = jnp.mean(per_t)
    assert jnp.allclose(jnp.mean(out.value), expected, atol=1e-5)


def test_pytree_substate_sums_leaf_contributions():
    """When `where` returns a PyTree of two arrays, the term sums per-leaf."""
    trial, time, units = 3, 9, 4
    omega1, omega2 = 0.5, 0.9
    amp = 1.0
    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
    leaf1 = jnp.broadcast_to(amp * jnp.sin(omega1 * t), (trial, time, units))
    leaf2 = jnp.broadcast_to(amp * jnp.sin(omega2 * t), (trial, time, units))
    states_dict = {"a": leaf1, "b": leaf2}

    loss = OutputJerkLoss(label="output_jerk", where=lambda s: s)
    out = loss(states_dict, trial_specs=None, model=None)

    # Closed form for each leaf, summed.
    t_inner = jnp.arange(1, time - 1, dtype=jnp.float32)

    def _expected(omega):
        coeff = units * (amp ** 2) * 4.0 * (jnp.cos(omega) - 1.0) ** 2
        return jnp.mean(coeff * jnp.sin(omega * t_inner) ** 2)

    expected = _expected(omega1) + _expected(omega2)
    assert jnp.allclose(jnp.mean(out.value), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient sign
# ---------------------------------------------------------------------------


def test_gradient_sign_penalises_jerk():
    """Scaling a sinusoidal velocity by α scales loss by α². So d(loss)/dα > 0
    for α > 0 (loss increases with the magnitude of the temporal jerk)."""

    def loss_of_alpha(alpha):
        time, units = 11, 3
        omega = 0.5
        t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
        v = jnp.broadcast_to(alpha * jnp.sin(omega * t), (4, time, units))
        states = _states_from_vel(v)
        loss = OutputJerkLoss(label="output_jerk")
        out = loss(states, trial_specs=None, model=None)
        return jnp.mean(out.value)

    g = jax.grad(loss_of_alpha)(1.0)
    assert g > 0.0


# ---------------------------------------------------------------------------
# Compositional (zero-weight is a no-op)
# ---------------------------------------------------------------------------


class _ConstLoss(eqx.Module):
    """Test-only baseline term that always returns a fixed scalar."""

    label: str

    def __call__(self, states, trial_specs, model):
        from feedbax.objectives.loss import TermTree

        return TermTree.leaf(self.label, jnp.array(11.0))

    def skeleton(self, batch_dims):
        from feedbax.objectives.loss import TermTree

        return TermTree.leaf(self.label, jnp.empty(batch_dims))


def test_zero_weight_is_a_no_op_in_composite():
    """Adding OutputJerkLoss with weight 0 leaves total loss unchanged."""
    states = _sin_vel_states()

    base = CompositeLoss(
        terms={"const": _ConstLoss("const")},
        weights={"const": 1.0},
    )
    with_jerk = CompositeLoss(
        terms={
            "const": _ConstLoss("const"),
            "output_jerk": OutputJerkLoss(label="output_jerk"),
        },
        weights={"const": 1.0, "output_jerk": 0.0},
    )

    base_total = jnp.mean(base(states, trial_specs=None, model=None).total)
    new_total = jnp.mean(with_jerk(states, trial_specs=None, model=None).total)
    assert jnp.allclose(base_total, new_total, atol=1e-6)


def test_positive_weight_increases_total_loss():
    """Non-zero weight raises the composite total in the expected direction."""
    states = _sin_vel_states()

    composite = CompositeLoss(
        terms={
            "const": _ConstLoss("const"),
            "output_jerk": OutputJerkLoss(label="output_jerk"),
        },
        weights={"const": 1.0, "output_jerk": 1.0},
    )
    out = composite(states, trial_specs=None, model=None)

    bare = OutputJerkLoss(label="output_jerk")
    bare_value = jnp.mean(bare(states, trial_specs=None, model=None).value)
    expected_total = 11.0 + bare_value  # const term contributes 11.0
    assert jnp.allclose(jnp.mean(out.total), expected_total, atol=1e-6)


# ---------------------------------------------------------------------------
# Vmap / ensemble invariance
# ---------------------------------------------------------------------------


def test_ensemble_vmap_preserves_value():
    """Each replicate sees a non-vmapped state — vmap produces a per-replicate loss."""
    n_replicates = 3
    trial, time, units = 4, 11, 3
    omega, amp = 0.5, 1.0

    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
    v_single = jnp.broadcast_to(amp * jnp.sin(omega * t), (trial, time, units))
    # Stack identical replicates along a leading ensemble axis.
    v_ens = jnp.broadcast_to(v_single, (n_replicates,) + v_single.shape)

    loss = OutputJerkLoss(label="output_jerk")

    def per_replicate(v):
        states = _states_from_vel(v)
        return jnp.mean(loss(states, trial_specs=None, model=None).value)

    per_rep_values = jax.vmap(per_replicate)(v_ens)

    t_inner = jnp.arange(1, time - 1, dtype=jnp.float32)
    coeff = units * (amp ** 2) * 4.0 * (jnp.cos(omega) - 1.0) ** 2
    expected = jnp.mean(coeff * jnp.sin(omega * t_inner) ** 2)

    assert per_rep_values.shape == (n_replicates,)
    assert jnp.allclose(per_rep_values, jnp.full((n_replicates,), expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-timestep order declaration (for streaming-loss support, feedbax d67e303)
# ---------------------------------------------------------------------------


def test_order_attribute_is_two():
    """OutputJerkLoss declares cross-timestep order 2 (needs t-1, t, t+1)."""
    loss = OutputJerkLoss(label="output_jerk")
    assert loss.order == 2


# ---------------------------------------------------------------------------
# Cell-type / hidden-size invariance smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("units", [1, 8, 64])
def test_works_across_feature_sizes(units):
    """Term is invariant to feature dimensionality — works on any vector trajectory."""
    states = _sin_vel_states(units=units)
    loss = OutputJerkLoss(label="output_jerk")
    out = loss(states, trial_specs=None, model=None)
    # Just check finite + non-negative (positive on this non-trivial signal).
    val = jnp.mean(out.value)
    assert jnp.isfinite(val)
    assert val > 0.0
