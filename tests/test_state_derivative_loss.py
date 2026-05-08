"""Unit tests for `StateDerivativeLoss`.

Bug: efc4d68 (rlrmp). Verifies that the compositional ||h_t - h_{t-1}||²
hidden-state smoothness loss term:

  - returns the analytically expected value on a tiny synthetic rollout,
  - has the correct gradient sign (penalises larger time-differences),
  - works with both single Arrays and PyTree-of-Arrays substates,
  - composes cleanly into `CompositeLoss` so a zero weight is a no-op,
  - is invariant to a leading ensemble (vmap) axis on the state.

:copyright: Copyright 2025 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from feedbax.loss import CompositeLoss, StateDerivativeLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeNet(eqx.Module):
    hidden: jnp.ndarray


class _FakeStates(eqx.Module):
    """Minimal stand-in for the rollout state PyTree.

    Shapes follow feedbax's ``(trial, time, ...)`` convention.
    """

    net: _FakeNet


def _const_states(trial: int = 4, time: int = 7, units: int = 5) -> _FakeStates:
    """All-zeros rollout — derivative should be exactly zero everywhere."""
    return _FakeStates(net=_FakeNet(hidden=jnp.zeros((trial, time, units))))


def _ramp_states(trial: int = 4, time: int = 7, units: int = 5, slope: float = 1.0):
    """Hidden state h_t = slope * t * 1_units. Diff has constant magnitude.

    Per-step squared norm of the difference is ``units * slope²``; mean over
    time is the same constant; mean over trials is the same constant.
    """
    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]  # (1, T, 1)
    h = jnp.broadcast_to(slope * t, (trial, time, units))
    return _FakeStates(net=_FakeNet(hidden=h))


# ---------------------------------------------------------------------------
# Core analytical correctness
# ---------------------------------------------------------------------------


def test_zero_state_gives_zero_loss():
    """All-zeros hidden trajectory ⇒ loss is exactly 0."""
    states = _const_states()
    loss = StateDerivativeLoss(label="nn_hidden_derivative")
    out = loss(states, trial_specs=None, model=None)
    val = jnp.mean(out.value)
    assert jnp.isclose(val, 0.0)


def test_constant_slope_matches_closed_form():
    """h_t = slope * t * 1_n → ||Δh_t||² = units * slope² for every t.

    Mean over time and trial of that constant is the same constant.
    """
    units, slope = 5, 0.3
    states = _ramp_states(units=units, slope=slope)
    loss = StateDerivativeLoss(label="nn_hidden_derivative")
    out = loss(states, trial_specs=None, model=None)
    expected = units * (slope ** 2)
    assert jnp.allclose(jnp.mean(out.value), expected, atol=1e-6)


def test_pytree_substate_sums_leaf_contributions():
    """When `where` returns a PyTree of two arrays, the term sums per-leaf.

    Two ramp-leaves with slopes (s1, s2) ⇒ value = units*(s1² + s2²).
    """
    trial, time, units = 3, 6, 4
    s1, s2 = 0.5, 0.2
    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
    leaf1 = jnp.broadcast_to(s1 * t, (trial, time, units))
    leaf2 = jnp.broadcast_to(s2 * t, (trial, time, units))

    states_dict = {"a": leaf1, "b": leaf2}

    loss = StateDerivativeLoss(
        label="nn_hidden_derivative",
        where=lambda s: s,
    )
    out = loss(states_dict, trial_specs=None, model=None)
    expected = units * (s1 ** 2 + s2 ** 2)
    assert jnp.allclose(jnp.mean(out.value), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient sign
# ---------------------------------------------------------------------------


def test_gradient_sign_penalises_jumps():
    """∂(loss)/∂h_t > 0 when h_t exceeds its temporal neighbours.

    Concretely: with h_t = α * ramp, scaling α should increase the loss, so
    d(loss)/dα > 0 for α > 0.
    """
    units, slope = 5, 0.4

    def loss_of_alpha(alpha):
        t = jnp.arange(7, dtype=jnp.float32)[None, :, None]
        h = jnp.broadcast_to(alpha * slope * t, (4, 7, units))
        states = _FakeStates(net=_FakeNet(hidden=h))
        loss = StateDerivativeLoss(label="nn_hidden_derivative")
        out = loss(states, trial_specs=None, model=None)
        return jnp.mean(out.value)

    g = jax.grad(loss_of_alpha)(1.0)
    assert g > 0.0


# ---------------------------------------------------------------------------
# Compositional (zero-weight is a no-op)
# ---------------------------------------------------------------------------


class _ConstLoss(eqx.Module):
    """Test-only baseline term that always returns a fixed scalar.

    Implemented as a duck-typed AbstractLoss (delegates via __call__ to
    TermTree.leaf) so we don't drag in TargetStateLoss's full machinery.
    """

    label: str

    def __call__(self, states, trial_specs, model):
        from feedbax.loss import TermTree
        return TermTree.leaf(self.label, jnp.array(7.0))

    def skeleton(self, batch_dims):
        from feedbax.loss import TermTree
        return TermTree.leaf(self.label, jnp.empty(batch_dims))


def test_zero_weight_is_a_no_op_in_composite():
    """Adding StateDerivativeLoss with weight 0 leaves total loss unchanged."""
    units, slope = 5, 0.3
    states = _ramp_states(units=units, slope=slope)

    base = CompositeLoss(
        terms={"const": _ConstLoss("const")},
        weights={"const": 1.0},
    )
    with_smooth = CompositeLoss(
        terms={
            "const": _ConstLoss("const"),
            "nn_hidden_derivative": StateDerivativeLoss(label="nn_hidden_derivative"),
        },
        weights={"const": 1.0, "nn_hidden_derivative": 0.0},
    )

    base_total = jnp.mean(base(states, trial_specs=None, model=None).total)
    new_total = jnp.mean(with_smooth(states, trial_specs=None, model=None).total)
    assert jnp.allclose(base_total, new_total, atol=1e-6)


def test_positive_weight_increases_total_loss():
    """Non-zero weight raises the composite total in the expected direction."""
    units, slope = 5, 0.3
    states = _ramp_states(units=units, slope=slope)

    composite = CompositeLoss(
        terms={
            "const": _ConstLoss("const"),
            "nn_hidden_derivative": StateDerivativeLoss(label="nn_hidden_derivative"),
        },
        weights={"const": 1.0, "nn_hidden_derivative": 1e-3},
    )
    out = composite(states, trial_specs=None, model=None)
    expected_smooth = 1e-3 * units * (slope ** 2)
    expected_total = 7.0 + expected_smooth
    assert jnp.allclose(jnp.mean(out.total), expected_total, atol=1e-6)


# ---------------------------------------------------------------------------
# Vmap / ensemble invariance
# ---------------------------------------------------------------------------


def test_ensemble_vmap_preserves_value():
    """Each replicate sees a non-vmapped state — vmap produces a per-replicate loss."""
    n_replicates = 3
    trial, time, units = 4, 7, 5
    slope = 0.2

    t = jnp.arange(time, dtype=jnp.float32)[None, :, None]
    h_single = jnp.broadcast_to(slope * t, (trial, time, units))
    # Stack identical replicates along a leading ensemble axis.
    h_ens = jnp.broadcast_to(h_single, (n_replicates,) + h_single.shape)

    loss = StateDerivativeLoss(label="nn_hidden_derivative")

    def per_replicate(h):
        states = _FakeStates(net=_FakeNet(hidden=h))
        return jnp.mean(loss(states, trial_specs=None, model=None).value)

    per_rep_values = jax.vmap(per_replicate)(h_ens)
    expected = units * (slope ** 2)
    assert per_rep_values.shape == (n_replicates,)
    assert jnp.allclose(per_rep_values, jnp.full((n_replicates,), expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Cell-type invariance (smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("units", [1, 8, 64])
def test_works_across_hidden_sizes(units):
    """Term is invariant to hidden-size — a stand-in for cell-type invariance.

    The class never inspects the recurrent cell; it only operates on the
    rollout substate selected by `where`. Different cell types just produce
    different numerical hidden trajectories of the same PyTree shape.
    """
    states = _ramp_states(units=units, slope=0.1)
    loss = StateDerivativeLoss(label="nn_hidden_derivative")
    out = loss(states, trial_specs=None, model=None)
    expected = units * (0.1 ** 2)
    assert jnp.allclose(jnp.mean(out.value), expected, atol=1e-6)
