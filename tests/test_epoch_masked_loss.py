"""Unit tests for ``EpochMaskedLoss``.

Bug: efc4d68 (rlrmp consumer; feedbax-side new issue).

Verifies that the compositional epoch-mask wrapper:
  - reduces to the base loss when the mask covers every epoch,
  - returns zero when the mask covers no epochs,
  - zeros out post-go contributions when wrapping a per-step
    ``TargetStateLoss`` with epoch_indices=(0, 1),
  - zeros out post-go contributions when wrapping cross-timestep terms
    (``StateDerivativeLoss`` order=1, ``OutputJerkLoss`` order=2),
  - varies its mask correctly across trials with different per-trial go_idx.

:copyright: Copyright 2025 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from feedbax.config.mapping import WhereDict
from feedbax.objectives.loss import (
    EpochMaskedLoss,
    OutputJerkLoss,
    StateDerivativeLoss,
    TargetSpec,
    TargetStateLoss,
)
from feedbax.tasks import TaskTrialSpec, TrialTimeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeNet(eqx.Module):
    hidden: jnp.ndarray
    output: jnp.ndarray


class _FakeEffector(eqx.Module):
    pos: jnp.ndarray
    vel: jnp.ndarray


class _FakeMechanics(eqx.Module):
    effector: _FakeEffector


class _FakeStates(eqx.Module):
    """Minimal stand-in for a feedbax rollout state."""

    net: _FakeNet
    mechanics: _FakeMechanics


def _make_states(
    *,
    n_trials: int = 3,
    n_steps: int = 11,
    units: int = 4,
    hidden_slope: float = 0.5,
    output_const: float = 1.0,
    vel_amp: float = 1.0,
):
    """Build a fake rollout where:
      - hidden ramps in time (constant nonzero per-step derivative),
      - output is a constant nonzero scalar (for TargetStateLoss vs zero target),
      - effector velocity is sinusoidal in time (nonzero jerk).
    """
    t = jnp.arange(n_steps, dtype=jnp.float32)[None, :, None]  # (1, T, 1)
    hidden = jnp.broadcast_to(hidden_slope * t, (n_trials, n_steps, units))
    output = jnp.broadcast_to(jnp.float32(output_const), (n_trials, n_steps, units))
    pos = jnp.zeros((n_trials, n_steps, 2))
    vel = jnp.broadcast_to(vel_amp * jnp.sin(0.7 * t), (n_trials, n_steps, 2))
    return _FakeStates(
        net=_FakeNet(hidden=hidden, output=output),
        mechanics=_FakeMechanics(effector=_FakeEffector(pos=pos, vel=vel)),
    )


def _make_trial_specs(
    *,
    n_trials: int = 3,
    n_steps: int = 11,
    epoch_bounds=None,
    epoch_names=("hold", "target_on", "movement"),
    output_target_zero: bool = True,
) -> TaskTrialSpec:
    """Build batched TaskTrialSpec with per-trial `epoch_bounds`.

    `epoch_bounds` should be `(n_trials, E+1)` aligned with `epoch_names`.
    """
    if epoch_bounds is None:
        # Default: every trial has go-cue at t=4 and bounds=(0, 2, 4, n_steps).
        eb = jnp.broadcast_to(
            jnp.asarray([0, 2, 4, n_steps], dtype=jnp.int32),
            (n_trials, 4),
        )
    else:
        eb = jnp.asarray(epoch_bounds, dtype=jnp.int32)
        if eb.ndim == 1:
            eb = jnp.broadcast_to(eb, (n_trials, eb.shape[0]))

    timeline = TrialTimeline.from_epochs_events(
        n_steps=n_steps,
        epoch_bounds=eb,
        epoch_names=epoch_names,
    )

    targets = WhereDict()
    if output_target_zero:
        # We'll wrap a TargetStateLoss with where=lambda s: s.net.output,
        # zero target, so the loss is (T-1)-element-wise constant non-zero
        # (= units * output_const^2 per step).
        targets[lambda state: state.net.output] = TargetSpec(
            value=jnp.float32(0.0)
        )

    return TaskTrialSpec(
        inits=WhereDict(),
        targets=targets,
        inputs={},
        intervene={},
        timeline=timeline,
    )


# ---------------------------------------------------------------------------
# Mask = full ⇒ wrapper equivalent to base loss
# ---------------------------------------------------------------------------


def test_full_mask_equals_base_state_derivative():
    n_trials, n_steps, units = 3, 11, 4
    states = _make_states(n_trials=n_trials, n_steps=n_steps, units=units)
    specs = _make_trial_specs(n_trials=n_trials, n_steps=n_steps)

    base = StateDerivativeLoss(label="nn_hidden_derivative")
    wrapped = EpochMaskedLoss(
        label="nn_hidden_derivative_full",
        base_loss=base,
        epoch_indices=(0, 1, 2),  # every epoch
    )

    base_val = jnp.mean(base(states, trial_specs=specs, model=None).value)
    wrapped_val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)
    assert jnp.allclose(base_val, wrapped_val, atol=1e-6)


def test_full_mask_equals_base_output_jerk():
    n_trials, n_steps = 3, 11
    states = _make_states(n_trials=n_trials, n_steps=n_steps)
    specs = _make_trial_specs(n_trials=n_trials, n_steps=n_steps)

    base = OutputJerkLoss(label="output_jerk")
    wrapped = EpochMaskedLoss(
        label="output_jerk_full",
        base_loss=base,
        epoch_indices=(0, 1, 2),
    )

    base_val = jnp.mean(base(states, trial_specs=specs, model=None).value)
    wrapped_val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)
    assert jnp.allclose(base_val, wrapped_val, atol=1e-6)


# ---------------------------------------------------------------------------
# Empty mask ⇒ exactly zero
# ---------------------------------------------------------------------------


def test_empty_mask_is_zero_state_derivative():
    states = _make_states()
    specs = _make_trial_specs()
    wrapped = EpochMaskedLoss(
        label="nn_hidden_derivative_none",
        base_loss=StateDerivativeLoss(label="nn_hidden_derivative"),
        epoch_indices=(),  # no epochs ⇒ all-zero mask
    )
    val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)
    assert jnp.isclose(val, 0.0, atol=1e-6)


def test_empty_mask_is_zero_output_jerk():
    states = _make_states()
    specs = _make_trial_specs()
    wrapped = EpochMaskedLoss(
        label="output_jerk_none",
        base_loss=OutputJerkLoss(label="output_jerk"),
        epoch_indices=(),
    )
    val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)
    assert jnp.isclose(val, 0.0, atol=1e-6)


def test_empty_mask_is_zero_target_state_loss():
    states = _make_states()
    specs = _make_trial_specs()
    base = TargetStateLoss(
        label="nn_output",
        where=lambda state: state.net.output,
        spec=TargetSpec(value=jnp.float32(0.0)),
    )
    wrapped = EpochMaskedLoss(
        label="nn_output_none",
        base_loss=base,
        epoch_indices=(),
    )
    val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)
    assert jnp.isclose(val, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Pre-go mask zeros the post-go contribution (cross-timestep)
# ---------------------------------------------------------------------------


def test_pre_go_mask_zeros_post_go_contribution_state_derivative():
    """Wrap StateDerivativeLoss; epoch_indices=(0,1) ⇒ post-go diffs are zero.

    Hidden ramps with constant per-step derivative, so the per-time-step
    squared norm of the diff is the SAME constant everywhere. Mask covers the
    first 4 steps (epochs 0+1 cover [0, 4)). After right-edge alignment with
    order=1, the mask on the differenced array has length T-1=10 and is
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0] (mask[1:] of mask shape (T,)).

    So the non-masked density (units * slope^2) appears at 3 of 10 differenced
    timesteps; the wrapper takes mean over T-1=10 ⇒ value = 3/10 * units * slope^2.
    """
    n_trials, n_steps, units, slope = 3, 11, 4, 0.5
    states = _make_states(
        n_trials=n_trials, n_steps=n_steps, units=units, hidden_slope=slope
    )
    specs = _make_trial_specs(n_trials=n_trials, n_steps=n_steps)

    wrapped = EpochMaskedLoss(
        label="nn_hidden_derivative_pre_go",
        base_loss=StateDerivativeLoss(label="nn_hidden_derivative"),
        epoch_indices=(0, 1),
    )
    val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)

    full_density = units * slope ** 2
    # Mask of length T (=11) is [1,1,1,1,0,0,0,0,0,0,0].
    # Right-edge align for order=1: mask[1:] = [1,1,1,0,0,0,0,0,0,0] (len 10).
    expected = full_density * (3.0 / (n_steps - 1))
    assert jnp.allclose(val, expected, atol=1e-6)


def test_pre_go_mask_zeros_post_go_contribution_output_jerk():
    """Wrap OutputJerkLoss; epoch_indices=(0,1) ⇒ post-go second-diffs zero.

    Sinusoidal velocity gives a nonzero squared-second-difference everywhere.
    With order=2, mask becomes mask[2:] of length T-2.
    """
    n_trials, n_steps, vel_amp = 3, 11, 1.0
    omega = 0.7
    states = _make_states(
        n_trials=n_trials, n_steps=n_steps, vel_amp=vel_amp,
    )
    specs = _make_trial_specs(n_trials=n_trials, n_steps=n_steps)

    wrapped = EpochMaskedLoss(
        label="output_jerk_pre_go",
        base_loss=OutputJerkLoss(label="output_jerk"),
        epoch_indices=(0, 1),
    )
    val = jnp.mean(wrapped(states, trial_specs=specs, model=None).value)

    # Closed form. Per-(trial, t_inner) squared 2nd-diff of sin(omega*t) is
    #     2 * vel_amp^2 * 4 sin^2(omega*t) (cos(omega) - 1)^2  (per feature)
    # times `units` for two features ⇒ coeff * sin^2(omega*t).
    units = 2  # vel has shape (..., 2)
    coeff = units * (vel_amp ** 2) * 4.0 * (jnp.cos(omega) - 1.0) ** 2
    t_inner = jnp.arange(1, n_steps - 1, dtype=jnp.float32)  # length T-2 = 9
    per_t = coeff * jnp.sin(omega * t_inner) ** 2

    # Right-edge alignment: mask[2:] of mask=[1,1,1,1,0,0,0,0,0,0,0] is
    # [1, 1, 0, 0, 0, 0, 0, 0, 0] (length 9).
    mask_inner = jnp.array([1.0] * 2 + [0.0] * 7)
    expected = jnp.mean(per_t * mask_inner)

    assert jnp.allclose(val, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Per-step base loss (TargetStateLoss) wrapping
# ---------------------------------------------------------------------------


def test_pre_go_mask_zeros_post_go_target_state_loss():
    """Wrap TargetStateLoss(zero target on net.output); pre-go-only mask.

    `net.output` is a constant 1.0 of shape (units,) ⇒ squared L2 norm density
    is `units` at every step. After [:, 1:] in TargetStateLoss, T-axis length
    is n_steps-1=10. epoch_mask[:, 1:] = [1,1,1,0,0,0,0,0,0,0] (len 10).
    `reduce_over_time_with_weights` does a weighted SUM (not mean), so the
    wrapped value equals `units * 3` per trial (3 steps with mask=1).
    """
    n_trials, n_steps, units, output_const = 3, 11, 4, 1.0
    states = _make_states(
        n_trials=n_trials, n_steps=n_steps, units=units,
        output_const=output_const,
    )
    specs = _make_trial_specs(n_trials=n_trials, n_steps=n_steps)

    base = TargetStateLoss(
        label="nn_output",
        where=lambda state: state.net.output,
        spec=TargetSpec(value=jnp.float32(0.0)),
    )
    wrapped = EpochMaskedLoss(
        label="nn_output_pre_go",
        base_loss=base,
        epoch_indices=(0, 1),
    )
    val = wrapped(states, trial_specs=specs, model=None).value
    # Trial-mean is taken implicitly by TermTree's default leaf_fn=jnp.mean.
    val_mean = jnp.mean(val)

    expected_per_trial = units * (output_const ** 2) * 3.0
    assert jnp.allclose(val_mean, expected_per_trial, atol=1e-5)


# ---------------------------------------------------------------------------
# Per-trial mask varies with per-trial epoch_bounds
# ---------------------------------------------------------------------------


def test_per_trial_mask_varies_with_epoch_bounds():
    """Two trials with different go-cues ⇒ wrapper masks each differently.

    Trial 0: bounds=(0, 1, 2, T)  → epochs 0+1 cover [0, 2).
    Trial 1: bounds=(0, 3, 6, T)  → epochs 0+1 cover [0, 6).

    Wrapping a TargetStateLoss(zero) on net.output (constant 1):
      Trial 0: 1 step with mask=1 in [:, 1:] view (mask = [1,1,0,0,...] then
               sliced [:, 1:] → [1, 0, 0, ...] ⇒ count=1).
      Trial 1: 5 steps with mask=1 ([1,1,1,1,1,1,0,0,0,0,0] sliced [:, 1:] →
               [1,1,1,1,1,0,0,0,0,0] ⇒ count=5).

    Per-trial values: trial0 = units * 1, trial1 = units * 5. Trial-mean = units * 3.
    """
    n_trials, n_steps, units = 2, 11, 4
    states = _make_states(
        n_trials=n_trials, n_steps=n_steps, units=units, output_const=1.0,
    )

    bounds = jnp.array([
        [0, 1, 2, n_steps],
        [0, 3, 6, n_steps],
    ], dtype=jnp.int32)
    specs = _make_trial_specs(
        n_trials=n_trials, n_steps=n_steps, epoch_bounds=bounds,
    )

    base = TargetStateLoss(
        label="nn_output",
        where=lambda state: state.net.output,
        spec=TargetSpec(value=jnp.float32(0.0)),
    )
    wrapped = EpochMaskedLoss(
        label="nn_output_pre_go",
        base_loss=base,
        epoch_indices=(0, 1),
    )
    val = wrapped(states, trial_specs=specs, model=None).value
    # `value` should be (N,) per-trial sum; verify:
    assert val.shape == (n_trials,)
    expected = jnp.asarray([units * 1.0, units * 5.0])
    assert jnp.allclose(val, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Non-supported base losses raise
# ---------------------------------------------------------------------------


def test_unsupported_base_loss_raises():
    from feedbax.objectives.loss import ModelLoss
    base = ModelLoss(label="model_only", loss_fn=lambda m: jnp.float32(0.0))
    wrapped = EpochMaskedLoss(
        label="model_only_pre_go",
        base_loss=base,
        epoch_indices=(0, 1),
    )
    states = _make_states()
    specs = _make_trial_specs()
    with pytest.raises(NotImplementedError):
        wrapped(states, trial_specs=specs, model=None)
