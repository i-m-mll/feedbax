"""Streaming loss computation for memory-efficient training.

Constructs a per-step loss function from a loss tree, precomputing all targets
and time weights so the scan body only needs the current state and timestep
index.  This eliminates the need to store the full state trajectory during
training.

The helper ``make_streaming_loss_fn`` analyses a ``CompositeLoss`` (or any
``AbstractLoss``) and builds a closure ``(state_view, t) -> scalar`` that sums
all per-step contributions.  Only loss types whose per-step contribution can be
computed from the current state alone are supported; cross-timestep losses
(e.g. ``EffectorStraightPathLoss``, ``StopAtGoalLoss``) raise
``NotImplementedError``.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from feedbax._model import AbstractModel
from feedbax.loss import (
    AbstractLoss,
    CompositeLoss,
    ModelLoss,
    TargetSpec,
    TargetStateLoss,
)


def make_streaming_loss_fn(
    loss_func: AbstractLoss,
    trial_specs,
    model: AbstractModel,
    n_steps: int,
) -> Callable:
    """Construct a per-step streaming loss function from a loss tree.

    The returned function has signature ``(state_view, t) -> scalar`` and is
    intended to be called inside ``lax.scan`` via ``iterate_component``'s
    ``streaming_loss_fn`` parameter.

    This function should be called *inside* ``filter_vmap`` so that
    ``trial_specs`` refers to a single trial (no batch dimension).

    Args:
        loss_func: The loss function (typically a ``CompositeLoss``).
        trial_specs: Trial specifications for a **single** trial (no batch
            dimension — call inside ``filter_vmap``).
        model: The model instance (needed for ``ModelLoss`` terms).
        n_steps: Number of simulation timesteps.

    Returns:
        A function ``(state_view, t) -> scalar`` that computes the streaming
        loss contribution at timestep ``t``.

    Raises:
        NotImplementedError: If any leaf loss term does not support streaming.
    """
    # Collect (leaf_loss, cumulative_weight) pairs from the loss tree.
    terms = _collect_leaf_terms(loss_func, parent_weight=1.0)

    # Separate into per-step terms (TargetStateLoss) and constant terms
    # (ModelLoss — independent of state, evaluated once).
    step_term_closures: list[Callable] = []
    constant_loss = jnp.float32(0.0)

    for leaf_loss, cum_weight in terms:
        if isinstance(leaf_loss, ModelLoss):
            # ModelLoss is state-independent; evaluate once and fold into the
            # constant.  Divide by n_steps so the total matches what
            # full-trajectory evaluation would produce (it calls .mean() over
            # T timesteps, but the value is the same at every step).
            constant_loss = constant_loss + cum_weight * leaf_loss.term(
                None, None, model
            )
            continue

        if isinstance(leaf_loss, TargetStateLoss):
            closure = _make_target_state_closure(
                leaf_loss, trial_specs, cum_weight, n_steps,
            )
            step_term_closures.append(closure)
            continue

        raise NotImplementedError(
            f"{type(leaf_loss).__name__} does not support streaming loss. "
            "It requires the full trajectory and cannot be evaluated per-step. "
            "Remove it from the loss function or use full-trajectory training."
        )

    # Freeze the list so the closure captures a tuple (JIT-friendly).
    _closures = tuple(step_term_closures)
    _const = constant_loss

    def streaming_fn(state_view, t):
        """Per-step loss contribution.

        Args:
            state_view: The component's state view at this timestep (single
                trial, no batch dimension).
            t: The 0-based timestep index (a JAX integer scalar).

        Returns:
            A scalar loss contribution for this timestep.
        """
        total = _const
        for fn in _closures:
            total = total + fn(state_view, t)
        return total

    return streaming_fn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_leaf_terms(
    loss: AbstractLoss,
    parent_weight: float,
) -> list[tuple[AbstractLoss, float]]:
    """Recursively collect ``(leaf_loss, cumulative_weight)`` pairs.

    Only ``CompositeLoss`` nodes are expanded; other ``AbstractTermedLoss``
    subclasses (like ``FuncTermsLoss``) are treated as leaves and must
    individually declare streaming support.
    """
    if isinstance(loss, CompositeLoss):
        result = []
        for name, child in loss.terms.items():
            w = loss.weights.get(name, 1.0)
            result.extend(_collect_leaf_terms(child, parent_weight * w))
        return result

    # Leaf loss (TargetStateLoss, ModelLoss, FuncTermsLoss, etc.)
    return [(loss, parent_weight)]


def _make_target_state_closure(
    loss: TargetStateLoss,
    trial_specs,
    cum_weight: float,
    n_steps: int,
) -> Callable:
    """Build a per-step closure for a single ``TargetStateLoss`` term.

    Precomputes the target value and time weights for the trial so the
    closure only needs ``(state_view, t)`` at call time.

    The closure operates on a *single trial* (no batch dimension) because it
    is constructed inside ``filter_vmap``.
    """
    # --- resolve target spec (mirrors TargetStateLoss.term logic) ---
    task_target_spec = trial_specs.targets.get(loss.key, None)

    if task_target_spec is None:
        if loss.spec is None:
            raise ValueError(
                f"TargetSpec must be provided on construction of "
                f"TargetStateLoss '{loss.label}', or as part of the trial "
                f"specifications"
            )
        target_spec = loss.spec
    elif isinstance(task_target_spec, TargetSpec):
        target_spec = eqx.combine(loss.spec, task_target_spec)
    elif isinstance(task_target_spec, Mapping):
        target_spec = eqx.combine(loss.spec, task_target_spec[loss.label])
    else:
        raise ValueError(f"Invalid target spec type: {type(task_target_spec)}")

    target_value = target_spec.value  # (features,) or scalar for single trial

    # --- precompute time weights as a (T,) vector ---
    # T = n_steps because state_history[:, 1:] has n_steps entries (the initial
    # state is excluded, matching TargetStateLoss.term's `[:, 1:]`).
    T = n_steps

    time_mask = target_spec.time_mask
    if time_mask is None:
        time_mask = target_spec.get_time_mask(T)

    masks = [x for x in [time_mask, target_spec.discount] if x is not None]

    # Evaluate masks for a single trial.  _combine_weights expects (N, T)
    # shapes; we fake N=1 and squeeze afterward.
    if masks:
        time_weights = _compute_single_trial_weights(
            masks, trial_specs, T,
        )  # (T,)
    else:
        time_weights = jnp.ones((T,), dtype=jnp.float32)

    where_fn = loss.where
    norm_fn = loss.norm
    w = cum_weight

    def _step(state_view, t):
        state_component = where_fn(state_view)  # (features,) — single trial
        error = norm_fn(state_component - target_value)  # scalar
        weight_t = time_weights[t]
        return w * error * weight_t

    return _step


def _compute_single_trial_weights(
    masks: list,
    trial_specs,
    T: int,
) -> Array:
    """Compute combined time weights for a single trial.

    This mirrors ``_combine_weights`` from ``loss.py`` but for a single trial
    (no batch dimension).

    Args:
        masks: List of ``WeightsSpec`` selectors (scalars, arrays, or
            callables taking a single trial's spec).
        trial_specs: A single trial's specification (no batch dim).
        T: Number of timesteps.

    Returns:
        A ``(T,)`` array of combined time weights.
    """
    dtype = jnp.float32
    w = jnp.ones((T,), dtype=dtype)

    for selector in masks:
        if callable(selector):
            wi = selector(trial_specs)
        else:
            wi = selector

        wi = jnp.asarray(wi, dtype=dtype)
        if wi.ndim == 0:
            wi = jnp.full((T,), wi, dtype=dtype)
        elif wi.shape != (T,):
            raise ValueError(
                f"Mask/discount must be scalar or shape ({T},), got {wi.shape}"
            )
        w = w * wi

    return w
