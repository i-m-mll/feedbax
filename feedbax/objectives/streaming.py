"""Streaming loss computation for memory-efficient training.

Constructs a per-step loss function from a loss tree, precomputing all targets
and time weights so the scan body only needs the current state and timestep
index.  This eliminates the need to store the full state trajectory during
training.

The helper ``make_streaming_loss_fn`` analyses a ``CompositeLoss`` (or any
``AbstractLoss``) and builds a closure that sums all per-step contributions.
Per-step losses consume the current state. Cross-timestep losses declare their
finite-difference order and consume a rolling window of recent states.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import functools as ft
from collections.abc import Callable, Mapping

import equinox as eqx
import jax.tree as jt
import jax.numpy as jnp
from jaxtyping import Array

from feedbax.runtime.model import AbstractModel
from feedbax.runtime.streaming import current_state_from_window
from feedbax.objectives.loss import (
    AbstractLoss,
    CompositeLoss,
    CrossTimestepLoss,
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

    The returned function normally has signature ``(state_view, t) -> scalar``.
    If any term is a ``CrossTimestepLoss``, it advertises
    ``streaming_order > 0`` and expects a rolling state window with a leading
    time axis of length ``streaming_order + 1`` instead of a single state.

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

    # Separate into per-step terms, cross-timestep terms, and constant terms
    # (ModelLoss — independent of state, evaluated once).
    step_term_closures: list[Callable] = []
    constant_loss = jnp.float32(0.0)
    streaming_order = 0

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

        if isinstance(leaf_loss, CrossTimestepLoss):
            closure = _make_cross_timestep_closure(
                leaf_loss, cum_weight, n_steps,
            )
            step_term_closures.append(closure)
            streaming_order = max(streaming_order, leaf_loss.order)
            continue

        raise NotImplementedError(
            f"{type(leaf_loss).__name__} does not support streaming loss. "
            "It requires the full trajectory and cannot be evaluated per-step. "
            "Remove it from the loss function or use full-trajectory training."
        )

    # Freeze the list so the closure captures a tuple (JIT-friendly).
    _closures = tuple(step_term_closures)
    _const = constant_loss
    _streaming_order = streaming_order

    def streaming_fn(state_or_window, t):
        """Per-step loss contribution.

        Args:
            state_or_window: The component's state view at this timestep
                (single trial, no batch dimension), or a rolling state window
                when ``streaming_order > 0``.
            t: The 0-based timestep index (a JAX integer scalar).

        Returns:
            A scalar loss contribution for this timestep.
        """
        total = _const
        current_state = (
            current_state_from_window(state_or_window)
            if _streaming_order > 0
            else state_or_window
        )
        for fn in _closures:
            total = total + fn(state_or_window, current_state, t)
        return total

    streaming_fn.streaming_order = streaming_order
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

    target_value = target_spec.value  # (features,), (T, features), or scalar

    # --- precompute time weights as a (T,) vector ---
    #
    # The scan runs ``n_steps`` iterations (0 .. n_steps-1).  In
    # full-trajectory mode, ``TargetStateLoss.term`` slices the state
    # history with ``[:, 1:]``, discarding the first scan output (which is
    # the state immediately after the initial step).  The resulting ``T-1``
    # entries carry the loss weights.  In streaming mode, the scan still
    # produces ``n_steps`` states, so we need ``n_steps`` weights with a
    # leading zero to mirror the ``[:, 1:]`` exclusion.
    #
    # Mask/discount callables (``get_epoch_weights``, ``make_mid_period_ramp``,
    # etc.) return ``(n_steps - 1,)`` arrays.  We compute these at that
    # natural length, then prepend a zero to align with the scan index.
    T = n_steps
    T_loss = T - 1  # number of timesteps the full-trajectory loss covers

    time_mask = target_spec.time_mask
    if time_mask is None:
        time_mask = target_spec.get_time_mask(T_loss)

    masks = [x for x in [time_mask, target_spec.discount] if x is not None]

    if masks:
        time_weights_inner = _compute_single_trial_weights(
            masks, trial_specs, T_loss,
        )  # (T_loss,)
    else:
        time_weights_inner = jnp.ones((T_loss,), dtype=jnp.float32)

    # Prepend a zero weight for scan step 0 (excluded by full-trajectory
    # ``[:, 1:]``), giving a ``(T,)`` vector aligned with the scan index.
    time_weights = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.float32), time_weights_inner],
    )  # (T,)

    # --- align time-varying targets with the scan index ---
    #
    # Tasks broadcast the target to ``(T_loss, features)`` for
    # full-trajectory evaluation.  Prepend a dummy entry so that
    # ``target_value[t]`` at scan step ``t`` matches full-trajectory
    # index ``t - 1`` for ``t >= 1``.  The entry at ``t = 0`` is
    # zero-weighted.
    if (
        isinstance(target_value, Array)
        and target_value.ndim >= 2
        and target_value.shape[0] == T_loss
    ):
        target_value = jnp.concatenate(
            [target_value[:1], target_value], axis=0,
        )  # (T, features)

    # After possible padding, a (T, features) target is indexed per step.
    target_is_time_varying = (
        isinstance(target_value, Array)
        and target_value.ndim >= 2
        and target_value.shape[0] == T
    )

    where_fn = loss.where
    norm_fn = loss.norm
    w = cum_weight

    def _step(state_window, current_state, t):
        state_component = where_fn(current_state)  # (features,) — single trial
        tv = target_value[t] if target_is_time_varying else target_value
        error = norm_fn(state_component - tv)  # scalar
        weight_t = time_weights[t]
        return w * error * weight_t

    return _step


def _make_cross_timestep_closure(
    loss: CrossTimestepLoss,
    cum_weight: float,
    n_steps: int,
) -> Callable:
    """Build a per-step closure for one cross-timestep finite-difference term."""
    if loss.order < 1:
        raise ValueError(f"{type(loss).__name__} order must be >= 1")
    if n_steps <= loss.order:
        raise ValueError(
            f"{type(loss).__name__} order {loss.order} requires more than "
            f"{loss.order} timesteps, got {n_steps}."
        )

    order = loss.order
    denominator = jnp.asarray(n_steps - order, dtype=jnp.float32)
    where_fn = loss.where
    norm_fn = loss.norm
    w = cum_weight

    def _step(state_window, current_state, t):
        substate = where_fn(state_window)

        def _per_leaf(arr):
            arr = arr[-(order + 1):]
            d = jnp.diff(arr, n=order, axis=0)
            d_norm = norm_fn(d)
            return jnp.sum(d_norm, axis=0) / denominator

        per_leaf = jt.map(_per_leaf, substate)
        leaves = jt.leaves(per_leaf)
        if not leaves:
            raise ValueError(
                f"{type(loss).__name__}('{loss.label}'): `where` returned no "
                "array leaves to penalise."
            )
        value = ft.reduce(jnp.add, leaves)
        return jnp.where(t >= order, w * value, jnp.asarray(0.0, dtype=value.dtype))

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
        elif wi.shape == (T - 1,):
            # Common off-by-one: masks computed from epoch_bounds may have T-1
            # elements (timeline n_steps includes init state). Pad with last value.
            wi = jnp.concatenate([wi, wi[-1:]])
        elif wi.shape != (T,):
            raise ValueError(
                f"Mask/discount must be scalar or shape ({T},) or ({T-1},), got {wi.shape}"
            )
        w = w * wi

    return w
