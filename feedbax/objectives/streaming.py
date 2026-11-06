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
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array

from feedbax.objectives.loss import (
    AbstractLoss,
    CompositeLoss,
    ModelLoss,
    NthDifferenceLoss,
    TargetStateLoss,
    _target_start_step,
    _align_target_shape,
    _multiply_time_weights,
    _prepare_time_weight,
    _resolve_target_spec,
    _target_density,
    _target_weight_selectors,
)
from feedbax.runtime.graph import Component
from feedbax.runtime.streaming import current_state_from_window


def make_streaming_loss_fn(
    loss_func: AbstractLoss,
    trial_specs,
    model: Component,
    n_steps: int,
) -> Callable:
    """Construct a per-step streaming loss function from a loss tree.

    The returned function normally has signature ``(state_view, t) -> scalar``.
    If any term is an ``NthDifferenceLoss``, it advertises
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
    if n_steps < 1:
        raise ValueError(f"Streaming loss requires positive n_steps, got {n_steps}.")

    # Collect (leaf_loss, cumulative_weight) pairs from the loss tree.
    terms = _collect_leaf_terms(loss_func, parent_weight=1.0)

    # Separate into per-step terms, cross-timestep terms, and constant terms
    # (ModelLoss — independent of state, evaluated once).
    step_term_closures: list[Callable] = []
    model_terms: list[Array] = []
    streaming_order = 0

    for leaf_loss, cum_weight in terms:
        if isinstance(leaf_loss, ModelLoss):
            value = jnp.asarray(cum_weight * leaf_loss.term(None, None, model))
            if value.shape != ():
                raise ValueError(f"ModelLoss '{leaf_loss.label}' must produce a scalar.")
            model_terms.append(value)
            continue

        if isinstance(leaf_loss, TargetStateLoss):
            closure = _make_target_state_closure(
                leaf_loss, trial_specs, cum_weight, n_steps,
            )
            step_term_closures.append(closure)
            continue

        if isinstance(leaf_loss, NthDifferenceLoss):
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

    step_terms = tuple(step_term_closures)
    model_total = ft.reduce(jnp.add, model_terms) if model_terms else None

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
        contributions = []
        if model_total is not None:
            contributions.append(jnp.where(t == 0, model_total, jnp.zeros_like(model_total)))
        current_state = (
            current_state_from_window(state_or_window)
            if streaming_order > 0
            else state_or_window
        )
        for fn in step_terms:
            contributions.append(fn(state_or_window, current_state, t))
        return ft.reduce(jnp.add, contributions) if contributions else jnp.asarray(0.0)

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
    target, _ = _resolve_target_spec(loss, trial_specs)
    value = jnp.asarray(target.value)
    target_start_step = _target_start_step(trial_specs)
    target_steps = n_steps - target_start_step
    weights = tuple(
        _prepare_time_weight(selector, trial_specs, target_steps)
        for selector in _target_weight_selectors(target, target_steps)
    )

    def _step(state_window, current_state, t):
        state = loss.where(current_state)
        trajectory_shape = (target_steps, *state.shape)
        aligned_target = value.reshape(_align_target_shape(value.shape, trajectory_shape))

        def density_at(target_step):
            density = _target_density(loss, state[None, None], target_step)
            if density.shape != (1, 1):
                raise ValueError(
                    f"TargetStateLoss '{loss.label}' must produce one scalar per step."
                )
            return density[0, 0]

        if target_steps == 0:
            target_step = jnp.zeros(aligned_target.shape[1:], dtype=value.dtype)
            return jnp.zeros_like(cum_weight * density_at(target_step))

        def contribution():
            target_index = t - target_start_step
            target_step = aligned_target[0 if aligned_target.shape[0] == 1 else target_index]
            density = density_at(target_step)
            if weights:
                density = density * _multiply_time_weights(
                    [weight[target_index] for weight in weights], density.dtype
                )
            return cum_weight * density

        return _when_valid(t >= target_start_step, contribution)

    return _step


def _make_cross_timestep_closure(
    loss: NthDifferenceLoss,
    cum_weight: float,
    n_steps: int,
) -> Callable:
    """Build a per-step closure for one cross-timestep finite-difference term."""
    order = loss.order
    loss._validate_length(n_steps)

    def _step(state_window, current_state, t):
        leaves = jt.leaves(loss.where(state_window))
        if not leaves:
            raise ValueError(
                f"{type(loss).__name__}('{loss.label}'): `where` returned no "
                "array leaves to penalise."
            )

        def contribution():
            values = []
            for arr in leaves:
                # Reintroduce the singleton trial axis used by retained losses.
                window = arr[None, -(order + 1):]
                if loss.time_axis not in (1, 1 - window.ndim):
                    raise ValueError(
                        f"{type(loss).__name__}('{loss.label}') must use the "
                        "rollout time axis for streaming loss."
                    )
                density, _ = loss._difference_density(window, time_axis=1)
                if density.shape != (1, 1):
                    raise ValueError(
                        f"{type(loss).__name__}('{loss.label}') must produce one "
                        "scalar per step."
                    )
                values.append(density[0, 0] / (n_steps - order))
            return cum_weight * ft.reduce(jnp.add, values)

        return _when_valid(t >= order, contribution)

    return _step


def _when_valid(valid, contribution: Callable) -> Array:
    # Skipping excluded samples also skips their norm derivatives. Multiplying
    # by zero after evaluation would still propagate undefined derivatives.
    dtype = jax.eval_shape(contribution).dtype
    return jax.lax.cond(valid, contribution, lambda: jnp.zeros((), dtype=dtype))
