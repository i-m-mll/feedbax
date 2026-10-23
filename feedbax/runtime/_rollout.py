"""Shared scan harness for multi-timestep rollouts.

``Graph._call_with_iteration`` and ``iterate_component`` drive the same
three-branch scan — accumulate a streaming loss, stack a filtered state
history, or just stack outputs. They differ only in how one timestep is
evaluated and in whether the scan carry threads extra per-step values
(cycle-wire port values, for cyclic graphs). This module holds that harness
once; both call sites supply a ``step_fn`` and, optionally, an initial step
carry.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree

from equinox.nn import State
from feedbax.runtime.streaming import (
    init_streaming_state_window,
    update_streaming_state_window,
)

#: ``(step_input, state, step_carry, key, t) -> (outputs, state, step_carry)``.
RolloutStepFn = Callable[
    [PyTree, State, PyTree, PRNGKeyArray, PyTree],
    tuple[PyTree, State, PyTree],
]


def select_step_inputs(inputs: PyTree, n_steps: int) -> PyTree:
    """Restrict time-major ``inputs`` to ``n_steps`` leading entries.

    Leaves are expected to be JAX arrays of shape ``(n_steps, ...)``. When they
    all are, selecting every entry is an identity and ``inputs`` is returned
    unchanged. Longer leaves are truncated by gathering the requested leading
    entries. Non-scalar leaves shorter than ``n_steps`` are rejected before
    gathering so indexed JAX access cannot silently clamp the timestep.

    Args:
        inputs: PyTree whose array leaves carry a leading time dimension.
        n_steps: Number of timesteps the scan will run; a static Python int.
    """
    short_lengths = [
        x.shape[0]
        for x in jt.leaves(inputs)
        if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] < n_steps
    ]
    if short_lengths:
        raise ValueError(
            f"Rollout input leaves must have at least n_steps={n_steps} leading entries; "
            f"found leading dimension(s) {short_lengths}"
        )

    if all(
        isinstance(x, jax.Array) and x.ndim >= 1 and x.shape[0] == n_steps
        for x in jt.leaves(inputs)
    ):
        return inputs
    return jax.vmap(lambda i: jt.map(lambda x: x[i], inputs))(jnp.arange(n_steps))


def _prepend_initial(x0, x):
    if x0 is None or x is None:
        return None
    return jnp.concatenate([x0[None], x], axis=0)


def scan_rollout(
    step_fn: RolloutStepFn,
    init_state: State,
    step_inputs: PyTree,  # leading time dimension of length n_steps
    keys: PRNGKeyArray,  # shape (n_steps, ...)
    n_steps: int,
    *,
    state_view_fn: Callable[[State], PyTree | None],
    init_step_carry: PyTree = None,
    checkpoint: bool = False,
    save_history: bool = True,
    state_filter: PyTree[bool] = True,
    streaming_loss_fn: Optional[Callable] = None,
) -> tuple[PyTree, State, PyTree | None]:
    """Scan ``step_fn`` over ``n_steps`` timesteps.

    Args:
        step_fn: Advances one timestep, returning ``(outputs, state,
            step_carry)``. The returned ``step_carry`` is threaded into the
            next step; components without per-step carry return it unchanged.
        init_state: Initial ``equinox.nn.State``.
        step_inputs: Per-step inputs, time-major.
        keys: Per-step PRNG keys.
        n_steps: Number of timesteps; a static Python int.
        state_view_fn: Maps a state to its state view, or ``None`` when the
            component exposes no view.
        init_step_carry: Initial value of the per-step carry, ``None`` when
            the caller threads no extra values. ``None`` is an empty PyTree
            node, so it adds no leaves to the scan carry.
        checkpoint: Wrap the scan body in ``jax.checkpoint``.
        save_history: Stack the filtered state view of every step and prepend
            the initial view. Ignored when ``streaming_loss_fn`` is given.
        state_filter: ``equinox.filter`` spec applied to each state view.
        streaming_loss_fn: ``(state_view, t) -> scalar``. When given, the scan
            accumulates a scalar loss instead of storing history. A function
            advertising ``streaming_order > 0`` is passed a rolling state
            window instead of a single state view.

    Returns:
        ``(outputs, final_state, aux)``, where ``aux`` is the accumulated
        streaming loss, the prepended state history, or ``None``.
    """
    # --- streaming-loss path: accumulate scalar, skip history storage ---
    if streaming_loss_fn is not None:
        streaming_order = getattr(streaming_loss_fn, "streaming_order", 0)
        init_state_view = state_view_fn(init_state)
        state_window = init_streaming_state_window(init_state_view, streaming_order)

        def step_streaming(carry, args):
            state, step_carry, state_window, loss_accum = carry
            (step_input, step_key), t = args

            outputs, state, step_carry = step_fn(step_input, state, step_carry, step_key, t)

            state_view = state_view_fn(state)
            loss_input = state_view
            if streaming_order > 0:
                state_window = update_streaming_state_window(state_window, state_view)
                loss_input = state_window
            step_loss = streaming_loss_fn(loss_input, t)
            return (state, step_carry, state_window, loss_accum + step_loss), outputs

        if checkpoint:
            step_streaming = jax.checkpoint(step_streaming)

        (final_state, _, _, total_loss), outputs = lax.scan(
            step_streaming,
            (init_state, init_step_carry, state_window, jnp.float32(0.0)),
            ((step_inputs, keys), jnp.arange(n_steps)),
        )
        return outputs, final_state, total_loss

    # --- standard paths (history or no-history) ---
    init_state_view = None
    if save_history:
        init_state_view = state_view_fn(init_state)
        if init_state_view is None:
            save_history = False

    def step_body(carry, args):
        state, step_carry = carry
        (step_input, step_key), t = args

        outputs, state, step_carry = step_fn(step_input, state, step_carry, step_key, t)

        if save_history:
            state_view = state_view_fn(state)
            if state_view is None:
                return (state, step_carry), (outputs, None)
            state_view = eqx.filter(state_view, state_filter)
            return (state, step_carry), (outputs, state_view)
        return (state, step_carry), outputs

    if checkpoint:
        step_body = jax.checkpoint(step_body)

    if save_history:
        (final_state, _), (outputs, state_history) = lax.scan(
            step_body,
            (init_state, init_step_carry),
            ((step_inputs, keys), jnp.arange(n_steps)),
        )

        # Prepend initial state to history
        init_state_view = eqx.filter(init_state_view, state_filter)
        state_history = jt.map(_prepend_initial, init_state_view, state_history)
        return outputs, final_state, state_history

    (final_state, _), outputs = lax.scan(
        step_body,
        (init_state, init_step_carry),
        ((step_inputs, keys), jnp.arange(n_steps)),
    )
    return outputs, final_state, None
