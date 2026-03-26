"""Iteration utilities for eager components.

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
import jax.random as jr
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.graph import Component, Graph
from equinox.nn import State


def iterate_component(
    component: Component,
    inputs: PyTree,  # leading time dimension
    init_state: State,
    n_steps: int,
    key: PRNGKeyArray,
    state_filter: PyTree[bool] = True,
    checkpoint: bool = False,
    streaming_loss_fn: Optional[Callable] = None,
) -> tuple[PyTree, State, PyTree | None]:
    """Iterate an acyclic component over multiple timesteps.

    When ``streaming_loss_fn`` is provided the scan accumulates a scalar loss
    instead of storing state history, eliminating trajectory memory entirely.
    ``streaming_loss_fn`` should have signature ``(state_view, t) -> scalar``
    where ``state_view`` is the component's state view at timestep ``t``
    (0-based) and the return value is the per-step loss contribution already
    reduced over batch and features.  ``state_filter`` is ignored in this mode.
    """
    keys = jr.split(key, n_steps)
    step_inputs = jax.vmap(lambda i: jt.map(lambda x: x[i], inputs))(jnp.arange(n_steps))

    # --- streaming-loss path: accumulate scalar, skip history storage ---
    if streaming_loss_fn is not None:
        def step(carry, args):
            state, loss_accum = carry
            (step_input, step_key), t = args
            outputs, new_state = component(step_input, state, key=step_key)
            state_view = component.state_view(new_state)
            step_loss = streaming_loss_fn(state_view, t)
            return (new_state, loss_accum + step_loss), outputs

        if checkpoint:
            step = jax.checkpoint(step)

        (final_state, total_loss), outputs = lax.scan(
            step,
            (init_state, jnp.float32(0.0)),
            ((step_inputs, keys), jnp.arange(n_steps)),
        )
        return outputs, final_state, total_loss

    # --- standard paths (history or no-history) ---
    save_history = state_filter is not False
    init_state_view = None
    if save_history:
        init_state_view = component.state_view(init_state)
        if init_state_view is None:
            save_history = False

    def step(carry, args):
        state = carry
        step_input, step_key = args

        outputs, new_state = component(step_input, state, key=step_key)

        if save_history:
            state_view = component.state_view(new_state)
            if state_view is None:
                return new_state, (outputs, None)
            state_view = eqx.filter(state_view, state_filter)
            return new_state, (outputs, state_view)
        return new_state, outputs

    if checkpoint:
        step = jax.checkpoint(step)

    if save_history:
        final_state, (outputs, state_history) = lax.scan(
            step, init_state, (step_inputs, keys)
        )
        init_state_view = eqx.filter(init_state_view, state_filter)

        def _prepend(x0, x):
            if x0 is None or x is None:
                return None
            return jnp.concatenate([x0[None], x], axis=0)

        state_history = jt.map(_prepend, init_state_view, state_history)
        return outputs, final_state, state_history

    final_state, outputs = lax.scan(step, init_state, (step_inputs, keys))
    return outputs, final_state, None


def run_component(
    component: Component,
    inputs: PyTree,
    init_state: State,
    *,
    key: PRNGKeyArray,
    n_steps: Optional[int] = None,
    state_filter: PyTree[bool] = True,
    streaming_loss_fn: Optional[Callable] = None,
):
    """Run a component, iterating if needed, returning outputs and state history.

    When ``streaming_loss_fn`` is provided, the third return element is the
    accumulated scalar loss instead of state history.
    """
    if isinstance(component, Graph) and component._needs_iteration:
        if streaming_loss_fn is not None:
            # TODO: Streaming loss for Graph components requires changes in
            # graph.py to thread the loss accumulator through its internal
            # iteration.  For now, only the iterate_component path supports it.
            raise NotImplementedError(
                "Streaming loss is not yet supported for Graph components "
                "that handle their own iteration. Use iterate_component "
                "directly or restructure the model as an acyclic component."
            )
        return component(
            inputs,
            init_state,
            key=key,
            n_steps=n_steps,
            return_state_history=True,
            state_filter=state_filter,
        )
    if n_steps is None:
        raise ValueError("n_steps is required for acyclic components")
    checkpoint = getattr(component, 'checkpoint', False)
    return iterate_component(
        component,
        inputs,
        init_state,
        n_steps,
        key,
        state_filter=state_filter,
        checkpoint=checkpoint,
        streaming_loss_fn=streaming_loss_fn,
    )
