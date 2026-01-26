"""Iteration utilities for eager components.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

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
) -> tuple[PyTree, State, PyTree | None]:
    """Iterate an acyclic component over multiple timesteps."""
    keys = jr.split(key, n_steps)

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

    step_inputs = jax.vmap(lambda i: jt.map(lambda x: x[i], inputs))(jnp.arange(n_steps))

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
):
    """Run a component, iterating if needed, returning outputs and state history."""
    if isinstance(component, Graph) and component._needs_iteration:
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
    return iterate_component(
        component,
        inputs,
        init_state,
        n_steps,
        key,
        state_filter=state_filter,
    )
