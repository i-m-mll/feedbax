"""Iteration utilities for eager components.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import jax.random as jr
from jaxtyping import PRNGKeyArray, PyTree

from equinox.nn import State
from feedbax.runtime._rollout import scan_rollout, select_step_inputs
from feedbax.runtime.graph import Component, Graph, RolloutStepHook


def _run_component_step(
    component: Component,
    step_input: PyTree,
    state: State,
    *,
    key: PRNGKeyArray,
    t: PyTree,
    rollout_step_hook: Optional[RolloutStepHook],
):
    if isinstance(component, Graph):
        return component(
            step_input,
            state,
            key=key,
            t=t,
            rollout_step_hook=rollout_step_hook,
        )
    return component(step_input, state, key=key)


def iterate_component(
    component: Component,
    inputs: PyTree,  # leading time dimension
    init_state: State,
    n_steps: int,
    key: PRNGKeyArray,
    state_filter: PyTree[bool] = True,
    checkpoint: bool = False,
    streaming_loss_fn: Optional[Callable] = None,
    rollout_step_hook: Optional[RolloutStepHook] = None,
) -> tuple[PyTree, State, PyTree | None]:
    """Iterate an acyclic component over multiple timesteps.

    When ``streaming_loss_fn`` is provided the scan accumulates a scalar loss
    instead of storing state history, eliminating trajectory memory entirely.
    ``streaming_loss_fn`` should have signature ``(state_view, t) -> scalar``
    for per-step losses. Functions produced by ``make_streaming_loss_fn`` may
    instead advertise ``streaming_order > 0``; in that case a rolling state
    window is passed so cross-timestep losses can be evaluated without storing
    the full trajectory. ``state_filter`` is ignored in this mode.
    """
    keys = jr.split(key, n_steps)
    step_inputs = select_step_inputs(inputs, n_steps)

    def step_fn(step_input, state, step_carry, step_key, t):
        outputs, new_state = _run_component_step(
            component,
            step_input,
            state,
            key=step_key,
            t=t,
            rollout_step_hook=rollout_step_hook,
        )
        return outputs, new_state, step_carry

    return scan_rollout(
        step_fn,
        init_state,
        step_inputs,
        keys,
        n_steps,
        state_view_fn=component.state_view,
        checkpoint=checkpoint,
        save_history=state_filter is not False,
        state_filter=state_filter,
        streaming_loss_fn=streaming_loss_fn,
    )


def run_component(
    component: Component,
    inputs: PyTree,
    init_state: State,
    *,
    key: PRNGKeyArray,
    n_steps: Optional[int] = None,
    state_filter: PyTree[bool] = True,
    streaming_loss_fn: Optional[Callable] = None,
    rollout_step_hook: Optional[RolloutStepHook] = None,
):
    """Run a component, iterating if needed, returning outputs and state history.

    When ``streaming_loss_fn`` is provided, the third return element is the
    accumulated scalar loss instead of state history.
    """
    if isinstance(component, Graph) and component._needs_iteration:
        return component(
            inputs,
            init_state,
            key=key,
            n_steps=n_steps,
            return_state_history=streaming_loss_fn is None,
            state_filter=state_filter,
            streaming_loss_fn=streaming_loss_fn,
            rollout_step_hook=rollout_step_hook,
        )
    if n_steps is None:
        raise ValueError("n_steps is required for acyclic components")
    checkpoint = getattr(component, "checkpoint", False)
    return iterate_component(
        component,
        inputs,
        init_state,
        n_steps,
        key,
        state_filter=state_filter,
        checkpoint=checkpoint,
        streaming_loss_fn=streaming_loss_fn,
        rollout_step_hook=rollout_step_hook,
    )
