"""Discrete-time control system components.

Provides discrete-time accumulators, unit delays, and zero-order holds.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

import equinox as eqx
from equinox import field
from equinox.nn import State, StateIndex
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.graph import Component


class IntegratorDiscrete(Component):
    """Discrete-time accumulator (summation).

    Computes y[k] = y[k-1] + u[k] * dt at each step.

    Args:
        dt: Scaling factor per step (defaults to 1.0 for pure accumulation).
        n_dims: Dimensionality of the accumulated signal.
        initial_value: Initial accumulator value.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    dt: float
    n_dims: int = field(static=True)
    state_index: StateIndex
    _initial_state: Array = field(static=True)

    def __init__(
        self,
        dt: float = 1.0,
        n_dims: int = 1,
        initial_value: float = 0.0,
    ):
        self.dt = float(dt)
        self.n_dims = n_dims
        self._initial_state = jnp.zeros(n_dims) + initial_value
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        y = state.get(self.state_index)
        y_new = y + inputs["input"] * self.dt
        state = state.set(self.state_index, y_new)
        return {"output": y_new}, state


class UnitDelay(Component):
    """Unit delay (z^{-1}): output is the previous input.

    y[k] = u[k-1]

    Args:
        n_dims: Dimensionality of the signal.
        initial_value: Initial output value (before any input arrives).
    """

    input_ports = ("input",)
    output_ports = ("output",)

    n_dims: int = field(static=True)
    state_index: StateIndex
    _initial_state: Array = field(static=True)

    def __init__(
        self,
        n_dims: int = 1,
        initial_value: float = 0.0,
    ):
        self.n_dims = n_dims
        self._initial_state = jnp.zeros(n_dims) + initial_value
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        prev = state.get(self.state_index)
        state = state.set(self.state_index, inputs["input"])
        return {"output": prev}, state


class ZeroOrderHold(Component):
    """Zero-order hold: sample input every N steps, hold between samples.

    Args:
        hold_steps: Number of steps to hold each sample.
        n_dims: Dimensionality of the signal.
        initial_value: Initial held value.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    hold_steps: int = field(static=True)
    n_dims: int = field(static=True)
    state_index: StateIndex
    _initial_state: tuple[Array, Array] = field(static=True)

    def __init__(
        self,
        hold_steps: int = 1,
        n_dims: int = 1,
        initial_value: float = 0.0,
    ):
        self.hold_steps = int(hold_steps)
        self.n_dims = n_dims
        held_value = jnp.zeros(n_dims) + initial_value
        step_counter = jnp.zeros((), dtype=jnp.int32)
        self._initial_state = (held_value, step_counter)
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        held_value, step_counter = state.get(self.state_index)
        u = inputs["input"]

        # Sample on counter == 0, otherwise hold
        should_sample = step_counter == 0
        new_held = jnp.where(should_sample, u, held_value)

        # Advance counter, wrap around
        new_counter = jnp.where(
            step_counter >= self.hold_steps - 1,
            jnp.zeros((), dtype=jnp.int32),
            step_counter + 1,
        )

        state = state.set(self.state_index, (new_held, new_counter))
        return {"output": new_held}, state
