"""PID controller components.

Provides continuous and discrete PID controllers with anti-windup.

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


class PID(Component):
    """Continuous PID controller with anti-windup clamping.

    Computes the standard PID law:
        u = Kp * e + Ki * integral(e) + Kd * de/dt

    The integral term is clamped to ``[-integral_limit, integral_limit]``
    to prevent windup. The derivative is computed via backward difference.

    Args:
        Kp: Proportional gain.
        Ki: Integral gain.
        Kd: Derivative gain.
        dt: Time step for integration and differentiation.
        integral_limit: Symmetric clamp on the integral accumulator.
        n_dims: Dimensionality of the error signal.
    """

    input_ports = ("error",)
    output_ports = ("output",)

    Kp: float
    Ki: float
    Kd: float
    dt: float
    integral_limit: float
    n_dims: int = field(static=True)
    state_index: StateIndex
    _initial_state: tuple[Array, Array] = field(static=True)

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        dt: float = 0.01,
        integral_limit: float = float("inf"),
        n_dims: int = 1,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.dt = float(dt)
        self.integral_limit = float(integral_limit)
        self.n_dims = n_dims
        integral = jnp.zeros(n_dims)
        prev_error = jnp.zeros(n_dims)
        self._initial_state = (integral, prev_error)
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        integral, prev_error = state.get(self.state_index)
        error = inputs["error"]

        # Proportional
        p_term = self.Kp * error

        # Integral with anti-windup
        integral = integral + error * self.dt
        integral = jnp.clip(integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * integral

        # Derivative (backward difference)
        d_term = self.Kd * (error - prev_error) / self.dt

        output = p_term + i_term + d_term
        state = state.set(self.state_index, (integral, error))
        return {"output": output}, state


class PIDDiscrete(Component):
    """Discrete PID controller in velocity (incremental) form.

    Uses the velocity form for improved anti-windup behavior:
        du[k] = Kp*(e[k] - e[k-1]) + Ki*e[k]*dt + Kd*(e[k] - 2*e[k-1] + e[k-2])/dt
        u[k]  = u[k-1] + du[k]

    The accumulated output is clamped to ``[-output_limit, output_limit]``.

    Args:
        Kp: Proportional gain.
        Ki: Integral gain.
        Kd: Derivative gain.
        dt: Time step.
        output_limit: Symmetric clamp on the accumulated output.
        n_dims: Dimensionality of the error signal.
    """

    input_ports = ("error",)
    output_ports = ("output",)

    Kp: float
    Ki: float
    Kd: float
    dt: float
    output_limit: float
    n_dims: int = field(static=True)
    state_index: StateIndex
    _initial_state: tuple[Array, Array, Array] = field(static=True)

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        dt: float = 0.01,
        output_limit: float = float("inf"),
        n_dims: int = 1,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.dt = float(dt)
        self.output_limit = float(output_limit)
        self.n_dims = n_dims
        prev_output = jnp.zeros(n_dims)
        prev_error = jnp.zeros(n_dims)
        prev_prev_error = jnp.zeros(n_dims)
        self._initial_state = (prev_output, prev_error, prev_prev_error)
        self.state_index = StateIndex(self._initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        prev_output, prev_error, prev_prev_error = state.get(self.state_index)
        error = inputs["error"]

        # Velocity form increments
        dp = self.Kp * (error - prev_error)
        di = self.Ki * error * self.dt
        dd = self.Kd * (error - 2.0 * prev_error + prev_prev_error) / self.dt

        du = dp + di + dd
        output = prev_output + du
        output = jnp.clip(output, -self.output_limit, self.output_limit)

        state = state.set(self.state_index, (output, error, prev_error))
        return {"output": output}, state
