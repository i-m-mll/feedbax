"""PID controller components.

Provides continuous and discrete PID controllers with anti-windup.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import partial
from typing import ClassVar

import jax.numpy as jnp
from equinox import field
from equinox.nn import State, StateIndex
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray, PyTree
from pydantic import BaseModel
from pydantic import Field as ConfigField

from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, ConfigBinding, Ports
from feedbax.control._vector import (
    _CONTROL_ERROR_PROTOTYPE,
    _control_output_prototype,
    initial_control_vector,
    require_control_vector,
)
from feedbax.runtime.graph import Component


class _PIDParams(BaseModel):
    Kp: float = ConfigField(default=1.0)
    Ki: float = 0.0
    Kd: float = 0.0
    dt: float = ConfigField(default=0.01, ge=0.0)
    n_dims: int = ConfigField(default=1, ge=1)
    input_dtype: str | None = ConfigField(
        default=None,
        description="Real floating-point dtype inferred from the connected signal when omitted.",
    )


class _ContinuousPIDParams(_PIDParams):
    integral_limit: float | None = None


class _DiscretePIDParams(_PIDParams):
    output_limit: float | None = None


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
        integral_limit: Symmetric clamp on the integral accumulator. None leaves it unclamped.
        n_dims: Dimensionality of the error signal.
        input_dtype: Real floating dtype of the input and carry.
    """

    ports: ClassVar[Ports] = Ports(
        inputs={"error": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(_control_output_prototype, port="error"),
        input_prototype_parameter=_CONTROL_ERROR_PROTOTYPE,
    )
    config: ClassVar[type[BaseModel]] = _ContinuousPIDParams
    config_bindings: ClassVar[tuple[ConfigBinding, ...]] = (
        ConfigBinding("integral_limit_param", name="integral_limit"),
    )

    Kp: float
    Ki: float
    Kd: float
    dt: float
    integral_limit: float
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    state_index: StateIndex

    def __init__(
        self,
        Kp: float = _ContinuousPIDParams.model_fields["Kp"].default,
        Ki: float = _ContinuousPIDParams.model_fields["Ki"].default,
        Kd: float = _ContinuousPIDParams.model_fields["Kd"].default,
        dt: float = _ContinuousPIDParams.model_fields["dt"].default,
        integral_limit: float | None = _ContinuousPIDParams.model_fields["integral_limit"].default,
        n_dims: int = _ContinuousPIDParams.model_fields["n_dims"].default,
        *,
        input_dtype: DTypeLike | None = _ContinuousPIDParams.model_fields["input_dtype"].default,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.dt = float(dt)
        self.integral_limit = float("inf") if integral_limit is None else float(integral_limit)
        zero = initial_control_vector(n_dims, input_dtype)
        self.n_dims = zero.shape[0]
        self.input_dtype = zero.dtype.name
        initial_state = (zero, zero.copy())
        self.state_index = StateIndex(initial_state)

    @property
    def integral_limit_param(self) -> float | None:
        """Return the integral limit in a JSON-compatible form."""
        return None if self.integral_limit == float("inf") else self.integral_limit

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        integral, prev_error = state.get(self.state_index)
        error = require_control_vector(inputs["error"], prev_error, port="error")

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
        output_limit: Symmetric clamp on the accumulated output. None leaves it unclamped.
        n_dims: Dimensionality of the error signal.
        input_dtype: Real floating dtype of the input and carry.
    """

    ports: ClassVar[Ports] = Ports(
        inputs={"error": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(_control_output_prototype, port="error"),
        input_prototype_parameter=_CONTROL_ERROR_PROTOTYPE,
    )
    config: ClassVar[type[BaseModel]] = _DiscretePIDParams
    config_bindings: ClassVar[tuple[ConfigBinding, ...]] = (
        ConfigBinding("output_limit_param", name="output_limit"),
    )

    Kp: float
    Ki: float
    Kd: float
    dt: float
    output_limit: float
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    state_index: StateIndex

    def __init__(
        self,
        Kp: float = _DiscretePIDParams.model_fields["Kp"].default,
        Ki: float = _DiscretePIDParams.model_fields["Ki"].default,
        Kd: float = _DiscretePIDParams.model_fields["Kd"].default,
        dt: float = _DiscretePIDParams.model_fields["dt"].default,
        output_limit: float | None = _DiscretePIDParams.model_fields["output_limit"].default,
        n_dims: int = _DiscretePIDParams.model_fields["n_dims"].default,
        *,
        input_dtype: DTypeLike | None = _DiscretePIDParams.model_fields["input_dtype"].default,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.dt = float(dt)
        self.output_limit = float("inf") if output_limit is None else float(output_limit)
        zero = initial_control_vector(n_dims, input_dtype)
        self.n_dims = zero.shape[0]
        self.input_dtype = zero.dtype.name
        initial_state = (zero, zero.copy(), zero.copy())
        self.state_index = StateIndex(initial_state)

    @property
    def output_limit_param(self) -> float | None:
        """Return the output limit in a JSON-compatible form."""
        return None if self.output_limit == float("inf") else self.output_limit

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        prev_output, prev_error, prev_prev_error = state.get(self.state_index)
        error = require_control_vector(inputs["error"], prev_error, port="error")

        # Velocity form increments
        dp = self.Kp * (error - prev_error)
        di = self.Ki * error * self.dt
        dd = self.Kd * (error - 2.0 * prev_error + prev_prev_error) / self.dt

        du = dp + di + dd
        output = prev_output + du
        output = jnp.clip(output, -self.output_limit, self.output_limit)

        state = state.set(self.state_index, (output, error, prev_error))
        return {"output": output}, state
