"""Continuous-time control system components.

Provides integrators, derivatives, state-space models, and transfer
functions that operate using Euler discretization per call.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax.numpy as jnp
from equinox import field
from equinox.nn import State, StateIndex
from jax.typing import DTypeLike
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.component_registry import component
from feedbax.component_registry.declarations import ParameterField
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.control._vector import (
    _CONTROL_INPUT_DTYPE,
    _CONTROL_INPUT_PROTOTYPE,
    _control_output_prototype,
    initial_control_vector,
    require_control_vector,
)
from feedbax.runtime.graph import Component


@component(
    "Integrator",
    category="Control",
    description="Continuous-time integrator (Euler).",
    parameter_fields=[
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=True),
        ParameterField(name="initial_value", type="float", default=0.0, required=False),
        _CONTROL_INPUT_DTYPE,
    ],
    icon="Integral",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=partial(_control_output_prototype, port="input"),
    input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
)
class Integrator(Component):
    """Continuous-time integrator using Euler integration.

    Computes dy/dt = u, discretized as y += u * dt each call.

    Args:
        dt: Integration time step.
        n_dims: Dimensionality of the integrated signal.
        initial_value: Initial value for all dimensions.
        input_dtype: Real floating dtype of the input and carry.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    dt: float
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        dt: float = 0.01,
        n_dims: int = 1,
        initial_value: float = 0.0,
        *,
        input_dtype: DTypeLike | None = None,
    ):
        self.dt = float(dt)
        self.initial_value = float(initial_value)
        initial_state = initial_control_vector(n_dims, input_dtype, self.initial_value)
        self.n_dims = initial_state.shape[0]
        self.input_dtype = initial_state.dtype.name
        self.state_index = StateIndex(initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        y = state.get(self.state_index)
        u = require_control_vector(inputs["input"], y, port="input")
        y_new = y + u * self.dt
        state = state.set(self.state_index, y_new)
        return {"output": y_new}, state


@component(
    "Derivative",
    category="Control",
    description="Finite-difference derivative.",
    parameter_fields=[
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=True),
        ParameterField(name="initial_value", type="float", default=0.0, required=False),
        _CONTROL_INPUT_DTYPE,
    ],
    icon="TrendingUp",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=partial(_control_output_prototype, port="input"),
    input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
)
class Derivative(Component):
    """Finite-difference derivative approximation.

    Computes dy/dt ~ (u - u_prev) / dt using backward differencing.

    Args:
        dt: Time step for the derivative computation.
        n_dims: Dimensionality of the input signal.
        initial_value: Initial previous-input value.
        input_dtype: Real floating dtype of the input and carry.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    dt: float
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        dt: float = 0.01,
        n_dims: int = 1,
        initial_value: float = 0.0,
        *,
        input_dtype: DTypeLike | None = None,
    ):
        self.dt = float(dt)
        self.initial_value = float(initial_value)
        initial_state = initial_control_vector(n_dims, input_dtype, self.initial_value)
        self.n_dims = initial_state.shape[0]
        self.input_dtype = initial_state.dtype.name
        self.state_index = StateIndex(initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        u_prev = state.get(self.state_index)
        u = require_control_vector(inputs["input"], u_prev, port="input")
        deriv = (u - u_prev) / self.dt
        state = state.set(self.state_index, u)
        return {"output": deriv}, state


@component(
    "StateSpace",
    runtime_type=None,
    category="Control",
    description="Continuous LTI state-space (Euler).",
    parameter_fields=[
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
    ],
    icon="Grid3x3",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
)
class StateSpace(Component):
    """Continuous-time linear state-space model with Euler discretization.

    Implements:
        x' = Ax + Bu
        y  = Cx + Du

    Each call advances one Euler step: x_new = x + (Ax + Bu) * dt.

    Args:
        A: State matrix, shape ``[n, n]``.
        B: Input matrix, shape ``[n, m]``.
        C: Output matrix, shape ``[p, n]``.
        D: Feedthrough matrix, shape ``[p, m]``.
        dt: Integration time step.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    A: Array  # [n, n]
    B: Array  # [n, m]
    C: Array  # [p, n]
    D: Array  # [p, m]
    dt: float
    state_index: StateIndex

    def __init__(
        self,
        A: Array,
        B: Array,
        C: Array,
        D: Array,
        dt: float = 0.01,
    ):
        self.A = jnp.asarray(A, dtype=float)
        self.B = jnp.asarray(B, dtype=float)
        self.C = jnp.asarray(C, dtype=float)
        self.D = jnp.asarray(D, dtype=float)
        self.dt = float(dt)
        n = self.A.shape[0]
        initial_state = jnp.zeros(n)
        self.state_index = StateIndex(initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        x = state.get(self.state_index)
        u = inputs["input"]
        # Euler step: x_new = x + (Ax + Bu) * dt
        x_dot = self.A @ x + self.B @ u
        x_new = x + x_dot * self.dt
        y = self.C @ x_new + self.D @ u
        state = state.set(self.state_index, x_new)
        return {"output": y}, state


@component(
    "TransferFunction",
    runtime_type=None,
    category="Control",
    description="Transfer function H(s)=num/den.",
    parameter_fields=[
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
    ],
    icon="FunctionSquare",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
)
class TransferFunction(Component):
    """Transfer function H(s) = num(s) / den(s) in controllable canonical form.

    Converts a SISO transfer function to state-space representation
    internally, then delegates to a ``StateSpace`` component.

    The numerator and denominator are given as coefficient lists in
    descending power of s. For example, H(s) = (2s + 3) / (s^2 + 4s + 5)
    would be ``num=[2, 3], den=[1, 4, 5]``.

    Args:
        num: Numerator polynomial coefficients (descending powers of s).
        den: Denominator polynomial coefficients (descending powers of s).
        dt: Integration time step for the internal state-space model.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    _ss: StateSpace

    def __init__(
        self,
        num: Sequence[float],
        den: Sequence[float],
        dt: float = 0.01,
    ):
        num = [float(c) for c in num]
        den = [float(c) for c in den]

        if len(den) == 0:
            raise ValueError("Denominator must have at least one coefficient.")
        if den[0] == 0.0:
            raise ValueError("Leading denominator coefficient must be nonzero.")

        # Normalize so leading denominator coefficient is 1
        lead = den[0]
        den = [c / lead for c in den]
        num = [c / lead for c in num]

        n = len(den) - 1  # system order

        if n == 0:
            # Static gain: H(s) = num[0]
            A = jnp.zeros((1, 1))
            B = jnp.zeros((1, 1))
            C = jnp.zeros((1, 1))
            D = jnp.array([[num[0] if num else 0.0]])
            self._ss = StateSpace(A, B, C, D, dt)
            return

        # Pad numerator with leading zeros to length n + 1
        num = [0.0] * (n + 1 - len(num)) + num

        # Controllable canonical form
        # A is companion matrix
        A = jnp.zeros((n, n))
        if n > 1:
            A = A.at[: n - 1, 1:].set(jnp.eye(n - 1))
        # Last row: -a_0, -a_1, ..., -a_{n-1} (reversed den[1:])
        A = A.at[n - 1, :].set(jnp.array([-den[n - i] for i in range(n)]))

        B = jnp.zeros((n, 1))
        B = B.at[n - 1, 0].set(1.0)

        # C and D from numerator
        # D = b_0 (coefficient of s^n in numerator after padding)
        d_val = num[0]
        # C = [b_n - b_0*a_n, ..., b_1 - b_0*a_1] rearranged for canonical form
        c_coeffs = jnp.array([num[n - i] - d_val * den[n - i] for i in range(n)])
        C = c_coeffs.reshape(1, n)
        D = jnp.array([[d_val]])

        self._ss = StateSpace(A, B, C, D, dt)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        return self._ss(inputs, state, key=key)
