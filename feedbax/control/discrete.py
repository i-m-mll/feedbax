"""Discrete-time control system components.

Provides discrete-time accumulators, unit delays, and zero-order holds.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from equinox import field
from equinox.nn import State, StateIndex
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray, PyTree

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
    "IntegratorDiscrete",
    category="Discrete",
    description="Discrete-time accumulator.",
    parameter_fields=[
        ParameterField(name="dt", type="float", default=1.0, min=0.0, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=True),
        ParameterField(name="initial_value", type="float", default=0.0, required=False),
        _CONTROL_INPUT_DTYPE,
    ],
    icon="PlusSquare",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=partial(_control_output_prototype, port="input"),
    input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
)
class IntegratorDiscrete(Component):
    """Discrete-time accumulator (summation).

    Computes y[k] = y[k-1] + u[k] * dt at each step.

    Args:
        dt: Scaling factor per step (defaults to 1.0 for pure accumulation).
        n_dims: Dimensionality of the accumulated signal.
        initial_value: Initial accumulator value.
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
        dt: float = 1.0,
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
    "UnitDelay",
    category="Discrete",
    description="Unit delay (z^-1).",
    parameter_fields=[
        ParameterField(name="n_dims", type="int", default=1, min=1, required=True),
        ParameterField(name="initial_value", type="float", default=0.0, required=False),
        _CONTROL_INPUT_DTYPE,
    ],
    icon="Clock",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=partial(_control_output_prototype, port="input"),
    input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
)
class UnitDelay(Component):
    """Unit delay (z^{-1}): output is the previous input.

    y[k] = u[k-1]

    Args:
        n_dims: Dimensionality of the signal.
        initial_value: Initial output value (before any input arrives).
        input_dtype: Real floating dtype of the input and carry.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        n_dims: int = 1,
        initial_value: float = 0.0,
        *,
        input_dtype: DTypeLike | None = None,
    ):
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
        prev = state.get(self.state_index)
        u = require_control_vector(inputs["input"], prev, port="input")
        state = state.set(self.state_index, u)
        return {"output": prev}, state


@component(
    "ZeroOrderHold",
    category="Discrete",
    description="Sample and hold every N steps.",
    parameter_fields=[
        ParameterField(name="hold_steps", type="int", default=1, min=1, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=True),
        ParameterField(name="initial_value", type="float", default=0.0, required=False),
        _CONTROL_INPUT_DTYPE,
    ],
    icon="Pause",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=partial(_control_output_prototype, port="input"),
    input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
)
class ZeroOrderHold(Component):
    """Zero-order hold: sample input every N steps, hold between samples.

    Args:
        hold_steps: Number of steps to hold each sample.
        n_dims: Dimensionality of the signal.
        initial_value: Initial held value.
        input_dtype: Real floating dtype of the input and carry.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    hold_steps: int = field(static=True)
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        hold_steps: int = 1,
        n_dims: int = 1,
        initial_value: float = 0.0,
        *,
        input_dtype: DTypeLike | None = None,
    ):
        self.hold_steps = int(hold_steps)
        self.initial_value = float(initial_value)
        held_value = initial_control_vector(n_dims, input_dtype, self.initial_value)
        self.n_dims = held_value.shape[0]
        self.input_dtype = held_value.dtype.name
        step_counter = jnp.zeros((), dtype=jnp.int32)
        initial_state = (held_value, step_counter)
        self.state_index = StateIndex(initial_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        held_value, step_counter = state.get(self.state_index)
        u = require_control_vector(inputs["input"], held_value, port="input")

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
