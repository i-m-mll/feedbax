"""Discrete-time control system components.

Provides discrete-time accumulators, unit delays, and zero-order holds.

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
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.control._vector import (
    _CONTROL_INPUT_PROTOTYPE,
    _control_output_prototype,
    initial_control_vector,
    require_control_vector,
)
from feedbax.runtime.graph import Component


class _IntegratorDiscreteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dt: float = ConfigField(default=1.0, ge=0.0)
    n_dims: int = ConfigField(default=1, ge=1)
    initial_value: float = ConfigField(
        default=0.0
    )
    input_dtype: str | None = None


class IntegratorDiscrete(Component):
    """Discrete-time accumulator (summation).

    Computes y[k] = y[k-1] + u[k] * dt at each step.

    Args:
        dt: Scaling factor per step (defaults to 1.0 for pure accumulation).
        n_dims: Dimensionality of the accumulated signal.
        initial_value: Initial accumulator value.
        input_dtype: Real floating dtype of the input and carry.
    """

    config: ClassVar[type[BaseModel]] = _IntegratorDiscreteConfig

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(_control_output_prototype, port="input"),
        input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
    )

    dt: float
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        dt: float = _IntegratorDiscreteConfig.model_fields["dt"].default,
        n_dims: int = _IntegratorDiscreteConfig.model_fields["n_dims"].default,
        initial_value: float = _IntegratorDiscreteConfig.model_fields["initial_value"].default,
        *,
        input_dtype: DTypeLike | None = _IntegratorDiscreteConfig.model_fields[
            "input_dtype"
        ].default,
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


class _UnitDelayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    n_dims: int = ConfigField(default=1, ge=1)
    initial_value: float = ConfigField(
        default=0.0
    )
    input_dtype: str | None = None


class UnitDelay(Component):
    """Unit delay (z^{-1}): output is the previous input.

    y[k] = u[k-1]

    Args:
        n_dims: Dimensionality of the signal.
        initial_value: Initial output value (before any input arrives).
        input_dtype: Real floating dtype of the input and carry.
    """

    config: ClassVar[type[BaseModel]] = _UnitDelayConfig

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(_control_output_prototype, port="input"),
        input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
    )

    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        n_dims: int = _UnitDelayConfig.model_fields["n_dims"].default,
        initial_value: float = _UnitDelayConfig.model_fields["initial_value"].default,
        *,
        input_dtype: DTypeLike | None = _UnitDelayConfig.model_fields["input_dtype"].default,
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


class _ZeroOrderHoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    hold_steps: int = ConfigField(
        default=1, ge=1
    )
    n_dims: int = ConfigField(default=1, ge=1)
    initial_value: float = ConfigField(
        default=0.0
    )
    input_dtype: str | None = None


class ZeroOrderHold(Component):
    """Zero-order hold: sample input every N steps, hold between samples.

    Args:
        hold_steps: Number of steps to hold each sample.
        n_dims: Dimensionality of the signal.
        initial_value: Initial held value.
        input_dtype: Real floating dtype of the input and carry.
    """

    config: ClassVar[type[BaseModel]] = _ZeroOrderHoldConfig

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(_control_output_prototype, port="input"),
        input_prototype_parameter=_CONTROL_INPUT_PROTOTYPE,
    )

    hold_steps: int = field(static=True)
    n_dims: int = field(static=True)
    input_dtype: str = field(static=True)
    initial_value: float
    state_index: StateIndex

    def __init__(
        self,
        hold_steps: int = _ZeroOrderHoldConfig.model_fields["hold_steps"].default,
        n_dims: int = _ZeroOrderHoldConfig.model_fields["n_dims"].default,
        initial_value: float = _ZeroOrderHoldConfig.model_fields["initial_value"].default,
        *,
        input_dtype: DTypeLike | None = _ZeroOrderHoldConfig.model_fields["input_dtype"].default,
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
