"""General-purpose filters for dynamical systems.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Optional, Type

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from equinox import field
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from pydantic import BaseModel, ConfigDict
from pydantic import Field as ConfigField

from feedbax._ode import ode_init, ode_step
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import ComponentOptions, Ports
from feedbax.runtime.components import (
    stateful_vector_output_prototype,
    stateful_vector_prototype,
)
from feedbax.runtime.graph import Component, GraphNode
from feedbax.runtime.prototypes import (
    array_input_configuration,
    materialize_array_input,
    stateful_passthrough_output_prototype,
)
from feedbax.runtime.state import StateBounds


def require_real_floating_filter_prototype(prototype: PyTree[Array]) -> None:
    """Require the real floating state domain of FirstOrderFilter."""
    for index, leaf in enumerate(jt.leaves(prototype)):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError(
                "FirstOrderFilter requires real floating state leaves; "
                f"prototype leaf {index} has dtype {dtype!s}"
            )


class FilterState(eqx.Module):
    """Holds the current filtered signal and solver state."""

    output: PyTree[Array]
    solver: PyTree


def first_order_filter_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the filter signal prototype after checking its numerical domain."""
    outputs = stateful_passthrough_output_prototype(params, inputs)
    require_real_floating_filter_prototype(outputs["output"])
    return outputs


class _FirstOrderFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tau_rise: float = ConfigField(default=0.05, ge=0.0)
    tau_decay: float = ConfigField(default=0.05, ge=0.0)
    dt: float = ConfigField(default=0.001, ge=0.0)
    init_value: float = ConfigField(default=0.0)
    input_shape: Any = ConfigField(default=None, json_schema_extra={"feedbax": {"type": "array"}})
    input_dtype: str | None = ConfigField(default=None)


class LowPassFilter(GraphNode):
    """Continuous-time first-order low-pass filter."""

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )

    input_ports: ClassVar = tuple(ports.inputs)
    output_ports: ClassVar = tuple(ports.outputs)

    tau_rise: float = 0.050
    tau_decay: float = 0.050
    dt: float = 0.001
    solver: dfx.AbstractSolver = field(default_factory=lambda: dfx.Euler())
    input_proto: PyTree[Array] = field(default_factory=lambda: jnp.zeros(1))
    init_value: float = 0.0
    state_index: StateIndex

    def __init__(
        self,
        tau_rise: float = _FirstOrderFilterConfig.model_fields["tau_rise"].default,
        tau_decay: float = _FirstOrderFilterConfig.model_fields["tau_decay"].default,
        dt: float = _FirstOrderFilterConfig.model_fields["dt"].default,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        input_proto: Optional[PyTree[Array]] = None,
        init_value: float = _FirstOrderFilterConfig.model_fields["init_value"].default,
    ):
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.dt = dt
        self.solver = solver_type()
        if input_proto is None:
            input_proto = jnp.zeros(1)
        self.input_proto = input_proto
        self.init_value = init_value

        initial_state = self._initial_state_value(input_proto)
        self.state_index = StateIndex(initial_state)

    def _initial_state_value(self, input_proto: PyTree[Array]) -> FilterState:
        require_real_floating_filter_prototype(input_proto)
        output_init = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
        solver_init = ode_init(
            self.solver, self._term, self.dt, output_init, jt.map(jnp.zeros_like, input_proto)
        )
        return FilterState(output=output_init, solver=solver_init)

    def vector_field(
        self,
        t: Scalar,
        state: PyTree[Array],
        input: PyTree[jax.Array],
    ) -> PyTree[Array]:
        def derivative(value, output):
            tau = jnp.where(
                value >= output,
                jnp.asarray(self.tau_rise, dtype=output.dtype),
                jnp.asarray(self.tau_decay, dtype=output.dtype),
            )
            return (value - output) / tau

        return jt.map(derivative, input, state)

    @property
    def _term(self) -> dfx.AbstractTerm:
        return dfx.ODETerm(self.vector_field)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        filter_state: FilterState = state.get(self.state_index)
        input_value = inputs["input"]

        output, solver_state = ode_step(
            self.solver,
            self._term,
            self.dt,
            filter_state.output,
            input_value,
            filter_state.solver,
        )
        output_state = FilterState(output=output, solver=solver_state)
        state = state.set(self.state_index, output_state)
        return {"output": output_state.output}, state

    def change_input(self, input_proto: PyTree[Array]) -> "FirstOrderFilter":
        new_initial_state = self._initial_state_value(input_proto)
        new_state_index = StateIndex(new_initial_state)
        return eqx.tree_at(
            lambda model: (model.input_proto, model.state_index),
            self,
            (input_proto, new_state_index),
        )

    @property
    def bounds(self) -> StateBounds[FilterState]:
        return StateBounds(
            low=FilterState(output=None, solver=None),
            high=FilterState(output=None, solver=None),
        )


class FirstOrderFilter(LowPassFilter, Component):
    """Portable FirstOrderFilter with a reconstructible typed configuration."""

    ports: ClassVar = LowPassFilter.ports

    config: ClassVar[type[BaseModel]] = _FirstOrderFilterConfig

    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=first_order_filter_output_prototype
    )

    @classmethod
    def from_config(cls, config, **context):
        return materialize_array_input(cls, config, **context)

    def configuration_values(self):
        if type(self.solver) is not dfx.Euler:
            raise ValueError(
                "Portable FirstOrderFilter requires Euler; use LowPassFilter for a Python solver"
            )
        return array_input_configuration(self)


def _nonnegative_time(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


class _HeldLowPass(eqx.Module):
    input_proto: Array
    state_index: StateIndex

    def __init__(self, input_proto: Array):
        self.input_proto = input_proto
        self.state_index = StateIndex(FilterState(output=jnp.zeros_like(input_proto), solver=None))

    def advance(
        self, sample: Array, state: State, *, tau: float, dt: float
    ) -> tuple[Array, Array, State]:
        """Return the updated low-pass and its same-sample residual."""
        previous = state.get(self.state_index).output
        elapsed = dt / tau if tau else (math.inf if dt else 0.0)
        dtype = jnp.result_type(previous.dtype, jnp.float32)
        decay = jnp.asarray(math.exp(-elapsed), dtype=dtype)
        gain = jnp.asarray(-math.expm1(-elapsed), dtype=dtype)
        retained = decay * previous
        lowpass = (retained + gain * sample).astype(previous.dtype)
        residual = (decay * sample - retained).astype(previous.dtype)
        return lowpass, residual, state.set(self.state_index, FilterState(lowpass, None))


class _HighPassFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tau: float = ConfigField(default=0.1, ge=0.0)
    dt: float = ConfigField(default=0.01, ge=0.0)
    n_dims: int = ConfigField(default=1, ge=1)
    input_shape: list[int] | None = None
    input_dtype: str | None = None


class HighPassFilter(Component):
    """High-pass filter: output = input - lowpass(input).

    The first-order low-pass state advances exactly over one interval with
    the current input held constant. The output is the residual after this
    update, so the current input contributes to the current output.

    Args:
        tau: Analog time constant (cutoff at `1 / (2 * pi * tau)`). Zero
            makes the low-pass track the input instantly when dt is positive.
        dt: Sample interval. Zero leaves the low-pass state fixed.
        n_dims: Dimensionality used when no input prototype is supplied.
        input_proto: Shape and dtype of the signal.
    """

    config: ClassVar[type[BaseModel]] = _HighPassFilterConfig

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=stateful_vector_output_prototype
    )

    tau: float = field(static=True)
    dt: float = field(static=True)
    n_dims: int = field(static=True)
    _lowpass: _HeldLowPass

    def __init__(
        self,
        tau: float = _HighPassFilterConfig.model_fields["tau"].default,
        dt: float = _HighPassFilterConfig.model_fields["dt"].default,
        n_dims: int = _HighPassFilterConfig.model_fields["n_dims"].default,
        *,
        input_proto: Array | None = None,
    ):
        self.tau = _nonnegative_time(tau, "tau")
        self.dt = _nonnegative_time(dt, "dt")
        self.n_dims = n_dims
        self._lowpass = _HeldLowPass(stateful_vector_prototype(n_dims, input_proto))

    @property
    def input_proto(self) -> Array:
        return self._lowpass.input_proto

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        _, output, state = self._lowpass.advance(inputs["input"], state, tau=self.tau, dt=self.dt)
        return {"output": output}, state

    @classmethod
    def from_config(cls, config, **context):
        return materialize_array_input(cls, config, **context)

    def configuration_values(self):
        return array_input_configuration(self)


class _BandPassFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tau_low: float = ConfigField(default=0.1, ge=0.0)
    tau_high: float = ConfigField(default=0.01, ge=0.0)
    dt: float = ConfigField(default=0.01, ge=0.0)
    n_dims: int = ConfigField(default=1, ge=1)
    input_shape: list[int] | None = None
    input_dtype: str | None = None


class BandPassFilter(Component):
    """Band-pass filter: cascade of high-pass and low-pass.

    The high-pass output for the current sample is held constant over the
    upper-cutoff stage's interval. Both stages use exact held-input updates
    and the time constants describe the analog cutoffs of the individual stages.

    Args:
        tau_low: Time constant for the high-pass stage (lower frequency bound;
            larger tau means lower cutoff frequency).
        tau_high: Time constant for the low-pass stage (upper frequency bound;
            smaller tau means higher cutoff frequency). Zero makes this stage
            track its input instantly when dt is positive.
        dt: Sample interval. Zero leaves both stages' state fixed.
        n_dims: Dimensionality used when no input prototype is supplied.
        input_proto: Shape and dtype of the signal.
    """

    config: ClassVar[type[BaseModel]] = _BandPassFilterConfig

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="vector")}, outputs={"output": PortType(dtype="vector")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=stateful_vector_output_prototype
    )

    tau_high: float = field(static=True)
    _highpass: HighPassFilter
    _lowpass: _HeldLowPass

    def __init__(
        self,
        tau_low: float = _BandPassFilterConfig.model_fields["tau_low"].default,
        tau_high: float = _BandPassFilterConfig.model_fields["tau_high"].default,
        dt: float = _BandPassFilterConfig.model_fields["dt"].default,
        n_dims: int = _BandPassFilterConfig.model_fields["n_dims"].default,
        *,
        input_proto: Array | None = None,
    ):
        self.tau_high = _nonnegative_time(tau_high, "tau_high")
        self._highpass = HighPassFilter(tau=tau_low, dt=dt, n_dims=n_dims, input_proto=input_proto)
        self._lowpass = _HeldLowPass(self.input_proto)

    @property
    def tau_low(self) -> float:
        return self._highpass.tau

    @property
    def dt(self) -> float:
        return self._highpass.dt

    @property
    def n_dims(self) -> int:
        return self._highpass.n_dims

    @property
    def input_proto(self) -> Array:
        return self._highpass.input_proto

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        hp_out, state = self._highpass(inputs, state, key=key)
        output, _, state = self._lowpass.advance(
            hp_out["output"], state, tau=self.tau_high, dt=self.dt
        )
        return {"output": output}, state

    @classmethod
    def from_config(cls, config, **context):
        return materialize_array_input(cls, config, **context)

    def configuration_values(self):
        return array_input_configuration(self)
