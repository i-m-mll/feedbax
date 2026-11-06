"""General-purpose filters for dynamical systems.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import math
from collections.abc import Mapping
from typing import Any, Optional, Type

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from equinox import field
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from feedbax._ode import ode_init, ode_step
from feedbax.component_registry import component
from feedbax.component_registry.declarations import ComponentConstructorPlan, ParameterField
from feedbax.component_registry.prototypes import (
    migrate_stateful_v1,
    stateful_passthrough_output_prototype,
)
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.migrations import ComponentMigration
from feedbax.runtime.components import (
    _STATEFUL_VECTOR_CONSTRUCTOR,
    _STATEFUL_VECTOR_PROTOTYPE_FIELDS,
    stateful_vector_output_prototype,
    stateful_vector_prototype,
)
from feedbax.runtime.graph import Component
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


@component(
    "FirstOrderFilter",
    constructor=ComponentConstructorPlan(
        context_adapters={"input_shape": ("input_proto", "array_prototype")}
    ),
    param_schema_version="2",
    category="Channels",
    description="First-order low-pass filter.",
    parameter_fields=[
        ParameterField(name="tau_rise", type="float", default=0.05, min=0.0, required=True),
        ParameterField(name="tau_decay", type="float", default=0.05, min=0.0, required=True),
        ParameterField(name="dt", type="float", default=0.001, min=0.0, required=True),
        ParameterField(name="init_value", type="float", default=0.0, required=False),
        ParameterField(name="input_shape", type="array", default=None, required=False),
        ParameterField(name="input_dtype", type="str", default=None, required=False),
    ],
    icon="Filter",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=first_order_filter_output_prototype,
    migrations=(
        ComponentMigration(
            source_type="FirstOrderFilter",
            target_type="FirstOrderFilter",
            owner="feedbax",
            source_param_schema_version="1",
            target_param_schema_version="2",
            migration_id="feedbax.component.FirstOrderFilter.params.v1-to-v2",
            migrate_params=migrate_stateful_v1,
            description="Preserve stateful declarations with optional inferred input dtype.",
        ),
    ),
)
class FirstOrderFilter(Component):
    """Continuous-time first-order low-pass filter."""

    input_ports = ("input",)
    output_ports = ("output",)

    tau_rise: float = 0.050
    tau_decay: float = 0.050
    dt: float = 0.001
    solver: dfx.AbstractSolver = field(default_factory=lambda: dfx.Euler())
    input_proto: PyTree[Array] = field(default_factory=lambda: jnp.zeros(1))
    init_value: float = 0.0
    state_index: StateIndex

    def __init__(
        self,
        tau_rise: float = 0.050,
        tau_decay: float = 0.050,
        dt: float = 0.001,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        input_proto: Optional[PyTree[Array]] = None,
        init_value: float = 0.0,
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


@component(
    "HighPassFilter",
    constructor=_STATEFUL_VECTOR_CONSTRUCTOR,
    category="Signal Processing",
    description="High-pass filter (input - lowpass).",
    parameter_fields=[
        ParameterField(name="tau", type="float", default=0.1, min=0.0, required=True),
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=False),
        *_STATEFUL_VECTOR_PROTOTYPE_FIELDS,
    ],
    icon="Filter",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=stateful_vector_output_prototype,
)
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

    input_ports = ("input",)
    output_ports = ("output",)

    tau: float = field(static=True)
    dt: float = field(static=True)
    n_dims: int = field(static=True)
    _lowpass: _HeldLowPass

    def __init__(
        self,
        tau: float = 0.1,
        dt: float = 0.01,
        n_dims: int = 1,
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


@component(
    "BandPassFilter",
    constructor=_STATEFUL_VECTOR_CONSTRUCTOR,
    category="Signal Processing",
    description="Band-pass: high-pass then low-pass.",
    parameter_fields=[
        ParameterField(name="tau_low", type="float", default=0.1, min=0.0, required=True),
        ParameterField(name="tau_high", type="float", default=0.01, min=0.0, required=True),
        ParameterField(name="dt", type="float", default=0.01, min=0.0, required=True),
        ParameterField(name="n_dims", type="int", default=1, min=1, required=False),
        *_STATEFUL_VECTOR_PROTOTYPE_FIELDS,
    ],
    icon="Filter",
    port_types=PortTypeSpec(
        inputs={"input": PortType(dtype="vector")},
        outputs={"output": PortType(dtype="vector")},
    ),
    output_prototype_fn=stateful_vector_output_prototype,
)
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

    input_ports = ("input",)
    output_ports = ("output",)

    tau_high: float = field(static=True)
    _highpass: HighPassFilter
    _lowpass: _HeldLowPass

    def __init__(
        self,
        tau_low: float = 0.1,
        tau_high: float = 0.01,
        dt: float = 0.01,
        n_dims: int = 1,
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
