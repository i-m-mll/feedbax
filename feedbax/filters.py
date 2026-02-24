"""General-purpose filters for dynamical systems.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from functools import cached_property
import dataclasses
from typing import Optional, Type

import diffrax as dfx
import equinox as eqx
from equinox import field
from equinox.nn import State, StateIndex
import jax
import jax.tree as jt
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from feedbax.graph import Component
from feedbax.state import StateBounds


class FilterState(eqx.Module):
    """Holds the current filtered signal and solver state."""

    output: jax.Array
    solver: PyTree


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
    _initial_state: FilterState = field(static=True)

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

        self._initial_state = self._initial_state_value(input_proto)
        self.state_index = StateIndex(self._initial_state)

    def _initial_state_value(self, input_proto: PyTree[Array]) -> FilterState:
        output_init = jt.map(lambda x: jnp.full_like(x, self.init_value), input_proto)
        solver_init = self.solver.init(self._term, 0, self.dt, output_init, None)
        return FilterState(output=output_init, solver=solver_init)

    def vector_field(
        self,
        t: Scalar,
        state: FilterState,
        input: PyTree[jax.Array],
    ) -> FilterState:
        tau = jnp.where(input >= state.output, self.tau_rise, self.tau_decay)
        return eqx.tree_at(
            lambda s: s.output,
            state,
            (input - state.output) / tau,
        )

    @cached_property
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

        output_state, _, _, solver_state, _ = self.solver.step(
            self._term,
            0,
            self.dt,
            filter_state,
            input_value,
            filter_state.solver,
            made_jump=False,
        )

        output_state = eqx.tree_at(
            lambda s: s.solver,
            output_state,
            solver_state,
        )
        state = state.set(self.state_index, output_state)
        return {"output": output_state.output}, state

    def change_input(self, input_proto: PyTree[Array]) -> "FirstOrderFilter":
        new_initial_state = self._initial_state_value(input_proto)
        new_state_index = StateIndex(new_initial_state)
        return dataclasses.replace(
            self,
            input_proto=input_proto,
            state_index=new_state_index,
            _initial_state=new_initial_state,
        )

    @property
    def bounds(self) -> StateBounds[FilterState]:
        return StateBounds(
            low=FilterState(output=None, solver=None),
            high=FilterState(output=None, solver=None),
        )


class HighPassFilter(Component):
    """High-pass filter: output = input - lowpass(input).

    Uses a first-order low-pass filter internally. The high-pass response
    is obtained by subtracting the low-pass filtered signal from the input.

    Args:
        tau: Time constant for both rise and decay of the internal low-pass.
        dt: Integration time step.
        n_dims: Dimensionality of the signal.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    tau: float
    dt: float
    _lowpass: FirstOrderFilter

    def __init__(
        self,
        tau: float = 0.1,
        dt: float = 0.01,
        n_dims: int = 1,
    ):
        self.tau = float(tau)
        self.dt = float(dt)
        input_proto = jnp.zeros(n_dims)
        self._lowpass = FirstOrderFilter(
            tau_rise=tau,
            tau_decay=tau,
            dt=dt,
            input_proto=input_proto,
        )

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        lp_out, state = self._lowpass(inputs, state, key=key)
        output = inputs["input"] - lp_out["output"]
        return {"output": output}, state


class BandPassFilter(Component):
    """Band-pass filter: cascade of high-pass and low-pass.

    Passes frequencies between the two cutoffs by first applying a
    high-pass filter (to reject low frequencies) then a low-pass filter
    (to reject high frequencies).

    Args:
        tau_low: Time constant for the high-pass stage (lower frequency bound;
            larger tau means lower cutoff frequency).
        tau_high: Time constant for the low-pass stage (upper frequency bound;
            smaller tau means higher cutoff frequency).
        dt: Integration time step.
        n_dims: Dimensionality of the signal.
    """

    input_ports = ("input",)
    output_ports = ("output",)

    tau_low: float
    tau_high: float
    dt: float
    _highpass: HighPassFilter
    _lowpass: FirstOrderFilter

    def __init__(
        self,
        tau_low: float = 0.1,
        tau_high: float = 0.01,
        dt: float = 0.01,
        n_dims: int = 1,
    ):
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.dt = float(dt)
        input_proto = jnp.zeros(n_dims)
        self._highpass = HighPassFilter(tau=tau_low, dt=dt, n_dims=n_dims)
        self._lowpass = FirstOrderFilter(
            tau_rise=tau_high,
            tau_decay=tau_high,
            dt=dt,
            input_proto=input_proto,
        )

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        hp_out, state = self._highpass(inputs, state, key=key)
        lp_inputs = {"input": hp_out["output"]}
        lp_out, state = self._lowpass(lp_inputs, state, key=key)
        return {"output": lp_out["output"]}, state
