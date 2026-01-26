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
