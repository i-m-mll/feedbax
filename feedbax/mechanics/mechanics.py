"""Discretize and step plant models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import cached_property
import logging
from typing import Optional, Type

import diffrax as dfx  # type: ignore
import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.graph import Component
from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


class MechanicsState(Module):
    """State for a mechanical plant integration step."""

    plant: PlantState
    effector: CartesianState
    solver: PyTree


class Mechanics(Component):
    """Discretizes and steps a plant with Diffrax."""

    input_ports = ("force",)
    output_ports = ("effector",)

    plant: AbstractPlant
    dt: float
    solver: dfx.AbstractSolver
    state_index: StateIndex
    _initial_state: MechanicsState = field(static=True)

    def __init__(
        self,
        plant: AbstractPlant,
        dt: float,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        self.plant = plant
        self.solver = solver_type()
        self.dt = dt

        if key is None:
            key = jax.random.PRNGKey(0)

        plant_state = self.plant.init(key=key)
        init_input = jnp.zeros((self.plant.input_size,))
        solver_state = self.solver.init(self._term, 0, self.dt, plant_state, init_input)
        effector = self.plant.skeleton.effector(plant_state.skeleton)
        self._initial_state = MechanicsState(
            plant=plant_state, effector=effector, solver=solver_state
        )
        self.state_index = StateIndex(self._initial_state)

    @cached_property
    def _term(self) -> dfx.AbstractTerm:
        return dfx.ODETerm(self.plant.vector_field)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        mechanics_state: MechanicsState = state.get(self.state_index)
        force = inputs["force"]

        # Convert effector force back into configuration forces, if applicable.
        skeleton_state = self.plant.skeleton.update_state_given_effector_force(
            mechanics_state.effector.force,
            mechanics_state.plant.skeleton,
            key=key,
        )
        plant_state = eqx.tree_at(
            lambda s: s.skeleton,
            mechanics_state.plant,
            skeleton_state,
        )

        # Kinematics update (non-ODE ops).
        plant_state = self.plant.kinematics_update(force, plant_state, key=key)

        plant_state, _, _, solver_state, _ = self.solver.step(
            self._term,
            0,
            self.dt,
            plant_state,
            force,
            mechanics_state.solver,
            made_jump=False,
        )

        effector = self.plant.skeleton.effector(plant_state.skeleton)
        new_state = MechanicsState(plant=plant_state, effector=effector, solver=solver_state)
        state = state.set(self.state_index, new_state)
        return {"effector": effector}, state

    def state_view(self, state: State) -> MechanicsState:
        return state.get(self.state_index)
