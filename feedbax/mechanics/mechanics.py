"""Discretize and step plant models.

Supports two execution paths:

- **Legacy path** (default): Direct Diffrax solver stepping, backward
  compatible with all existing code.
- **Backend path**: Delegates to a ``PhysicsBackend`` (Diffrax or MJX)
  with configurable sub-stepping and optional gradient checkpointing.

:copyright: Copyright 2023-2025 by MLL <mll@mll.bio>.
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
from feedbax.mechanics.backend import PhysicsBackend, PhysicsState
from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


class MechanicsState(Module):
    """State for a mechanical plant integration step.

    Attributes:
        plant: The current plant state.
        effector: Cartesian state of the end-effector.
        solver: Backend-specific auxiliary state. Diffrax solver state
            for the legacy/DiffraxBackend paths, ``None`` for MJXBackend.
    """

    plant: PlantState
    effector: CartesianState
    solver: PyTree


class Mechanics(Component):
    """Discretizes and steps a plant model.

    When constructed without a ``backend``, uses the legacy Diffrax path
    (single-step Euler by default). When a ``PhysicsBackend`` is provided,
    delegates stepping to it, enabling MJX native integration and
    configurable sub-stepping.

    Attributes:
        plant: The biomechanical plant model.
        dt: Control timestep.
        solver: Diffrax solver (legacy path; ignored when backend is set).
        backend: Optional physics backend for the new stepping path.
        remat_substep: Whether to apply gradient checkpointing to substeps
            (backend path only).
    """

    input_ports = ("force",)
    output_ports = ("effector", "state")

    plant: AbstractPlant
    dt: float
    solver: dfx.AbstractSolver
    backend: Optional[PhysicsBackend]
    remat_substep: bool = field(static=True)
    state_index: StateIndex
    _initial_state: MechanicsState = field(static=True)

    def __init__(
        self,
        plant: AbstractPlant,
        dt: float,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        *,
        backend: Optional[PhysicsBackend] = None,
        remat_substep: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize Mechanics.

        Args:
            plant: The plant model to integrate.
            dt: Control timestep in seconds.
            solver_type: Diffrax solver class (legacy path). Ignored when
                ``backend`` is provided.
            backend: Optional physics backend. When provided, stepping is
                delegated to the backend. ``None`` uses the legacy Diffrax
                path.
            remat_substep: If ``True`` and a backend is provided, apply
                ``eqx.filter_checkpoint`` to each substep to reduce memory
                usage during backpropagation.
            key: PRNG key for initialization. Defaults to ``PRNGKey(0)``.
        """
        self.plant = plant
        self.solver = solver_type()
        self.dt = dt
        self.backend = backend
        self.remat_substep = remat_substep

        if key is None:
            key = jax.random.PRNGKey(0)

        if backend is not None:
            # Backend path: initialize via backend
            physics_state = backend.init_state(plant, key=key)
            self._initial_state = MechanicsState(
                plant=physics_state.plant,
                effector=physics_state.effector,
                solver=physics_state.aux,
            )
        else:
            # Legacy path: initialize via Diffrax
            plant_state = self.plant.init(key=key)
            init_input = jnp.zeros((self.plant.input_size,))
            solver_state = self.solver.init(
                self._term, 0, self.dt, plant_state, init_input
            )
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
        """Step the plant by one control timestep.

        Dispatches to the backend path or legacy path depending on
        whether a ``PhysicsBackend`` was provided at construction.

        Args:
            inputs: Dict with ``"force"`` key containing control input.
            state: Equinox ``State`` container.
            key: PRNG key.

        Returns:
            Tuple of (outputs dict, updated State).
        """
        if self.backend is not None:
            return self._call_backend(inputs, state, key=key)
        return self._call_legacy(inputs, state, key=key)

    def _call_legacy(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Legacy Diffrax stepping path (unchanged from original).

        Args:
            inputs: Dict with ``"force"`` key.
            state: Equinox ``State`` container.
            key: PRNG key.

        Returns:
            Tuple of (outputs dict, updated State).
        """
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
        new_state = MechanicsState(
            plant=plant_state, effector=effector, solver=solver_state
        )
        state = state.set(self.state_index, new_state)
        return {"effector": effector, "state": new_state}, state

    def _call_backend(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Backend-delegated stepping with sub-step scanning.

        Converts the current ``MechanicsState`` to a ``PhysicsState``,
        scans over ``backend.n_substeps`` calling ``backend.substep()``,
        optionally applying gradient checkpointing, then converts back.

        Args:
            inputs: Dict with ``"force"`` key.
            state: Equinox ``State`` container.
            key: PRNG key.

        Returns:
            Tuple of (outputs dict, updated State).
        """
        backend = self.backend
        mechanics_state: MechanicsState = state.get(self.state_index)
        action = inputs["force"]

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

        # Build PhysicsState from MechanicsState
        physics_state = PhysicsState(
            plant=plant_state,
            effector=mechanics_state.effector,
            aux=mechanics_state.solver,
        )

        # Define the substep function
        def do_substep(carry: PhysicsState, _: None) -> tuple[PhysicsState, None]:
            return backend.substep(self.plant, carry, action), None

        # Optionally apply gradient checkpointing
        # Bug: 928d494 — remat_substep reduces memory for long substep chains
        if self.remat_substep:
            do_substep = eqx.filter_checkpoint(do_substep)

        # Scan over substeps
        if backend.n_substeps == 1:
            # Avoid scan overhead for single substep
            physics_state = backend.substep(self.plant, physics_state, action)
        else:
            physics_state, _ = jax.lax.scan(
                do_substep,
                physics_state,
                None,
                length=backend.n_substeps,
            )

        # Extract effector via backend
        effector = backend.observe(self.plant, physics_state)

        # Convert back to MechanicsState
        new_state = MechanicsState(
            plant=physics_state.plant,
            effector=effector,
            solver=physics_state.aux,
        )
        state = state.set(self.state_index, new_state)
        return {"effector": effector, "state": new_state}, state

    def state_view(self, state: State) -> MechanicsState:
        return state.get(self.state_index)
