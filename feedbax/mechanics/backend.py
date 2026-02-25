"""Physics backend abstraction for feedbax Mechanics.

Provides a unified interface for physics stepping, allowing Mechanics to
delegate time integration to either:

- ``DiffraxBackend``: Wraps existing analytical plant + Diffrax solver
  (preserves legacy behavior with n_substeps=1).
- ``MJXBackend``: Calls ``mjx.step()`` for native MuJoCo discrete stepping
  with configurable sub-stepping (frame_skip).

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import cached_property
import logging
from typing import Optional, Protocol, runtime_checkable

import diffrax as dfx
import equinox as eqx
from equinox import Module, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


class PhysicsState(Module):
    """Unified physics state returned by all backends.

    Attributes:
        plant: The current plant state (skeleton + optional muscles).
        effector: Cartesian state of the end-effector.
        aux: Backend-specific auxiliary data. Solver state for Diffrax,
            ``None`` for MJX.
    """

    plant: PlantState
    effector: CartesianState
    aux: PyTree


@runtime_checkable
class PhysicsBackend(Protocol):
    """Protocol for physics stepping backends.

    Each backend defines how a single substep of physics integration is
    performed, along with timing parameters that determine how many
    substeps comprise one control step.

    Attributes:
        control_dt: Time per control step (seconds).
        sub_dt: Time per physics substep (seconds).
        n_substeps: Number of physics substeps per control step.
    """

    control_dt: float
    sub_dt: float
    n_substeps: int

    def init_state(
        self, plant: AbstractPlant, *, key: PRNGKeyArray
    ) -> PhysicsState:
        """Initialize a PhysicsState from a plant.

        Args:
            plant: The plant model.
            key: PRNG key for random initialization.

        Returns:
            Initial PhysicsState.
        """
        ...

    def observe(
        self, plant: AbstractPlant, state: PhysicsState
    ) -> CartesianState:
        """Extract end-effector Cartesian state.

        Args:
            plant: The plant model (for kinematics).
            state: Current physics state.

        Returns:
            Effector CartesianState.
        """
        ...

    def substep(
        self, plant: AbstractPlant, state: PhysicsState, action: Array
    ) -> PhysicsState:
        """Perform one physics substep.

        Args:
            plant: The plant model.
            state: Current physics state.
            action: Control input for this substep.

        Returns:
            Updated PhysicsState after one substep.
        """
        ...


class DiffraxBackend(Module):
    """Physics backend wrapping existing Diffrax ODE integration.

    With ``n_substeps=1`` and ``sub_dt=control_dt``, this exactly
    reproduces the legacy ``Mechanics.__call__`` behavior.

    Attributes:
        control_dt: Time per control step.
        sub_dt: Time per physics substep.
        solver: Diffrax solver instance (default: Euler).
    """

    control_dt: float
    sub_dt: float
    solver: dfx.AbstractSolver

    def __init__(
        self,
        control_dt: float,
        sub_dt: float | None = None,
        solver: dfx.AbstractSolver | None = None,
    ):
        """Initialize DiffraxBackend.

        Args:
            control_dt: Time per control step.
            sub_dt: Time per physics substep. Defaults to ``control_dt``
                (single substep, matching legacy behavior).
            solver: Diffrax solver. Defaults to ``dfx.Euler()``.
        """
        self.control_dt = control_dt
        self.sub_dt = sub_dt if sub_dt is not None else control_dt
        self.solver = solver if solver is not None else dfx.Euler()

    @property
    def n_substeps(self) -> int:
        """Number of substeps per control step."""
        return max(1, int(round(self.control_dt / self.sub_dt)))

    def _term(self, plant: AbstractPlant) -> dfx.AbstractTerm:
        """Build the ODE term from a plant's vector field."""
        return dfx.ODETerm(plant.vector_field)

    @jax.named_scope("fbx.DiffraxBackend.init_state")
    def init_state(
        self, plant: AbstractPlant, *, key: PRNGKeyArray
    ) -> PhysicsState:
        """Initialize PhysicsState with Diffrax solver state.

        Args:
            plant: The plant model.
            key: PRNG key.

        Returns:
            Initial PhysicsState with solver state in ``aux``.
        """
        plant_state = plant.init(key=key)
        init_input = jnp.zeros((plant.input_size,))
        solver_state = self.solver.init(
            self._term(plant), 0, self.sub_dt, plant_state, init_input
        )
        effector = plant.skeleton.effector(plant_state.skeleton)
        return PhysicsState(
            plant=plant_state, effector=effector, aux=solver_state
        )

    @jax.named_scope("fbx.DiffraxBackend.observe")
    def observe(
        self, plant: AbstractPlant, state: PhysicsState
    ) -> CartesianState:
        """Extract effector state via plant skeleton.

        Args:
            plant: The plant model.
            state: Current physics state.

        Returns:
            Effector CartesianState.
        """
        return plant.skeleton.effector(state.plant.skeleton)

    @jax.named_scope("fbx.DiffraxBackend.substep")
    def substep(
        self, plant: AbstractPlant, state: PhysicsState, action: Array
    ) -> PhysicsState:
        """One Diffrax solver substep.

        Applies kinematics update (clipping, muscle geometry) then
        steps the ODE solver, matching the legacy Mechanics path.

        Args:
            plant: The plant model.
            state: Current physics state.
            action: Control input (forces/activations).

        Returns:
            Updated PhysicsState.
        """
        plant_state = plant.kinematics_update(action, state.plant)
        plant_state, _, _, solver_state, _ = self.solver.step(
            self._term(plant),
            0,
            self.sub_dt,
            plant_state,
            action,
            state.aux,
            made_jump=False,
        )
        effector = plant.skeleton.effector(plant_state.skeleton)
        return PhysicsState(
            plant=plant_state, effector=effector, aux=solver_state
        )


class MJXBackend(Module):
    """Physics backend using native MJX discrete stepping.

    Each substep calls ``MJXSkeleton.step()`` (which invokes
    ``mjx.step()``), bypassing Diffrax entirely. The ``n_substeps``
    parameter implements frame_skip: the action is held constant
    across substeps while the physics integrator advances at ``sub_dt``.

    Attributes:
        control_dt: Time per control step.
        sub_dt: Time per MuJoCo timestep (should match ``model.opt.timestep``).
        n_substeps: Number of ``mjx.step()`` calls per control step.
    """

    control_dt: float
    sub_dt: float
    n_substeps: int = field(static=True)

    def __init__(
        self,
        control_dt: float,
        sub_dt: float,
        n_substeps: int | None = None,
    ):
        """Initialize MJXBackend.

        Args:
            control_dt: Time per control step.
            sub_dt: Time per MuJoCo timestep.
            n_substeps: Number of substeps per control step. If ``None``,
                computed as ``round(control_dt / sub_dt)``.
        """
        self.control_dt = control_dt
        self.sub_dt = sub_dt
        if n_substeps is not None:
            self.n_substeps = n_substeps
        else:
            self.n_substeps = max(1, int(round(control_dt / sub_dt)))

    @jax.named_scope("fbx.MJXBackend.init_state")
    def init_state(
        self, plant: AbstractPlant, *, key: PRNGKeyArray
    ) -> PhysicsState:
        """Initialize PhysicsState for MJX (no solver aux state).

        Args:
            plant: The plant model (must have an MJXSkeleton).
            key: PRNG key.

        Returns:
            Initial PhysicsState with ``aux=None``.
        """
        plant_state = plant.init(key=key)
        effector = plant.skeleton.effector(plant_state.skeleton)
        return PhysicsState(
            plant=plant_state, effector=effector, aux=None
        )

    @jax.named_scope("fbx.MJXBackend.observe")
    def observe(
        self, plant: AbstractPlant, state: PhysicsState
    ) -> CartesianState:
        """Extract effector state via plant skeleton.

        Args:
            plant: The plant model.
            state: Current physics state.

        Returns:
            Effector CartesianState.
        """
        return plant.skeleton.effector(state.plant.skeleton)

    @jax.named_scope("fbx.MJXBackend.substep")
    def substep(
        self, plant: AbstractPlant, state: PhysicsState, action: Array
    ) -> PhysicsState:
        """One discrete MJX substep.

        Converts muscle activations to joint torques via the plant's
        ``_muscle_activations_to_joint_torques`` method, then calls
        ``MJXSkeleton.step()`` for native MuJoCo integration.

        Args:
            plant: The plant model (must be an ``MJXPlant``).
            state: Current physics state.
            action: Muscle activations or control input.

        Returns:
            Updated PhysicsState.
        """
        # Convert muscle activations to joint torques if the plant
        # provides such a mapping (MJXPlant does).
        if hasattr(plant, '_muscle_activations_to_joint_torques'):
            ctrl = plant._muscle_activations_to_joint_torques(action)
        else:
            ctrl = action

        new_skeleton = plant.skeleton.step(state.plant.skeleton, ctrl)
        new_plant = eqx.tree_at(
            lambda s: s.skeleton, state.plant, new_skeleton
        )
        effector = plant.skeleton.effector(new_skeleton)
        return PhysicsState(
            plant=new_plant, effector=effector, aux=None
        )
