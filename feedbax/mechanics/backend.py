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

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import diffrax as dfx
import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax._ode import ode_init, ode_step
from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.runtime.state import CartesianState


# Allow independently rounded float32 control and substep periods to agree.
_TIMING_RTOL = 2e-7


def _positive_period(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def _substep_count(
    control_dt: float, sub_dt: float, n_substeps: int | None = None
) -> int:
    control_dt = _positive_period("control_dt", control_dt)
    sub_dt = _positive_period("sub_dt", sub_dt)
    ratio = control_dt / sub_dt
    if math.isfinite(ratio):
        count = round(ratio)
        tolerance = min(_TIMING_RTOL * ratio, 0.125)
        if count > 0 and abs(ratio - count) <= tolerance:
            if n_substeps is None:
                return count
            if (
                isinstance(n_substeps, Integral)
                and not isinstance(n_substeps, bool)
                and n_substeps == count
            ):
                return count
            raise ValueError(
                f"n_substeps must be the integer {count} for control_dt={control_dt!r} "
                f"and sub_dt={sub_dt!r}, got {n_substeps!r}"
            )
    raise ValueError(
        f"control_dt={control_dt!r} must be a positive integer multiple "
        f"of sub_dt={sub_dt!r}"
    )


def _mechanics_control_dt(dt: float, backend: PhysicsBackend | None) -> float:
    dt = _positive_period("Mechanics.dt", dt)
    if backend is None:
        return dt
    control_dt = _positive_period("backend.control_dt", backend.control_dt)
    if not math.isclose(dt, control_dt, rel_tol=_TIMING_RTOL):
        raise ValueError(
            f"Mechanics.dt={dt!r} must agree with backend.control_dt={control_dt!r}"
        )
    return control_dt


def step_plant_ode(
    plant: AbstractPlant,
    solver: dfx.AbstractSolver,
    dt: float,
    state: PlantState,
    action: PyTree[Array],
    solver_state: PyTree,
    *,
    key: PRNGKeyArray | None = None,
    storage_like: PlantState | None = None,
) -> tuple[PlantState, PyTree]:
    """Prepare actuation once, integrate, and project the accepted plant state."""
    prepared = plant.kinematics_update(action, state, key=key)
    next_state, solver_state = ode_step(
        solver,
        dfx.ODETerm(plant.vector_field),
        dt,
        prepared,
        action,
        solver_state,
        storage_like=state if storage_like is None else storage_like,
    )
    return plant.project_state(next_state), solver_state

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


@dataclass(frozen=True)
class PhysicsCapabilities:
    """Execution and differentiation capabilities of a compiled model."""

    native_step: bool
    continuous: bool
    reverse_mode: bool


@runtime_checkable
class PhysicsBackend(Protocol):
    """Protocol for physics stepping backends.

    Each backend defines how a single substep of physics integration is
    performed. The control period must equal the substep period times
    the positive integer number of substeps, within numerical tolerance.

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
        self.control_dt = _positive_period("control_dt", control_dt)
        n_substeps = _substep_count(control_dt, control_dt if sub_dt is None else sub_dt)
        self.sub_dt = self.control_dt / n_substeps
        self.solver = solver if solver is not None else dfx.Euler()

    @property
    def n_substeps(self) -> int:
        """Number of substeps per control step."""
        return _substep_count(self.control_dt, self.sub_dt)

    @property
    def _integration_dt(self) -> float:
        return self.control_dt / self.n_substeps

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
        solver_state = ode_init(
            self.solver, self._term(plant), self._integration_dt, plant_state, init_input
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

        Prepares kinematics and actuation, then integrates and projects the state.

        Args:
            plant: The plant model.
            state: Current physics state.
            action: Control input (forces/activations).

        Returns:
            Updated PhysicsState.
        """
        plant_state, solver_state = step_plant_ode(
            plant, self.solver, self._integration_dt, state.plant, action, state.aux,
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
        sub_dt: Time per MuJoCo timestep (must agree with ``model.opt.timestep``).
        n_substeps: Number of ``mjx.step()`` calls per control step.
    """

    control_dt: float
    sub_dt: float

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
            n_substeps: Number of substeps per control step. If given, it
                must agree with the subdivision implied by the two periods.
        """
        count = _substep_count(control_dt, sub_dt, n_substeps)
        self.control_dt = float(control_dt)
        self.sub_dt = self.control_dt / count

    def __setstate__(self, state: dict) -> None:
        # Older checkpoints stored a separate count that could override the schedule.
        _substep_count(state["control_dt"], state["sub_dt"], state.pop("n_substeps", None))
        self.__dict__.update(state)

    @property
    def n_substeps(self) -> int:
        """Number of MuJoCo steps per control step."""
        return _substep_count(self.control_dt, self.sub_dt)

    def _check_model_timestep(self, plant: AbstractPlant, value: PyTree) -> PyTree:
        model_dt = plant.skeleton.mjx_model.opt.timestep
        sub_dt = self.control_dt / self.n_substeps
        message = f"MJX model.opt.timestep must agree with backend.sub_dt={sub_dt!r}"
        if isinstance(model_dt, jax.core.Tracer):
            mismatch = (model_dt <= 0) | ~jnp.isclose(
                model_dt, sub_dt, rtol=_TIMING_RTOL, atol=0.0,
            )
            return eqx.error_if(value, mismatch, message, on_error="raise")
        if not math.isclose(float(model_dt), sub_dt, rel_tol=_TIMING_RTOL):
            raise ValueError(f"{message}, got {float(model_dt)!r}")
        return value

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
        key = self._check_model_timestep(plant, key)
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

        skeleton_state = self._check_model_timestep(plant, state.plant.skeleton)
        new_skeleton = plant.skeleton.step(skeleton_state, ctrl)
        new_plant = eqx.tree_at(
            lambda s: s.skeleton, state.plant, new_skeleton
        )
        new_plant = plant.project_state(new_plant)
        effector = plant.skeleton.effector(new_plant.skeleton)
        return PhysicsState(
            plant=new_plant, effector=effector, aux=None
        )
