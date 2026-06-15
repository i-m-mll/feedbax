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

from feedbax.mechanics.dynamics import LinearSystem
from feedbax.runtime.graph import Component
from feedbax.mechanics.backend import PhysicsBackend, PhysicsState
from feedbax.mechanics.plant import AbstractPlant, PlantState
from feedbax.runtime.state import CartesianState


logger = logging.getLogger(__name__)


def _warm_up_cached_properties(obj: object) -> None:
    """Force evaluation of all ``cached_property`` attributes on an object tree.

    Some modules (e.g. ``PointMass``) use ``functools.cached_property`` to
    lazily compute matrices. When these modules are captured in closures
    traced by JAX (``lax.scan``, ``lax.fori_loop``, ``eqx.filter_checkpoint``),
    the first access inside the traced context creates a mutable side effect
    (setting the cached value on the object), causing an
    ``UnexpectedTracerError``. Calling this function before entering the
    traced context materializes all cached values, avoiding the issue.
    """
    from functools import cached_property as _cached_property

    cls = type(obj)
    for name in dir(cls):
        if isinstance(getattr(cls, name, None), _cached_property):
            try:
                getattr(obj, name)
            except Exception:
                pass

    # Recurse into Equinox module fields
    if hasattr(obj, '__dataclass_fields__'):
        for field_name in obj.__dataclass_fields__:
            try:
                child = getattr(obj, field_name)
            except Exception:
                continue
            if isinstance(child, Module):
                _warm_up_cached_properties(child)


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

        # Warm up cached_property attributes on the plant before entering
        # JAX-traced code (scan/fori_loop/checkpoint). Modules like PointMass
        # use functools.cached_property for matrices (A, B, etc.), which
        # creates side effects when first accessed inside a traced context.
        # Accessing them here (outside the transform) materializes the cache
        # so traced code finds them already populated.
        # Bug: 928d494 — prevents UnexpectedTracerError from cached_property
        _warm_up_cached_properties(self.plant)

        plant = self.plant

        def do_substep(carry: PhysicsState) -> PhysicsState:
            return backend.substep(plant, carry, action)

        # Optionally apply gradient checkpointing
        # Bug: 928d494 — remat_substep reduces memory for long substep chains
        if self.remat_substep:
            do_substep = eqx.filter_checkpoint(do_substep)

        # Run substeps via fori_loop (compatible with cached_property modules)
        if backend.n_substeps == 1:
            physics_state = do_substep(physics_state)
        else:
            def _fori_body(_, carry):
                return do_substep(carry)

            physics_state = jax.lax.fori_loop(
                0, backend.n_substeps, _fori_body, physics_state,
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

    def initial_outputs(self, state_value: MechanicsState | None) -> dict[str, PyTree]:
        """Return outputs inferred from current mechanics state.

        Overrides the base ``Component.initial_outputs`` to handle the
        ``"state"`` output port, which is the full ``MechanicsState`` and
        therefore cannot be derived from a state attribute lookup by name.

        Args:
            state_value: Current ``MechanicsState``, or ``None``.

        Returns:
            Dict with ``"effector"`` and ``"state"`` entries if
            ``state_value`` is not ``None``, otherwise an empty dict.
        """
        if state_value is None:
            return {}
        return {
            "effector": state_value.effector,
            "state": state_value,
        }

    def state_view(self, state: State) -> MechanicsState:
        return state.get(self.state_index)

    def linearize_with_force_filter(
        self,
        skeleton_state: PyTree | None = None,
        *,
        tau: Optional[float] = None,
    ) -> LinearSystem:
        """Continuous-time linearisation augmented with a force-filter LPF.

        Composes the bare-skeleton linearisation
        (``self.plant.skeleton.linearize(skeleton_state)``) with a first-order
        force-filter LPF row block:

        - Without filter (``tau=None``): returns the skeleton's linearisation
          unchanged, with ``dt=self.dt``.
        - With filter (``tau > 0``): augments the state vector to
          ``[skeleton_state, F]``. The skeleton's ``B`` becomes the
          ``F`` -> ``state`` block, the new ``B`` is the control input
          ``u`` -> ``F`` (rate ``1/tau``), and ``B_w`` (the disturbance
          channel) is taken from the skeleton's ``B_w`` (i.e. the
          disturbance bypasses the filter, entering as additive force on
          the velocity row, matching ``FixedField`` / ``CurlField``).

        This mirrors the construction in
        ``rlrmp.analysis.hinf_riccati.linearize_pointmass``.

        Args:
            skeleton_state: Nominal skeleton state at which to linearise.
                Ignored by exactly-linear skeletons.
            tau: Force-filter time constant. ``None`` (or ``0.0``) skips
                augmentation.

        Returns:
            ``LinearSystem`` with the augmented matrices and ``dt=self.dt``.
        """
        sys = self.plant.skeleton.linearize(skeleton_state)
        A_s, B_s = sys.A, sys.B
        Bw_s = sys.B_w  # may be None
        n_s = A_s.shape[0]
        m_u = B_s.shape[1]

        if tau is None or tau == 0.0:
            return LinearSystem(
                A=A_s,
                B=B_s,
                B_w=Bw_s,
                state_indices=dict(sys.state_indices),
                dt=float(self.dt),
            )

        # Augment state with F (force-filter output): x_aug = [x_s, F]
        I_u = jnp.eye(m_u, dtype=A_s.dtype)
        Z_u = jnp.zeros((m_u, m_u), dtype=A_s.dtype)
        Z_su = jnp.zeros((n_s, m_u), dtype=A_s.dtype)
        Z_us = jnp.zeros((m_u, n_s), dtype=A_s.dtype)

        # A_aug = [[A_s, B_s], [0, -I/tau]]
        A_aug = jnp.block([[A_s, B_s], [Z_us, -(1.0 / tau) * I_u]])
        # B_aug = [[0_su], [I/tau]]  (control u → force-filter input)
        B_aug = jnp.concatenate([Z_su, (1.0 / tau) * I_u], axis=0)
        # B_w_aug: disturbance bypasses the filter (additive force on
        # the velocity row, identical to skeleton's B_w with zero force-row).
        if Bw_s is not None:
            m_w = Bw_s.shape[1]
            Z_uw = jnp.zeros((m_u, m_w), dtype=A_s.dtype)
            Bw_aug = jnp.concatenate([Bw_s, Z_uw], axis=0)
        else:
            Bw_aug = None

        new_indices = dict(sys.state_indices)
        new_indices["force"] = (n_s, n_s + m_u)

        return LinearSystem(
            A=A_aug,
            B=B_aug,
            B_w=Bw_aug,
            state_indices=new_indices,
            dt=float(self.dt),
        )
