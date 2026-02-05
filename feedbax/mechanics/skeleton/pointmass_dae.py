"""Point mass dynamics using DAE/implicit integration.

This module provides a DAE-based point mass for testing and validating
the implicit solver infrastructure before moving to more complex systems.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from functools import cached_property
import logging
from typing import Optional, Type

import diffrax as dfx
import equinox as eqx
from equinox import Module, field
from equinox.nn import State
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.mechanics.dae import DAEComponent
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


# Spatial dimensions
N_DIM = 2


class PointMassDAEParams(Module):
    """Parameters for point mass dynamics.

    Attributes:
        mass: Mass of the point particle. Units: [M].
        damping: Viscous damping coefficient. Units: [M/T].
    """

    mass: float = 1.0
    damping: float = 0.0


class PointMassDAE(DAEComponent[CartesianState]):
    """Point mass dynamics with implicit integration.

    Uses DAE infrastructure to integrate Newtonian point mass dynamics.
    While a point mass doesn't require implicit integration, this serves
    as a validation of the DAE solver setup and a template for more
    complex systems.

    The dynamics follow Newton's second law:
        m * a = F - b * v

    where m is mass, a is acceleration, F is applied force,
    b is damping coefficient, and v is velocity.

    Attributes:
        params: Physical parameters (mass, damping).
        dt: Integration timestep.
    """

    input_ports = ("force",)
    output_ports = ("effector", "state")

    params: PointMassDAEParams

    def __init__(
        self,
        mass: float = 1.0,
        damping: float = 0.0,
        dt: float = 0.01,
        solver_type: Type[dfx.AbstractImplicitSolver] = dfx.ImplicitEuler,
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize point mass DAE component.

        Args:
            mass: Mass of the point particle.
            damping: Viscous damping coefficient.
            dt: Integration timestep.
            solver_type: Implicit solver type (default: ImplicitEuler).
            root_finder: Root finder for implicit steps.
            key: PRNG key for initialization.
        """
        self.params = PointMassDAEParams(mass=mass, damping=damping)
        super().__init__(
            dt=dt,
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    @jax.named_scope("fbx.PointMassDAE.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: CartesianState,
        input: Float[Array, "ndim=2"],
    ) -> CartesianState:
        """Compute time derivatives of position and velocity.

        Args:
            t: Current time (unused, dynamics are time-invariant).
            state: Current Cartesian state (pos, vel, force).
            input: Applied force vector.

        Returns:
            Time derivatives: d_pos = vel, d_vel = (F - b*v) / m.
        """
        # Total force includes external force from state
        total_force = input + state.force

        # Damping force
        damping_force = -self.params.damping * state.vel

        # Acceleration
        acceleration = (total_force + damping_force) / self.params.mass

        return CartesianState(
            pos=state.vel,
            vel=acceleration,
            force=jnp.zeros_like(state.force),  # Force derivative is zero
        )

    def init_system_state(self, *, key: PRNGKeyArray) -> CartesianState:
        """Initialize the point mass state at rest at origin."""
        return CartesianState(
            pos=jnp.zeros(N_DIM),
            vel=jnp.zeros(N_DIM),
            force=jnp.zeros(N_DIM),
        )

    def extract_outputs(self, state: CartesianState) -> dict[str, PyTree]:
        """Extract effector state (same as system state for point mass)."""
        # For a point mass, effector = system state
        effector = CartesianState(
            pos=state.pos,
            vel=state.vel,
            force=jnp.zeros_like(state.force),  # Reset force for next step
        )
        return {"effector": effector}

    def _get_zero_input(self) -> Array:
        """Zero force input."""
        return jnp.zeros(N_DIM)

    @property
    def input_size(self) -> int:
        """Number of input dimensions (2D force vector)."""
        return N_DIM

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute integration step with force input.

        Overrides to use 'force' input port instead of generic 'input'.
        """
        # Get force from inputs
        force = inputs.get("force", self._get_zero_input())

        # Call parent with force as 'input'
        modified_inputs = {"input": force}
        return super().__call__(modified_inputs, state, key=key)

    def compute_kinetic_energy(self, state: CartesianState) -> Array:
        """Compute kinetic energy: KE = 0.5 * m * v^2."""
        return 0.5 * self.params.mass * jnp.sum(state.vel**2)

    def compute_momentum(self, state: CartesianState) -> Array:
        """Compute linear momentum: p = m * v."""
        return self.params.mass * state.vel
