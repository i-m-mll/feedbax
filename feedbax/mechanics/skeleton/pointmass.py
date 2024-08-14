"""Newtonian point mass dynamics, as a trivial skeleton model.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from feedbax.dynamics import LTISystem
from feedbax.mechanics.skeleton import AbstractSkeleton
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


# global defaults
ORDER = 2  # maximum ORDER of the state derivatives
N_DIM = 2  # number of spatial dimensions


class PointMass(AbstractSkeleton[CartesianState]):
    """A point with mass but no spatial extent, that obeys Newton's laws of motion.

    Attributes:
        mass: The mass of the point mass.
        damping: The factor by which to damp the motion of the point mass, relative to its velocity.
        A: The state evolution matrix according to Newton's first law of motion.
        B: The control matrix according to Newton's second law.
    """

    mass: float
    damping: float = 0

    @cached_property
    def A(self) -> Array:
        A_undamped = jnp.sum(jnp.stack(
            [
                jnp.diagflat(jnp.ones((ORDER - i) * N_DIM), i * N_DIM)
                for i in range(1, ORDER)
            ]
        ), axis=0)

        vel_slice = slice(N_DIM, 2 * N_DIM)
        A_damping = jnp.zeros_like(A_undamped).at[vel_slice, vel_slice].set(
            -(self.damping / self.mass) * jnp.eye(N_DIM)
        )

        return A_undamped + A_damping

    @cached_property
    def B(self) -> Array:
        return jnp.concatenate(
            [jnp.zeros((N_DIM, N_DIM)), jnp.eye(N_DIM) / self.mass],
            axis=0,
        )

    @cached_property
    def C(self) -> Array:
        return jnp.array(1)  # TODO

    @cached_property
    def _lti_system(self) -> LTISystem:
        return LTISystem(self.A, self.B, self.C)


    def vector_field(
        self, t: Scalar, state: CartesianState, input: Float[Array, "input"]
    ) -> CartesianState:
        """Returns time derivatives of the system's states.

        Arguments:
            t: The current simulation time. (Unused.)
            state: The state of the point mass.
            input: The input force on the point mass.
        """
        # `state.force` may contain forces due to
        # `update_state_given_effector_force` executed during the
        # "update_effector_force" stage of `Mechanics`
        force = input + state.force
        state_ = jnp.concatenate([state.pos, state.vel])
        d_y = self._lti_system.vector_field(t, state_, force)

        return CartesianState(pos=d_y[:2], vel=d_y[2:])

    def forward_kinematics(self, state: CartesianState) -> CartesianState:
        """Trivially, returns the Cartesian state of the point mass itself."""
        return state

    def inverse_kinematics(self, effector_state: CartesianState) -> CartesianState:
        """Trivially, returns the Cartesian state of the point mass itself."""
        return effector_state

    def effector(  # type: ignore
        self,
        config_state: CartesianState,
    ) -> CartesianState:
        """Return the effector state given the configuration state.

        !!! Note  ""
            For a point mass, these are identical. However, we make sure to
            return zero `force` to avoid positive feedback loops as effector
            forces are converted back to system forces by `Mechanics` on the
            next time step.
        """
        return CartesianState(
            pos=config_state.pos,
            vel=config_state.vel,
        )

    def update_state_given_effector_force(  # type: ignore
        self,
        effector_force: Array,
        system_state: CartesianState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> CartesianState:
        """Updates the force on the point mass.

        !!! Note ""
            Effector forces are equal to configuration forces for a point mass,
            so this just inserts the effector force into the state directly.
            This method exists because for more complex skeletons, the
            conversion is not trivial.
        """
        return eqx.tree_at(
            lambda state: state.force,
            system_state,
            effector_force,
        )

    def init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> CartesianState:
        """Return a default state for the point mass."""
        return CartesianState()

    @property
    def input_size(self) -> int:
        """Number of control variables."""
        return self.B.shape[1]