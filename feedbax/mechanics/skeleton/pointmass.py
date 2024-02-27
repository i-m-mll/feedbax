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
from jaxtyping import Array, Float, PRNGKeyArray

from feedbax.dynamics import AbstractLTISystem
from feedbax.mechanics.skeleton import AbstractSkeleton
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


# global defaults
ORDER = 2  # maximum ORDER of the state derivatives
N_DIM = 2  # number of spatial dimensions


class PointMass(AbstractLTISystem, AbstractSkeleton[CartesianState]):
    """A point with mass but no spatial extent, that obeys Newton's laws of motion.

    Attributes:
        A: The state evolution matrix according to Newton's first law of motion.
        B: The control matrix according to Newton's second law.
    """

    mass: float

    def __init__(self, mass):
        self.mass = mass

    @cached_property
    def A(self) -> Array:
        return sum(
            [
                jnp.diagflat(jnp.ones((ORDER - i) * N_DIM), i * N_DIM)
                for i in range(1, ORDER)
            ]
        )

    @cached_property
    def B(self) -> Array:
        return jnp.concatenate(
            [jnp.zeros((N_DIM, N_DIM)), jnp.eye(N_DIM) / self.mass],
            axis=0,
        )

    @cached_property
    def C(self) -> Array:
        return 1  # TODO

    def vector_field(
        self, t: float, state: CartesianState, input: Float[Array, "input"]
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
        d_y = super().vector_field(t, state_, force)

        return CartesianState(pos=d_y[:2], vel=d_y[2:])

    def forward_kinematics(self, state: Float[Array, "state"]) -> Float[Array, "state"]:
        """Trivially, returns the Cartesian state of the point mass itself."""
        return state

    def inverse_kinematics(self, effector_state: CartesianState) -> CartesianState:
        """Trivially, returns the Cartesian state of the point mass itself."""
        return effector_state

    def effector(
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

    def update_state_given_effector_force(
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
        key: Optional[PRNGKeyArray] = None,
    ) -> CartesianState:
        """Return a default state for the point mass."""
        return CartesianState()
