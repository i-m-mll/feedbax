"""Modules defining continuous linear dynamical systems.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging 
from typing import Any, Optional, TypeVar

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from feedbax.dynamics import AbstractLTISystem
from feedbax.mechanics.skeleton import AbstractSkeleton
from feedbax.state import AbstractState, CartesianState2D, StateBounds


logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractState)


# global defaults
ORDER = 2  # maximum ORDER of the state derivatives
N_DIM = 2  # number of spatial dimensions


class PointMass(AbstractLTISystem, AbstractSkeleton[CartesianState2D]):
    """An LTI system where the effector is assumed identical to the system.
    
    This generally makes sense for linear systems with only one moving part, 
    such as a point mass.
    
    
    """
    # A: Float[Array, "state state"]  # state evolution matrix
    # B: Float[Array, "state input"]  # control matrix
    # C: Float[Array, "obs state"]  # observation matrix    
    mass: float 

    def __init__(self, mass):
        self.mass = mass 
    
    @cached_property
    def A(self):
        return sum([jnp.diagflat(jnp.ones((ORDER - i) * N_DIM), 
                                 i * N_DIM)
                    for i in range(1, ORDER)])
    
    @cached_property
    def B(self): 
        return jnp.concatenate(
            [
                jnp.zeros((N_DIM, N_DIM)), 
                jnp.eye(N_DIM) / self.mass
            ],
            axis=0,
        )
    
    @cached_property
    def C(self):
        return 1  # TODO 
    
    def forward_kinematics(
        self, 
        state: Float[Array, "state"]
    ) -> Float[Array, "state"]:
        return state 
    
    def inverse_kinematics(
        self, 
        effector_state: CartesianState2D
    ) -> CartesianState2D:
        return effector_state
    
    def effector(
        self, 
        system_state: CartesianState2D,
    ) -> CartesianState2D:
        """Return the effector state given the system state.
        
        For a point mass, these are identical. However, we make sure to return
        zero `force` to avoid positive feedback loops as effector forces are
        converted back to system forces by `Mechanics` on the next time step.
        """
        return CartesianState2D(
            pos=system_state.pos,
            vel=system_state.vel,
            # force=jnp.zeros_like(system_state.force),
        )

    def update_state_given_effector_force(
        self, 
        effector_force: jax.Array,
        system_state: CartesianState2D,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> CartesianState2D:
        return eqx.tree_at(
            lambda state: state.force,
            system_state,
            effector_force,
        )