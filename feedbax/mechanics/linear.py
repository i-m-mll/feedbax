"""Modules defining continuous linear dynamical systems.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Any, Optional

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from feedbax.state import CartesianState2D, StateBounds


logger = logging.getLogger(__name__)


# global defaults
ORDER = 2  # maximum order of the state derivatives
N_DIM = 2  # number of spatial dimensions


class AbstractLTISystem(eqx.Module):
    """Linear time-invariant system.
    
    Note that the system is defined in continuous time.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
    
    TODO:
    - Don't hardcode the state split.
    """
    A: AbstractVar[Float[Array, "state state"]]  # state evolution matrix
    B: AbstractVar[Float[Array, "state input"]]  # control matrix
    C: AbstractVar[Float[Array, "obs state"]]  # observation matrix
    
    @jax.named_scope("fbx.AbstractLTISystem.vector_field")
    def vector_field(
        self, 
        t: float, 
        state: CartesianState2D,
        args: Float[Array, "input"]
    ) -> Float[Array, "state"]:
        input = args  
        state_ = jnp.concatenate([state.pos, state.vel])
        force = input + state.force
        d_y = self.A @ state_ + self.B @ force
        d_pos, d_vel = d_y[:2], d_y[2:]
        return CartesianState2D(d_pos, d_vel)
    
    @property
    def control_size(self) -> int:
        return self.B.shape[1]
    
    @property 
    def state_size(self) -> int:
        return self.A.shape[1]
    
    @property
    def bounds(self) -> StateBounds[CartesianState2D]:
        return StateBounds(low=None, high=None)


class SimpleLTISystem(AbstractLTISystem):
    """An LTI system where the effector is assumed identical to the system.
    
    This generally makes sense for linear systems with only one moving part, 
    such as a point mass.
    
    NOTE: I'm not actually sure if there are other systems that this applies to.
    I guess there may be other linear but non-Newtonian point systems we might
    want to use, but I'm not sure why.
    """
    A: Float[Array, "state state"]  # state evolution matrix
    B: Float[Array, "state input"]  # control matrix
    C: Float[Array, "obs state"]  # observation matrix    
    
    # def forward_kinematics(
    #     self, 
    #     state: Float[Array, "state"]
    # ) -> Float[Array, "state"]:
    #     return state 
    
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
        key: Optional[jax.Array] = None,
    ) -> CartesianState2D:
        return eqx.tree_at(
            lambda state: state.force,
            system_state,
            effector_force,
        )
    
    def init(
        self,
    ) -> CartesianState2D:
        return CartesianState2D()
    

def point_mass(
    mass: float = 1., 
    order: int = ORDER, 
    n_dim: int = N_DIM
) -> SimpleLTISystem:
    A = sum([jnp.diagflat(jnp.ones((order - i) * n_dim), 
                          i * n_dim)
             for i in range(1, order)])
    B = jnp.concatenate(
        [jnp.zeros((n_dim, n_dim)), 
         jnp.eye(n_dim) / mass], 
        axis=0
    )
    C = jnp.array(1)  # TODO 
    return SimpleLTISystem(A, B, C)