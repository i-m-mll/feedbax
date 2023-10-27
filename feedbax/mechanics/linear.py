"""

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

from feedbax.types import CartesianState2D


logger = logging.getLogger(__name__)


# global defaults
ORDER = 2  # maximum order of the state derivatives
N_DIM = 2  # number of spatial dimensions


class AbstractLTISystem(eqx.Module):
    """Linear time-invariant system.
    
    Note that the system is defined in continuous time.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
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
        d_y = self.A @ state_ + self.B @ input
        # TODO: don't hardcode the split; define on instantiation
        d_pos, d_vel = d_y[:2], d_y[2:]
        return CartesianState2D(d_pos, d_vel)
    
    @property
    def control_size(self) -> int:
        return self.B.shape[1]
    
    @property 
    def state_size(self) -> int:
        return self.A.shape[1]


class SimpleLTISystem(AbstractLTISystem):
    """An LTI system where the effector is taken trivially as the state.
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
    
    def effector(self, system_state):
        return system_state
    
    def init(
        self, 
        effector_state: Optional[CartesianState2D] = None
    ):
        if effector_state is None:
            effector_state = CartesianState2D(
                jnp.zeros(N_DIM), 
                jnp.zeros(N_DIM),
            )
        return effector_state
    

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