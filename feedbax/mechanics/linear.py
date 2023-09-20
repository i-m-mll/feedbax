""""""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# global defaults
ORDER = 2  # maximum order of the state derivatives
N_DIM = 2  # number of spatial dimensions


class LTISystem(eqx.Module):
    """Linear time-invariant system.
    
    Note that the system is defined in continuous time.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
    """
    A: Float[Array, "state state"]  # state evolution matrix
    B: Float[Array, "state input"]  # control matrix
    C: Float[Array, "obs state"]  # observation matrix
    
    def vector_field(
        self, 
        t: float, 
        y: Float[Array, "state"],
        args: Float[Array, "input"]
    ) -> Float[Array, "state"]:
        u = args  #? right way?
        d_y = self.A @ y + self.B @ u
        return d_y
    
    @property
    def control_size(self) -> int:
        return self.B.shape[1]
    
    @property 
    def state_size(self) -> int:
        return self.A.shape[1]
    

def point_mass(
    mass: float = 1., 
    order: int = ORDER, 
    n_dim: int = N_DIM
) -> LTISystem:
    A = sum([jnp.diagflat(jnp.ones((order - i) * n_dim), 
                          i * n_dim)
             for i in range(1, order)])
    B = jnp.concatenate(
        [jnp.zeros((n_dim, n_dim)), 
         jnp.eye(n_dim) / mass], 
        axis=0
    )
    C = jnp.array(1)  # TODO 
    return LTISystem(A, B, C)