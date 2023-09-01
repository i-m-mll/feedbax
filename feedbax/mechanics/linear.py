""""""

import equinox as eqx
import jax.numpy as jnp


# global defaults
ORDER = 2  # maximum order of the state derivatives
N_DIM = 2  # number of spatial dimensions


class LTISystem(eqx.Module):
    """Linear time-invariant system.
    
    Note that the system is defined in continuous time.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
    """
    A: jnp.ndarray  # state evolution matrix
    B: jnp.ndarray  # control matrix
    C: jnp.ndarray  # observation matrix
    
    def field(self, t, y, args):
        u = args  #? right way?
        d_y = self.A @ y + self.control(t, u)
        return d_y
        
    def control(self, t, u):
        return self.B @ u
        

def point_mass(
    mass: float = 1., 
    order: int = ORDER, 
    n_dim: int = N_DIM
) -> LTISystem:
    A = sum([jnp.diagflat(jnp.ones((order - i) * n_dim), i * n_dim)
             for i in range(1, order)])
    B = jnp.concatenate([jnp.zeros((n_dim, n_dim)), jnp.eye(n_dim) / mass], axis=0)
    C = jnp.array(1)
    return LTISystem(A, B, C)