""" """


import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array


class MuscleGroup: #(eqx.Module):  # TODO rename
    M: Array = jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0), 
                          (0.0, 0.0, 2.0, -2.0, 2.0, -1.50)))  # [cm], apparently
    theta0: Array = 2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), 
                                            (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360. # [deg] TODO: radians
    l0: Array = jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04))  # [m]