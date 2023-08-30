""" """


import jax.numpy as jnp


class MuscleGroup:  # TODO rename
    M: jnp.array = jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0), 
                              (0.0, 0.0, 2.0, -2.0, 2.0, -1.50)))  # [cm], apparently
    theta0: jnp.array = 2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), 
                                               (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360. # [deg] TODO: radians
    l0: jnp.array = jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04))  # [m]