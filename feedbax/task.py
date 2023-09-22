

import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
import numpy as np


def uniform_endpoints(
    key: jrandom.PRNGKey,
    batch_size: int, 
    ndim: int = 2, 
    workspace: Float[Array, "ndim 2"] = jnp.array([[-1., 1.], 
                                                   [-1., 1.]]),
):
    """Segment endpoints uniformly distributed in a rectangular workspace."""
    pos_endpoints = jrandom.uniform(
        key, 
        (2, batch_size, ndim),   # (start/end, ...)
        minval=workspace[:, 0], 
        maxval=workspace[:, 1]
    )
    # add 0 velocity to init and target state
    state_endpoints = jnp.pad(pos_endpoints, ((0, 0), (0, 0), (0, ndim)))
    return state_endpoints


def centreout_endpoints(
    center: Float[Array, "2"], 
    n_directions: int, 
    angle_offset: float, 
    length: float,
): 
    ndim = 2  # TODO: generalize to sphere?
    """Segment endpoints starting in the centre and ending equally spaced on a circle."""
    angles = jnp.linspace(0, 2 * np.pi, n_directions + 1)[:-1]
    angles = angles + angle_offset

    starts = jnp.tile(center, (n_directions, 1))
    ends = center + length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    pos_endpoints = jnp.stack([starts, ends], axis=0)  
    state_endpoints = jnp.pad(pos_endpoints, ((0, 0), (0, 0), (0, ndim)))
    
    return state_endpoints