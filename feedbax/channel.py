"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree 


logger = logging.getLogger(__name__)


class ChannelState(eqx.Module):
    output: PyTree[Array]
    queue: Tuple[PyTree[Array], ...]
    

class Channel(eqx.Module):
    """Distant connection implemented as a queue, with optional added noise.
    
    For example, for modeling an axon, tract, or wire.
    
    This uses a tuple implementation since this is significantly faster than
    using `jnp.roll` and `.at` to shift and update a JAX array.
    
    TODO: 
    - Infer delay steps from time.
    """
    delay: int 
    noise_std: Optional[float]
    
    def __init__(self, delay, noise_std=None):
        self.delay = delay
        self.noise_std = noise_std
    
    @jax.named_scope("fbx.Channel")
    def __call__(self, input, state, key):      
        queue = state.queue[1:] + (input,)
        output = state.queue[0]
        if self.noise_std is not None:
            output = output + self.noise_std * jrandom.normal(key, output.shape) 
        return ChannelState(output, queue)
    
    def init(self, input):
        input_zeros = jax.tree_map(jnp.zeros_like, input)
        return ChannelState(
            input_zeros, 
            (self.delay - 1) * (input_zeros,) + (input,),
        )
