"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""


import logging
from typing import Optional, Tuple

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
import jax.random as jr
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
    - Use a shape dtype struct for `input_proto`.
    """
    delay: int 
    noise_std: Optional[float] = None
    input_proto: PyTree[Array] = field(default_factory=lambda: jnp.zeros(1))
    
    @jax.named_scope("fbx.Channel")
    def __call__(self, input, state, key):      
        queue = state.queue[1:] + (input,)
        output = state.queue[0]
        if self.noise_std is not None:
            output = jax.tree_map(
                lambda x: x + self.noise_std * jr.normal(key, x.shape),
                output,
            )
        return ChannelState(output, queue)
    
    def init(self, *, key: Optional[Array] = None):
        input_zeros = jax.tree_map(jnp.zeros_like, self.input_proto)
        return ChannelState(
            input_zeros, 
            (self.delay - 1) * (input_zeros,) + (self.input_proto,)
        )
        
    def change_input(self, input) -> "Channel":
        """Returns a similar `Channel` object for a different input type."""
        return eqx.tree_at(
            lambda channel: channel.input_proto,
            self,
            input
        )       
