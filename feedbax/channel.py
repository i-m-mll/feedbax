"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
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
    noise: Optional[PyTree[Array]] = None


class Channel(eqx.Module):
    """Distant connection implemented as a queue, with optional added noise.
    
    For example, for modeling an axon, tract, or wire.
    
    This uses a tuple implementation since this is significantly faster than
    using `jnp.roll` and `.at` to shift and update a JAX array.
    
    TODO: 
    - Infer delay steps from time.
    - Use a shape dtype struct for `input_proto`.
    """
    delay: int = field(converter=lambda x: x + 1)
    noise_std: Optional[float] = None
    init_value: float = jnp.nan
    input_proto: PyTree[Array] = field(default_factory=lambda: jnp.zeros(1))
    
    def __check_init__(self):
        if not isinstance(self.delay, int):
            raise ValueError("Delay must be an integer")
    
    @jax.named_scope("fbx.Channel")
    def __call__(self, input, state, key):             
        queue = state.queue[1:] + (input,)
        output = state.queue[0]
        if self.noise_std is not None:
            noise = jax.tree_map(
                lambda x: self.noise_std * jr.normal(key, x.shape),
                output,
            )
            output = jax.tree_map(lambda x, y: x + y, output, noise)
        else:
            noise = None
        return ChannelState(output, queue, noise)
    
    def init(self, *, key: Optional[Array] = None):
        input_init = jax.tree_map(
            lambda x: jnp.full_like(x, self.init_value), 
            self.input_proto
        )
        if self.noise_std is not None:
            noise_init = input_init
        else:
            noise_init = None
            
        return ChannelState(
            input_init, 
            self.delay * (input_init,),
            noise_init,
        )
        
    def change_input(self, input) -> "Channel":
        """Returns a similar `Channel` object for a different input type."""
        return eqx.tree_at(
            lambda channel: channel.input_proto,
            self,
            input
        )       
