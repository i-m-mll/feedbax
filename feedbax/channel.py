"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""


from collections import OrderedDict
from collections.abc import Mapping, Sequence
import logging
from typing import Optional, Tuple, Union

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree 

from feedbax.intervene import AbstractIntervenor
from feedbax.model import AbstractStagedModel


logger = logging.getLogger(__name__)
    

class ChannelState(eqx.Module):
    output: PyTree[Array, 'T']
    queue: Tuple[PyTree[Array, 'T'], ...]
    noise: Optional[PyTree[Array, 'T']] = None


class Channel(AbstractStagedModel[ChannelState]):
    """Distant connection implemented as a queue, with optional added noise.
    
    For example, for modeling an axon, tract, or wire.
    
    This uses a tuple implementation since this is significantly faster than
    using `jnp.roll` and `.at` to shift and update a JAX array.
    
    TODO: 
    - Infer delay steps from time.
    - Use a shape dtype struct for `input_proto`.
    """
    delay: int
    noise_std: Optional[float]
    init_value: float 
    input_proto: PyTree[Array] 
    intervenors: Mapping[str, Sequence[AbstractIntervenor]] 
        
    def __init__(
        self, 
        delay: int, 
        noise_std: Optional[float] = None,
        init_value: float = jnp.nan,
        input_proto: PyTree[Array] = jnp.zeros(1),
        intervenors: Optional[Union[
                Sequence[AbstractIntervenor],
                Mapping[str, Sequence[AbstractIntervenor]]]
            ] = None,
    ):
        self.delay = delay 
        self.noise_std = noise_std
        self.init_value = init_value
        self.input_proto = input_proto
        self.intervenors = self._get_intervenors_dict(intervenors)
    
    def __check_init__(self):
        if not isinstance(self.delay, int):
            raise ValueError("Delay must be an integer")
    
    # @jax.named_scope("fbx.Channel")
    # def __call__(self, input, state, key):             
    #     queue = state.queue[1:] + (input,)
    #     output = state.queue[0]
    #     if self.noise_std is not None:
    #         noise = jax.tree_map(
    #             lambda x: self.noise_std * jr.normal(key, x.shape),
    #             output,
    #         )
    #         output = jax.tree_map(lambda x, y: x + y, output, noise)
    #     else:
    #         noise = None
    #     return ChannelState(output, queue, noise)
    
    def _update_queue(self, input, state, *, key):
        return (
            state.queue[0], 
            state.queue[1:] + (input,),
        )
        
    def _add_noise(self, input, state, *, key):
        noise = jax.tree_map(
            lambda x: self.noise_std * jr.normal(key, x.shape),
            input,
        )
        output = jax.tree_map(lambda x, y: x + y, state.output, noise)
        return noise, output
    
    @property
    def model_spec(self):
        spec = OrderedDict({
            "update_queue": (
                lambda self: self._update_queue,
                lambda input, state: input, 
                lambda state: (state.output, state.queue),
            ),            
        })
        
        if self.noise_std is not None:
            spec |= {
                "add_noise": (
                    lambda self: self._add_noise,
                    lambda input, state: state.output,
                    lambda state: (state.noise, state.output),
                ),
            }
            
        return spec
        
    @property
    def memory_spec(self):
        return ChannelState(
            output=True, 
            queue=False,
            noise=False,
        )
    
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
