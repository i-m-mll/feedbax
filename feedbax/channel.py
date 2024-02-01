"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""


from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Optional, Tuple, Union

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree 

from feedbax.intervene import AbstractIntervenor
from feedbax.model import AbstractModelState
from feedbax.staged import AbstractStagedModel, ModelStageSpec
from feedbax.tree import random_split_like_tree


logger = logging.getLogger(__name__)
    

class ChannelState(eqx.Module):
    output: PyTree[Array, 'T']
    queue: Tuple[PyTree[Array, 'T'], ...]
    noise: Optional[PyTree[Array, 'T']] = None
    
    
class ChannelSpec(eqx.Module):
    """Specification for constructing channel, with `input_proto` obtained from 
    `where`.
    """
    where: Callable[[AbstractModelState], PyTree[Array]]
    delay: int = 0
    noise_std: Optional[float] = None


class Channel(AbstractStagedModel[ChannelState]):
    """A connection with delay and noise.
    
    This can be used for modeling an axon, tract, or wire.
    
    Delay is implemented as a tuple queue. 
    
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
        # TODO: Allow the delay to actually be 0 (i.e. return the input immediately; queue is always empty)
        self.delay = delay + 1  # otherwise when delay=0, nothing is stored
        self.noise_std = noise_std
        self.init_value = init_value
        self.input_proto = input_proto
        self.intervenors = self._get_intervenors_dict(intervenors)
    
    def __check_init__(self):
        if not isinstance(self.delay, int):
            raise ValueError("Delay must be an integer")
    
    def _update_queue(self, input, state, *, key):
        return ChannelState(
            output=state.queue[0], 
            queue=state.queue[1:] + (input,),
        )
        
    def _add_noise(self, input, state, *, key):
        noise = jax.tree_map(
            lambda x, key: self.noise_std * jr.normal(key, x.shape),
            input,
            random_split_like_tree(key, input),
        )
        output = jax.tree_map(lambda x, y: x + y, input, noise)
        return noise, output
    
    @property
    def model_spec(self):
        spec = OrderedDict({
            "update_queue": ModelStageSpec(
                callable=lambda self: self._update_queue,
                where_input=lambda input, state: input, 
                where_state=lambda state: state,
            ),            
        })
        
        if self.noise_std is not None:
            spec |= {
                "add_noise": ModelStageSpec(
                    callable=lambda self: self._add_noise,
                    where_input=lambda input, state: state.output,
                    where_state=lambda state: (state.noise, state.output),
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
