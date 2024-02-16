"""Queue-based modules for modeling distant, possibly noisy connections.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""


from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
from typing import Optional, Tuple, Union

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree 

from feedbax.intervene import AbstractIntervenor
from feedbax.staged import AbstractStagedModel, ModelStage
from feedbax.state import AbstractState
from feedbax.tree import random_split_like_tree


logger = logging.getLogger(__name__)
    

class ChannelState(eqx.Module):
    """Type of state PyTree operated on by [`Channel`][feedbax.channel.Channel] instances.
    
    Attributes:
        output: The current output of the channel.
        queue: A tuple of previous inputs to the channel, with the most recent appearing last.
        noise: The noise added to the current output, if any.
    """
    output: PyTree[Array, 'T']
    queue: Tuple[PyTree[Array, 'T'], ...]
    noise: Optional[PyTree[Array, 'T']] = None
    
    
class ChannelSpec(eqx.Module):
    """Specifies how to build a [`Channel`][feedbax.channel.Channel], with respect to the state PyTree of its owner.
    
    Attributes:
        where: A function that selects the subtree of feedback states.
        delay: The number of previous inputs to store in the queue.
        noise_std: The standard deviation of the noise to add to the output.
    """
    where: Callable[[AbstractState], PyTree[Array]]
    delay: int = 0
    noise_std: Optional[float] = None


class Channel(AbstractStagedModel[ChannelState]):
    """A noisy queue.
    
    !!! NOTE
        This can be used for modeling an axon, tract, wire, or other delayed 
        and semi-reliable connection between model components.
        
    Attributes:
        delay: The number of previous inputs stored in the queue.
        noise_std: The standard deviation of the noise added to the output.
        input_proto: A PyTree of arrays with the same structure/shapes as the
            inputs to the channel will have.
        intervenors: [Intervenors][feedbax.intervene.AbstractIntervenor] to add 
            to the model at construction time.
    """
    delay: int
    noise_std: Optional[float]
    input_proto: PyTree[Array] 
    intervenors: Mapping[str, Sequence[AbstractIntervenor]] 
    _init_value: float 
        
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
        self._init_value = init_value
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
    
    @cached_property
    def model_spec(self):
        """Returns an `OrderedDict` that specifies the stages of the channel model.
        
        Always includes a queue input-output stage. Optionally includes 
        a stage that adds noise to the output, if `noise_std` was not 
        `None` at construction time.
        """
        spec = OrderedDict({
            "update_queue": ModelStage(
                callable=lambda self: self._update_queue,
                where_input=lambda input, state: input, 
                where_state=lambda state: state,
            ),            
        })
        
        if self.noise_std is not None:
            spec |= {
                "add_noise": ModelStage(
                    callable=lambda self: self._add_noise,
                    where_input=lambda input, state: state.output,
                    where_state=lambda state: (state.noise, state.output),
                ),
            }
            
        return spec
        
    @property
    def memory_spec(self) -> ChannelState:  # type: ignore
        return ChannelState(
            output=True, 
            queue=False,
            noise=False,
        )
    
    def init(self, *, key: Optional[PRNGKeyArray] = None) -> ChannelState:
        """Returns an empty `ChannelState` for the channel."""
        input_init = jax.tree_map(
            lambda x: jnp.full_like(x, self._init_value), 
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
        
    def change_input(self, input_proto: PyTree[Array]) -> "Channel":
        """Returns a similar `Channel` with a changed input structure."""
        return eqx.tree_at(
            lambda channel: channel.input_proto,
            self,
            input_proto
        )       
