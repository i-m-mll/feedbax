"""Modules that compose other modules into control contexts.

For example, classes defined here could be used to to model a sensorimotor 
loop, a body, or (perhaps) multiple interacting bodies. 

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
import logging
from typing import Callable, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

from feedbax.channel import Channel, ChannelState
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.networks import NetworkState 
from feedbax.state import AbstractState
from feedbax.types import CartesianState2D

logger = logging.getLogger(__name__)


State = TypeVar("State", bound=ChannelState)


class AbstractContext(eqx.Module, Generic[State]):
    @abstractmethod
    def __call__(
        self, 
        input, 
        state: State, 
        key: jr.PRNGKeyArray,
    ) -> State:
        ...
        
    # @abstractmethod
    # def init(
    #     self,
    #     state, 
    # ) -> State:
    #     ...
        
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...


class SimpleFeedbackState(AbstractState):
    mechanics: MechanicsState
    network: NetworkState
    feedback: ChannelState


class SimpleFeedback(AbstractContext):
    """Simple feedback loop with a single RNN and single mechanical system.
    
    TODO:
    - PyTree of force inputs (in addition to control inputs) to mechanics
    """
    net: eqx.Module  
    mechanics: Mechanics 
    feedback_channel: Channel
    delay: int 
    feedback_leaves_func: Callable[[PyTree], PyTree]  
    #perturbation: Optional[eqx.Module]
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: Mechanics, 
        delay: int = 0, 
        feedback_leaves_func: Callable = \
            lambda mechanics_state: mechanics_state.system
    ):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
        self.feedback_channel = Channel(delay)
        self.feedback_leaves_func = feedback_leaves_func
    
    @jax.named_scope("fbx.SimpleFeedback")
    def __call__(
        self, 
        input,  # AbstractTaskInput 
        state: SimpleFeedbackState, 
        key: jr.PRNGKeyArray,
    ) -> SimpleFeedbackState:
        
        key1, key2 = jr.split(key)
        
        feedback_state = self.feedback_channel(
            self.feedback_leaves_func(state.mechanics),
            state.feedback,
            key1,
        )
        
        # mechanics state feedback plus task inputs (e.g. target state)
        network_state = self.net(
            (input, feedback_state.output), 
            state.network.activity, 
            key2
        )
        
        mechanics_state = self.mechanics(
            network_state.output, 
            state.mechanics
        )        
        
        return SimpleFeedbackState(
            mechanics_state, 
            network_state,
            feedback_state,
        )
    
    def init(
        self, 
        effector_state: CartesianState2D,
        # TODO:
        mechanics_state: MechanicsState = None, 
        network_state: NetworkState = None, 
        feedback_state: ChannelState = None,
    ): 
        mechanics_state = self.mechanics.init(effector_state=effector_state)
        return SimpleFeedbackState(
            mechanics=mechanics_state,
            network=self.net.init(),
            feedback=self.feedback_channel.init(
                self.feedback_leaves_func(mechanics_state)
            ),
        )
    
    @property
    def memory_spec(self):
        """Specifies which states should typically be remembered by callers.
        
        For example, `fbx.Recursion` stores trajectories of states, however it
        doesn't usually make sense to store `states.feedback.queue` for every
        timestep, because it contains info that is already available to
        `Recursion` if `states.mechanics` is stored at every timestep. If the
        feedback delay is 5 steps, `Recursion` will end up with 5 extra copies
        of all the parts of `states.mechanics` that are part of the feedback.
        
        NOTE: It makes sense for this to be here since it has to do with the
        logic of the feedback loop, i.e. that queue is just transient internal 
        memory of another variable in the loop. 
        """
        return SimpleFeedbackState(
            mechanics=True, 
            network=True,
            feedback=ChannelState(output=True, queue=False)
        )