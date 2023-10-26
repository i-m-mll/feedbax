"""Modules that compose other modules into control contexts.

For example, classes defined here could be used to to model a sensorimotor 
loop, a body, or (perhaps) multiple interacting bodies. 

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree

from feedbax.channel import Channel, ChannelState
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.state import AbstractState
from feedbax.types import CartesianState2D

logger = logging.getLogger(__name__)


class SimpleFeedbackState(AbstractState):
    mechanics: MechanicsState
    control: Array
    hidden: PyTree
    feedback: ChannelState


class SimpleFeedback(eqx.Module):
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
    
    def __call__(
        self, 
        input,  # AbstractTaskInput 
        state: SimpleFeedbackState, 
        key: jrandom.PRNGKeyArray,
    ) -> SimpleFeedbackState:
        
        key1, key2 = jrandom.split(key)
        
        feedback = self.feedback_channel(
            self.feedback_leaves_func(state.mechanics),
            state.feedback,
            key1,
        )
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((input, feedback.output), state.hidden, key2)
        
        mechanics_state = self.mechanics(state.control, state.mechanics)        
        
        return SimpleFeedbackState(mechanics_state, control, hidden, feedback)
    
    def init(
        self, 
        effector_state: CartesianState2D,
    ): 
        mechanics_state = self.mechanics.init(effector_state)
        return SimpleFeedbackState(
            mechanics=mechanics_state,
            control=jnp.zeros((self.mechanics.system.control_size,)),
            hidden=self.net.init(),
            feedback=self.feedback_channel.init(
                self.feedback_leaves_func(mechanics_state)
            ),
        )
    
    # def states_include(self):
    #     return SimpleFeedbackState(
    #         mechanics=True, 
    #         control=True, 
    #         hidden=True, 
    #         feedback=ChannelState(output=True, queue=False)
    #     )