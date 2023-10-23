"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree

from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.state import AbstractState


logger = logging.getLogger(__name__)


class SimpleFeedbackState(AbstractState):
    mechanics: MechanicsState
    control: Array
    hidden: PyTree


class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    delay: int = eqx.field(static=True)
    #perturbation: Optional[eqx.Module]
    
    def __init__(self, net, mechanics, delay=0):
        self.net = net
        self.mechanics = mechanics
        #! delay isn't implemented here yet
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(
        self, 
        input, 
        state: SimpleFeedbackState, 
        args, #! part of feedback hack
        key: jrandom.PRNGKeyArray,
    ) -> SimpleFeedbackState:
        
        feedback_state = args  #! part of feedback hack
        
        key1, key2 = jrandom.split(key)
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((input, feedback_state), state.hidden, key1)
        
        # TODO: construct pytree of controls + extra inputs
        # TODO: transform extra inputs to appropriate forces
        
        mechanics_state = self.mechanics(state.control, state.mechanics)        
        
        return SimpleFeedbackState(mechanics_state, control, hidden)
    
    def init(self, effector_state): 
        return SimpleFeedbackState(
            self.mechanics.init(effector_state),
            jnp.zeros((self.mechanics.system.control_size,)),
            self.net.init(),
        )