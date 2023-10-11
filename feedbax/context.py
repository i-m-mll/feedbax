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
    ee: PyTree[Array]
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
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(
        self, 
        input, 
        state: SimpleFeedbackState, 
        args, 
        key: jrandom.PRNGKeyArray,
    ) -> SimpleFeedbackState:
        
        feedback_state = args  #! part of feedback hack
        
        key1, key2 = jrandom.split(key)
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((input, feedback_state), state.hidden, key1)
        
        # TODO: construct pytree of controls + extra inputs
        # TODO: transform extra inputs to appropriate forces
        
        mechanics_state = self.mechanics(state.control, state.mechanics)        
        
        # TODO: could be replaced with a general call to a `Mechanics` method that knows about the operational space
        twolink = self.mechanics.system.twolink
        ee_state = tuple(arr[:, -1] for arr in twolink.forward_kinematics(
            mechanics_state.system[0], mechanics_state.system[1]
        ))
        
        return SimpleFeedbackState(mechanics_state, ee_state, control, hidden)
    
    def init(self, ee_state): 
              
        # dataset gives init state in terms of effector position, but we need joint angles
        init_joints_pos = self.mechanics.system.twolink.inverse_kinematics(
            ee_state[0]
        )
        # TODO the tuple structure of pos-vel should be introduced in data generation, and kept throughout
        #! assumes zero initial velocity; TODO convert initial velocity also
        system_state = (
            init_joints_pos, 
            jnp.zeros_like(init_joints_pos),  
            jnp.zeros((self.mechanics.system.control_size,)),  # per-muscle activation
        )

        return SimpleFeedbackState(
            self.mechanics.init(system_state),
            ee_state,
            jnp.zeros((self.net.out_size,)),
            self.net.init(),
        )