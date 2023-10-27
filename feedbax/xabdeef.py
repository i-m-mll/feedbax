"""Basic API.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

import equinox as eqx
import jax
import jax.random as jr

from feedbax.channel import ChannelState
from feedbax.context import SimpleFeedback, SimpleFeedbackState
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.muscle import ActivationFilter, TodorovLiVirtualMuscle
from feedbax.mechanics.muscled_arm import TwoLinkMuscled
from feedbax.networks import RNN
from feedbax.recursion import Recursion


logger = logging.getLogger(__name__)


N_DIM = 2


def get_model(
        key=None,
        dt=0.05, 
        n_hidden=50, 
        n_steps=50, 
        feedback_delay=0, 
        tau=0.01, 
        out_nonlinearity=jax.nn.sigmoid,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
    key1, key2 = jr.split(key)
    
    system = TwoLinkMuscled(
        muscle_model=TodorovLiVirtualMuscle(), 
        activator=ActivationFilter(
            tau_act=tau,  
            tau_deact=tau,
        )
    )
    mechanics = Mechanics(system, dt)
    
    feedback_leaves_func = lambda mechanics_state: (
        mechanics_state.system.theta,
        mechanics_state.system.d_theta,
        mechanics_state.effector,         
    )
    
    # joint state feedback + effector state + target state
    n_input = system.twolink.state_size + 2 * N_DIM + 2 * N_DIM
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(
        cell, 
        system.control_size, 
        out_nonlinearity=out_nonlinearity,
        persistence=False, 
        key=key2
    )
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,  
        feedback_leaves_func=feedback_leaves_func,
    )

    return Recursion(body, n_steps)