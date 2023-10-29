"""Basic API providing pre-built models.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.random as jr

from feedbax.channel import ChannelState
from feedbax.context import SimpleFeedback, SimpleFeedbackState
from feedbax.mechanics import Mechanics
from feedbax.mechanics.linear import point_mass
from feedbax.mechanics.muscle import ActivationFilter, TodorovLiVirtualMuscle
from feedbax.mechanics.muscled_arm import TwoLinkMuscled
from feedbax.networks import RNN
from feedbax.recursion import Recursion


logger = logging.getLogger(__name__)


N_DIM = 2


def point_mass_RNN_loop(
    task,
    dt: float = 0.05, 
    mass: float = 1., 
    n_hidden: int = 50, 
    n_steps: int = 100, 
    feedback_delay: int = 0,
    out_nonlinearity: Callable = lambda x: x,
    key: jr.PRNGKeyArray = None,
):
    """From nb8"""
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
    
    key1, key2 = jr.split(key)
    
    system = point_mass(mass=mass, n_dim=N_DIM)
    mechanics = Mechanics(system, dt, solver=diffrax.Euler)
    
    # automatically determine network input size
    n_input = SimpleFeedback.get_nn_input_size(
        task, mechanics
    )
    
    # the cell determines what kind of RNN layer to use
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(
        cell, 
        system.control_size, 
        out_nonlinearity=out_nonlinearity, 
        persistence=False,
        key=key2
    )
    body = SimpleFeedback(net, mechanics, feedback_delay)
    
    model = Recursion(body, n_steps)
    
    return model