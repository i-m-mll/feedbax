"""Pre-built models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable
from functools import partial
import logging
from typing import Optional

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax.bodies import SimpleFeedback
from feedbax.mechanics.plant import DirectForceInput
from feedbax.model import AbstractModel, get_ensemble
from feedbax.iterate import ForgetfulIterator, Iterator
from feedbax.mechanics import Mechanics
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.muscle import ActivationFilter
from feedbax.networks import SimpleStagedNetwork


logger = logging.getLogger(__name__)


N_DIM = 2


def point_mass_nn(
    task,
    key: Optional[PRNGKeyArray] = None,
    dt: float = 0.05, 
    mass: float = 1., 
    hidden_size: int = 50, 
    hidden_type: eqx.Module = eqx.nn.GRUCell,
    encoding_size: Optional[int] = None,
    n_steps: int = 100, 
    feedback_delay_steps: int = 0,
    feedback_noise_std: float = 0.0,
    motor_noise_std: float = 0.0,  # TODO
    out_nonlinearity: Callable = lambda x: x,
):
    """From nb8"""
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
        
    key1, key2 = jr.split(key)
    
    system = PointMass(mass=mass)
    mechanics = Mechanics(DirectForceInput(system), dt)
    
    feedback_spec = dict(
        where=lambda state: (
            state.plant.skeleton.pos,
            state.plant.skeleton.vel,
        ),
        delay=feedback_delay_steps,
        noise_std=feedback_noise_std,
    )
    
    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics, feedback_spec=feedback_spec
    )
    
    net = SimpleStagedNetwork(
        input_size,
        hidden_size,
        out_size=system.control_size, 
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        out_nonlinearity=out_nonlinearity, 
        key=key1,
    )
    body = SimpleFeedback(net, mechanics, feedback_spec=feedback_spec, key=key2)
    
    model = Iterator(body, n_steps)
    
    return model


