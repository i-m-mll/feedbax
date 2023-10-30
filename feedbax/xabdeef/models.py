"""Basic API providing pre-built models.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Callable, Optional

import diffrax
import equinox as eqx
import jax
import jax.random as jr
import optax

from feedbax.channel import ChannelState
from feedbax.context import AbstractModel, SimpleFeedback
from feedbax.iterate import Iterator
from feedbax.mechanics import Mechanics
from feedbax.mechanics.linear import point_mass
from feedbax.mechanics.muscle import ActivationFilter, TodorovLiVirtualMuscle
from feedbax.mechanics.muscled_arm import TwoLinkMuscled
from feedbax.networks import RNN
from feedbax.task import AbstractTask
from feedbax.trainer import TaskTrainer


logger = logging.getLogger(__name__)


N_DIM = 2
DEFAULT_LEARNING_RATE = 0.01


class ModelManager(eqx.Module):
    """Simple interface to pre-built models, losses, and tasks.
    
    Not sure about naming... `Model` conflicts with references to
    `AbstractModel` instances as `model` elsewhere, which would be confusing
    to refer to as `context`. Also the task isn't part of the model, whereas a
    task is associated with an instance of this class.
    
    We could choose not to associate `task` with this class. Let users choose 
    tasks from `xabdeef.tasks` and pass them to `train`...
    """
    model: AbstractModel 
    task: AbstractTask
    optimizer: optax.GradientTransformation
    
    def __init__(
        self,
        task: AbstractTask,
        context: AbstractModel,
        optimizer: Optional[optax.GradientTransformation] = None,
    ):
        if optimizer is None:
            optimizer = optax.adam(DEFAULT_LEARNING_RATE)
        
        self._trainer = TaskTrainer(
            optimizer,  
            
        )    
        
        self.context = context 
        self.task = task
    
    def train(
        self,
        batch_size,
        n_batches,
    ):
        """Train the model on the task."""
        return self.trainer(
            task=self.task,
            model=self.model, 
        )


def point_mass_RNN(
    task,
    key: Optional[jax.Array],
    dt: float = 0.05, 
    mass: float = 1., 
    n_hidden: int = 50, 
    n_steps: int = 100, 
    feedback_delay: int = 0,
    out_nonlinearity: Callable = lambda x: x,
):
    """From nb8"""
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
    
    key1, key2 = jr.split(key)
    
    system = point_mass(mass=mass, n_dim=N_DIM)
    mechanics = Mechanics(system, dt, solver=diffrax.Euler)
    
    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics
    )
    
    # the cell determines what kind of RNN layer to use
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNNCellWithReadout(
        input_size,
        hidden_size,
        out_size,
        system.control_size, 
        out_nonlinearity=out_nonlinearity, 
        persistence=False,
        key=key2,
    )
    body = SimpleFeedback(net, mechanics, feedback_delay)
    
    model = Iterator(body, n_steps)
    
    return model