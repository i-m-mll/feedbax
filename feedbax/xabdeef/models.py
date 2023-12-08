"""Basic API providing pre-built models.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Callable, Optional, Type

import diffrax
import equinox as eqx
import jax
import jax.random as jr
import optax

from feedbax.channel import ChannelState
from feedbax.model import AbstractModel, SimpleFeedback
from feedbax.iterate import Iterator, SimpleIterator
from feedbax.mechanics import Mechanics
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.muscle import ActivationFilter, TodorovLiVirtualMuscle
from feedbax.mechanics.muscled_arm import TwoLinkMuscled
from feedbax.networks import SimpleNetwork
from feedbax.task import AbstractTask, RandomReaches
from feedbax.trainer import TaskTrainer
from feedbax.xabdeef.losses import simple_reach_loss


logger = logging.getLogger(__name__)


N_DIM = 2
DEFAULT_LEARNING_RATE = 0.01


class ContextManager(eqx.Module):
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
    trainable_leaves_func: Optional[Callable] = None
    
    
    def train(
        self,
        n_batches,
        batch_size,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        log_step=100,
        optimizer_cls: Optional[Type[optax.GradientTransformation]] = optax.adam,
        key
    ):
        optimizer = optax.inject_hyperparams(optimizer_cls)(
            learning_rate
        )
        
        trainer = TaskTrainer(
            optimizer=optimizer,
            checkpointing=False,
        )
        
        """Train the model on the task."""
        return trainer(
            task=self.task,
            model=self.model, 
            n_batches=n_batches,
            batch_size=batch_size,
            log_step=log_step,
            trainable_leaves_func=self.trainable_leaves_func,
            key=key,
        )

def point_mass_RNN(
    task,
    key: Optional[jax.Array] = None,
    dt: float = 0.05, 
    mass: float = 1., 
    hidden_size: int = 50, 
    n_steps: int = 100, 
    feedback_delay_steps: int = 0,
    out_nonlinearity: Callable = lambda x: x,
):
    """From nb8"""
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
        
    key1, key2 = jr.split(key)
    
    system = PointMass(mass=mass)
    mechanics = Mechanics(system, dt, solver=diffrax.Euler)
    
    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics
    )
    
    net = SimpleNetwork(
        input_size,
        hidden_size,
        out_size=system.control_size, 
        out_nonlinearity=out_nonlinearity, 
        key=key1,
    )
    body = SimpleFeedback(net, mechanics, delay=feedback_delay_steps, key=key2)
    
    model = SimpleIterator(body, n_steps)
    
    return model


def point_mass_RNN_simple_reaches(
    n_steps: int = 100, 
    dt: float = 0.05, 
    mass: float = 1., 
    workspace = ((-1., -1.),
                 (1., 1.)),
    hidden_size: int = 50, 
    feedback_delay_steps: int = 0,
    eval_grid_n: int = 2,
    *,
    key: jax.Array,
):
    """"""
    
    task = RandomReaches(
        loss_func=simple_reach_loss(n_steps),
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=eval_grid_n,
        eval_n_directions=8,
        eval_reach_length=0.5,    
    )
    
    model = point_mass_RNN(
        task,
        key=key,
        dt=dt,
        mass=mass,
        hidden_size=hidden_size, 
        n_steps=n_steps,
        feedback_delay_steps=feedback_delay_steps,
    )
    
    trainable_leaves_func = lambda model: (
        model.step.net.cell.weight_hh, 
        model.step.net.cell.weight_ih, 
        model.step.net.cell.bias
    )
    
    manager = ContextManager(
        model=model,
        task=task,
        trainable_leaves_func=trainable_leaves_func,
    )
    
    return manager


def point_mass_RNN_simple_reaches(
    n_steps: int = 100, 
    dt: float = 0.05, 
    mass: float = 1., 
    workspace = ((-1., -1.),
                 (1., 1.)),
    hidden_size: int = 50, 
    feedback_delay_steps: int = 0,
    eval_grid_n: int = 2,
    *,
    key: jax.Array,
):
    """"""
    
    task = RandomReaches(
        loss_func=simple_reach_loss(n_steps),
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=eval_grid_n,
        eval_n_directions=8,
        eval_reach_length=0.5,    
    )
    
    model = point_mass_RNN(
        task,
        key=key,
        dt=dt,
        mass=mass,
        hidden_size=hidden_size, 
        n_steps=n_steps,
        feedback_delay_steps=feedback_delay_steps,
    )
    
    trainable_leaves_func = lambda model: (
        model.step.net.cell.weight_hh, 
        model.step.net.cell.weight_ih, 
        model.step.net.cell.bias
    )
    
    manager = ContextManager(
        model=model,
        task=task,
        trainable_leaves_func=trainable_leaves_func,
    )
    
    return manager