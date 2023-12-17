"""Basic API providing pre-built models.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from functools import partial
import logging
from typing import Callable, Optional, Type

import diffrax
import equinox as eqx
import jax
import jax.random as jr
import optax

from feedbax.channel import ChannelState
from feedbax.mechanics.plant import SimplePlant
from feedbax.model import AbstractModel, SimpleFeedback
from feedbax.iterate import Iterator, SimpleIterator
from feedbax.mechanics import Mechanics
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.muscle import ActivationFilter, TodorovLiVirtualMuscle
from feedbax.mechanics.muscled_arm import TwoLinkMuscled
from feedbax.networks import SimpleNetwork
from feedbax.task import AbstractTask, RandomReaches
from feedbax.trainer import TaskTrainer
from feedbax.utils import get_model_ensemble
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
    
    TODO:
    - Rename this, since it conflicts with a common Python pattern.
    """
    model: AbstractModel 
    task: AbstractTask
    trainable_leaves_func: Optional[Callable] = None
    ensembled: bool = False
    
    
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
            ensembled=self.ensembled,
        )
    

def point_mass_NN(
    task,
    key: Optional[jax.Array] = None,
    dt: float = 0.05, 
    mass: float = 1., 
    hidden_size: int = 50, 
    hidden_type: eqx.Module = eqx.nn.GRUCell,
    encoding_size: Optional[int] = None,
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
    mechanics = Mechanics(SimplePlant(system), dt)
    
    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics
    )
    
    net = SimpleNetwork(
        input_size,
        hidden_size,
        out_size=system.control_size, 
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        out_nonlinearity=out_nonlinearity, 
        key=key1,
    )
    body = SimpleFeedback(net, mechanics, delay=feedback_delay_steps, key=key2)
    
    model = SimpleIterator(body, n_steps)
    
    return model


def point_mass_NN_simple_reaches(
    n_replicates: int = 1,
    n_steps: int = 100, 
    dt: float = 0.05, 
    mass: float = 1., 
    workspace = ((-1., -1.),
                 (1., 1.)),
    hidden_size: int = 50, 
    encoding_size: Optional[int] = None,
    hidden_type: eqx.Module = eqx.nn.GRUCell,
    feedback_delay_steps: int = 0,
    eval_grid_n: int = 2,
    *,
    key: jax.Array,
):
    """
    
    TODO: 
    - It would be nice to avoid the vmapping overhead when not ensembling.
      However, then there's an issue with return values. Should we unsqueeze
      the model/states in the non-ensembled case to look like the ensembled 
      case? Or allow different returns? Or make separate functions here in 
      `xabdeef` for the two cases?
    """
    
    task = RandomReaches(
        loss_func=simple_reach_loss(n_steps),
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=eval_grid_n,
        eval_n_directions=8,
        eval_reach_length=0.5,    
    )
    
    if n_replicates == 1:
        model = point_mass_NN(
            task,
            key=key,
            dt=dt,
            mass=mass,
            hidden_size=hidden_size, 
            encoding_size=encoding_size,
            hidden_type=hidden_type,
            n_steps=n_steps,
            feedback_delay_steps=feedback_delay_steps,
        )
        ensembled = False
    elif n_replicates > 1:
        model = get_model_ensemble(
            partial(
                point_mass_NN,
                task,
                dt=dt,
                mass=mass,
                hidden_size=hidden_size, 
                encoding_size=encoding_size,
                hidden_type=hidden_type,
                n_steps=n_steps,
                feedback_delay_steps=feedback_delay_steps,
            ),
            n_replicates=n_replicates,
            key=key,
        )
        ensembled = True
    else:
        raise ValueError("n_replicates must be an integer >= 1")
    
    trainable_leaves_func = lambda model: model.step.net
    
    manager = ContextManager(
        model=model,
        task=task,
        trainable_leaves_func=trainable_leaves_func,
        ensembled=ensembled,
    )
    
    return manager