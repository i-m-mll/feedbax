"""Pairings of pre-built models and tasks for easy setup and training.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable
from functools import partial
import logging
from typing import Optional, Type

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray
import optax

from feedbax.model import AbstractModel, get_ensemble
from feedbax.task import AbstractTask, SimpleReaches
from feedbax.trainer import TaskTrainer
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_nn


logger = logging.getLogger(__name__)


N_LOG_STEPS_DEFAULT = 10


class TrainingContext(eqx.Module):
    """A model-task pairing with automatic instantiation of a trainer.
    """
    model: AbstractModel 
    task: AbstractTask
    where_train: Optional[Callable] = None
    ensembled: bool = False
    
    
    def train(
        self,
        n_batches,
        batch_size,
        *,
        learning_rate: float = 0.01,
        log_step=None,
        optimizer_cls: Optional[Type[optax.GradientTransformation]] = optax.adam,
        key: PRNGKeyArray,
        **kwargs,
    ):
        optimizer = optax.inject_hyperparams(optimizer_cls)(
            learning_rate
        )
        
        trainer = TaskTrainer(
            optimizer=optimizer,
            checkpointing=True,
        )
        
        if log_step is None:
            log_step = n_batches // N_LOG_STEPS_DEFAULT
        
        """Train the model on the task."""
        return trainer(
            task=self.task,
            model=self.model, 
            n_batches=n_batches,
            batch_size=batch_size,
            log_step=log_step,
            where_train=self.where_train,
            key=key,
            ensembled=self.ensembled,
            **kwargs,
        )
        
        
def point_mass_nn_simple_reaches(
    n_steps: int = 100, 
    dt: float = 0.05, 
    mass: float = 1., 
    workspace = ((-1., -1.),
                 (1., 1.)),
    encoding_size: Optional[int] = None,
    hidden_size: int = 50, 
    hidden_type: eqx.Module = eqx.nn.GRUCell,
    where_train: Optional[Callable] = lambda model: model.step.net,
    feedback_delay_steps: int = 0,
    n_replicates: int = 1,
    eval_grid_n: int = 1,
    eval_n_directions: int = 7,
    *,
    key: PRNGKeyArray,
):
    """A simple reach task paired with a simple point mass-neural network model.
    
    Arguments:
    
    - `n_steps` is the number of time steps in each trial.
    - `dt` is the duration of each time step.
    - `mass` is the mass of the point mass.
    - `workspace` gives the bounds of the rectangular workspace.
    - `encoding_size` is the number of units in the encoding layer of the 
    network. Defaults to `None` (no encoding layer).
    - `hidden_size` is the number of units in the hidden layer of the network.
    - `hidden_type` is the type of the hidden layer of the network. Defaults to 
    a GRU cell.
    - `where_train` is a function that takes a model and returns the part of
    the model that should be trained.
    - `feedback_delay_steps` is the number of time steps by which sensory 
    feedback is delayed. Defaults to 0.
    - `n_replicates` is the number of models to generate, with different random 
    initializations.
    - `eval_grid_n` is the number of grid points for center-out reaches in the 
    validation task. For example, a value of 2 gives a grid of 2x2=4 center-out 
    reach sets. Defaults to 1.
    - `eval_n_directions` is the number of evenly-spread reach directions per 
    set of center-out reaches. 
    - `key` is a `jax.random.PRNGKey` that determines the pseudo-random 
    numbers used to initialize the model(s).
    
    TODO: 
    - It would be nice to avoid the vmapping overhead when not ensembling.
      However, then there's an issue with return values. Should we unsqueeze
      the model/states in the non-ensembled case to look like the ensembled 
      case? Or allow different returns? Or make separate functions here in 
      `xabdeef` for the two cases?
    """
    
    task = SimpleReaches(
        loss_func=simple_reach_loss(),
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=eval_grid_n,
        eval_n_directions=eval_n_directions,
        eval_reach_length=0.5,    
    )
    
    # TODO: Generalize this for all pre-built models
    if n_replicates == 1:
        model = point_mass_nn(
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
        model = get_ensemble(
            partial(
                point_mass_nn,
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
    
    return TrainingContext(
        model=model,
        task=task,
        where_train=where_train,
        ensembled=ensembled,
    )