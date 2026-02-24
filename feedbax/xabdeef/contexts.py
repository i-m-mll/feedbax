"""Pairings of pre-built models and tasks for easy setup and training.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable, Sequence
import logging
from typing import Any, Optional

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray
import optax  # type: ignore

from feedbax import get_ensemble
from feedbax.graph import Component
from feedbax.task import AbstractTask, SimpleReaches
from feedbax.train import TaskTrainer, TaskTrainerHistory
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_nn


logger = logging.getLogger(__name__)


N_LOG_STEPS_DEFAULT = 10


class TrainingContext(eqx.Module):
    """A model-task pairing with automatic construction of a
    [`TaskTrainer`][feedbax.train.TaskTrainer] instance.

    Attributes:
        model: The model.
        task: The task.
        where_train: A function that takes the model and returns the parts of the
            model to be trained.
        ensembled: Whether `model` is an ensemble of models.
    """

    model: Component
    task: AbstractTask
    where_train: Callable = lambda model: model.net
    ensembled: bool = False

    def train(
        self,
        *,
        n_batches: int,
        batch_size: int,
        learning_rate: float = 0.01,
        log_step: Optional[int] = None,
        optimizer_cls: Callable[..., optax.GradientTransformation] = optax.adam,
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> tuple[Component, TaskTrainerHistory]:
        """Train the model on the task.

        Arguments:
            n_batches: The number of batches of trials to train on.
            batch_size: The number of trials per batch.
            learning_rate: The learning rate for the optimizer.
            log_step: The number of batches between logs of training progress.
                If `None`, 10 evenly-spaced logs will be made along the training run.
            optimizer_cls: The class of Optax optimizer to use.
            key: A PRNG key for initializing the model.
            **kwargs: Additional keyword arguments to pass to the `TaskTrainer`.
        """
        optimizer = optax.inject_hyperparams(optimizer_cls)(learning_rate)

        trainer = TaskTrainer(
            optimizer=optimizer,
            checkpointing=True,
        )

        if log_step is None:
            log_step = n_batches // N_LOG_STEPS_DEFAULT

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
    n_replicates: int = 1,
    n_steps: int = 100,
    dt: float = 0.05,
    mass: float = 1.0,
    workspace: Sequence[tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    encoding_size: Optional[int] = None,
    hidden_size: int = 50,
    hidden_type: type[eqx.Module] = eqx.nn.GRUCell,
    where_train: Callable = lambda model: model.net,
    feedback_delay_steps: int = 0,
    eval_grid_n: int = 1,
    eval_n_directions: int = 7,
    *,
    key: PRNGKeyArray,
) -> TrainingContext:
    """A simple reach task paired with a neural network controlling a point mass.

    Arguments:
        n_replicates: The number of models to generate, with different random
            initializations.
        n_steps: The number of time steps in each trial.
        dt: The duration of each time step.
        mass: The mass of the point mass.
        workspace: The bounds of the rectangular workspace.
        encoding_size: The number of units in the encoding layer of the
            network. Defaults to `None` (no encoding layer).
        hidden_size: The number of units in the hidden layer of the network.
        hidden_type: The type of the hidden layer of the network.
        where_train: A function that takes a model and returns the part of
            the model that should be trained.
        feedback_delay_steps: The number of time steps by which sensory
            feedback is delayed.
        eval_grid_n: The number of grid points for center-out reaches in the
            validation task. For example, a value of 2 gives a grid of 2x2=4 center-out
            reach sets.
        eval_n_directions: The number of evenly-spread reach directions per
            set of center-out reaches.
        key: A random key used to initialize the model(s).
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
            n_steps=n_steps,
            dt=dt,
            mass=mass,
            encoding_size=encoding_size,
            hidden_size=hidden_size,
            hidden_type=hidden_type,
            feedback_delay_steps=feedback_delay_steps,
            key=key,
        )
        ensembled = False
    elif n_replicates > 1:
        model = get_ensemble(
            point_mass_nn,
            task,
            n_steps=n_steps,
            dt=dt,
            mass=mass,
            encoding_size=encoding_size,
            hidden_size=hidden_size,
            hidden_type=hidden_type,
            feedback_delay_steps=feedback_delay_steps,
            n=n_replicates,
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
