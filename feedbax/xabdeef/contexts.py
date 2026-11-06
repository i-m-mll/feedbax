"""Pairings of pre-built models and tasks for easy setup and training.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Optional

import equinox as eqx
from jaxtyping import PRNGKeyArray, PyTree

import feedbax._vendor.jax_cookbook.tree as jtree
from feedbax.runtime.graph import Component
from feedbax.tasks import AbstractTask, SimpleReaches
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_nn


class TrainingContext(eqx.Module):
    """A model-task pairing for constructing supervised training run specs.

    Attributes:
        model: The model.
        task: The task.
        where_train: A function that takes the model and returns the parts of the
            model to be trained.
        n_replicates: Number of models constructed.
        model_axes: Axes used to construct the ensemble, for passing to task
            evaluation. A scalar model has no mapped axes.
    """

    model: Component
    task: AbstractTask
    where_train: Callable = lambda model: model.nodes["net"]
    n_replicates: int = 1
    model_axes: PyTree = None

    @property
    def ensembled(self) -> bool:
        return self.n_replicates > 1


def point_mass_nn_simple_reaches(
    n_replicates: int = 1,
    n_steps: int = 100,
    dt: float = 0.05,
    mass: float = 1.0,
    workspace: Sequence[tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    encoding_size: Optional[int] = None,
    hidden_size: int = 50,
    hidden_type: type[eqx.Module] = eqx.nn.GRUCell,
    where_train: Callable = lambda model: model.nodes["net"],
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

    if n_replicates < 1:
        raise ValueError("n_replicates must be an integer >= 1")

    build = partial(
        point_mass_nn,
        task,
        n_steps=n_steps,
        dt=dt,
        mass=mass,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        hidden_type=hidden_type,
        feedback_delay_steps=feedback_delay_steps,
    )
    if n_replicates == 1:
        model_axes = None
        model = build(key=key)
    else:
        model_axes = eqx.if_array(0)
        model = jtree.get_ensemble(
            build,
            n=n_replicates,
            key=key,
            out_axes=model_axes,
        )

    return TrainingContext(
        model=model,
        task=task,
        where_train=where_train,
        n_replicates=n_replicates,
        model_axes=model_axes,
    )
