"""Tasks on which models are trained and evaluated.

TODO:
- Maybe allow initial mods to model parameters, in addition to substates.
- Some of the private functions could be public.
- Refactor `get_target_seq` and `get_scalar_epoch_seq` redundancy.
    - Also, the way `seq` and `seqs` are generated is similar to `states` in
      `ForgetfulIterator.init`...

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations

from abc import abstractmethod, abstractproperty
from collections.abc import Callable, Mapping, Sequence
import dis
from functools import cached_property
import logging
from typing import (
    TYPE_CHECKING,
    Optional,
    Self,
    Tuple,
)

import equinox as eqx
from equinox import AbstractVar, Module, field
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Shaped
import numpy as np

from feedbax import tree_take
from feedbax.intervene import (
    AbstractIntervenor,
    AbstractIntervenorInput,
    InterventionSpec,
    TimeSeriesParam,
    add_fixed_intervenor,
    add_intervenors,
    schedule_intervenor,
)
from feedbax.intervene.remove import remove_all_intervenors
from feedbax.intervene.schedule import _eval_intervenor_param_spec
from feedbax.loss import AbstractLoss, LossDict
from feedbax._mapping import AbstractTransformedOrderedDict
from feedbax._model import ModelInput
from feedbax._staged import AbstractStagedModel
from feedbax.misc import get_unique_label, is_module
from feedbax.state import CartesianState, StateT
from feedbax._tree import tree_call

if TYPE_CHECKING:
    from feedbax._model import AbstractModel

logger = logging.getLogger(__name__)


N_DIM = 2


def _get_where_str(where_func: Callable) -> str:
    """
    Given a function that accesses a tree of attributes of a single parameter,
    return a string repesenting the attributes.

    This is useful for getting a unique string representation of a substate
    of an `AbstractState` or `AbstractModel` object, as defined by a `where`
    function, so we can compare two such functions and see if they refer to
    the same substate.

    TODO:
    - I'm not sure it's good practice to introspect on bytecode like this.
    """
    bytecode = dis.Bytecode(where_func)
    return ".".join(instr.argrepr for instr in bytecode if instr.opname == "LOAD_ATTR")


@jtu.register_pytree_node_class
class WhereDict(
    AbstractTransformedOrderedDict[
        str, Callable[[PyTree[Array]], PyTree[Array, "T"]], PyTree[Array, "T"]
    ]
):
    """An `OrderedDict` that allows limited use of `where` lambdas as keys.

    In particular, keys can be lambdas that take a single argument,
    and return a single (nested) attribute accessed from that argument.

    Lambdas are parsed to equivalent strings, which can be used
    interchangeably as keys. For example, the following are equivalent when
    `init_spec` is a `WhereDict`:

    ```python
    init_spec[lambda state: state.mechanics.effector]
    ```
    ```python
    init_spec['mechanics.effector']
    ```

    ??? dev-note "Performance tests"
        - Construction is about 100x slower than `OrderedDict`. For dicts of
        size relevant to our use case, this means ~100 us instead of ~1 us.
        A modest increase in dict size only changes this modestly.
        - Access is about 800x slower, and as expected this doesn't
        change with dict size because there's just a constant overhead
        for doing the key transformation on each access.
            - Each access is about 26 us, and this is also the duration of
            a call to `get_where_str`.
        - `list(init_spec.items())` is about 100x slower (24 us versus 234 ns)
        for a single-entry `init_spec`, and about 260x slower (149 us versus
        571 ns) for 6 entries, which is about as many as I expect anyone to use
        in the near future, when constructing a task.

        This is pretty slow, but since we only need to do a single
        construction and a single access of `init_spec` per batch/evaluation,
        it shouldn't matter too much in practice—overhead of about 125 us/batch,
        with a batch normally taking about 20,000+ us to train.

        Optimizations should focus on `get_where_str`.
    """

    def _key_transform(self, key: str | Callable) -> str:
        if isinstance(key, Callable):
            where_str = _get_where_str(key)
            if not where_str:
                raise ValueError(
                    "WhereDict keys must be lambdas that perform " "attribute access."
                )
            return where_str
        return key

    def __repr__(self):
        # Make a pretty representation of the lambdas
        items_str = ", ".join(
            f"(lambda state: state{'.' if k else ''}{k}, {v})"
            for k, (_, v) in self.store.items()
        )
        return f"{type(self).__name__}([{items_str}])"


class AbstractTaskInputs(Module):
    """Abstract base class for model inputs provided by a task.

    !!! Note ""
        Normally, each field of a subclass will be a PyTree of arrays where
        each array has a leading dimension corresponding to the time step—
        which becomes the second dimension, when the PyTree describes a batch
        of trials.
    """

    ...


# TODO: Once `target` is specified as a `WhereDict` as well, the only thing
# that will really change between classes of tasks is `inputs`. In that case,
# we could just make this `TaskTrialSpec` and have it be a generic of
# `AbstractTaskInputs`
class AbstractTaskTrialSpec(Module):
    """Abstract base class for trial specifications provided by a task.

    Attributes:
        inits: A mapping from `lambdas` that select model substates to be
            initialized, to substates to initialize them with.
        inputs: A PyTree of inputs to the model.
        target: A PyTree of target states.
        intervene: A mapping from unique intervenor names, to per-trial
            intervention parameters.
    """

    inits: AbstractVar[WhereDict]
    # inits: OrderedDict[Callable[[AbstractState], PyTree[Array]],
    #                        PyTree[Array]]
    inputs: AbstractVar[AbstractTaskInputs]
    target: AbstractVar[PyTree[Array]]
    intervene: AbstractVar[Mapping[str, Array]]


class AbstractReachTrialSpec(AbstractTaskTrialSpec):
    """Abstract base class for trial specifications for reaching tasks.

    Attributes:
        inits: A mapping from `lambdas` that select model substates to be
            initialized, to substates to initialize them with.
        inputs: A PyTree of inputs to the model, including data about the
            reach target.
        target: The target trajectory for the mechanical end effector,
            used for computing the loss.
        intervene: A mapping from unique intervenor names, to per-trial
            intervention parameters.

    """

    inits: AbstractVar[WhereDict]
    inputs: AbstractVar[AbstractTaskInputs]
    target: AbstractVar[CartesianState]
    intervene: AbstractVar[Mapping[str, Array]]

    @cached_property
    def goal(self):
        """The final state in the target trajectory for the mechanical end effector."""
        return jax.tree_map(lambda x: x[:, -1], self.target)


class SimpleReachTaskInputs(AbstractTaskInputs):
    """Model input for a simple reaching task.

    Attributes:
        effector_target: The trajectory of effector target states to be presented to
            the model.
    """

    effector_target: CartesianState  #! column vector: why here?


class DelayedReachTaskInputs(Module):
    """Model input for a delayed reaching task.

    Attributes:
        effector_target: The trajectory of effector target states to be presented to
            the model.
        hold: The hold/go (1/0 signal) to be presented to the model.
        target_on: A signal indicating to the model when the value of `effector_target`
            should be interpreted as a reach target. Otherwise, if zeros are passed for
            the target during (say) the hold period, the model may interpret this as
            meaningful—that is, "your reach target is at 0".
    """

    effector_target: CartesianState  # PyTree[Float[Array, "time ..."]]
    hold: Int[
        Array, "time 1"
    ]  # TODO: do these need to be typed as column vectors, here?
    target_on: Int[Array, "time 1"]


class SimpleReachTrialSpec(AbstractReachTrialSpec):
    """Trial specification for a simple reaching task.

    Attributes:
        inits: A mapping from `lambdas` that select model substates to be
            initialized, to substates to initialize them with at the start of trials.
        inputs: For providing the model with the reach target.
        target: The target trajectory for the mechanical end effector.
        intervene: A mapping from unique intervenor names, to per-trial
            intervention parameters.
    """

    inits: WhereDict
    inputs: SimpleReachTaskInputs
    target: CartesianState
    intervene: Mapping[str, Array] = field(default_factory=dict)


class DelayedReachTrialSpec(AbstractReachTrialSpec):
    """Trial specification for a delayed reaching task.

    Attributes:
        inits: A mapping from `lambdas` that select model substates to be
            initialized, to substates to initialize them with at the start of trials.
        inputs: For providing the model with the reach target and hold signal.
        target: The target trajectory for the mechanical end effector.
        epoch_start_idxs: The indices of the start of each epoch in the trial.
        intervene: A mapping from unique intervenor names, to per-trial
            intervention parameters.
    """

    inits: WhereDict
    inputs: DelayedReachTaskInputs
    target: CartesianState
    epoch_start_idxs: Int[Array, "n_epochs"]
    intervene: Mapping[str, Array] = field(default_factory=dict)


class AbstractTask(Module):
    """Abstract base class for tasks.

    Provides methods for evaluating suitable models or ensembles of models on
    training and validation trials.

    !!! Note ""
        Subclasses must provide:

        - a method that generates training trials
        - a property that provides a set of validation trials
        - a field for a loss function that grades performance on the task

    Attributes:
        loss_func: The loss function that grades task performance.
        n_steps: The number of time steps in the task trials.
        seed_validation: The random seed for generating the validation trials.
        # TODO
        intervention_specs: A mapping from unique intervenor names, to specifications
            for generating per-trial intervention parameters on training trials.
        intervention_specs_validation: A mapping from unique intervenor names, to
            specifications for generating per-trial intervention parameters on
            validation trials.
    """

    loss_func: AbstractVar[AbstractLoss]
    n_steps: AbstractVar[int]
    seed_validation: AbstractVar[int]

    intervention_specs: AbstractVar[Mapping[str, InterventionSpec]]
    intervention_specs_validation: AbstractVar[Mapping[str, InterventionSpec]]

    def __check_init__(self):
        if not isinstance(self.loss_func, AbstractLoss):
            raise ValueError(
                "The loss function must be an instance of `AbstractLoss`"
            )

    @abstractmethod
    def get_train_trial(
        self,
        key: PRNGKeyArray,
    ) -> AbstractTaskTrialSpec:
        """Return a single training trial specification.

        Arguments:
            key: A random key for generating the trial.
        """
        ...

    @eqx.filter_jit
    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
    ) -> AbstractTaskTrialSpec:
        """Return a single training trial specification, including intervention parameters.

        Arguments:
            key: A random key for generating the trial.
        """
        key, key_intervene = jr.split(key)

        with jax.named_scope(f"{type(self).__name__}.get_train_trial"):
            trial_spec = self.get_train_trial(key)

        trial_spec = eqx.tree_at(
            lambda x: x.intervene,
            trial_spec,
            self._intervenor_params(
                self.intervention_specs,
                trial_spec,
                key_intervene,
            ),
            is_leaf=lambda x: x is None,
        )
        return trial_spec

    def _intervenor_params(
        self,
        intervention_specs: Mapping[str, InterventionSpec],
        trial_spec: AbstractTaskTrialSpec,
        key: PRNGKeyArray,
    ):
        spec_intervenor_params = {k: v.intervenor.params for k, v in intervention_specs.items()}

        # TODO: Try not to repeat `intervene._eval_intervenor_param_spec`
        # Evaluate any parameters that vary by trial.
        intervenor_params = tree_call(
            spec_intervenor_params,
            trial_spec,
            key=key,
            # Treat `TimeSeriesParam`s as leaves, but don't call (unwrap) them yet.
            exclude=lambda x: isinstance(x, TimeSeriesParam),
            is_leaf=lambda x: isinstance(x, TimeSeriesParam),
        )

        timeseries, other = eqx.partition(
            intervenor_params,
            lambda x: isinstance(x, TimeSeriesParam),
            is_leaf=lambda x: isinstance(x, TimeSeriesParam),
        )

        # Unwrap the `TimeSeriesParam` instances.
        timeseries_arrays = tree_call(
            timeseries, is_leaf=lambda x: isinstance(x, TimeSeriesParam)
        )
        # timeseries_arrays = jax.tree_map(
        #     lambda x: x(),
        #     timeseries,
        #     is_leaf=lambda x: isinstance(x, TimeSeriesParam),
        # )

        # Broadcast the non-timeseries arrays.
        other_broadcasted = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)),
            jax.tree_map(jnp.array, other),
        )

        return eqx.combine(timeseries_arrays, other_broadcasted)

    @abstractmethod
    def get_validation_trials(
        self,
        key: PRNGKeyArray,
    ) -> AbstractTaskTrialSpec:
        """Return a set of validation trials, given a random key.

        !!! Note ""
            Subclasses must override this method. However, the validation
            used during training and provided by `self.validation_set`
            will be determined by the field `self.seed_validation`, which must
            also be implemented by subclasses.

        Arguments:
            key: A random key for generating the validation set.
        """
        ...

    @cached_property
    def validation_trials(self) -> AbstractTaskTrialSpec:
        """The set of validation trials associated with the task."""
        key = jr.PRNGKey(self.seed_validation)
        keys = jr.split(key, self.n_validation_trials)

        trial_specs = self.get_validation_trials(key)

        trial_specs = eqx.tree_at(
            lambda x: x.intervene,
            trial_specs,
            eqx.filter_vmap(self._intervenor_params, in_axes=(None, 0, 0))(
                self.intervention_specs_validation,
                trial_specs,
                keys,
            ),
            is_leaf=lambda x: x is None,
        )

        return trial_specs

    @abstractproperty
    def n_validation_trials(self) -> int:
        """Number of trials in the validation set."""
        ...

    @eqx.filter_jit
    @jax.named_scope("fbx.AbstractTask.eval_trials")
    def eval_trials(
        self,
        model: "AbstractModel[StateT]",
        trial_specs: AbstractTaskTrialSpec,
        keys: PRNGKeyArray,
    ) -> Tuple[StateT, LossDict]:
        """Evaluate a model on a set of trials.

        Arguments:
            model: The model to evaluate.
            trial_specs: The set of trials to evaluate the model on.
            keys: For providing randomness during model evaluation.
        """
        init_states = jax.vmap(model.init)(key=keys)

        for where_substate, init_substates in trial_specs.inits.items():
            init_states = eqx.tree_at(
                where_substate,
                init_states,
                init_substates,
            )

        init_states = jax.vmap(model.step.state_consistency_update)(init_states)

        states = eqx.filter_vmap(model)(  # ), in_axes=(eqx.if_array(0), 0, 0))(
            ModelInput(trial_specs.inputs, trial_specs.intervene),
            init_states,
            keys,
        )

        losses = self.loss_func(states, trial_specs)

        return states, losses

    def eval_with_loss(
        self,
        model: "AbstractModel[StateT]",
        key: PRNGKeyArray,
    ) -> Tuple[StateT, LossDict]:
        """Evaluate a model on the task's validation set of trials.

        Arguments:
            model: The model to evaluate.
            key: For providing randomness during model evaluation.

        Returns:
            The losses for the trials in the validation set.
            The evaluated model states.
        """

        keys = jr.split(key, self.n_validation_trials)
        trial_specs = self.validation_trials

        return self.eval_trials(model, trial_specs, keys)

    def eval(
        self,
        model: "AbstractModel[StateT]",
        key: PRNGKeyArray,
    ) -> StateT:
        """Return states for a model evaluated on the tasks's set of validation trials.

        Arguments:
            model: The model to evaluate.
            key: For providing randomness during model evaluation.
        """
        states, _ = self.eval_with_loss(model, key)
        return states

    @eqx.filter_jit
    def eval_ensemble_with_loss(
        self,
        models: "AbstractModel[StateT]",
        n_replicates: int,
        key: PRNGKeyArray,
        ensemble_random_trials: bool = True,
    ) -> tuple[StateT, LossDict]:
        """Return states and losses for an ensemble of models evaluated on the tasks's set of
        validation trials.

        Arguments:
            models: The ensemble of models to evaluate.
            n_replicates: The number of models in the ensemble.
            key: For providing randomness during model evaluation.
                Will be split into `n_replicates` keys.
            ensemble_random_trials: If `False`, each model in the ensemble will be
                evaluated on the same set of trials.
        """
        # TODO: Why not just use `eqx.filter_vmap`? It should handle the array partitioning.
        models_arrays, models_other = eqx.partition(
            models,
            eqx.is_array,
            is_leaf=lambda x: isinstance(x, AbstractIntervenor),
        )

        def evaluate_single(model_arrays, model_other, key):
            model = eqx.combine(model_arrays, model_other)
            return self.eval_with_loss(model, key)

        # TODO: Instead, we should expect the user to provide `keys` instead of `key`,
        # if they are vmapping `eval`.
        if ensemble_random_trials:
            return eqx.filter_vmap(evaluate_single, in_axes=(0, None, 0))(
                models_arrays, models_other, jr.split(key, n_replicates)
            )
        else:
            return eqx.filter_vmap(evaluate_single, in_axes=(0, None, None))(
                models_arrays, models_other, key
            )

    def eval_ensemble(
        self,
        models: "AbstractModel[StateT]",
        n_replicates: int,
        key: PRNGKeyArray,
        ensemble_random_trials: bool = True,
    ) -> StateT:
        """Return states for an ensemble of models evaluated on the tasks's set of
        validation trials.

        Arguments:
            models: The ensemble of models to evaluate.
            n_replicates: The number of models in the ensemble.
            key: For providing randomness during model evaluation.
                Will be split into `n_replicates` keys.
            ensemble_random_trials: If `False`, each model in the ensemble will be
                evaluated on the same set of trials.
        """
        states, _ = self.eval_ensemble_with_loss(
            models, n_replicates, key, ensemble_random_trials=ensemble_random_trials
        )
        return states

    @eqx.filter_jit
    def eval_train_batch(
        self,
        model: "AbstractModel[StateT]",
        batch_size: int,
        key: PRNGKeyArray,
    ) -> Tuple[StateT, LossDict, AbstractTaskTrialSpec]:
        """Evaluate a model on a single batch of training trials.

        Arguments:
            model: The model to evaluate.
            batch_size: The number of trials in the batch.
            key: For providing randomness during model evaluation.

        Returns:
            The losses for the trials in the batch.
            The evaluated model states.
            The trial specifications for the batch.
        """
        key_batch, key_eval = jr.split(key)
        keys_batch = jr.split(key_batch, batch_size)
        keys_eval = jr.split(key_eval, batch_size)

        trials = jax.vmap(self.get_train_trial_with_intervenor_params)(keys_batch)

        states, losses = self.eval_trials(model, trials, keys_eval)

        return states, losses, trials

    @eqx.filter_jit
    def eval_ensemble_train_batch(
        self,
        models: "AbstractModel[StateT]",
        n_replicates: int,
        batch_size: int,
        key: PRNGKeyArray,
        ensemble_random_trials: bool = True,
    ) -> Tuple[StateT, LossDict, AbstractTaskTrialSpec]:
        """Evaluate an ensemble of models on a single training batch.

        Arguments:
            models: The ensemble of models to evaluate.
            n_replicates: The number of models in the ensemble.
            batch_size: The number of trials in the batch to evaluate.
            key: For providing randomness during model evaluation.
            ensemble_random_trials: If `False`, each model in the ensemble will be
                evaluated on the same set of trials.

        Returns:
            The losses for the trials in the batch, for each model in the ensemble.
            The evaluated model states, for each trial and each model in the ensemble.
            The trial specifications for the batch.
        """
        models_arrays, models_other = eqx.partition(
            models,
            eqx.is_array,
            is_leaf=lambda x: isinstance(x, AbstractIntervenor),
        )

        def evaluate_single(model_arrays, model_other, batch_size, key):
            model = eqx.combine(model_arrays, model_other)
            return self.eval_train_batch(model, batch_size, key)

        if ensemble_random_trials:
            return eqx.filter_vmap(evaluate_single, in_axes=(0, None, None, 0))(
                models_arrays, models_other, batch_size, jr.split(key, n_replicates)
            )
        else:
            return eqx.filter_vmap(evaluate_single, in_axes=(0, None, None, None))(
                models_arrays, models_other, batch_size, key
            )

    def add_intervenors_to_base_model(
        self,
        model: AbstractStagedModel[StateT],
    ) -> AbstractStagedModel[StateT]:
        """Add the task's scheduled intervenors to a model.

        Assumes that the model has the appropriate structure to admit the
        intervention. This depends on the the `where` and `stage_name`
        properties stored in the task's `intervention_specs` field, since the
        original call to `schedule_intervenor` that added them to the task.

        - `where` should pick out an `AbstractStagedModel` component of `model`.
        - If defined, `stage_name` should be the name of one of the stages of
            the `AbstractStagedModel` component picked out by `where`.

        Any existing intervenors in the model that were scheduled with another
        task, will be removed to prevent conflicts. Other intervenors which were
        added directly to the model without being scheduled with a task will
        not be removed.

        !!! Note
            This method is mostly useful when evaluating a trained model on a task
            with a different set of interventions than the one it was trained on.
        """
        # Remove all intervenors from the model that don't have underscores
        base_model = remove_all_intervenors(model, scheduled_only=True)

        # Make a copy of `self`, without its spec intervenors
        base_task = eqx.tree_at(
            lambda task: (task.intervention_specs, task.intervention_specs_validation),
            self,
            (dict(), dict()),
        )

        # Use schedule_intervenors to reproduce `self`, along with the modified model
        task, model_ = base_task, base_model
        for label, spec in self.intervention_specs.items():
            if label in self.intervention_specs_validation:
                intervenor_params_val = (
                    self.intervention_specs_validation[label].intervenor.params
                )
            else:
                intervenor_params_val = None

            task, model_ = schedule_intervenor(
                task,
                model_,
                intervenor=spec.intervenor,
                where=spec.where,
                default_active=spec.default_active,
                intervenor_params_validation=intervenor_params_val,
                # Only applies if `intervenor_params_val` is None:
                validation_same_schedule=False,
            )

        return model_


def _pos_only_states(positions: Float[Array, "... ndim=2"]):
    """Construct Cartesian init and target states with zero force and velocity."""
    velocities = jnp.zeros_like(positions)
    forces = jnp.zeros_like(positions)

    states = jax.tree_map(
        lambda x: CartesianState(*x),
        list(zip(positions, velocities, forces)),
        is_leaf=lambda x: isinstance(x, tuple),
    )

    return states


def internal_grid_points(
    bounds: Float[Array, "bounds=2 ndim=2"], n: int = 2
) -> Float[Array, "n**ndim ndim=2"]:
    """Return a list of evenly-spaced grid points internal to the bounds.

    Arguments:
        bounds: The outer bounds of the grid.
        n: The number of internal grid points along each dimension.

    !!! Example
        ```python
        internal_grid_points(
            bounds=((0, 0), (9, 9)),
            n=2,
        )
        ```
        ```>> Array([[3., 3.], [6., 3.], [3., 6.], [6., 6.]]).```
    """
    ticks = jax.vmap(lambda b: jnp.linspace(b[0], b[1], n + 2)[1:-1])(bounds.T)
    points = jnp.vstack(jax.tree_map(jnp.ravel, jnp.meshgrid(*ticks))).T
    return points


def _centerout_endpoints_grid(
    workspace: Float[Array, "bounds=2 ndim=2"],
    eval_grid_n: int,
    eval_n_directions: int,
    eval_reach_length: float,
):
    """Sets of center-out reaches, their centers in a grid across a workspace."""
    centers = internal_grid_points(workspace, eval_grid_n)
    pos_endpoints = jax.vmap(
        centreout_endpoints,
        in_axes=(0, None, None),
        out_axes=1,
    )(
        centers, eval_n_directions, eval_reach_length
    ).reshape((2, -1, N_DIM))
    return pos_endpoints


def _forceless_task_inputs(
    target_states: CartesianState,
) -> CartesianState:
    """Only position and velocity of targets are supplied to the model."""
    return CartesianState(
        pos=target_states.pos,
        vel=target_states.vel,
        force=None,
    )


class SimpleReaches(AbstractTask):
    """Reaches between random endpoints in a rectangular workspace. No hold signal.

    Validation set is center-out reaches.

    !!! Note
        This passes a trajectory of target velocities all equal to zero, assuming
        that the user will choose a loss function that penalizes only the initial
        or final velocities. If the loss function penalizes the intervening velocities,
        this task no longer makes sense as a reaching task.

    Attributes:
        n_steps: The number of time steps in each task trial.
        loss_func: The loss function that grades performance on each trial.
        workspace: The rectangular workspace in which the reaches are distributed.
        seed_validation: The random seed for generating the validation trials.
        intervention_specs: A mapping from unique intervenor names, to specifications
            for generating per-trial intervention parameters on training trials.
        intervention_specs_validation: A mapping from unique intervenor names, to
            specifications for generating per-trial intervention parameters on
            validation trials.
        eval_grid_n: The number of evenly-spaced internal grid points of the
            workspace at which a set of center-out reach is placed.
        eval_n_directions: The number of evenly-spread center-out reaches
            starting from each workspace grid point in the validation set. The number
            of trials in the validation set is equal to
            `eval_n_directions * eval_grid_n ** 2`.
        eval_reach_length: The length (in space) of each reach in the validation set.
    """

    n_steps: int
    loss_func: AbstractLoss
    workspace: Float[Array, "bounds=2 ndim=2"] = field(converter=jnp.asarray)
    seed_validation: int = 5555
    intervention_specs: Mapping[str, InterventionSpec] = field(default_factory=dict)
    intervention_specs_validation: Mapping[str, InterventionSpec] = field(
        default_factory=dict
    )
    eval_n_directions: int = 7
    eval_reach_length: float = 0.5
    eval_grid_n: int = 1  # e.g. 2 -> 2x2 grid of center-out reach sets

    def get_train_trial(self, key: PRNGKeyArray) -> SimpleReachTrialSpec:
        """Random reach endpoints across the rectangular workspace.

        Arguments:
            key: A random key for generating the trial.
        """

        effector_pos_endpoints = uniform_tuples(key, n=2, bounds=self.workspace)

        effector_init_state, effector_target_state = _pos_only_states(
            effector_pos_endpoints
        )

        # Broadcast the fixed targets to a sequence with the desired number of
        # time steps, since that's what `ForgetfulIterator` and `Loss` will expect.
        # Hopefully this should not use up any extra memory.
        effector_target_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)),
            effector_target_state,
        )
        effector_target = _forceless_task_inputs(effector_target_state)
        # effector_target = _forceless_task_inputs(
        #     jax.tree_map(
        #         lambda x: x[:-1],
        #         effector_target_state,
        #     )
        # )
        # TODO: It might be better here to use an `Intervenor`-like callable
        # instead of `WhereDict`, which is slow. Though the callable would
        # ideally provide the initial state as a
        # def init_func(state):
        #     return eqx.tree_at(
        #         lambda state: state.mechanics.effector,
        #         state,
        #         effector_init_state,
        #     )

        return SimpleReachTrialSpec(
            inits=WhereDict(
                {(lambda state: state.mechanics.effector): effector_init_state}
            ),
            inputs=SimpleReachTaskInputs(effector_target=effector_target),
            target=effector_target_state,
        )

    def get_validation_trials(self, key: PRNGKeyArray) -> SimpleReachTrialSpec:
        """Center-out reach sets in a grid across the rectangular workspace."""

        effector_pos_endpoints = _centerout_endpoints_grid(
            self.workspace,
            self.eval_grid_n,
            self.eval_n_directions,
            self.eval_reach_length,
        )

        effector_init_states, effector_target_states = _pos_only_states(
            effector_pos_endpoints
        )

        # Broadcast to the desired number of time steps. Awkwardly, we also
        # need to use `swapaxes` because the batch dimension is explicit, here.
        effector_target_states = jax.tree_map(
            lambda x: jnp.swapaxes(jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)), 0, 1),
            effector_target_states,
        )

        # task_inputs = _forceless_task_inputs(
        #     jax.tree_map(
        #         lambda x: x[:, :-1],
        #         effector_target_states,
        #     )
        # )

        return SimpleReachTrialSpec(
            inits=WhereDict(
                {(lambda state: state.mechanics.effector): effector_init_states}
            ),
            inputs=SimpleReachTaskInputs(
                effector_target=_forceless_task_inputs(effector_target_states)
            ),
            target=effector_target_states,
        )

    @property
    def n_validation_trials(self) -> int:
        """Number of trials in the validation set."""
        return self.eval_grid_n**2 * self.eval_n_directions


class DelayedReaches(AbstractTask):
    """Uniform random endpoints in a rectangular workspace.

    e.g. allows for a stimulus epoch, followed by a delay period, then movement.

    Attributes:
        loss_func: The loss function that grades performance on each trial.
        workspace: The rectangular workspace in which the reaches are distributed.
        n_steps: The number of time steps in each task trial.
        epoch_len_ranges: The ranges from which to uniformly sample the durations of
            the task phases for each task trial.
        target_on_epochs: The epochs in which the "target on" signal is turned on.
        hold_epochs: The epochs in which the hold signal is turned on.
        eval_n_directions: The number of evenly-spread center-out reaches
            starting from each workspace grid point in the validation set. The number
            of trials in the validation set is equal to
            `eval_n_directions * eval_grid_n ** 2`.
        eval_reach_length: The length (in space) of each reach in the validation set.
        eval_grid_n: The number of evenly-spaced internal grid points of the
            workspace at which a set of center-out reach is placed.
        seed_validation: The random seed for generating the validation trials.
    """

    loss_func: AbstractLoss
    workspace: Float[Array, "bounds=2 ndim=2"] = field(converter=jnp.asarray)
    n_steps: int
    epoch_len_ranges: Tuple[Tuple[int, int], ...] = field(
        default=(
            (5, 15),  # start
            (10, 20),  # target on ("stim")
            (10, 25),  # delay
        )
    )
    target_on_epochs: Int[Array, "_"] = field(default=(1,), converter=jnp.asarray)
    hold_epochs: Int[Array, "_"] = field(default=(0, 1, 2), converter=jnp.asarray)
    eval_n_directions: int = 7
    eval_reach_length: float = 0.5
    eval_grid_n: int = 1
    seed_validation: int = 5555
    intervention_specs: Mapping[str, InterventionSpec] = field(default_factory=dict)
    intervention_specs_validation: Mapping[str, InterventionSpec] = field(
        default_factory=dict
    )

    def get_train_trial(self, key: PRNGKeyArray) -> DelayedReachTrialSpec:
        """Random reach endpoints across the rectangular workspace.

        Arguments:
            key: A random key for generating the trial.
        """

        key1, key2 = jr.split(key)
        effector_pos_endpoints = uniform_tuples(key1, n=2, bounds=self.workspace)

        effector_init_state, effector_target_state = _pos_only_states(
            effector_pos_endpoints
        )

        task_inputs, effector_target_states, epoch_start_idxs = self._get_sequences(
            effector_init_state, effector_target_state, key2
        )

        return DelayedReachTrialSpec(
            inits=WhereDict(
                {(lambda state: state.mechanics.effector): effector_init_state}
            ),
            inputs=task_inputs,
            target=effector_target_states,
            epoch_start_idxs=epoch_start_idxs,
        )

    def get_validation_trials(self, key: PRNGKeyArray) -> DelayedReachTrialSpec:
        """Center-out reach sets in a grid across the rectangular workspace."""

        effector_pos_endpoints = _centerout_endpoints_grid(
            self.workspace,
            self.eval_grid_n,
            self.eval_n_directions,
            self.eval_reach_length,
        )

        effector_init_states, effector_target_states = _pos_only_states(
            effector_pos_endpoints
        )

        key_val = jr.PRNGKey(self.seed_validation)
        epochs_keys = jr.split(key_val, effector_init_states.pos.shape[0])
        task_inputs, effector_target_states, epoch_start_idxs = jax.vmap(
            self._get_sequences
        )(effector_init_states, effector_target_states, epochs_keys)

        return DelayedReachTrialSpec(
            inits=WhereDict(
                {(lambda state: state.mechanics.effector): effector_init_states}
            ),
            inputs=task_inputs,
            target=effector_target_states,
            epoch_start_idxs=epoch_start_idxs,
        )

    def _get_sequences(
        self,
        init_states: CartesianState,
        target_states: CartesianState,
        key: PRNGKeyArray,
    ) -> Tuple[DelayedReachTaskInputs, CartesianState, Int[Array, "n_epochs"]]:
        """Convert static task inputs to sequences, and make hold signal."""
        epoch_lengths = gen_epoch_lengths(key, self.epoch_len_ranges)
        epoch_start_idxs = jnp.pad(
            jnp.cumsum(epoch_lengths), (1, 0), constant_values=(0, -1)
        )
        epoch_masks = get_masks(self.n_steps - 1, epoch_start_idxs)
        move_epoch_mask = jnp.logical_not(jnp.prod(epoch_masks, axis=0))[None, :]

        stim_seqs = get_masked_seqs(
            _forceless_task_inputs(target_states), epoch_masks[self.target_on_epochs]
        )
        target_seqs = jax.tree_map(
            lambda x, y: x + y,
            get_masked_seqs(target_states, move_epoch_mask),
            get_masked_seqs(init_states, epoch_masks[self.hold_epochs]),
        )
        stim_on_seq = get_scalar_epoch_seq(
            epoch_start_idxs, self.n_steps - 1, 1.0, self.target_on_epochs
        )
        hold_seq = get_scalar_epoch_seq(
            epoch_start_idxs, self.n_steps - 1, 1.0, self.hold_epochs
        )

        task_input = DelayedReachTaskInputs(stim_seqs, hold_seq, stim_on_seq)
        target_states = target_seqs

        return task_input, target_states, epoch_start_idxs

    @property
    def n_validation_trials(self) -> int:
        """Number of trials in the validation set."""
        return self.eval_grid_n**2 * self.eval_n_directions


class Stabilization(AbstractTask):
    """Postural stabilization task at random points in workspace.

    Validation set is center-out reaches.
    """

    n_steps: int
    loss_func: AbstractLoss
    workspace: Float[Array, "bounds=2 ndim=2"] = field(converter=jnp.asarray)
    seed_validation: int = 5555
    eval_grid_n: int = 1  # e.g. 2 -> 2x2 grid
    eval_workspace: Optional[Float[Array, "bounds=2 ndim=2"]] = field(
        converter=jnp.asarray, default=None
    )
    intervention_specs: Mapping[str, InterventionSpec] = field(default_factory=dict)
    intervention_specs_validation: Mapping[str, InterventionSpec] = field(
        default_factory=dict
    )

    def get_train_trial(self, key: PRNGKeyArray) -> SimpleReachTrialSpec:
        """Random reach endpoints in a 2D rectangular workspace."""

        points = uniform_tuples(key, n=1, bounds=self.workspace)

        target_state, = _pos_only_states(points)

        init_state = target_state

        target_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)),
            target_state,
        )

        # TODO: don't use SimpleReachTrialSpec...
        return SimpleReachTrialSpec(
            inits=WhereDict({lambda state: state.mechanics.effector: init_state}),
            inputs=SimpleReachTaskInputs(
                effector_target=_forceless_task_inputs(target_state)
            ),
            target=target_state,
        )

    def get_validation_trials(self, key: PRNGKeyArray) -> SimpleReachTrialSpec:
        """Center-out reaches across a regular workspace grid."""

        # if self.eval_workspace is None:
        #     workspace = self.workspace
        # else:
        #     workspace = self.eval_workspace

        points = _points_grid(
            self.workspace,
            self.eval_grid_n,
        )

        target_states, = _pos_only_states(points)

        init_states = target_states

        # Broadcast to the desired number of time steps. Awkwardly, we also
        # need to use `swapaxes` because the batch dimension is explicit, here.
        target_states = jax.tree_map(
            lambda x: jnp.swapaxes(
                jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)), 0, 1
            ),
            target_states,
        )

        return SimpleReachTrialSpec(
            inits=WhereDict({lambda state: state.mechanics.effector: init_states}),
            inputs=SimpleReachTaskInputs(
                effector_target=_forceless_task_inputs(target_states)
            ),
            target=target_states,
        )

    @property
    def n_validation_trials(self) -> int:
        """Size of the validation set."""
        return self.eval_grid_n ** 2


def _points_grid(
    workspace: Float[Array, "bounds=2 ndim=2"],
    grid_n: int | Tuple[int, int],
):
    """A regular grid of points over a rectangular workspace.

    Args:
        grid_n: Number of grid points in each dimension.
    """
    if isinstance(grid_n, int):
        grid_n = (grid_n, grid_n)

    xy_1d = map(lambda x: jnp.linspace(x[0][0], x[0][1], x[1]), zip(workspace.T, grid_n))
    grid = jnp.stack(jnp.meshgrid(*xy_1d))
    grid_points = grid.reshape(2, -1).T[None]
    return grid_points


def uniform_tuples(
    key: PRNGKeyArray,
    n: int,
    bounds: Float[Array, "bounds=2 ndim=2"],
):
    """Tuples of points uniformly distributed in some (2D) bounds."""
    return jr.uniform(key, (n, N_DIM), minval=bounds[0], maxval=bounds[1])


def centreout_endpoints(
    center: Float[Array, "2"],
    n_directions: int,
    length: float,
    angle_offset: float = 0,
) -> Float[Array, "2 n_directions 2"]:
    """Segment endpoints starting in the centre and ending equally spaced on a circle."""
    angles = jnp.linspace(0, 2 * np.pi, n_directions + 1)[:-1]
    angles = angles + angle_offset

    starts = jnp.tile(center, (n_directions, 1))
    ends = center + length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    return jnp.stack([starts, ends], axis=0)


def gen_epoch_lengths(
    key: PRNGKeyArray,
    ranges: Tuple[Tuple[int, int], ...] = (
        (1, 3),  # (min, max) for first epoch
        (2, 5),  # second epoch
        (1, 3),
    ),
) -> Int[Array, "n_epochs"]:
    """Generate a random integer in each given ranges."""
    ranges_arr = jnp.array(ranges, dtype=int)
    return jr.randint(key, (ranges_arr.shape[0],), *ranges_arr.T)


def get_masks(
    length: int,
    idx_bounds: Int[Array, "_"],
):
    """Get a 1D mask of length `length` with `False` values at `idxs`."""
    idxs = jnp.arange(length)
    # ? could also use `arange` to get ranges of idxs
    mask_fn = lambda e: (idxs < idx_bounds[e]) + (idxs > idx_bounds[e + 1] - 1)
    return jnp.stack([mask_fn(e) for e in range(len(idx_bounds) - 1)])


def get_masked_seqs(
    arrays: PyTree,
    masks: Int[Array, "masks n"],  # TODO
    init_fn: Callable[[Tuple[int, ...]], Shaped[Array, "..."]] = jnp.zeros,
) -> PyTree:
    """Expand arrays with an initial axis of length `n`, and fill with
    original array values where the intersection of `masks` is `False`.

    That is, each expanded array will be filled with the values from `array`
    for all indices where *any* of the masks is `False`.

    Returns a PyTree with the same structure as `target`, but where each
    array has an additional sequence dimension, and the original `target`
    values are assigned only during the target epoch, as bounded by
    `target_idxs`.

    TODO:
    - Find a better name.
    """

    seqs = jax.tree_map(lambda x: init_fn((masks.shape[1], *x.shape)), arrays)
    # seqs = tree_set(seqs, targets, slice(*epoch_idxs[target_epoch:target_epoch + 2]))
    mask = jnp.prod(masks, axis=0)
    seqs = jax.tree_map(
        lambda x, y: jnp.where(
            jnp.expand_dims(mask, np.arange(y.ndim) + 1), x, y[None, :]
        ),
        seqs,
        arrays,
    )
    return seqs


def get_scalar_epoch_seq(
    epoch_idxs: Int[Array, "n_epochs-1"],
    n_steps: int,
    hold_value: float,
    hold_epochs: Sequence[int] | Int[Array, "_"],
):
    """A scalar sequence with a non-zero value held during `hold_epochs`.

    Similar to `get_target_steps`, but not for a PyTree.
    """
    seq = jnp.zeros((n_steps,))
    idxs = jnp.arange(n_steps)
    # fill_idxs = jnp.arange(epoch_idxs[hold_epoch], epoch_idxs[hold_epoch + 1])
    mask_fn = lambda e: (idxs < epoch_idxs[e]) + (idxs > epoch_idxs[e + 1] - 1)
    mask = jnp.prod(jnp.stack([mask_fn(e) for e in hold_epochs]), axis=0)
    seq = jnp.where(mask, seq, jnp.array(hold_value))
    return jnp.expand_dims(seq, -1)
