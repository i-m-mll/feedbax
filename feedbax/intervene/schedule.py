"""

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence, Callable
import copy
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.intervene import AbstractIntervenor, AbstractIntervenorInput, is_intervenor
from feedbax.misc import get_unique_label, is_module
from feedbax._model import AbstractModel
from feedbax.state import StateT
from feedbax._tree import tree_call

if TYPE_CHECKING:
    from feedbax._staged import AbstractStagedModel


logger = logging.getLogger(__name__)


class InterventionSpec(eqx.Module):
    # TODO: The type of `intervenor` is wrong: each entry will have the same PyTree
    # structure as `AbstractIntervenorInput` but may be filled with callables that
    # specify a trial distribution for the leaves
    intervenor: AbstractIntervenor
    where: Callable[[AbstractModel], "AbstractStagedModel"]
    stage_name: Optional[str]
    default_active: bool


def add_intervenor(
    model: "AbstractStagedModel[StateT]",
    intervenor: AbstractIntervenor,
    stage_name: Optional[str] = None,
    **kwargs: Any,
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with an added intervenor.

    !!! Note ""
        This is a helper for calling `add_intervenors` with a single intervenor.

    Arguments:
        model: The model to which the intervenor will be added.
        intervenor: The intervenor to add.
        stage_name: The stage named in `model.model_spec` to which the intervenor will
            be added. The intervenor will execute at the end of the stage.
            If `None`, the intervenor will execute before the first model stage.
        kwargs: Additional keyword arguments to
            [`add_intervenors`][feedbax.intervene.add_intervenors].
    """
    return add_intervenors(model, {stage_name: [intervenor]}, **kwargs)


StateS = TypeVar("StateS", Module, Array)
InputT = TypeVar("InputT", bound=AbstractIntervenorInput)


def add_intervenors(
    model: "AbstractStagedModel[StateT]",
    intervenors: Union[
        Sequence[AbstractIntervenor[StateS, InputT]],
        Mapping[Optional[str], Sequence[AbstractIntervenor[StateS, InputT]]],
    ],
    where: Callable[
        [AbstractModel[StateT]], Any  # "AbstractStagedModel[StateS]"
    ] = lambda model: model.step,
    keep_existing: bool = True,
    scheduled: bool = False,
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with added intervenors.

    Arguments:
        model: The model to which the intervenors will be added.
        intervenors: The intervenors to add. May be a sequence of intervenors to
            execute before the execution of the first model stage, or a mapping from
            stage name to the sequence of intervenors to execute at the end of that
            model stage.
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
            it (which may be `model` itself) to which to add the intervenors.
        keep_existing: Whether to keep the existing intervenors belonging directly to
            the instance of `AbstractStagedModel` to which the new intervenors are added.
            If `True`, the new intervenors are appended to the existing ones; if `False`,
            the old intervenors are replaced.
        scheduled: By default, intervenor labels are prepended by an underscore when the
            intervenor is not associate with trial scheduling, by a task. Functions such
            as `schedule_intervenor` that pair an intervention with a task schedule
            should set `scheduled` to `True`.
    """
    if keep_existing:
        existing_intervenors = {
            stage_name: list(intervenors)
            for stage_name, intervenors in where(model).intervenors.items()
        }
    else:
        # TODO: is this necessary?
        existing_intervenors = {stage_name: [] for stage_name in where(model).model_spec}

    if not scheduled:
        intervenors = jax.tree_map(
            lambda intervenor: eqx.tree_at(
                lambda x: x.label,
                intervenor,
                '_' + intervenor.label,
            ),
            intervenors,
            is_leaf=is_intervenor,
        )

    if isinstance(intervenors, Sequence):
        # If a sequence is given, append to the intervenors not associated with a stage
        intervenors_dict = existing_intervenors | {
            None: existing_intervenors.get(None, []) + list(intervenors)
        }
    elif isinstance(intervenors, dict):
        intervenors_dict = copy.deepcopy(existing_intervenors)
        for label, new_intervenors in intervenors.items():
            intervenors_dict[label] = (
                intervenors_dict.get(label, []) + list(new_intervenors)
            )
    else:
        raise ValueError("intervenors not a sequence or dict of sequences")

    for stage_name in intervenors_dict:
        if stage_name not in where(model).model_spec and stage_name is not None:
            raise ValueError(
                f"{stage_name} is not a valid model stage for intervention"
            )

    return eqx.tree_at(
        lambda model: where(model).intervenors,
        model,
        OrderedDict(intervenors_dict),
    )


class TimeSeriesParam(Module):
    """Wraps intervenor parameters that should be interpreted as time series.

    Attributes:
        param: The parameter to interpret as a time series.
    """

    param: Array

    def __call__(self):
        """Return the wrapped parameter."""
        return self.param


def _eval_intervenor_param_spec(
    intervention_spec: InterventionSpec,
    trial_spec, #: AbstractTaskTrialSpec,
    key: PRNGKeyArray,
):
    # Unwrap any `TimeSeriesParam` instances:
    return tree_call(
        # Evaluate any trial-generation lambdas:
        tree_call(
            intervention_spec.intervenor.params,
            trial_spec,
            key=key,
            # Don't unwrap `TimeSeriesParam`s yet:
            exclude=lambda x: isinstance(x, TimeSeriesParam),
            is_leaf=lambda x: isinstance(x, TimeSeriesParam),
        ),
        is_leaf=lambda x: isinstance(x, TimeSeriesParam),
    )


def schedule_intervenor(
    tasks: PyTree["AbstractTask"],
    models: PyTree[AbstractModel[StateT]],
    intervenor: AbstractIntervenor | Type[AbstractIntervenor],
    # ensembled: bool = False,  # TODO
    where: Callable[[AbstractModel[StateT]], Any] = lambda model: model,
    stage_name: Optional[str] = None,
    validation_same_schedule: bool = True,
    intervenor_params: Optional[
        AbstractIntervenorInput
    ] = None,  #! wrong! distribution functions are allowed. only the PyTree structure is the same
    intervenor_params_validation: Optional[AbstractIntervenorInput] = None,
    default_active: bool = False,
) -> Tuple[PyTree["AbstractTask"], PyTree[AbstractModel[StateT]]]:
    """Adds an intervention to a model and a task.

    !!! Note ""
        Accepts either an intervenor instance, or an intervenor class. Passing
        an intervenor instance but no `intervenor_params`, the instance's
        `params` attribute is used as `intervenor_params`. This can be combined
        with the intervenor's `with_params` constructor to define the
        schedule. For example:

        ```python
        schedule_intervenor(
            tasks,
            models,
            CurlField.with_params(
                amplitude=lambda trial_spec, *, key: jr.normal(key, (1,)),
                active=True,
            ),
            ...
        )
        ```

        Passing an intervenor class and an `intervenor_params`, an instance
        will be constructed from the two.

        Passing an intervenor instance *and* an `intervenor_params`, the
        instance's `params` will be replaced with the `intervenor_params`
        before adding to the model.

        Passing an intervenor class but no `intervenor_params`, an error is
        raised due to insufficient information to schedule the intervention.

        Passing a value for `intervenor_params_validation` allows for separate
        control over the intervention schedule for the task's validation set.

    Arguments:
        tasks: The task(s) in whose trials the intervention will be scheduled
        models: The model(s) to which the intervention will be added
        intervenor: The intervenor (or intervenor class) to schedule.
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
            it (which may be `model` itself) to which to add the intervenors.
        stage_name: The name of the stage in `where(model).model_spec` at the end of
            which the intervenor will be executed. If `None`, executes before the
            first stage of the model.
        validation_same_schedule: Whether the interventions should be scheduled
            in the same way for the validation set as for the training set.
        intervenor_params: The parameters of to the intervenor, which may be
            constants, or callables that are used by `task` to construct the
            parameters for the intervention on each trial.
        intervenor_params_validation: Same as `intervenor_input`, but for the
            task's validation set. Overrides `validation_same_schedule`.
        default_active: If the intervenor added to the model should have
            `active=True` by default, so that the intervention will be
            turned on even if the intervenor doesn't explicitly receive values
            for its parameters.
    """
    intervenor_: AbstractIntervenor
    if isinstance(intervenor, type(AbstractIntervenor)):
        if intervenor_params is None:
            raise ValueError("Must pass intervenor_params if intervenor is a class")
        intervenor_ = intervenor(params=intervenor_params)  # type: ignore
    elif isinstance(intervenor, AbstractIntervenor):
        intervenor_ = intervenor
    else:
        raise ValueError("intervenor must be an AbstractIntervenor instance or class")

    # A unique label is needed because `AbstractTask` uses a single dict to
    # pass intervention parameters for all intervenors in an `AbstractStagedModel`,
    # regardless of where they appear in the model hierarchy.
    #
    # The label should be unique across all models and tasks that the intervenor
    # will be registered with.
    invalid_labels_models = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda model: model.step._all_intervenor_labels,
            models,
            is_leaf=is_module,  # AbstractModel
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels_tasks = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda task: tuple(task.intervention_specs.keys()),
            tasks,
            is_leaf=is_module,  # AbstractTask
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels = set(invalid_labels_models + invalid_labels_tasks)
    label = get_unique_label(intervenor_.label, invalid_labels)

    # Construct specification-intervenors
    intervention_specs = {label: InterventionSpec(
        intervenor=intervenor_,
        where=where,
        stage_name=stage_name,
        default_active=default_active,
    )}

    if intervenor_params_validation is not None:
        intervention_specs_validation = {label: InterventionSpec(
            intervenor=eqx.tree_at(
                lambda intervenor: intervenor.params,
                intervenor_,
                intervenor_params_validation,
            ),
            where=where,
            stage_name=stage_name,
            default_active=default_active,
        )}
    elif validation_same_schedule:
        intervention_specs_validation = intervention_specs
    else:
        intervention_specs_validation = dict()

    # Add the spec intervenors to every task in `tasks`
    tasks = jax.tree_map(
        lambda task: eqx.tree_at(
            lambda task: (task.intervention_specs, task.intervention_specs_validation),
            task,
            (
                task.intervention_specs | intervention_specs,
                task.intervention_specs_validation | intervention_specs_validation,
            ),
        ),
        tasks,
        is_leaf=is_module,  # AbstractTask
    )

    # Apply the unique label to the intervenor to be added to the model.
    intervenor_relabeled = eqx.tree_at(lambda x: x.label, intervenor_, label)

    # Construct the intervenor with default parameters, to add to the model.
    # TODO: Should we let the user pass a `default_intervenor_params`?
    key_example = jax.random.PRNGKey(0)
    # Assume that all the tasks are compatible with the way trial specs are used
    # to generate trial-by-trial intervenor parameters.
    task_example = jax.tree_leaves(
        tasks, is_leaf=is_module  # AbstractTask
    )[0]
    trial_spec_example = task_example.get_train_trial(key_example)
    intervenor_defaults = eqx.tree_at(
        lambda intervenor: intervenor.params,
        intervenor_relabeled,
        _eval_intervenor_param_spec(
            # Prefer the validation parameters, if they exist.
            (intervention_specs | intervention_specs_validation)[label],
            trial_spec_example,
            key_example,
        )
    )

    intervenor_final = eqx.tree_at(
        lambda intervenor: intervenor.params.active,
        intervenor_defaults,
        default_active,
    )

    # Add the intervenor to all of the models.
    models = jax.tree_map(
        lambda model: add_intervenors(
            model,
            {stage_name: [intervenor_final]},
            where=where,
            scheduled=True,
        ),
        models,
        is_leaf=lambda x: isinstance(x, AbstractModel),
    )

    return tasks, models

