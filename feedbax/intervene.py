"""Add-ins to a model that modify its state operations.

Intervenors are intended to be used with instances of types of `AbstractStagedModel`, to
patch state operations onto existing models. For example, it is common to take
an existing task and apply a certain force field on the biomechanical effector.
Instead of rewriting the biomechanical model itself, which would lead to a
proliferation of model classes with slightly different task conditions, we can
instead use `add_intervenor` to do surgery on the model using only a few lines
of code. 

Likewise, since the exact parameters of an intervention often change between
(or within) task trials, it is convenient to be able to specify the distribution 
of these parameters across trials. The function `schedule_intervenors` does 
simultaneous surgery on task and model instances so that the model interventions
are paired with their trial-by-trial parameters, as provided by the task.

TODO:
- Could just use `operation` to distinguish `NetworkClamp` from 
  `NetworkConstantInput`. Though might need to rethink the NaN unit spec thing
- Could enforce that `out_where = in_where` for some intervenors, in particular 
  `AddNoise`, `NetworkClamp`, and `NetworkConstantInput`. These are intended to 
  make modifications to a part of the state, not to transform between parts.
  
:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractclassmethod, abstractmethod
from collections.abc import Mapping, Sequence, Callable
import copy
from dataclasses import fields
import logging
import operator as op
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import equinox as eqx
from equinox import AbstractVar, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree

from feedbax.misc import get_unique_label
from feedbax.model import AbstractModel
from feedbax.state import AbstractState, StateT
from feedbax._tree import tree_call

if TYPE_CHECKING:
    # from feedbax.model import AbstractModel
    from feedbax.mechanics.mechanics import MechanicsState
    from feedbax.nn import NetworkState
    from feedbax._staged import AbstractStagedModel
    from feedbax.task import AbstractTask


logger = logging.getLogger(__name__)


class AbstractIntervenorInput(eqx.Module):
    """Base class for PyTrees of intervention parameters.

    Attributes:
        active: Whether the intervention is active.
    """

    active: AbstractVar[bool]


InputT = TypeVar("InputT", bound=AbstractIntervenorInput)


class AbstractIntervenor(eqx.Module, Generic[StateT, InputT]):
    """Base class for modules that intervene on a model's state.

    Attributes:
        params: Default intervention parameters.
        in_where: Takes an instance of the model state, and returns the substate
          corresponding to the intervenor's input.
        out_where: Takes an instance of the model state, and returns the substate
          corresponding to the intervenor's output. In many cases, `out_where` will
          be the same as `in_where`.
        operation: Which operation to use to combine the original and altered
          `out_where` substates. For example, an intervenor that clamps a state
          variable to a particular value should use an operation like `lambda x, y: y`
          to replace the original with the altered state. On the other hand, an
          additive intervenor would use the equivalent of `lambda x, y: x + y`.
        label: The intervenor's label.
    """

    params: AbstractVar[InputT]
    in_where: AbstractVar[Callable[[StateT], PyTree[ArrayLike, "T"]]]
    out_where: AbstractVar[Callable[[StateT], PyTree[ArrayLike, "S"]]]
    operation: AbstractVar[Callable[[ArrayLike, ArrayLike], ArrayLike]]
    label: AbstractVar[str]

    @classmethod
    def with_params(cls, **kwargs) -> "AbstractIntervenor[StateT, InputT]":
        """Constructor that accepts field names of `InputT` as keywords.

        This is a convenience so we don't need to import the parameter class,
        to instantiate an intervenor class it is associated with.

        !!! Example
            ```python
            CurlField.with_params(amplitude=10.0, label="MyCurlField")
            ```
        """
        param_cls = next((f for f in fields(cls) if f.name == "params")).type
        param_fields = [f.name for f in fields(param_cls)]

        return cls(
            param_cls(**{k: v for k, v in kwargs.items() if k in param_fields}),
            **{k: v for k, v in kwargs.items() if k not in param_fields},
        )

    def __call__(self, input: InputT, state: StateT, *, key: PRNGKeyArray) -> StateT:
        """Return a state PyTree modified by the intervention.

        Arguments:
            input: PyTree of intervention parameters. If any leaves are `None`, they
              will be replaced by the corresponding leaves of `self.params`.
            state: The model state to be intervened upon.
            key: A key to provide randomness for the intervention.
        """
        params: InputT = eqx.combine(input, self.params)

        return jax.lax.cond(
            params.active,
            lambda: eqx.tree_at(  # Replace the `out_where` substate
                self.out_where,
                state,
                jax.tree_map(  # With the combined original and altered substates
                    lambda x, y: self.operation(x, y),
                    self.out_where(state),
                    self.transform(
                        params,
                        self.in_where(state),
                        key=key,
                    ),
                ),
            ),
            lambda: state,
        )

    @abstractmethod
    def transform(
        params: InputT,
        substate_in: PyTree[ArrayLike, "T"],
        *,
        key: Optional[PRNGKeyArray],
    ) -> PyTree[ArrayLike, "S"]:
        """Transforms the input substate to produce an altered output substate."""
        ...


class CurlFieldParams(AbstractIntervenorInput):
    """Parameters for a curl force field.

    Attributes:
        amplitude: The amplitude of the force field. Negative is clockwise, positive
          is counterclockwise.
        active: Whether the force field is active.
    """

    amplitude: float = 0.0
    active: bool = True


class CurlField(AbstractIntervenor["MechanicsState", CurlFieldParams]):
    """Apply a curl force field to a mechanical effector.

    Attributes:
        params: Default curl field parameters.
        in_where: Returns the substate corresponding to the effector's velocity.
        out_where: Returns the substate corresponding to the force on the effector.
        operation: How to combine the effector force due to the curl field,
          with the existing force on the effector. Default is addition.
        label: The intervenor's label.
    """

    params: CurlFieldParams = CurlFieldParams()
    in_where: Callable[["MechanicsState"], Float[Array, "... ndim=2"]] = (
        lambda state: state.effector.vel
    )
    out_where: Callable[["MechanicsState"], Float[Array, "... ndim=2"]] = (
        lambda state: state.effector.force
    )
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add
    label: str = "CurlField"

    def transform(
        self,
        params: CurlFieldParams,
        substate_in: Float[Array, "ndim=2"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "ndim=2"]:
        """Transform velocity into curl force."""
        scale = params.amplitude * jnp.array([-1, 1])
        return scale * substate_in[..., ::-1]


class AddNoiseParams(AbstractIntervenorInput):
    """Parameters for adding noise to a state.

    Attributes:
        scale: Constant factor to multiply the noise samples.
        noise_func: A function that returns noise samples.
        active: Whether the intervention is active.
    """

    scale: float = 1.0
    noise_func: Callable = jax.random.normal
    active: bool = True


class AddNoise(AbstractIntervenor[StateT, InputT]):
    """Add noise to a part of the state.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate to which noise is added.
        operation: How to combine the noise with the substate. Default is
          addition.
        label: The intervenor's label.
    """

    params: AddNoiseParams = AddNoiseParams()
    out_where: Callable[[StateT], PyTree[Array, "T"]] = lambda state: state
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add
    label: str = "AddNoise"

    @property
    def in_where(self) -> Callable[[StateT], PyTree[Array, "T"]]:
        return self.out_where

    def transform(
        self,
        params: AddNoiseParams,
        substate_in: PyTree[Array, "T"],
        *,
        key: Optional[PRNGKeyArray],
    ) -> PyTree[Array, "T"]:
        """Return a PyTree of scaled noise arrays with the same structure/shapes as
        `substate_in`."""
        return jax.tree_map(
            lambda x:  params.scale * params.noise_func(
                key,
                shape=x.shape,
                dtype=x.dtype,
            )
            substate_in,
        )


class NetworkIntervenorParams(AbstractIntervenorInput):
    """Parameters for interventions on network unit activity.

    Attributes:
        unit_spec: A PyTree of arrays with the same tree structure and array shapes
          as the substate of the network to be intervened upon, specifying the
          unit-wise activities that constitute the perturbation.
        active: Whether the intervention is active.

    !!! Note ""
        Note that `unit_spec` may be a single array—which is just a single-leaf PyTree
        of arrays—when `out_where` of the intervenor is also a single array.
    """

    unit_spec: Optional[PyTree] = None
    active: bool = True


class NetworkClamp(AbstractIntervenor["NetworkState", InputT]):
    """Clamps some of a network's units' activities to given values.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate of arrays giving the activity of the units
          whose activities may be clamped.
        operation: How to combine the original and clamped unit activities. Default
          is to replace the original with the altered.
        label: The intervenor's label.
    """

    params: NetworkIntervenorParams = NetworkIntervenorParams()
    out_where: Callable[["NetworkState"], PyTree[Array, "T"]] = (
        lambda state: state.hidden
    )
    operation: Callable[[Array, Array], Array] = lambda x, y: y
    label: str = "NetworkClamp"

    @property
    def in_where(self) -> Callable[[StateT], PyTree[Array, "T"]]:
        return self.out_where

    def transform(
        self,
        params: NetworkIntervenorParams,
        substate_in: PyTree[Array, "T"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array, "T"]:

        return jax.tree_map(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            substate_in,
            params.unit_spec,
        )


class NetworkConstantInput(AbstractIntervenor["NetworkState", InputT]):
    """Adds a constant input to some network units.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate of arrays giving the activity of the units
          to which a constant input may be added.
        operation: How to combine the original and altered unit activities. Default
          is addition.
        label: The intervenor's label.
    """

    params: NetworkIntervenorParams = NetworkIntervenorParams()
    out_where: Callable[["NetworkState"], PyTree[Array, "T"]] = (
        lambda state: state.hidden
    )
    operation: Callable[[Array, Array], Array] = op.add
    label: str = "NetworkConstantInput"

    @property
    def in_where(self) -> Callable[[StateT], PyTree[Array, "T"]]:
        return self.out_where

    def transform(
        self,
        params: NetworkIntervenorParams,
        substate_in: "NetworkState",
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array, "T"]:
        return jax.tree_map(jnp.nan_to_num, params.unit_spec)


class ConstantInputParams(AbstractIntervenorInput):
    """Parameters for adding a constant input to a state array.

    Attributes:
        scale: Constant factor to multiply the input.
        arrays: A PyTree of arrays with the same tree structure and array shapes
          as the substate of the state to be intervened upon, specifying the
          constant input to be added.
        active: Whether the intervention is active.
    """

    scale: float = 1.0
    arrays: Optional[PyTree] = None
    active: bool = True


class ConstantInput(AbstractIntervenor[StateT, InputT]):
    """Adds a constant input to a state array.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate of arrays to which a constant input is added.
        operation: How to combine the original and altered substates. Default is addition.
        label: The intervenor's label.
    """

    params: NetworkIntervenorParams = NetworkIntervenorParams()
    out_where: Callable[[StateT], PyTree[Array, "T"]] = lambda state: state
    operation: Callable[[Array, Array], Array] = op.add
    label: str = "ConstantInput"

    @property
    def in_where(self) -> Callable[[StateT], PyTree[Array, "T"]]:
        return self.out_where

    def transform(
        self,
        params: ConstantInputParams,
        substate_in: PyTree[Array, "T"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> PyTree[Array, "T"]:
        return params.scale * params.arrays


def add_intervenor(
    model: "AbstractStagedModel[StateT]",
    intervenor: AbstractIntervenor,
    stage_name: Optional[str] = None,
    **kwargs,
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with an added intervenor.

    !!! Note ""
        This is a helper for calling `add_intervenors` with a single intervenor.

    Arguments:
        model: The model to which the intervenor will be added.
        intervenor: The intervenor to add.
        stage_name: The stage named in `model.model_spec` to which the intervenor will
          be added.
        kwargs: Additional keyword arguments to
          [`add_intervenors`][feedbax.intervene.add_intervenors].
    """
    if stage_name is not None:
        return add_intervenors(model, {stage_name: [intervenor]}, **kwargs)
    else:
        return add_intervenors(model, [intervenor], **kwargs)


StateS = TypeVar("StateS", AbstractState, Array)


def add_intervenors(
    model: "AbstractStagedModel[StateT]",
    intervenors: Union[
        Sequence[AbstractIntervenor[StateS, InputT]],
        Mapping[str, Sequence[AbstractIntervenor[StateS, InputT]]],
    ],
    where: Callable[
        [AbstractModel[StateT]], "AbstractStagedModel[StateS]"
    ] = lambda model: model.step,
    keep_existing: bool = True,
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with added intervenors.

    Arguments:
        model: The model to which the intervenors will be added.
        intervenors: The intervenors to add. May be a sequence of intervenors to
          add to the first stage in `model.model_spec`, or a mapping from stage
          name to a sequence of intervenors to add to that stage.
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
          it (which may be `model` itself) to which to add the intervenors.
        keep_existing: Whether to keep the existing intervenors belonging directly to
          the instance of `AbstractStagedModel` to which the new intervenors are added.
          If `True`, the new intervenors are appended to the existing ones; if `False`,
          the old intervenors are replaced.
    """
    if keep_existing:
        existing_intervenors = where(model).intervenors

        if isinstance(intervenors, Sequence):
            # If a sequence is given, append to the first stage.
            first_stage_label = next(iter(existing_intervenors))
            intervenors_dict = eqx.tree_at(
                lambda intervenors: intervenors[first_stage_label],
                existing_intervenors,
                existing_intervenors[first_stage_label] + list(intervenors),
            )
        elif isinstance(intervenors, dict):
            intervenors_dict = copy.deepcopy(existing_intervenors)
            for label, new_intervenors in intervenors.items():
                intervenors_dict[label] += list(new_intervenors)
        else:
            raise ValueError("intervenors not a sequence or dict of sequences")

    for k in intervenors_dict:
        if k not in where(model).model_spec:
            raise ValueError(f"{k} is not a valid model stage for intervention")

    return eqx.tree_at(
        lambda model: where(model).intervenors,
        model,
        intervenors_dict,
    )


def remove_intervenors(
    model: AbstractModel,
    where: Callable[[AbstractModel], PyTree] = lambda model: model,
) -> AbstractModel:
    """Return a model with all intervenors removed."""
    return eqx.tree_at(
        where,
        model,
        jax.tree_map(
            lambda model: add_intervenors(
                model,
                intervenors={stage: [] for stage in model.model_spec},
                keep_existing=False,
            ),
            where(model),
            # Can't do `isinstance(x, AbstractModel)` because of circular import
            is_leaf=lambda x: getattr(x, "model_spec", None) is not None,
        ),
    )


class TimeSeriesParam(eqx.Module):
    """Wraps intervenor parameters that should be interpreted as time series.

    Attributes:
        param: The parameter to interpret as a time series.
    """

    param: Array

    def __call__(self):
        """Return the wrapped parameter."""
        return self.param


def schedule_intervenor(
    tasks: PyTree["AbstractTask"],
    models: PyTree[AbstractModel[StateT]],
    intervenor: AbstractIntervenor | Type[AbstractIntervenor],
    # TODO: intervenor_validation
    where: Callable[[AbstractModel[StateT]], Any] = lambda model: model,
    stage_name: Optional[str] = None,
    validation_same_schedule: bool = True,
    intervention_spec: Optional[
        AbstractIntervenorInput
    ] = None,  #! wrong! distribution functions are allowed. only the PyTree structure is the same
    intervention_spec_validation: Optional[AbstractIntervenorInput] = None,
    default_active: bool = False,
) -> Tuple["AbstractTask", AbstractModel[StateT]]:
    """Adds an intervention to a model and a task.

    !!! Note ""
        Accepts either an intervenor instance, or an intervenor class. Passing
        an intervenor instance but no `intervention_spec`, the instance's
        `params` attribute is used as `intervention_spec`. This can be combined
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

        Passing an intervenor class and an `intervention_spec`, an instance
        will be constructed from the two.

        Passing an intervenor instance *and* an `intervention_spec`, the
        instance's `params` will be replaced with the `intervention_spec`
        before adding to the model.

        Passing an intervenor class but no `intervention_spec`, an error is
        raised due to insufficient information to schedule the intervention.

        Passing a value for `intervention_spec_validation` allows for separate
        control over the intervention schedule for the task's validation set.

    Arguments:
        tasks: The task(s) in whose trials the intervention will be scheduled
        models: The model(s) to which the intervention will be added
        intervenor: The intervenor (or intervenor class) to schedule.
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
          it (which may be `model` itself) to which to add the intervenors.
        stage_name: The name of the stage in `where(model).model_spec` to which to
          add the intervenor. Defaults to the first stage.
        validation_same_schedule: Whether the interventions should be scheduled
          in the same way for the validation set as for the training set.
        intervention_spec: The input to the intervenor, which may be
          a constant, or a callable that is used by `task` to construct the
          intervention parameters for each trial.
        intervention_spec_validation: Same as `intervention_spec`, but for the
          task's validation set. Overrides `validation_same_schedule`.
        default_active: If the intervenor added to the model should have
          `active=True` by default, so that the intervention will be
          turned on even if the intervenor doesn't explicitly receive values
          for its parameters.
    """

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
            is_leaf=lambda x: isinstance(x, eqx.Module),  # AbstractModel
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels_tasks = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda task: tuple(task.intervention_specs.keys()),
            tasks,
            is_leaf=lambda x: isinstance(x, eqx.Module),  # AbstractTask
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels = set(invalid_labels_models + invalid_labels_tasks)
    label = get_unique_label(intervenor.label, invalid_labels)

    # Construct training and validation intervention specs
    if intervention_spec is not None:
        intervention_specs = {label: intervention_spec}
    else:
        if isinstance(intervenor, type(AbstractIntervenor)):
            raise ValueError("Must pass intervention_spec if intervenor is a class")
        intervention_specs = {label: intervenor.params}

    if intervention_spec_validation is not None:
        intervention_specs_validation = {label: intervention_spec_validation}
    elif validation_same_schedule:
        intervention_specs_validation = intervention_specs
    else:
        intervention_specs_validation = dict()

    # Add the intervention specs to every task in `tasks`
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
        is_leaf=lambda x: isinstance(x, eqx.Module),  # AbstractTask
    )

    # Instantiate the intervenor if necessary, give it the unique label,
    # and make sure it has a single set of default param values.
    if isinstance(intervenor, type(AbstractIntervenor)):
        intervenor = intervenor(params=intervention_specs[label])

    intervenor_relabeled = eqx.tree_at(lambda x: x.label, intervenor, label)

    # TODO: Should we let the user pass a `default_intervention_spec`?
    key_example = jax.random.PRNGKey(0)
    task_example = jax.tree_leaves(
        tasks, is_leaf=lambda x: isinstance(x, eqx.Module)  # AbstractTask
    )[0]
    trial_spec_example = task_example.get_train_trial(key_example)

    # Use the validation spec to construct the defaults.
    intervenor_defaults = eqx.tree_at(
        lambda intervenor: intervenor.params,
        intervenor_relabeled,
        # We apply `tree_call` twice:
        #   1. evaluate any lambdas;
        #   2. unwrap any `TimeSeriesParam` instances.
        tree_call(
            tree_call(
                intervention_specs_validation[label],
                trial_spec_example,
                key=key_example,
            )
        ),
    )

    intervenor_final = eqx.tree_at(
        lambda intervenor: intervenor.params.active,
        intervenor_defaults,
        default_active,
    )

    if stage_name is None:
        intervenors = [intervenor_final]
    else:
        intervenors = {stage_name: [intervenor_final]}

    # Add the intervenor to all of the models.
    models = jax.tree_map(
        lambda model: add_intervenors(
            model,
            intervenors,
            where=where,
        ),
        models,
        is_leaf=lambda x: isinstance(x, AbstractModel),
    )

    return tasks, models
