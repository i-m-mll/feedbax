"""Add-ins to a model that modify its state operations.

Intervenors are intended to be used with instances of types of `AbstractStagedModel`, to
patch state operations onto existing models. For example, it is common to take
an existing task and apply a certain force field on the biomechanical effector.
Instead of rewriting the biomechanical model itself, which would lead to a
proliferation of model classes with slightly different task conditions, we can
instead use `add_intervenor` to do surgery on the model using only a few lines
of cod. 

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
import logging
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
from jaxtyping import Array, ArrayLike, Float, PyTree

from feedbax.misc import get_unique_label
from feedbax.model import AbstractModel
from feedbax.state import AbstractState
from feedbax.task import AbstractTask
from feedbax.tree import tree_call

if TYPE_CHECKING:
    from feedbax.bodies import SimpleFeedback
    # from feedbax.model import AbstractModel
    from feedbax.mechanics.mechanics import MechanicsState
    from feedbax.networks import NetworkState
    from feedbax.staged import AbstractStagedModel


logger = logging.getLogger(__name__)


class AbstractIntervenorInput(eqx.Module):
    active: AbstractVar[bool]
    
    
StateT = TypeVar("StateT", bound=AbstractState)
InputT = TypeVar("InputT", bound=AbstractIntervenorInput)    


class AbstractIntervenor(eqx.Module, Generic[StateT, InputT]):
    """Base class for modules that intervene on a model's state.
    
    Concrete subclasses must define `_out` and `_in` class variables, which 
    are callables that return the substate to be updated and the substate to 
    use as input to the update method, respectively.
    
    Instances of the concrete subclasses of this class are intended to be
    registered with the `register_intervention` method of an instance of an
    `AbstractModel`, which causes the model to invoke the intervention 
    during model execution.
    
    Fields:
        params:
        in_where:
        out_where:
        operation: Which operation to use to combine the original and altered
            states. For example, an intervenor that clamps a state variable
            to a particular value should use an operation like 
            `lambda x, y: y` to replace the original with the altered state.
            On the other hand, an intervenor that 
        label:
    
    NOTE: 
    - Interestingly, this has the same call signature as `AbstractModel`.
      This might be relevant to generalizing the `AbstractModel` class and 
      the notion of model-model interactions.
        - Each "stage" of an `AbstractModel` is a modification of a substate.
    - `_get_updated_substate` in `AbstractClampIntervenor` and 
      `_get_substate_to_add` in `AbstractAdditiveIntervenor` serve a similar 
      role and have the same signature, but are defined separately for the 
      sake of clearer naming.
      
    TODO:
    - Subclass `AbstractModel`? Then what about `init` and `_step`?
    """
    
    params: AbstractVar[InputT]
    in_where: AbstractVar[Callable[[StateT], PyTree[ArrayLike, "T"]]]
    out_where: AbstractVar[Callable[[StateT], PyTree[ArrayLike, "S"]]]
    operation: AbstractVar[Callable[[ArrayLike, ArrayLike], ArrayLike]]
    label: AbstractVar[str]
    
    def __call__(self, input: InputT, state: StateT, *, key) -> StateT:
        """Return a modified state PyTree."""
        params = eqx.combine(input, self.params)
        
        return jax.lax.cond(
            params.active ,
            lambda: eqx.tree_at(
                self.out_where,
                state, 
                jax.tree_map(
                    lambda x, y: self.operation(x, y),
                    self.out_where(state),
                    self.intervene(params, state, key=key),
                ),
            ),
            lambda: state,
        )        
    
    @abstractmethod
    def intervene(
        params: InputT, 
        state: StateT, 
        *, 
        key: Optional[Array]
    ) -> PyTree[ArrayLike, "S"]:
        ...        
        
    @abstractclassmethod
    def with_params(cls, **kwargs) -> "AbstractIntervenor":
        """Constructor that takes parameters from `AbstractIntervenorInput`.
        """
        ...
    

class CurlFieldParams(AbstractIntervenorInput):
    amplitude: Optional[float] = 0.
    active: Optional[bool] = True
    

class CurlField(AbstractIntervenor["MechanicsState", CurlFieldParams]):
    """Apply a curl force field to an effector.
    
    Positive amplitude corresponds to a counterclockwise curl, in keeping
    with the conventional sign of rotation in the XY plane.
    
    By default, this is additive, i.e. it will contribute to the force on the 
    effector rather than override it.
    
    TODO:
    - Special case of `EffectorVelDepForceField`?
    """
    
    params: CurlFieldParams = CurlFieldParams()
    in_where: Callable[["MechanicsState"], Float[Array, '... ndim=2']] \
        = lambda state: state.effector.vel 
    out_where: Callable[["MechanicsState"], Float[Array, '... ndim=2']] \
        = lambda state: state.effector.force
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = lambda x, y: x + y
    label: str = "CurlField"
    
    @classmethod
    def with_params(
        cls,
        amplitude: Optional[float] = 0.,
        active: Optional[bool] = True,
        **kwargs,
    ) -> "CurlField":
        
        return cls(
            CurlFieldParams(amplitude=amplitude, active=active),
            **kwargs,
        )
    
    def intervene(
        self, 
        params: CurlFieldParams, 
        state: "MechanicsState", 
        *, 
        key: Optional[Array] = None 
    ):
        scale = params.amplitude * jnp.array([-1, 1]) 
        return scale * self.in_where(state)[..., ::-1]


class AddNoiseParams(AbstractIntervenorInput):
    amplitude: float = 1.0
    noise_func: Callable = jax.random.normal
    active: bool = True


class AddNoise(AbstractIntervenor[StateT, InputT]):
    """Add noise to a part of the state. 
    
    Default is standard normal noise.
    """
    params: AddNoiseParams = AddNoiseParams()
    in_where: Callable[[StateT], PyTree[Array, "T"]] = lambda state: state
    out_where: Callable[[StateT], PyTree[Array, "T"]] = lambda state: state
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = lambda x, y: x + y
    label: str = "AddNoise"
    
    @classmethod
    def with_params(
        cls,
        amplitude: float = 1.0,
        noise_func: Callable = jax.random.normal,
        active: bool = True,
        **kwargs,
    ) -> "AddNoise":
        return cls(
            AddNoiseParams(
                amplitude=amplitude, 
                noise_func=noise_func, 
                active=active
            ),
            **kwargs,
        )
    
    def intervene(
        self, 
        params: AddNoiseParams, 
        state: StateT, 
        *,
        key: Optional[Array] = None
    ):
        
        return jax.tree_map(
            lambda x: params.noise_func(
                shape=x.shape, 
                dtype=x.dtype,
            ) * params.amplitude,
            self.in_where(state),
        )


class NetworkIntervenorParams(AbstractIntervenorInput):
    active: bool = True
    unit_spec: Optional[PyTree] = None


class NetworkClamp(AbstractIntervenor["NetworkState", InputT]):
    """Replaces part of a network's state with constant values.

    NOTE:
    Originally this was intended to modify an arbitrary PyTree of network 
    states, which is why we use `tree_map` to replace parts of `network_state`
    with parts of `unit_spec`. However, currently it is only used to modify
    `NetworkState.hidden`, which is a single JAX array. Thus `unit_spec`
    should also be a single JAX array.

    Args:
        unit_spec (PyTree[Array]): A PyTree with the same structure as the
        network state, where all array values are NaN except for the constant
        values to be clamped.
    """
    
    params: NetworkIntervenorParams = NetworkIntervenorParams()
    in_where: Callable[["NetworkState"], PyTree[Array, "T"]] = lambda state: state.hidden 
    out_where: Callable[["NetworkState"], PyTree[Array, "T"]] = lambda state: state.hidden
    operation: Callable[[Array, Array], Array] = lambda x, y: y
    label: str = "NetworkClamp"
    
    @classmethod 
    def with_params(
        cls,
        active: bool = True,
        unit_spec: Optional[PyTree[Array, "T"]] = None,
        **kwargs,
    ) -> "NetworkClamp":
        
        return cls(
            NetworkIntervenorParams(active=active, unit_spec=unit_spec),
            **kwargs,
        )
    
    def intervene(
        self, 
        params: NetworkIntervenorParams, 
        state: "NetworkState",
        *,
        key: Optional[Array] = None
    ) -> PyTree[Array, "T"]:
        
        return jax.tree_map(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            state,
            params.unit_spec,
        )
        

class NetworkConstantInput(AbstractIntervenor["NetworkState", InputT]):
    """Adds a constant input to some network units.
    
    NOTE:
    Originally this was intended to modify an arbitrary PyTree of network 
    states. See `NetworkClamp` for more details.
    
    Args:
        unit_spec: A PyTree with the same structure as the
        network state, to be added to the network state.
    """    
    params: NetworkIntervenorParams = NetworkIntervenorParams()
    in_where: Callable[["NetworkState"], PyTree[Array, "T"]] \
        = lambda state: state.hidden 
    out_where: Callable[["NetworkState"], PyTree[Array, "T"]] \
        = lambda state: state.hidden    
    operation: Callable[[Array, Array], Array] = lambda x, y: x + y
    label: str = "NetworkConstantInput"
    
    @classmethod
    def with_params(
        cls,
        active: bool = True,
        unit_spec: Optional[PyTree[Array, "T"]] = None,
        **kwargs,
    ) -> "NetworkConstantInput":
        
        return cls(
            NetworkIntervenorParams(active=active, unit_spec=unit_spec),
            **kwargs,
        )
    
    def intervene(
        self, 
        params: NetworkIntervenorParams, 
        state: "NetworkState",
        *,
        key: Optional[Array] = None,
    ) -> PyTree[Array, "T"]:
        return jax.tree_map(jnp.nan_to_num, params.unit_spec)


class ConstantInputParams(AbstractIntervenorInput):
    active: bool = True
    amplitude: float = 1.0
    array_mask: Optional[PyTree] = None


class ConstantInput(AbstractIntervenor[StateT, InputT]):
    """Adds a constant input to a state array.
    
    Args:
        array_mask: 
    """    
    params: NetworkIntervenorParams = NetworkIntervenorParams()
    in_where: Callable[[StateT], PyTree[Array, "T"]] \
        = lambda state: state
    out_where: Callable[[StateT], PyTree[Array, "T"]] \
        = lambda state: state  
    operation: Callable[[Array, Array], Array] = lambda x, y: x + y
    label: str = "ConstantInput"
    
    @classmethod
    def with_params(
        cls,
        active: bool = True,
        amplitude: float = 1.0,
        array_mask: Optional[PyTree[Array, "T"]] = None,
        **kwargs,
    ) -> "ConstantInput":
        
        return cls(
            ConstantInputParams(
                active=active, 
                amplitude=amplitude,
                array_mask=array_mask,
            ),
            **kwargs,
        )
    
    def intervene(
        self, 
        params: ConstantInputParams, 
        state: StateT,
        *,
        key: Optional[Array] = None,
    ) -> PyTree[Array, "T"]:
        return params.amplitude * params.array_mask


def add_intervenor(
    model: "AbstractStagedModel[StateT]", 
    intervenor: AbstractIntervenor,
    stage_name: Optional[str] = None,
    **kwargs
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with an added intervenor.
    
    This is just a convenience for passing a single intervenor to `add_intervenor`.
    """
    if stage_name is not None:
        return add_intervenors(model, {stage_name: [intervenor]}, **kwargs)
    else:
        return add_intervenors(model, [intervenor], **kwargs)


def add_intervenors(
    model: "AbstractStagedModel[StateT]", 
    intervenors: Union[Sequence[AbstractIntervenor],
                       Mapping[str, Sequence[AbstractIntervenor]]], 
    where: Callable[["AbstractStagedModel[StateT]"], Any] = lambda model: model.step,
    keep_existing: bool = True,
    *,
    key: Optional[jax.Array] = None
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with added intervenors.
    
    Intervenors are added to `where(model)`, which by default is 
    just `model` itself.
    
    If intervenors are passed as a sequence, they are added to the first stage
    specified in `where(model).model_spec`, otherwise they should be passed in
    a dict where the keys refer to particular stages in the model spec.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
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
    model: "SimpleFeedback"
) -> "SimpleFeedback":
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)
    

def schedule_intervenor(
    tasks: PyTree["AbstractTask"],
    models: PyTree["AbstractModel[StateT]"],  # not AbstractStagedModel because it might be wrapped in `Iterator`
    intervenor: AbstractIntervenor | Type[AbstractIntervenor],
    # TODO: intervenor_validation
    where: Callable[["AbstractModel[StateT]"], Any] = lambda model: model.step,
    validation_same_schedule: bool = True,
    intervenor_spec: Optional[AbstractIntervenorInput] = None,  #! wrong! distribution functions are allowed. only the PyTree structure is the same
    intervenor_spec_validation: Optional[AbstractIntervenorInput] = None,
    default_active: bool = False,
) -> Tuple["AbstractTask", "AbstractModel[StateT]"]:
    """Adds an intervention to a model and a task.
    
    The user can pass an intervenor instance or an intervenor class. If they
    pass an intervenor instance but no `intervenor_spec`, the instance's
    `params` attribute will be used as `intervenor_spec`. This can be combined
    with the intervenor's `with_params` constructor to quickly define the 
    schedule. For example:
    
        schedule_intervenor(
            tasks,
            models,
            CurlField.with_params(
                amplitude=lambda trial_spec, *, key: jr.normal(key, (1,)),
                active=True,
            ),
            ...
        )
    
    If they pass an intervenor instance and an `intervenor_spec`, the
    instance's `params` will be replaced with the `intervenor_spec` before
    adding it to the model. If they pass an intervenor class but no
    `intervenor_spec`, an error is raised due to insufficient information to
    schedule the intervention.
    
    Args:
        intervenor: The intervenor (or intervenor class) to schedule
        task: The task(s) in whose trials the intervention will be scheduled
        model: The model(s) to which the intervention will be added
        where: Where in the model to insert the intervention
        validation_same_schedule: Whether the interventions should be scheduled
            in the same way for the validation set as for the training set.
        intervenor_spec: The input to the intervenor, which may be 
            a constant, or a callable that is used by `task` to construct the 
            intervention parameters for each trial.
        intervenor_spec_validation: Same as `intervenor_spec`, but for the 
            task's validation set. Overrides `validation_same_schedule`.
        default_active: If the intervenor added to the model should have 
            `active=True` by default, so that the intervention will be 
            turned on even if the intervenor doesn't explicitly receives values
            for its parameters. 
        
    TODO:
    - Generalize to scheduling multiple intervenors simultaneously.
        - Might want to check out how Equinox handles overloaded/PyTree argument types, 
          e.g. for `in_axes` of `eqx.filter_vmap` or something
    - If `validation_same_schedule` is `False` and no 
      `intervenor_spec_validation` is provided, what should happen? Presumably
      we should set constant `active=False` for the validation set.
    """
    
    # TODO: This is temporary, since we currently need to pass intervenor in a `dict`
    # if we want to choose the model stage it's added before
    if isinstance(intervenor, Mapping):
        # Assume a single intervenor
        intervenor_stage = next(iter(intervenor.keys()))
        intervenor = next(iter(intervenor.values()))
    else:
        intervenor_stage = None
    
    # A unique label is needed because `AbstractTask` uses a single dict to 
    # pass intervention parameters for all intervenors in an `AbstractStagedModel`,
    # regardless of where they appear in the model hierarchy.
    #
    # The label should be unique across all models and tasks that the intervenor 
    # will be registered with.
    invalid_labels_models = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda model: model._step._all_intervenor_labels, 
            models, 
            is_leaf=lambda x: isinstance(x, AbstractModel)
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels_tasks = jax.tree_util.tree_reduce(   
        lambda x, y: x + y,
        jax.tree_map(
            lambda task: tuple(task.intervention_spec.keys()), 
            tasks, 
            is_leaf=lambda x: isinstance(x, AbstractTask),
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels = set(invalid_labels_models + invalid_labels_tasks)
    label = get_unique_label(intervenor.label, invalid_labels)    
    
    # Construct training and validation intervention specs
    if intervenor_spec is not None:
        intervention_specs = {
            label: intervenor_spec
        }
    else:
        if isinstance(intervenor, type(AbstractIntervenor)):
            raise ValueError("Must pass intervenor_spec if intervenor is a class")
        intervention_specs = {
            label: intervenor.params
        }
        
    if intervenor_spec_validation is not None:
        intervention_specs_validation = {label: intervenor_spec_validation}
    elif validation_same_schedule:
        intervention_specs_validation = intervention_specs
    else:
        intervention_specs_validation = dict()

    # Add the intervention specs to every task in `tasks`
    tasks = jax.tree_map(
        lambda task: eqx.tree_at(
            lambda task: (
                task.intervention_spec, 
                task.intervention_spec_validation
            ),
            task,
            (
                task.intervention_spec | intervention_specs,
                task.intervention_spec_validation | intervention_specs_validation,
            ),
        ),
        tasks, 
        is_leaf=lambda x: isinstance(x, AbstractTask),
    )
    
    # Instantiate the intervenor if necessary, give it the unique label, 
    # and make sure it has a single set of default param values.
    if isinstance(intervenor, type(AbstractIntervenor)):
        intervenor = intervenor(params=intervention_specs[label])

    intervenor_relabeled = eqx.tree_at(lambda x: x.label, intervenor, label)   
    
    # TODO: Should we let the user pass a `default_intervenor_spec`?
    key_example = jax.random.PRNGKey(0)
    task_example = jax.tree_leaves(
        tasks, 
        is_leaf=lambda x: isinstance(x, AbstractTask)
    )[0]
    trial_spec_example = task_example.get_train_trial(key_example)
    
    # Use the validation spec to construct the defaults.
    intervenor_defaults = eqx.tree_at(
        lambda intervenor: intervenor.params,
        intervenor_relabeled,
        tree_call(
            intervention_specs_validation[label], 
            trial_spec_example, 
            key=key_example
        ),
    )
    
    intervenor_final = eqx.tree_at(
        lambda intervenor: intervenor.params.active,
        intervenor_defaults,
        default_active,
    )
    
    # TODO: This should be temporary. See TODO at start of function
    if intervenor_stage is None:
        intervenors = [intervenor_final]
    else:
        intervenors = {intervenor_stage: [intervenor_final]}        
    
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