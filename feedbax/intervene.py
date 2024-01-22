"""Modules that modify parts of PyTrees.

TODO:
- The type signature is backwards, compared to `AbstractModel`.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

#! Equinox 0.11.2 doesn't support stringified annotations of abstract vars
#! This is why we can't properly annotate `_in` and `_out` fields below.
# from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence, Callable
import copy
import logging
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar, Union, Tuple

import equinox as eqx
from equinox import AbstractVar, field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from feedbax.utils import get_unique_label

if TYPE_CHECKING:
    from feedbax.model import (
        AbstractModel, 
        AbstractStagedModel, 
        SimpleFeedback,
    )
    from feedbax.mechanics.mechanics import MechanicsState
    from feedbax.networks import NetworkState
    from feedbax.task import AbstractTask

from feedbax.state import AbstractState, CartesianState2D


logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractState)

class AbstractIntervenorInput(eqx.Module):
    active: AbstractVar[bool]
    
    
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
    - Should `Intervenor` only operate on a specific array in a state PyTree?    
      That is, should `_out` and `_in` return `Array` instead of `PyTree`?
    """
    
    params: AbstractVar[InputT]
    in_where: AbstractVar[Callable[[StateT], PyTree]]
    out_where: AbstractVar[Callable[[StateT], PyTree]]
    label: AbstractVar[str]
    
    def __call__(self, input, state, *, key):
        """Return a modified state PyTree."""
        params = eqx.combine(input, self.params)
        return eqx.tree_at(
            self.out_where,
            state, 
            self.intervene(params, state)
        )
    
    @abstractmethod
    def intervene(params, state):
        ...        
    

class CurlForceFieldParams(AbstractIntervenorInput):
    amplitude: Optional[float] = 0.
    active: Optional[bool] = True
    

class CurlForceField(AbstractIntervenor["MechanicsState", CurlForceFieldParams]):
    """Apply a curl force field to an effector.
    
    Positive amplitude corresponds to a counterclockwise curl, in keeping
    with the conventional sign of rotation in the XY plane.
    
    TODO:
    - Special case of `EffectorVelDepForceField`?
    """
    
    params: CurlForceFieldParams = CurlForceFieldParams()
    in_where: Callable[["MechanicsState"], Float[Array, '... ndim=2']] = lambda state: state.effector.vel 
    out_where: Callable[["MechanicsState"], Float[Array, '... ndim=2']] = lambda state: state.effector.force
    label: str = "CurlForceField"
    
    def intervene(self, params: CurlForceFieldParams, state: "MechanicsState"):
        return params.amplitude * jnp.array([-1, 1]) * self.in_where(state)[..., ::-1]
    
    # amplitude: float   # TODO: allow asymmetry 
    # _scale: jax.Array
    
    # label: str = "effector_curl_field"
    
    # in_where: Callable[[MechanicsState], PyTree]
    # lambda self, tree: tree.effector.vel 
    # _out = lambda self, tree: tree.effector.force
        
    # def __init__(self, amplitude: float = 1.0):
    #     self.amplitude = amplitude
    #     self._scale = amplitude * jnp.array([-1, 1])

    # def _get_substate_to_add(self, vel: Array, *, key: jax.Array):
    #     """Returns curl forces."""
    #     return self._scale * vel[..., ::-1]  
    
    
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
    where: Callable[["AbstractStagedModel[StateT]"], Any] = lambda model: model,
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
    intervenor: AbstractIntervenor, # TODO: | Type[AbstractIntervenor],
    intervenor_spec: AbstractIntervenorInput,  #! wrong! distribution functions are allowed. only the PyTree structure is the same
    task: "AbstractTask",
    model: "AbstractModel[StateT]",  # not AbstractStagedModel because it might be wrapped in `Iterator`
    model_where: Callable[["AbstractModel[StateT]"], Any],
    validation_same_schedule: bool = True,
    intervenor_spec_validation: Optional[AbstractIntervenorInput] = None,
) -> Tuple["AbstractTask", "AbstractModel[StateT]"]:
    """Adds an intervention to a model and a task.
    
    Args:
        intervenor: The intervenor (or intervenor class) to schedule
        intervenor_spec: The input to the intervenor, which may be 
            a constant or a stochastic per-trial callable
        validation_same_schedule: Whether the interventions should be scheduled
            in the same way for the validation set as for the training set.
        intervenor_spec_validation: Same as `intervenor_spec`, but for the 
            task's validation set. Only applies if `validation_same_schedule`
            is `False`. 
        task: The task in whose trials the intervention will be scheduled
        model: The model to which the intervention will be added
        model_where: Where in the model to insert the intervention
        
    TODO:
    - If `validation_same_schedule` is `False` and no 
      `intervenor_spec_validation` is provided, what should happen? Presumably
      we should set constant `active=False` for the validation set.
    """
    
    label = get_unique_label(
        intervenor.label, 
        invalid_labels=model._step._all_intervenor_labels
    )
    
    # if isinstance(intervenor, type(AbstractIntervenor)):
    #     intervenor = intervenor(tree_call(intervenor_input_spec, ...))
    
    intervenor_relabeled = eqx.tree_at(lambda x: x.label, intervenor, label)        
    
    intervention_spec = {
        label: intervenor_spec
    }
    
    task = eqx.tree_at(
        lambda task: task.intervention_spec,
        task,
        task.intervention_spec | intervention_spec,
    )   
    
    if validation_same_schedule:
        task = eqx.tree_at(
            lambda task: task.intervention_spec_validation,
            task,
            task.intervention_spec_validation | intervention_spec,
        )
    elif intervenor_spec_validation is not None:
        task = eqx.tree_at(
            lambda task: task.intervention_spec_validation,
            task,
            task.intervention_spec_validation | {
                label: intervenor_spec_validation
            },
        )
    
    model = add_intervenors(model, [intervenor_relabeled], where=model_where)
    
    return task, model