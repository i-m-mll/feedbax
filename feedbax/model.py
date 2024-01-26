"""Base classes for stateful models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import copy
from functools import cached_property, partial, wraps
import logging
import os
from typing import (
    TYPE_CHECKING,
    Generic, 
    Optional, 
    Tuple,
    TypeVar,
    TypeVarTuple, 
    Union,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.random as jr
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.intervene import AbstractIntervenor, AbstractIntervenorInput
from feedbax.tree import random_split_like_tree
from feedbax.misc import indent_str

if TYPE_CHECKING:
    from feedbax.task import AbstractTaskInput

logger = logging.getLogger(__name__)


N_DIM = 2


class AbstractModelState(eqx.Module):
    ...
    

StateT = TypeVar("StateT", bound=AbstractModelState)

# StateOrArrayT = TypeVar("StateOrArrayT", bound=Union[AbstractModelState, Array])

class AbstractModel(eqx.nn.StatefulLayer, Generic[StateT]):
    """
    
    TODO:
    - Should this be a generic of `StateT | Array`, or something?
      - i.e. what about models that return a single array or some other
        kind of PyTree? As in linear regression, PCA...
    """
    
    @abstractmethod
    def __call__(
        self,
        input,
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        """Update the model state given inputs and a prior state."""
        ...

    @abstractproperty
    def _step(self) -> "AbstractModel[StateT]":
        """Interface to a single model step.
        
        For non-iterated models, this should trivially return `step`.
        """
        ...
    
    def state_consistency_update(
        self, 
        state: StateT,
    ) -> StateT:
        """Make sure the model state is self-consistent.
        
        The default behaviour is to just return the same state. However, in
        models where it is customary to initialize (say) the effector state 
        but not the plant configuration state, this method should be used 
        to ensure that the configuration state is initialized properly. 
        
        This avoids having to define how both states should be initialized
        in `AbstractTask`, which would require knowledge not only of the 
        structure of the state, but also of the model interface that 
        provides particular operations for modifying the state. Though, I'm not 
        sure that would be a bad thing, as the model state objects tend to 
        mirror the operations done on them anyway.
        
        However, this approach also has the advantage that these consistency
        checks can be arbitrarily complex and important to the model 
        functioning properly; it is useful for the model to be able to 
        render a state consistent, with respect to its other normal operations.
        """
        return state
    
    @abstractmethod
    def init(
        self,
        *,
        key: Optional[jax.Array] = None,
    ) -> StateT:
        """Return an initial state for the model."""
        ...



class ModelInput(eqx.Module):
    input: "AbstractTaskInput"
    intervene: Mapping[AbstractIntervenorInput]

   

class MultiModel(AbstractModel[StateT]):
    """  """
    
    models: PyTree[AbstractModel[StateT], 'T']
    
    def __call__(
        self,
        inputs: ModelInput,
        states: PyTree[StateT, 'T'], 
        key: Array,
    ) -> StateT:

        return jax.tree_map(
            lambda model, input, state, key: model(ModelInput(input, inputs.intervene), state, key),
            self.models,
            inputs.input,
            states,
            self._get_keys(key),
            is_leaf=lambda x: isinstance(x, AbstractModel),
        )
        
    def init(
        self,
        *,
        key: Optional[jax.Array] = None,
    ) -> StateT:
        return jax.tree_map(
            lambda model, key: model.init(key=key),
            self.models,
            self._get_keys(key),
            is_leaf=lambda x: isinstance(x, AbstractModel),
        )
    
    def _step(self):
        return self
        
    def _get_keys(self, key):
        return random_split_like_tree(
            key, 
            target=self.models, 
            is_leaf=lambda x: isinstance(x, AbstractModel)
        )
   


class ModelStageSpec(eqx.Module, Generic[StateT]):
    """Specification for a stage in a subclass of `AbstractStagedModel`.
    
    To ensure that references remain fresh, `callable` takes a single argument,
    the instance of `AbstractStagedModel` (i.e. `self`), and typically returns 
    either an `eqx.Module` owned by that instance, or a method of a module. 
    However, it may be any callable/function with the appropriate signature. 
    """
    callable: Callable[
        ["AbstractStagedModel[StateT]"], 
        Callable[[Union[ModelInput, PyTree[Array]], StateT, Array], StateT]
    ]
    where_input: Callable[["AbstractTaskInput", StateT], PyTree]
    where_state: Callable[[StateT], PyTree]
    intervenors: Optional[Sequence[AbstractIntervenor]] = None


class AbstractStagedModel(AbstractModel[StateT]):
    """Base class for state-dependent models whose stages can be intervened upon.
    
    To define a new model, the following should be implemented:
    
        1. A concrete subclass of `AbstractModelState` that defines the PyTree
           structure of the full model state. The fields may be types of 
           `AbstractModelState`, in the case of nested models.
        2. A concrete subclass of this class with:
            i. the appropriate components for doing state transformations;
            either `eqx.Module`/`AbstractModel`-typed fields, or instance 
            methods, each with the signature `input, state, *, key`;
            ii. a `model_spec` property that specifies a series of named 
            operations on parts of the full model state, using those component 
            modules; and
            iii. an `init` method that returns an initial model state.
            
    The motivation behind the level of abstraction of this class is that it
    gives a name to each functional stage of a composite model, so that after 
    model instantiation or training, the user can choose to insert additional 
    state interventions to be executed prior to any of the named stages. 
    """
    intervenors: AbstractVar[Mapping[str, Sequence[AbstractIntervenor]]]
    
    def __call__(
        self, 
        input: ModelInput,
        state: StateT, 
        key: Array,
    ) -> StateT:
        
        with jax.named_scope(type(self).__name__):
            
            keys = jr.split(key, len(self._stages))
            
            for (label, spec), key in zip(self._stages.items(), keys):
                
                key1, key2 = jr.split(key)
                
                if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
                    debug_strs = [
                        indent_str(eqx.tree_pformat(x), indent=4) 
                        for x in 
                            (
                                spec.callable(self),
                                spec.where_input(input, state),
                                spec.where_state(state),
                            )
                    ]
                    
                    logger.debug(
                        f"Module: {type(self).__name__} \n"
                        f"Stage: {label} \n"
                        f"Stage module:\n{debug_strs[0]}\n"
                        f"Input:\n{debug_strs[1]}\n"
                        f"Substate:\n{debug_strs[2]}\n"
                    )                  
                
                for intervenor in spec.intervenors:
                    if intervenor.label in input.intervene:
                        params = input.intervene[intervenor.label]
                    else:
                        params = None
                    state = intervenor(params, state, key=key1)
                
                callable = spec.callable(self)
                subinput = spec.where_input(input.input, state)
                
                # TODO: What's a less hacky way of doing this?
                # I was trying to avoid introducing additional parameters to `AbstractStagedModel.__call__`
                if isinstance(callable, AbstractModel):
                    callable_input = ModelInput(subinput, input.intervene)
                else:
                    callable_input = subinput
                
                state = eqx.tree_at(
                    spec.where_state, 
                    state, 
                    callable(
                        callable_input, 
                        spec.where_state(state), 
                        key=key2,
                    ),
                )
        
        return state

    @abstractproperty
    def model_spec(self) -> OrderedDict[str, ModelStageSpec]:
        """Specifies the model in terms of substate operations.
        
        Each entry in the dict has the following elements:
        
            1. A callable (e.g. an `eqx.Module` or method) that takes `input`
               and a substate of the model state, and returns an updated copy of 
               that substate.
            2. A function that selects the parts of the model input and state 
               PyTrees that are passed to the module as its `input`. Note that 
               parts of the state PyTree can also be passed, because they may
               be inputs to the model stage but should not be part of the state 
               update associated with the model stage.
            3. A function that selects the substate PyTree out of the `StateT`
               PyTree; i.e. the `state` passed to the module, which then returns a
               PyTree with the same structure and array shapes/dtypes.
               
        Note: It's still necessary to use `OrderedDict` because `jax.tree_util`
        sorts `dict` keys, which puts our stages out of order.
        """
        ...
  
    @cached_property 
    def _stages(self) -> OrderedDict[str, ModelStageSpec]: 
        """Zips up the user-defined intervenors with `model_spec`.
        
        This should not be referred to in `__init__` before assigning `self.intervenors`!
        """
        
        return jax.tree_map(
            lambda x, y: eqx.tree_at(lambda x: x.intervenors, x, y),
            self.model_spec, 
            jax.tree_map(tuple, self.intervenors, 
                         is_leaf=lambda x: isinstance(x, list)),
            is_leaf=lambda x: isinstance(x, ModelStageSpec),
        )
    
    # TODO: I'm not sure we need to add intervenors, here.
    def _get_intervenors_dict(
        self, intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                          Mapping[str, Sequence[AbstractIntervenor]]]]
    ):
        intervenors_dict = jax.tree_map(
            lambda _: [], 
            self.model_spec,
            is_leaf=lambda x: isinstance(x, ModelStageSpec),
        )
        
        if intervenors is not None:
            if isinstance(intervenors, Sequence):
                # By default, place interventions in the first stage.
                intervenors_dict.update({'get_feedback': list(intervenors)})
            elif isinstance(intervenors, dict):
                intervenors_dict.update(
                    jax.tree_map(list, intervenors, 
                                 is_leaf=lambda x: isinstance(x, Sequence)))
            else:
                raise ValueError("intervenors not a sequence or dict of sequences")
        
        return intervenors_dict
    
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...
        
    @property 
    def _step(self):
        """This assumes all staged models will specify single update steps,
        not iterated models.
        
        This allows classes like `AbstractTask` and `TaskTrainer` to refer
        to methods of a single step of the model, without having to know whether
        the model step is wrapped in an iterator; `AbstractIterator` concretes will 
        return `self._step` instead.
        """
        return self 
    
    @property 
    def _all_intervenor_labels(self):
        model_leaves = jax.tree_util.tree_leaves(
            self, 
            is_leaf=lambda x: isinstance(x, AbstractIntervenor)
        )
        labels = [leaf.label for leaf in model_leaves 
                  if isinstance(leaf, AbstractIntervenor)]
        return labels


def wrap_stateless_callable(callable: Callable, pass_key=True):
    """Makes a 'stateless' callable trivially compatible with state-passing.
    
    `AbstractModel` defines everything in terms of transformations of parts of
    a state PyTree. In each case, the substate that is operated on is passed
    to the module that returns the updated substate. However, in some cases
    the new substate does not depend on the previous substate. For example,
    a linear network layer takes some inputs and returns some outputs, and
    on the next iteration, the linear layer's outputs do not conventionally 
    depend on its previous outputs, like an RNN cell's would.
    """
    if pass_key:
        @wraps(callable)
        def wrapped(input, state, *args, **kwargs):
            return callable(input, *args, **kwargs)
        
    else:
        @wraps(callable)
        def wrapped(input, state, *args, key: Optional[Array] = None, **kwargs):
            return callable(input, *args, **kwargs)
    
    return wrapped


def model_spec_format(
    model: AbstractStagedModel, 
    indent: int = 2, 
    newlines: bool = False,
) -> str:
    """Return a string representation of the model specification tree.
    
    Show what is called by `model`, and by any `AbstractStagedModel`s it calls.
    
    This assumes that the model spec is a tree/DAG. If there are cycles in 
    the model spec, this will recurse until an exception is raised.
    
    Args:
        model: The staged model to format.
        indent: Number of spaces to indent each nested level of the tree.
        newlines: Whether to add an extra blank line between each line.
    """
       
    def get_spec_strs(model: AbstractStagedModel):
        spec_strs = []

        for label, stage_spec in model._stages.items():
            intervenor_str = ''.join([
                f"intervenor: {type(intervenor).__name__}\n" 
                for intervenor in stage_spec.intervenors
            ])

            callable = stage_spec.callable(model)
            
            # Get a meaningful label for `BoundMethod`s
            if (func := getattr(callable, '__func__', None)) is not None:
                owner = type(getattr(callable, '__self__')).__name__
                spec_str = f"{label}: {owner}.{func.__name__}"

            else: 
                spec_str = f"{label}: {type(callable).__name__}"
                
            spec_strs += [intervenor_str + spec_str]

            if isinstance(callable, AbstractStagedModel):
                spec_strs += [
                    ' ' * indent + spec_str
                    for spec_str in get_spec_strs(callable)
                ]              
            
        return spec_strs 
    
    nl = '\n\n' if newlines else '\n'
    
    return nl.join(get_spec_strs(model))


Ts = TypeVarTuple("Ts")


def get_model_ensemble(
    get_model: Callable[[jax.Array, *Ts], eqx.Module], 
    n_replicates: int, 
    *args: *Ts, 
    key: jax.Array
) -> eqx.Module:
    """Helper to vmap model generation over a set of PRNG keys.
    
    TODO: 
    - Rename. This works for stuff other than `get_model`. It's basically
      a helper to split key, then vmap function.
    """
    keys = jr.split(key, n_replicates)
    get_model_ = partial(get_model, *args)
    return eqx.filter_vmap(get_model_)(keys)