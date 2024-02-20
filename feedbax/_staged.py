"""Base classes for stateful models with stages.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
import os
from typing import (
    TYPE_CHECKING,
    Generic, 
    Optional, 
    Union,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree
import numpy as np

from feedbax.model import AbstractModel, ModelInput
from feedbax.intervene import AbstractIntervenor
from feedbax.misc import indent_str
from feedbax.state import StateT

if TYPE_CHECKING:
    from feedbax.task import AbstractTaskInput


logger = logging.getLogger(__name__)


N_DIM = 2


class ModelStage(eqx.Module, Generic[StateT]):
    """Specification for a stage in a subclass of `AbstractStagedModel`.
    
    Each stage of a model is a callable that performs a modification to part 
    of the model state. 
    
    !!! Note
        To ensure that references to parts of the model instance remain fresh, 
        `callable_` takes the instance of `AbstractStagedModel` (i.e. `self`)
        and returns the callable associated with the stage.
        
        It is possible for references to become stale. For example, if we 
        assign `callable_=self.net` for the neural network update in 
        [`SimpleFeedback`][feedbax.bodies.SimpleFeedback], then it will 
        continue to refer to the neural network assigned to `self.net` 
        upon the model's construction, even after the network weights
        have been updated during training—so, the model will not train.
    
    Attributes: 
        callable_: The module, method, or function that transforms part of the 
          model state.
        where_input: Selects the  parts of the input and state to be passed 
          as input to `callable_`.
        where_state: Selects the substate that passed and return as state to 
          `callable_`.
        intervenors: Optionally, a sequence of state interventions to be
          applied at the beginning of this model stage. 
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
    
    !!! Info 
        To define a new staged model, the following complementary components 
        must be implemented:
        
        1. A [final](https://docs.kidger.site/equinox/pattern/) subclass of
           `AbstractState` that defines the PyTree structure of the model
           state. The type of the fields of this PyTree are typically JAX
           arrays, or else other `AbstractState` types associated with the
           model's components.     
        2. A final subclass of
           [`AbstractStagedModel`][feedbax.AbstractStagedModel]. Note that the
           abstract class is a `Generic`, and for proper type checking, the
           type argument of the subclass should be the type of `AbstractState`
           defined in (1).
           
            This subclass must implement the following:
           
            1. A `model_spec` property giving a mapping from stage labels 
               to [`ModelStage`][feedbax.ModelStage] instances, each
               specifying an operation performed on the model state.
            2. An `init` method that takes a random key and returns a default 
               model state.
                 
        For an example, consider 1) [`SimpleFeedbackState`][feedbax.bodies.SimpleFeedbackState] 
        and 2) [`SimpleFeedback`][feedbax.bodies.SimpleFeedback].
    """
    intervenors: AbstractVar[Mapping[str, Sequence[AbstractIntervenor]]]
    
    def __call__(
        self, 
        input: ModelInput,
        state: StateT, 
        key: PRNGKeyArray,
    ) -> StateT:
        """Return an updated model state, given input and a prior state.
        
        Arguments:
            input: The input to the model.
            state: The prior state of the model.
            key: A random key which will be split to provide separate keys for 
              each model stage and intervenor.
        """
        with jax.named_scope(type(self).__name__):
            
            keys = jr.split(key, len(self._stages))
            
            for (label, spec), key in zip(self._stages.items(), keys):
                
                key_intervene, key_stage = jr.split(key)
                
                keys_intervene = jr.split(key_intervene, len(spec.intervenors))
                
                for (intervenor, k) in zip(spec.intervenors, keys_intervene):
                    if intervenor.label in input.intervene:
                        params = input.intervene[intervenor.label]
                    else:
                        params = None
                    state = intervenor(params, state, key=k)
                
                callable_ = spec.callable(self)
                subinput = spec.where_input(input.input, state)
                
                # TODO: What's a less hacky way of doing this?
                # I was trying to avoid introducing additional parameters to `AbstractStagedModel.__call__`
                if isinstance(callable_, AbstractModel):
                    callable_input = ModelInput(subinput, input.intervene)
                else:
                    callable_input = subinput
                
                state = eqx.tree_at(
                    spec.where_state, 
                    state, 
                    callable_(
                        callable_input, 
                        spec.where_state(state), 
                        key=key_stage,
                    ),
                )
    
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
                    
                    log_str = '\n'.join([
                        f"Model type: {type(self).__name__}",
                        f"Stage: \"{label}\"",
                        f"Callable:\n{debug_strs[0]}",
                        f"Input:\n{debug_strs[1]}",
                        f"Substate:\n{debug_strs[2]}",
                    ])
                    
                    logger.debug(f'\n{indent_str(log_str, indent=2)}\n')    
        
        return state

    @abstractmethod
    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        """Return a default state for the model."""
        ...

    @abstractproperty
    def model_spec(self) -> OrderedDict[str, ModelStage]:
        """Specify the model's computation in terms of state operations.
        
        !!! Warning        
            It's necessary to return `OrderedDict` because `jax.tree_util`
            still sorts `dict` keys, which usually puts the stages out of order.        
        """
        ...
  
    @cached_property 
    def _stages(self) -> OrderedDict[str, ModelStage]: 
        """Zips up the user-defined intervenors with `model_spec`.
        
        This should not be referred to in `__init__` before assigning `self.intervenors`!
        """
        
        return jax.tree_map(
            lambda x, y: eqx.tree_at(lambda x: x.intervenors, x, y),
            self.model_spec, 
            jax.tree_map(tuple, self.intervenors, 
                         is_leaf=lambda x: isinstance(x, list)),
            is_leaf=lambda x: isinstance(x, ModelStage),
        )
    
    def _get_intervenors_dict(
        self, intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                          Mapping[str, Sequence[AbstractIntervenor]]]]
    ):
        intervenors_dict = jax.tree_map(
            lambda _: [], 
            self.model_spec,
            is_leaf=lambda x: isinstance(x, ModelStage),
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
        
    @property 
    def step(self):
        """The model step.
        
        For `AbstractStagedModel`s, this is trivially the model itself.
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
        return tuple(labels)


def pformat_model_spec(
    model: AbstractStagedModel, 
    indent: int = 2, 
    newlines: bool = False,
) -> str:
    """Returns a string representation of the model specification tree.
    
    Shows what is called by `model`, and by any `AbstractStagedModel`s it calls.
    
    !!! Warning     
        This assumes that the model spec is a tree/DAG. If there are cycles in 
        the model spec, this will recurse until an exception is raised.
    
    Arguments:
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


def pprint_model_spec(
    model: AbstractStagedModel, 
    indent: int = 2, 
    newlines: bool = False,
):
    """Prints a string representation of the model specification tree.
    
    Shows what is called by `model`, and by any `AbstractStagedModel`s it calls.
    
    !!! Warning     
        This assumes that the model spec is a tree. If there are cycles in 
        the model spec, this will recurse until an exception is raised.
    
    Arguments:
        model: The staged model to format.
        indent: Number of spaces to indent each nested level of the tree.
        newlines: Whether to add an extra blank line between each line.
    """
    print(pformat_model_spec(model, indent=indent, newlines=newlines))