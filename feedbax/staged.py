"""Base classes for stateful models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractproperty
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
import os
from typing import (
    TYPE_CHECKING,
    Generic, 
    Optional, 
    TypeVar,
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
from feedbax.state import AbstractState

if TYPE_CHECKING:
    from feedbax.task import AbstractTaskInput


logger = logging.getLogger(__name__)


N_DIM = 2


StateT = TypeVar("StateT", bound=AbstractState)

# StateOrArrayT = TypeVar("StateOrArrayT", bound=Union[AbstractState, Array])


class ModelStage(eqx.Module, Generic[StateT]):
    """Specification for a stage in a subclass of `AbstractStagedModel`.
    
    Each stage of a model is a callable that performs a modification to part 
    of the model state. 
    
    Fields: 
        callable_: The module, method, or function that transforms part of the 
          model state.
        where_input: Selects the  parts of the input and state to be passed 
          as input to `callable_`.
        where_state: Selects the substate that passed and return as state to 
          `callable_`.
        intervenors: Optionally, a sequence of state interventions to be
          applied at the beginning of this model stage.
    
    To ensure that references remain fresh, `callable_` takes a single argument,
    the instance of `AbstractStagedModel` (i.e. `self`), and typically returns 
    either an `eqx.Module` owned by that instance, or a method of some module. 
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
    
    1. A concrete subclass of `AbstractState` that defines the PyTree
        structure of the full model state. The fields may be types of
        `AbstractState`, in the case of nested models.
    2. A concrete subclass of this class with:
        i. the appropriate components for doing state transformations;
        either `eqx.Module`/`AbstractModel`-typed fields, or instance methods,
        each with the signature `input, state, *, key`; ii. a `model_spec`
        property that specifies a series of named operations on parts of the
        full model state, using those component modules; and iii. an `init`
        method that returns an initial model state.
            
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
        key: PRNGKeyArray,
    ) -> StateT:
        """Smee"""
        with jax.named_scope(type(self).__name__):
            
            keys = jr.split(key, len(self._stages))
            
            for (label, spec), key in zip(self._stages.items(), keys):
                
                key1, key2 = jr.split(key)
                
                for intervenor in spec.intervenors:
                    if intervenor.label in input.intervene:
                        params = input.intervene[intervenor.label]
                    else:
                        params = None
                    state = intervenor(params, state, key=key1)
                
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
                        key=key2,
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

    @abstractproperty
    def model_spec(self) -> OrderedDict[str, ModelStage]:
        """Specifies the model in terms of substate operations.
        
        Each entry in the dict has the following elements:
        
        1. A callable (e.g. an `eqx.Module` or method) that takes `input`
            and a substate of the model state, and returns an updated copy of
            that substate.
        2. A function that selects the parts of the model input and state 
            PyTrees that are passed to the module as its `input`. Note that
            parts of the state PyTree can also be passed, because they may be
            inputs to the model stage but should not be part of the state
            update associated with the model stage.
        3. A function that selects the substate PyTree out of the `StateT`
            PyTree; i.e. the `state` passed to the module, which then returns a
            PyTree with the same structure and array shapes/dtypes.
               
        Note: It's still necessary to use `OrderedDict` because `jax.tree_util`
        sorts `dict` keys, which puts our stages out of order.
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
    
    # TODO: I'm not sure we need to add intervenors, here.
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
        the model step is wrapped in an ForgetfulIterator; `AbstractIterator` concretes will 
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
        return tuple(labels)


def model_spec_format(
    model: AbstractStagedModel, 
    indent: int = 2, 
    newlines: bool = False,
) -> str:
    """Returns a string representation of the model specification tree.
    
    Shows what is called by `model`, and by any `AbstractStagedModel`s it calls.
    
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

