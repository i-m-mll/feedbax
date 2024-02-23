"""Base classes for stateful models.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections.abc import Callable, Mapping
from functools import wraps
import logging
from typing import (
    TYPE_CHECKING,
    Generic, 
    Optional, 
)

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree
import numpy as np

from feedbax.state import StateBounds, StateT
from feedbax._tree import random_split_like_tree

if TYPE_CHECKING:
    from feedbax.intervene import AbstractIntervenorInput
    from feedbax.task import AbstractTaskInputs


logger = logging.getLogger(__name__)


# StateOrArrayT = TypeVar("StateOrArrayT", bound=Union[AbstractState, Array])

class AbstractModel(eqx.Module, Generic[StateT]):
    """Base class for 
    
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
        key: PRNGKeyArray,
    ) -> StateT:
        """Update the model state given inputs and a prior state."""
        ...

    @abstractproperty
    def step(self) -> "AbstractModel[StateT]":
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
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        """Return a default state for the model."""
        ...

    @property 
    def bounds(self) -> PyTree[StateBounds]:  # type: ignore
        """Suggested bounds on the state.
        """
        return None
    
    @property
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        return True


class ModelInput(eqx.Module):
    input: "AbstractTaskInputs"
    intervene: Mapping["AbstractIntervenorInput"]

   

class MultiModel(AbstractModel[StateT]):
    """  """
    
    models: PyTree[AbstractModel[StateT], 'T']
    
    def __call__(
        self,
        inputs: ModelInput,
        states: PyTree[StateT, 'T'], 
        key: PRNGKeyArray,
    ) -> StateT:

        # TODO: This is hacky, because I want to pass intervenor stuff through entirely. See `staged`
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
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        return jax.tree_map(
            lambda model, key: model.init(key=key),
            self.models,
            self._get_keys(key),
            is_leaf=lambda x: isinstance(x, AbstractModel),
        )
    
    def step(self):
        return self
        
    def _get_keys(self, key):
        return random_split_like_tree(
            key, 
            tree=self.models, 
            is_leaf=lambda x: isinstance(x, AbstractModel)
        )
        

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
        def wrapped(input, state, *args, key: Optional[PRNGKeyArray] = None, **kwargs):
            return callable(input, *args, **kwargs)
    
    return wrapped





