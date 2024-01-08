"""Models that iterate other state-updating models.

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import os
from typing import Generic, Optional, Tuple, TypeVar

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree, Shaped
from tqdm.auto import tqdm

from feedbax.model import AbstractModel, AbstractModelState
from feedbax.utils import tree_get_idx, tree_set_idx

logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractModelState)
# ModelT = TypeVar("ModelT", bound=AbstractModel)


class AbstractIterator(AbstractModel[StateT]):
    """Base class for models which iterate other models.
    
    A main responsibility of this class, aside from defining abstract fields, 
    is to expose the interface of the iterated model, since we are often 
    interested in the methods belonging to the model defined as a single step. 
    
    Some issues with inheriting from `AbstractModel`: 
    - `Iterator.__call__` adds a batch dimension to the returned `StateT`.
      Technically I don't think we can (should?) type this differently anyway,
      but in principle it is not the same type.
    - `AbstractModel` is a generic over a type variable `StateT` of
      `AbstractModelState`; whereas `AbstractIterator` should be a generic over
      `AbstractModel[StateT]`, I think. I don't know that there is a way to achieve
      this without higher-kinded types. 
      - We could make this class a generic of both `AbstractModel[StateT]` 
        and a `ModelT`, but I don't think there's a way to associate them and 
        enforce that `ModelT` is a generic of `StateT`.
      - I think `AbstractModel` should keep being a generic of `StateT`, because
        of the signatures of its methods.
    """
    
    _step: AbstractVar[AbstractModel[StateT]]  
    n_steps: AbstractVar[int]

    def init(self, *, key: Array) -> StateT:
        """Initialize the state of the iterated model.
        """
        return self._step.init(key=key)
    
    @property 
    def step(self):
        return self._step


class Iterator(AbstractIterator[StateT]):
    """A module that recursively applies another module for `n_steps` steps.
    
    We automatically determine the shape of the arrays in the PyTree(s) 
    returned by `step`, and use this to initialize empty trajectory arrays in 
    which to store the states across all steps; `states_includes` can be used
    to specify which states to store. By default, all are stored.
    """
    _step: AbstractModel[StateT]
    n_steps: int 
    states_includes: PyTree[bool]  # can't do StateT[bool]
    
    def __init__(
        self,
        step: AbstractModel[StateT],
        n_steps: int,
        states_includes: Optional[PyTree[bool]] = None,
    ):
        if states_includes is None:
            states_includes = step.memory_spec
        self._step = step
        self.n_steps = n_steps
        self.states_includes = states_includes
    
    @jax.named_scope("fbx.Iterator._body_func")
    def _body_func(self, i: int, x: Tuple) -> Tuple:
        inputs, states, key = x
        
        key1, key2 = jr.split(key)
        
        # Since we optionally store the trajectories of only some of the states,
        # as specified by `states_includes`, we need to partition these out 
        # so we can index them, then recombine with the states for which only 
        # a single time step (the current one) is stored.
        states_mem, state_nomem = eqx.partition(states, self.states_includes)
        state_mem, input = tree_get_idx((states_mem, inputs), i)
        state = eqx.combine(state_mem, state_nomem)
        
        state = self._step(input, state, key1)
        
        # Likewise, split the resulting states into those which are stored,
        # which are then assigned to the next index in the trajectory, and 
        # recombined with the single-timestep states.
        state_mem, state_nomem = eqx.partition(state, self.states_includes)        
        states_mem = tree_set_idx(states_mem, state_mem, i + 1)
        states = eqx.combine(states_mem, state_nomem)
                
        return inputs, states, key2
    
    @jax.named_scope("fbx.Iterator")
    def __call__(
        self, 
        input: PyTree[Shaped[Array, "n_steps ..."]],  #! should have a batch dimension corresponding to time steps
        state: StateT,   # initial state
        key: Array,
    ) -> StateT:  #! Adds a batch dimension, actually.
        key1, key2, key3 = jr.split(key, 3)
        
        init_input = tree_get_idx(input, 0)
        states = self.init_arrays(init_input, state, key2)
        
        if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                input, states, key3 = self._body_func(i, (input, states, key3))
                
            return states
                 
        _, states, _ = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (input, states, key3),
        )
        
        return states
    
    @jax.named_scope("fbx.Iterator.init_arrays")
    def init_arrays(
        self, 
        input,  #! No batch dimension 
        init_state: StateT, 
        key: Array,
    ) -> StateT:  #! Adds a batch dimension
        # Get the shape of the state output by `self._step`
        outputs = eqx.filter_eval_shape(
            self._step, 
            input, 
            init_state, 
            key=key,
        )
        
        # Generate empty trajectories for mem states
        scalars, array_structs = eqx.partition(
            eqx.filter(outputs, self.states_includes), 
            eqx.is_array_like  # False for jax.ShapeDtypeStruct
        )
        asarrays = eqx.combine(
            jax.tree_map(jnp.asarray, scalars), 
            array_structs
        )
        states = jax.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            asarrays,
        )
    
        # Insert the init state for mem states; combine with no mem state
        init_state_mem, init_state_nomem = eqx.partition(
            init_state, self.states_includes
        )
        states = eqx.combine(
            tree_set_idx(states, init_state_mem, 0), 
            init_state_nomem
        )
        
        return states  


class SimpleIterator(AbstractIterator[StateT]):
    """A simple model iterator that stores the entire state.
    
    If memory is not an issue, this class is preferred as lacks the overhead
    of `Iterator` in terms of state partitioning, and is therefore faster. For 
    large state PyTrees, however, it may be preferable to use `Iterator` and
    choose which states are worth discarding to save memory.
    """
    _step: AbstractModel[StateT]
    n_steps: int 
    
    def __init__(self, step: AbstractModel[StateT], n_steps: int):
        self._step = step
        self.n_steps = n_steps
    
    def __call__(
        self, 
        input,
        state: StateT, 
        key: Array,
    ) -> StateT:
        
        keys = jr.split(key, self.n_steps)
        
        def step(state, args): 
            input, key = args
            state = self._step(input, state, key)
            return state, state
        
        _, states = lax.scan(
            step,
            state, 
            (input, keys),
            length=self.n_steps,
        )
        
        return states 
