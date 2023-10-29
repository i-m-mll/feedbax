"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import os
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
from tqdm.auto import tqdm

from feedbax.context import AbstractModel
from feedbax.utils import tree_get_idx, tree_set_idx

logger = logging.getLogger(__name__)


class Iterator(eqx.Module):
    """A module that recursively applies another module for `n_steps` steps.
    
    We automatically determine the shape of the arrays in the PyTree(s) 
    returned by `step`, and use this to initialize empty trajectory arrays in 
    which to store the states across all steps; `states_includes` can be used
    to specify which states to store. By default, all are stored.
    
    TODO:
    - is there a way to avoid assuming the `input, state` argument structure of `step`?
    - with the new partitioning of states into memory and no-memory,
    """
    step: AbstractModel
    n_steps: int 
    states_includes: PyTree[bool]
    
    def __init__(
        self,
        step: eqx.Module,
        n_steps: int,
        states_includes: Optional[PyTree[bool]] = None,
    ):
        if states_includes is None:
            states_includes = step.memory_spec
        self.step = step
        self.n_steps = n_steps
        self.states_includes = states_includes
    
    @jax.named_scope("fbx.Iterator._body_func")
    def _body_func(self, i, x):
        inputs, states, key = x
        
        key1, key2 = jr.split(key)
        
        # since we optionally store the trajectories of only some of the states,
        # as specified by `states_includes`, we need to partition these out 
        # so we can index them, then recombine with the states for which only 
        # the current step is stored
        states_mem, state_nomem = eqx.partition(states, self.states_includes)
        state_mem, input = tree_get_idx((states_mem, inputs), i)
        state = eqx.combine(state_mem, state_nomem)
        
        state = self.step(input, state, key1)
        
        state_mem, state_nomem = eqx.partition(state, self.states_includes)        
        states_mem = tree_set_idx(states_mem, state_mem, i + 1)
        states = eqx.combine(states_mem, state_nomem)
                
        return inputs, states, key2
    
    @jax.named_scope("fbx.Iterator")
    def __call__(self, inputs, init_effector_state, key):
        key1, key2, key3 = jr.split(key, 3)
        
        init_state = self.step.init(init_effector_state)  #! maybe this should be outside
        
        init_input = tree_get_idx(inputs, 0)
        states = self.init(init_input, init_state, key2)
        
        if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                inputs, states, key3 = self._body_func(i, (inputs, states, key3))
                
            return states
                 
        _, states, _ = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (inputs, states, key3),
        )
        
        return states
    
    def init(self, input, init_state, key):
        # get the shape of the state output by `self.step`
        outputs = eqx.filter_eval_shape(
            self.step, 
            input, 
            init_state, 
            key=key,
        )
        
        # generate empty trajectories for mem states
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
    
        # insert the init state for mem states; combine with no mem state
        init_state_mem, init_state_nomem = eqx.partition(
            init_state, self.states_includes
        )
        states = eqx.combine(
            tree_set_idx(states, init_state_mem, 0), 
            init_state_nomem
        )
        
        return states