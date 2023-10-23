""" 

TODO:
- `get_target_seq` and `get_scalar_epoch_seq` are similar. 
   Get rid of repetition.
    - Also, the way `seq` and `seqs` are generated is similar to `states` in 
      `Recursion.init`

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from functools import cached_property
import logging 
from typing import Tuple

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int, PyTree
import numpy as np

from feedbax.loss import AbstractLoss
from feedbax.utils import internal_grid_points, tree_set_idx


logger = logging.getLogger(__name__)


class AbstractTask(eqx.Module):
    """Abstract base class for tasks.
    
    Associates a trial generator with a loss function. Also provides methods
    for evaluating a suitable model. 
    
    TODO: 
    - Should allow for both dynamically generating trials, and predefined 
      datasets of trials. Currently training is dynamic, and evaluation is 
      static.
    """
    loss_func: AbstractVar[AbstractLoss]
    
    @abstractmethod
    def get_trial(self, key):
        """
        Returns `init_state`, `target_state`, `task_inputs`.
        """
        ...  
        
    @abstractproperty    
    def trials_eval(self):
        ...

    @eqx.filter_jit
    def eval(self, model, key):
        init_states, target_states, task_inputs = self.trials_eval
        
        states = jax.vmap(model, in_axes=(0, 0, None))(
            task_inputs, init_states, key
        ) 
        
        loss, loss_terms = self.loss_func(states, target_states)
        
        return loss, loss_terms, states


class RandomReaches(AbstractTask):
    loss_func: AbstractLoss
    workspace: Float[Array, "ndim 2"]
    eval_n_directions: int 
    eval_reach_length: float
    eval_grid_n: int  # e.g. 2 -> 2x2 grid of center-out reach sets
    
    N_DIM = 2
    
    def get_trial(self, key: jrandom.PRNGKeyArray):
        """Random reach endpoints in a 2D rectangular workspace."""
        pos_endpoints = jrandom.uniform(
            key, 
            (2, self.N_DIM), 
            minval=self.workspace[:, 0], 
            maxval=self.workspace[:, 1]
        )
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_state, target_state = tuple(zip(pos_endpoints, vel_endpoints))
        task_input = target_state
        return init_state, target_state, task_input
        
    @cached_property
    def trials_eval(self):
        centers = internal_grid_points(self.workspace, self.eval_grid_n)
        pos_endpoints = jnp.concatenate([
            centreout_endpoints(
                jnp.array(center), 
                self.eval_n_directions, 
                0, 
                self.eval_reach_length,
            ) for center in centers
        ], axis=1)
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_states, target_states = tuple(zip(pos_endpoints, vel_endpoints))
        task_inputs = target_states 
        return init_states, target_states, task_inputs
    
    def __call__(self, model, key):
        ...        


class RandomReachesGO(AbstractTask):
    ...


def uniform_endpoints(
    key: jrandom.PRNGKey,
    batch_size: int, 
    ndim: int = 2, 
    workspace: Float[Array, "ndim 2"] = jnp.array([[-1., 1.], 
                                                   [-1., 1.]]),
):
    """Segment endpoints uniformly distributed in a rectangular workspace."""
    return jrandom.uniform(
        key, 
        (2, batch_size, ndim),   # (start/end, ...)
        minval=workspace[:, 0], 
        maxval=workspace[:, 1]
    )


def centreout_endpoints(
    center: Float[Array, "2"], 
    n_directions: int, 
    angle_offset: float, 
    length: float,
): 
    ndim = 2  # TODO: generalize to sphere?
    """Segment endpoints starting in the centre and ending equally spaced on a circle."""
    angles = jnp.linspace(0, 2 * np.pi, n_directions + 1)[:-1]
    angles = angles + angle_offset

    starts = jnp.tile(center, (n_directions, 1))
    ends = center + length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    return jnp.stack([starts, ends], axis=0)  


def gen_epoch_lengths(
    key: jrandom.PRNGKey,
    ranges: Tuple[Tuple[int, int], ...]=((1, 3),  # (min, max) for first epoch
                                         (2, 5),  # second epoch
                                         (1, 3)),  
):
    """Generate a random integer in each given ranges."""
    ranges = jnp.array(ranges, dtype=int)
    return jrandom.randint(key, (ranges.shape[0],), *ranges.T)


def get_masks(length, idx_bounds):
    """Get a 1D mask of length `length` with `False` values at `idxs`."""
    idxs = jnp.arange(length)
    #? could also use `arange` to get ranges of idxs
    mask_fn = lambda e: (idxs < idx_bounds[e]) + (idxs > idx_bounds[e + 1] - 1)
    return jnp.stack([mask_fn(e) for e in range(len(idx_bounds) - 1)])

def get_masked_seqs(
    arrays: PyTree, 
    masks: Tuple[Int[Array, "n"], ...],
    init_fn=jnp.zeros,
):
    """Expand arrays with an initial axis of length `n`, and fill with 
    original array values where the intersection of `masks` is `False`.
    
    That is, each expanded array will be filled with the values from `array`
    for all indices where *any* of the masks is `False`.
    
    Returns a PyTree with the same structure as `target`, but where each
    array has an additional sequence dimension, and the original `target` 
    values are assigned only during the target epoch, as bounded by
    `target_idxs`.
    """
    
    seqs = jax.tree_map(
        lambda x: init_fn((masks.shape[1], *x.shape)),
        arrays
    )
    # seqs = tree_set_idx(seqs, targets, slice(*epoch_idxs[target_epoch:target_epoch + 2]))
    mask = jnp.prod(masks, axis=0)
    seqs = jax.tree_map(
        lambda x, y: jnp.where(
            jnp.expand_dims(mask, jnp.arange(y.ndim) + 1), 
            x, 
            y[None,:]
        ), 
        seqs, 
        arrays
    )
    return seqs


def get_scalar_epoch_seq(
    epoch_idxs: Int[Array, "n_epochs-1"], 
    n_steps: int, 
    hold_value: float, 
    hold_epochs: Tuple[int, ...],
):
    """A scalar sequence with a non-zero value held during `hold_epochs`.
    
    Similar to `get_target_steps`, but not for a PyTree.
    """
    seq = jnp.zeros((n_steps,))
    idxs = jnp.arange(n_steps)
    #fill_idxs = jnp.arange(epoch_idxs[hold_epoch], epoch_idxs[hold_epoch + 1])
    mask_fn = lambda e: (idxs < epoch_idxs[e]) + (idxs > epoch_idxs[e + 1] - 1)
    mask = jnp.prod(jnp.stack([mask_fn(e) for e in hold_epochs]), axis=0)
    seq = jnp.where(mask, seq, jnp.array(hold_value))
    return jnp.expand_dims(seq, -1)