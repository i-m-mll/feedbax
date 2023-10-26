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
from equinox import AbstractVar, field
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int, PyTree
import numpy as np

from feedbax.loss import AbstractLoss
from feedbax.types import CartesianState2D
from feedbax.utils import internal_grid_points


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

    # TODO: also try JIT on concrete `get_trial` methods
    @eqx.filter_jit
    def eval(self, model, key):
        init_states, target_states, task_inputs = self.trials_eval
        
        states = jax.vmap(model, in_axes=(0, 0, None))(
            task_inputs, init_states, key
        ) 
        
        loss, loss_terms = self.loss_func(states, target_states, task_inputs)
        
        return loss, loss_terms, states


class RandomReaches(AbstractTask):
    loss_func: AbstractLoss
    workspace: Float[Array, "ndim 2"] = eqx.field(converter=jnp.asarray)
    n_steps: int
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
        init_state, target_state = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints)),
            is_leaf=lambda x: isinstance(x, tuple)
        )
        # make targets as sequences, because `Recursion` and `Loss` want that
        target_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_steps, *x.shape)),
            target_state,
        )
        task_input = target_state
        return init_state, target_state, task_input
        
    @cached_property
    def trials_eval(self):
        centers = internal_grid_points(self.workspace, self.eval_grid_n)
        pos_endpoints = jax.vmap(
            centreout_endpoints, 
            in_axes=(0, None, None),
            out_axes=1,
        )(
            centers, self.eval_n_directions, self.eval_reach_length
        ).reshape((2, -1, self.N_DIM))
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_states, target_states = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints)),
            is_leaf=lambda x: isinstance(x, tuple)
        )
        # this is kind of annoying, but because the batch is explicit here, we
        # need to swap axes as well as broadcast
        target_states = jax.tree_map(
            lambda x: jnp.swapaxes(
                jnp.broadcast_to(x, (self.n_steps, *x.shape)),
                0, 1
            ),
            target_states,
        )
        task_inputs = target_states 
        return init_states, target_states, task_inputs
    
    def __call__(self, model, key):
        ...        


class DelayTaskInput(eqx.Module):
    stim: PyTree[Float[Array, "n_steps ..."]]
    hold: Int[Array, "n_steps 1"]
    stim_on: Int[Array, "n_steps 1"]


class RandomReachesDelayed(AbstractTask):
    """Random reaches with different task epochs.
    
    e.g. allows for a stimulus epoch, followed by a delay period, then movement.
    """
    loss_func: AbstractLoss 
    workspace: Float[Array, "ndim 2"] = eqx.field(converter=jnp.asarray)
    n_steps: int 
    epoch_len_ranges: Tuple[Tuple[int, int], ...]
    eval_n_directions: int
    eval_reach_length: float
    eval_grid_n: int  
    stim_epochs: Tuple[int, ...] = field(default=(1,), converter=jnp.asarray)
    hold_epochs: Tuple[int, ...] = field(default=(0, 1, 2), converter=jnp.asarray)
    key_eval: jrandom.PRNGKeyArray = field(default_factory=lambda: jrandom.PRNGKey(0))

    N_DIM = 2

    def get_trial(self, key: jrandom.PRNGKeyArray):
        """Random reach endpoints in a 2D rectangular workspace."""
        key1, key2 = jrandom.split(key)
        pos_endpoints = jrandom.uniform(
            key1, 
            (2, self.N_DIM), 
            minval=self.workspace[:, 0], 
            maxval=self.workspace[:, 1]
        )
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_state, target_state = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints)),
            is_leaf=lambda x: isinstance(x, tuple)
        )
        task_inputs, target_states, epoch_start_idxs = self.get_sequences(
            init_state, target_state, key2
        )      
        return init_state, target_states, task_inputs, epoch_start_idxs
        
    @cached_property
    def trials_eval(self):
        centers = internal_grid_points(self.workspace, self.eval_grid_n)
        pos_endpoints = jax.vmap(
            centreout_endpoints, 
            in_axes=(0, None, None),
            out_axes=1,
        )(
            centers, self.eval_n_directions, self.eval_reach_length
        ).reshape((2, -1, self.N_DIM))
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_states, target_states = jax.tree_map(
            lambda x: CartesianState2D(*x),
            list(zip(pos_endpoints, vel_endpoints)),
            is_leaf=lambda x: isinstance(x, tuple)
        )        
        epochs_keys = jrandom.split(self.key_eval, init_states.pos.shape[0])
        task_inputs, target_states, epoch_start_idxs = jax.vmap(self.get_sequences)(
            init_states, target_states, epochs_keys
        )    
           
        return init_states, target_states, task_inputs, epoch_start_idxs
    
    def get_sequences(
        self,  
        init_states, 
        target_states, 
        key,
    ):
        """Convert static task inputs to sequences, and make hold signal.
        
        TODO: 
        - this could be part of a `Task` subclass
        """        
        epoch_lengths = gen_epoch_lengths(key, self.epoch_len_ranges)
        epoch_idxs = jnp.pad(jnp.cumsum(epoch_lengths), (1, 0), constant_values=(0, -1))
        epoch_masks = get_masks(self.n_steps, epoch_idxs)
        move_epoch_mask = jnp.logical_not(jnp.prod(epoch_masks, axis=0))[None, :]
        
        stim_seqs = get_masked_seqs(target_states, epoch_masks[self.stim_epochs])
        target_seqs = jax.tree_map(
            lambda x, y: x + y, 
            get_masked_seqs(target_states, move_epoch_mask),
            get_masked_seqs(init_states, epoch_masks[self.hold_epochs]),
        )
        stim_on_seq = get_scalar_epoch_seq(
            epoch_idxs, self.n_steps, 1., self.stim_epochs
        )
        hold_seq = get_scalar_epoch_seq(
            epoch_idxs, self.n_steps, 1., self.hold_epochs
        )
        
        # TODO: catch trials: modify a proportion of hold_seq and target_seqs
        
        task_input = DelayTaskInput(stim_seqs, hold_seq, stim_on_seq)
        target = target_seqs
        
        return task_input, target, epoch_idxs
    
    def __call__(self, model, key):
        ...        


def uniform_endpoints_new(
    key: jrandom.PRNGKey,
    ndim: int = 2, 
    workspace: Float[Array, "ndim 2"] = jnp.array([[-1., 1.], 
                                                   [-1., 1.]]),
):
    """Segment endpoints uniformly distributed in a rectangular workspace."""
    return jrandom.uniform(
        key, 
        (2, ndim),   # (start/end, ...)
        minval=workspace[:, 0], 
        maxval=workspace[:, 1]
    )


#! retaining this for now in case I need it while converting the old notebooks
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
    length: float,
    angle_offset: float = 0, 
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
            jnp.expand_dims(mask, np.arange(y.ndim) + 1), 
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