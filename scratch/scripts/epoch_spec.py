# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from typing import Tuple, Callable

import equinox as eqx
from equinox import tree_pprint as tpp
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Int, Array, Float, PyTree, Shaped 
import numpy as np

from feedbax import task, loss
from feedbax.state import CartesianState2D
from feedbax.task import DelayTaskInput

# %%
seed = 1234
key = jr.PRNGKey(seed)
workspace = jnp.array([[-1, -1], [1, 1]])

loss_func = loss.EffectorPositionLoss()

t = task.RandomReachesDelayed(
    loss_func=loss_func, 
    workspace=workspace,
    n_steps=100,
)

# %%
pos_endpoints = task.uniform_tuples(key, n=2, bounds=workspace)

# %%
tpp(pos_endpoints)

# %%
init_state, target_state = task._pos_only_states(pos_endpoints)

# %%
tpp((init_state, target_state))


# %%
def gen_epoch_lengths(
    key: jr.PRNGKey,
    ranges: Tuple[Tuple[int, int], ...]=((1, 3),  # (min, max) for first epoch
                                         (2, 5),  # second epoch
                                         (1, 3)),  
) -> Int[Array, "n_epochs"]:
    """Generate a random integer in each given ranges."""
    ranges = jnp.array(ranges, dtype=int)
    return jr.randint(key, (ranges.shape[0],), *ranges.T)


def get_masks(
    length: int, 
    idx_bounds: Int[Array, "_"],
):
    """Get a 1D mask of length `length` with `False` values at `idxs`."""
    idxs = jnp.arange(length)
    # #? could also use `arange` to get ranges of idxs
    mask_fn = lambda e: (idxs < idx_bounds[e]) + (idxs > idx_bounds[e + 1] - 1)
    return jnp.stack([mask_fn(e) for e in range(len(idx_bounds) - 1)])


def get_masked_seqs(
    arrays: PyTree, 
    masks: Tuple[Int[Array, "n"], ...],
    init_fn: Callable[[Tuple[int, ...]], Shaped[Array, "..."]] = jnp.zeros,
) -> PyTree:
    """Expand arrays with an initial axis of length `n`, and fill with 
    original array values where the intersection of `masks` is `False`.
    
    That is, each expanded array will be filled with the values from `array`
    for all indices where *any* of the masks is `False`.
    
    Returns a PyTree with the same structure as `target`, but where each
    array has an additional sequence dimension, and the original `target` 
    values are assigned only during the target epoch, as bounded by
    `target_idxs`.
    
    TODO:
    - Find a better name.
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


# %%

def get_sequences(
    self,  
    init_states: CartesianState2D, 
    target_states: CartesianState2D, 
    key: jax.Array,
) -> Tuple[DelayTaskInput, CartesianState2D, Int[Array, "n_epochs"]]:
    """Convert static task inputs to sequences, and make hold signal.
    """        
    epoch_lengths = gen_epoch_lengths(key, self.epoch_len_ranges)
    epoch_start_idxs = jnp.pad(
        jnp.cumsum(epoch_lengths), 
        (1, 0), 
        constant_values=(0, -1)
    )
    epoch_masks = get_masks(self.n_steps, epoch_start_idxs)
    move_epoch_mask = jnp.logical_not(
        jnp.prod(epoch_masks, axis=0)
    )[None, :]
    
    stim_seqs = get_masked_seqs(
        _forceless_task_inputss(target_states), 
        epoch_masks[self.stim_epochs]
    )
    target_seqs = jax.tree_map(
        lambda x, y: x + y, 
        get_masked_seqs(target_states, move_epoch_mask),
        get_masked_seqs(init_states, epoch_masks[self.hold_epochs]),
    )
    stim_on_seq = get_scalar_epoch_seq(
        epoch_start_idxs, self.n_steps, 1., self.stim_epochs
    )
    hold_seq = get_scalar_epoch_seq(
        epoch_start_idxs, self.n_steps, 1., self.hold_epochs
    )
    
    task_input = DelayTaskInput(stim_seqs, hold_seq, stim_on_seq)
    target_states = target_seqs
    
    return task_input, target_states, epoch_start_idxs  
