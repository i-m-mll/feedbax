""" 

TODO:
- `get_target_seq` and `get_scalar_epoch_seq` are similar. 
   Get rid of repetition.
    - Also, the way `seq` and `seqs` are generated is similar to `states` in 
      `Recursion.init`

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int, PyTree
import numpy as np

from feedbax.utils import tree_set_idx


logger = logging.getLogger(__name__)


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


def get_target_seqs(
    epoch_idxs: Int[Array, "n_epochs-1"], 
    n_steps: int, 
    target: PyTree, 
    target_epochs: Tuple[int, ...]
):
    """Convert a static target to a sequence with a target epoch.
    
    Returns a PyTree with the same structure as `target`, but where each
    array has an additional sequence dimension, and the original `target` 
    values are assigned only during the target epoch, as bounded by
    `target_idxs`.
    """
    seqs = jax.tree_map(
        lambda x: jnp.zeros((n_steps, *x.shape)),
        target
    )
    # seqs = tree_set_idx(seqs, targets, slice(*epoch_idxs[target_epoch:target_epoch + 2]))
    idxs = jnp.arange(n_steps)
    mask_fn = lambda e: (idxs < epoch_idxs[e]) + (idxs > epoch_idxs[e + 1] - 1)
    mask = jnp.prod(jnp.stack([mask_fn(e) for e in target_epochs]), axis=0)
    seqs = jax.tree_map(
        lambda x, y: jnp.where(jnp.expand_dims(mask, jnp.arange(y.ndim) + 1), x, y[None,:]), 
        seqs, 
        target
    )
    return tree_set_idx(seqs, target, -1)


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