
from itertools import zip_longest, chain
import math

import jax
import jax.numpy as jnp


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin."""
    return [(x ** i) / math.factorial(i) for i in range(n)]


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)
    

def normalize(*args, min=0, max=1):
    """Normalize each column of each input array to [min, max]"""
    return [(arr - jnp.min(arr, axis=0)) / (jnp.max(arr, axis=0) - jnp.min(arr, axis=0)) * (max - min) + min
            for arr in args]


def tree_get_idx(tree, idx: int):
    """Retrieve the `idx`-th element of each leaf of `tree`."""
    return jax.tree_map(lambda xs: xs[idx], tree)


def tree_set_idx(tree, vals, idx: int):
    """Update the `idx`-th element of each leaf of `tree`.
    
    `vals` should be a pytree with the same structure as `tree`,
    except that each leaf should be missing the first dimension of 
    the corresponding leaf in `tree`.
    """
    return jax.tree_util.tree_map(lambda xs, x: xs.at[idx].set(x), tree, vals)

# 
# jax.debug.print(''.join([f"{s.shape}\t{p}\n" 
#                             for p, s in jax.tree_util.tree_leaves_with_path(state)]))


