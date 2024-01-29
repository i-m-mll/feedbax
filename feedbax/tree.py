"""Tools for manipulation of PyTrees.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import Callable
import logging 
from typing import Any, Optional, Tuple, TypeVar, TypeVarTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PyTree


logger = logging.getLogger(__name__)


def tree_index(tree: PyTree[Any, 'T'], index: int) -> PyTree[Any, 'T']:
    """Returns the same PyTree, indexing all of its array leaves.
    """
    models_arrays, models_other = eqx.partition(tree, eqx.is_array)
    return eqx.combine(
        jax.tree_map(lambda x: x[index], models_arrays),
        models_other,
    )


def filter_spec_leaves(tree, leaf_func):
    """Get a filter spec for tree leaves matching `leaf_func`.
    
    `leaf_func` should take `tree` and return leaves from `tree` to filter `True`.
    
    TODO:
    - Is this really the best way to do this?
    """
    filter_spec = jax.tree_util.tree_map(lambda _: False, tree)
    filter_spec = eqx.tree_at(
        leaf_func, filter_spec, replace_fn=lambda x: True,
    )
    return filter_spec


@jax.named_scope("fbx.tree_get_idx")
def tree_get_idx(tree: PyTree[Any, 'T'], idx: int) -> PyTree[Any, 'T']:
    """Retrieve the `idx`-th element of each array leaf of `tree`.
    
    Any non-array leaves are returned unchanged.
    """
    arrays, other = eqx.partition(tree, eqx.is_array)
    values = jax.tree_map(lambda xs: xs[idx], arrays)
    return eqx.combine(values, other)


@jax.named_scope("fbx.tree_get_idx")
def tree_take(tree: PyTree[Any, 'T'], idx: int, axis: int) -> PyTree[Any, 'T']:
    """Take elements from the specified axis of each array leaf of `tree`.
    
    Any non-array leaves are returned unchanged.
    
    TODO:
    - Get rid of `tree_get_idx` and just give this a default `axis=0`.
    """
    arrays, other = eqx.partition(tree, eqx.is_array)
    values = jax.tree_map(lambda xs: jnp.take(xs, idx, axis=axis), arrays)
    return eqx.combine(values, other)


@jax.named_scope("fbx.tree_set_idx")
def tree_set_idx(
    tree: PyTree[Any, 'T'], 
    vals: PyTree[Any, 'T'],
    idx: int
) -> PyTree[Any, 'T']:
    """Update the `idx`-th element of each array leaf of `tree`.
    
    `vals` should be a pytree with the same structure as `tree`,
    except that each leaf should be missing the first dimension of 
    the corresponding leaf in `tree`.
    
    Non-array leaves are simply replaced by their matching leaves in `vals`.
    """
    arrays = eqx.filter(tree, eqx.is_array)
    vals_update, other_update = eqx.partition(
        vals, jax.tree_map(lambda x: x is not None, arrays)
    )
    arrays_update = jax.tree_map(
        lambda xs, x: xs.at[idx].set(x), arrays, vals_update
    )
    return eqx.combine(arrays_update, other_update)


def random_split_like_tree(key, target=None, treedef=None, is_leaf=None):
    """Generate a split of PRNG keys with a target PyTree structure.
    
    See https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    if treedef is None:
        treedef = jax.tree_structure(target, is_leaf=is_leaf)
    keys = jr.split(key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_stack(trees):
    """Stack the leaves of each tree in `trees`.
    
    All trees should have the same structure.
    
    See https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
    """
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_sum_squares(tree):
    """Sum the sums of squares of the leaves of a PyTree.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: jnp.sum(x ** 2), tree)
    )


def tree_sum_n_features(tree):
    """Sum the sizes of the last dimensions of all leaves."""
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: x.shape[-1], tree)
    )


def tree_map_unzip(
    f: Callable[..., Tuple[Any, ...]], 
    tree: PyTree[Any, 'T'], 
    *rest, 
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Tuple[PyTree[Any, 'T'], ...]:
    """Map a function that returns a tuple over a PyTree, unzipping the results.
    
    For example, for a function `f(x) -> (y, z)`, we can do 
    `ys, zs = tree_map_unzip(f, xs)`, whereas with a normal `tree_map` we'd get
    a PyTree of tuples `(y, z)`. That is, we return a tuple of PyTrees instead 
    of a PyTree of tuples.
    """
    results = jax.tree_map(f, tree, *rest, is_leaf=is_leaf)
    return tree_unzip(results)


def tree_unzip(
    tree: PyTree[Any, 'T'],
    is_leaf: Optional[Callable[[Any], bool]] = lambda x: isinstance(x, tuple),
) -> Tuple[PyTree[Any, 'T'], ...]:
    """Unzips a PyTree of tuples into a tuple of PyTrees."""
    tree_flat, treedef = jax.tree_flatten(tree, is_leaf=is_leaf)
    tree_flat_unzipped = zip(*tree_flat)
    return tuple(jax.tree_unflatten(treedef, x) for x in tree_flat_unzipped)


def tree_call(tree, *args, **kwargs):
    """Calls a tree's callable leaves and returns a tree of their return values.
    
    Every callable receives identical `*args, **kwargs`.
    
    Non-callable leaves are passed through as-is.
    """
    callables, other_values = eqx.partition(
        tree, 
        lambda x: isinstance(x, Callable)
    )
    callables_values = jax.tree_map(
        lambda x: x(*args, **kwargs),
        callables,
    )
    return eqx.combine(callables_values, other_values)