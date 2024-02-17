"""Tools for manipulation of PyTrees.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import Callable, Sequence
import logging 
from typing import Any, Optional, Tuple, TypeVar, TypeVarTuple, get_type_hints

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, PyTreeDef, Shaped


logger = logging.getLogger(__name__)


def filter_spec_leaves(
    tree: PyTree[Any, 'T'], 
    leaf_func: Callable
) -> PyTree[bool, 'T']:
    """Returns a filter specification for tree leaves matching `leaf_func`.
    """
    filter_spec = jax.tree_util.tree_map(lambda _: False, tree)
    filter_spec = eqx.tree_at(
        leaf_func, filter_spec, replace_fn=lambda x: True,
    )
    return filter_spec


def tree_index(tree: PyTree[Any, 'T'], index: int) -> PyTree[Any, 'T']:
    """Returns the same PyTree, indexing all of its array leaves.
    """
    models_arrays, models_other = eqx.partition(tree, eqx.is_array)
    return eqx.combine(
        jax.tree_map(lambda x: x[index], models_arrays),
        models_other,
    )


@jax.named_scope("fbx.tree_take")
def tree_take(
    tree: PyTree[Any, 'T'], 
    indices: ArrayLike, 
    axis: int = 0,
    **kwargs,
) -> PyTree[Any, 'T']:
    """Indexes elements out of each array leaf of a PyTree.
    
    Any non-array leaves are returned unchanged.
    
    Arguments:
        tree: Any PyTree whose array leaves are equivalently indexable,
            according to the other arguments to this function. For example,
            `axis=0` could be used when the first dimension of every array leaf
            is a batch dimension, and `indices` specifies a subset of examples
            from the batch.
        indices: The indices of the values to take from each array leaf. 
        axis: The axis of the array leaves over which to take their values.
            Defaults to 0.

    Returns:
        A PyTree with the same structure as `tree`, where array leaves from `tree` have been replaced by indexed-out elements.
    """
    arrays, other = eqx.partition(tree, eqx.is_array)
    values = jax.tree_map(
        lambda xs: jnp.take(xs, indices, axis=axis, **kwargs), 
        arrays,
    )
    return eqx.combine(values, other)


@jax.named_scope("fbx.tree_set")
def tree_set(
    tree: PyTree[Any | Shaped[Array, "batch *?dims"], 'T'], 
    items: PyTree[Any | Shaped[Array, "*?dims"], 'T'],
    idx: int,
) -> PyTree[Any | Shaped[Array, "batch *?dims"], 'T']:
    """Perform an out-of-place update of each array leaf of a PyTree.
    
    Non-array leaves are simply replaced by their matching leaves in `items`.
    
    For example, if `tree` is a PyTree of states over time, whose first dimension
    is the time step, and `items` is a PyTree of states for a single time step,
    this function can be used to insert the latter into the former at a given time index.
    
    Arguments:
        tree: Any PyTree whose array leaves share a first dimension of the same
            length, for example a batch dimension.
        items: Any PyTree with the same structure as `tree`, and whose array
            leaves have the same shape as the corresponding leaves in `tree`, 
            but lacking the first dimension.
        idx: The index along the first dimension of the array leaves of `tree`
            into which to insert the array leaves of `items`.
    
    Returns:  
        A PyTree with the same structure as `tree`, where the array leaves of `items` have been inserted as the `idx`-th elements of the corresponding array leaves of `tree`.
    """
    arrays = eqx.filter(tree, eqx.is_array)
    vals_update, other_update = eqx.partition(
        items, jax.tree_map(lambda x: x is not None, arrays)
    )
    arrays_update = jax.tree_map(
        lambda xs, x: xs.at[idx].set(x), arrays, vals_update
    )
    return eqx.combine(arrays_update, other_update)


def random_split_like_tree(
    key: PRNGKeyArray, 
    tree: PyTree[Any, 'T'], 
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[PRNGKeyArray | None, 'T']:
    """Returns a split of random keys, as leaves of a target PyTree structure.
    
    Derived from [this](https://github.com/google/jax/discussions/9508#discussioncomment-2144076) comment
    on a discussion in the JAX GitHub repository.
    
    Arguments:
        key: The random key from which to split the tree of random keys. 
        tree: Any PyTree. 
        is_leaf: An optional function that decides whether each node in `tree` 
            should be treated as a leaf, or traversed as a subtree. 
    """
    treedef = jax.tree_structure(tree, is_leaf=is_leaf)
    return _random_split_like_treedef(key, treedef)


def _random_split_like_treedef(
    key: PRNGKeyArray,
    treedef: PyTreeDef,
):
    keys = jr.split(key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)    


def tree_stack(
    trees: Sequence[PyTree[Any, 'T']], 
    axis: int = 0,
) -> PyTree[Any, 'T']:
    """Returns a PyTree whose array leaves stack those of the PyTrees in `trees`.
    
    !!! Example    
        ```python
        a = [jnp.array([1, 2]), jnp.array([3, 4])]
        b = [jnp.array([5, 6]), jnp.array([7, 8])]
        
        tree_stack([a, b], axis=0)  
        # [jnp.array([[1, 2], [5, 6]]), jnp.array([[3, 4], [7, 8]])]
        ```
    
    Derived from [this](https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557)
    GitHub gist.
    
    Arguments:
        trees: A sequence of PyTrees with the same structure, and whose array
            leaves have the same shape.
        axis: The axis along which to stack the array leaves.
    """
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v, axis=axis), *trees)


def tree_sum_squares(tree: PyTree[Array]) -> ArrayLike:
    """Sum the sums of squares of the leaves of a PyTree.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: jnp.sum(x ** 2), tree)
    )


def tree_sum_n_features(tree: PyTree[Array]) -> int:
    """Returns the sum the sizes of the last dimensions of all leaves."""
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: x.shape[-1], tree)
    )


def _tree_map(
    f: Callable[..., Any], 
    tree: PyTree[Any, 'T'], 
    *rest, 
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[Any, 'T']:
    """Custom version of `jax.tree_util.tree_map`.
    
    The only difference is that by default, it will infer `is_leaf` from 
    the annotation of the first argument to `f`. This is useful when mapping
    user-defined functions over PyTrees of user-defined objects, when it is 
    acceptable to have slightly worse performance of `tree_map` in exchange
    for not needing to import the objects' class to be able to manually 
    define `is_leaf` with an `isinstance` check.
    
    Unfortunately this doesn't work with string annotations, which I've had to 
    use in places due to issues with circular imports, so it is not very 
    useful at the moment...
    """
    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, next(iter(f.__annotations__.values())))
    return jax.tree_map(f, tree, *rest, is_leaf=is_leaf)


def _is_module(obj: Any):
    return isinstance(obj, eqx.Module)


S = TypeVar('S')


def tree_map_module(
    f: Callable[[Any], S], 
    tree: PyTree[Any, 'T'], 
    *rest, 
) -> PyTree[S, 'T']:
    """Custom `tree_map` that treats `eqx.Module`s as leaves.
    
    This is a convenience for performing analyses involving mapping 
    repeatedly over PyTrees of `eqx.Module`, where it would be repetitive
    to write `is_leaf=lambda x: isinstance(x, eqx.Module)` every time.
    """
    
    return jax.tree_map(f, tree, *rest, is_leaf=_is_module)


def tree_map_unzip(
    f: Callable[..., Tuple[Any, ...]], 
    tree: PyTree[Any, 'T'], 
    *rest, 
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Tuple[PyTree[Any, 'T'], ...]:
    """Maps a tuple-valued function over a PyTree. Returns a tuple of PyTrees.
    
    For example, for a function `f(x) -> (y, z)`, we can do `ys, zs =
    tree_map_unzip(f, xs)` where `ys`, `zs` are PyTrees, whereas with a normal
    `tree_map` we'd get a single PyTree of tuples `(y, z)`.
    """
    results = jax.tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)
    return tree_unzip(results)


def tree_unzip(
    tree: PyTree[Tuple[Any, ...], 'T'],
) -> Tuple[PyTree[Any, 'T'], ...]:
    """Unzips a PyTree of tuples into a tuple of PyTrees.
    """
    tree_flat, treedef = jax.tree_flatten(tree, is_leaf=lambda x: isinstance(x, tuple))
    tree_flat_unzipped = zip(*tree_flat)
    return tuple(jax.tree_unflatten(treedef, x) for x in tree_flat_unzipped)


def tree_call(tree: PyTree[Any, 'T'], *args, **kwargs) -> PyTree[Any, 'T']:
    """Returns a tree of the return values of a PyTree's callable leaves.
    
    Every callable leaf is passed the same `*args, **kwargs`.
    
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


def tree_array_bytes(tree: PyTree) -> int:
    """Returns the total bytes of memory over all array leaves of a PyTree.
    """
    arrays = eqx.filter(tree, eqx.is_array)
    array_bytes = jax.tree_map(lambda x: x.nbytes, arrays)
    return jtu.tree_reduce(
        lambda x, y: x + y,
        array_bytes,
    )
    

def tree_struct_bytes(tree: PyTree[jax.ShapeDtypeStruct]) -> int:
    """Returns the total bytes of memory implied by a PyTree of `ShapeDtypeStruct`s."""
    structs = eqx.filter(tree, lambda x: isinstance(x, jax.ShapeDtypeStruct))
    struct_bytes = jax.tree_map(lambda x: x.size * x.dtype.itemsize, structs)
    return jtu.tree_reduce(
        lambda x, y: x + y,
        struct_bytes
    )