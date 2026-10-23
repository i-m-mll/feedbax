import functools
import itertools
import logging
import math
import string
from collections import defaultdict, namedtuple
from collections.abc import Callable, Hashable, Sequence
from types import EllipsisType
from typing import Any, Optional, Tuple, TypeAlias, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, PyTreeDef, Shaped

from ._func import falsef, is_type, truef
from ._ldict import LDict, LDictConstructor
from ._types import is_none
from .misc import construct_tuple_like, deep_merge, unique_generator

logger = logging.getLogger(__name__)


HR = "\u2500" * 80  # Horizontal rule


T = TypeVar("T")
S = TypeVar("S")
TupleT = TypeVar("TupleT", bound=tuple)


def filter_wrap(filter_spec=None, is_leaf=None):
    """Returns a decorator that ensures a function only operates on tree leaves satisfying `filter_spec`."""
    if filter_spec is None:
        filter_spec = truef

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(tree: PyTree, *args, **kwargs):
            filtered, other = eqx.partition(tree, filter_spec, is_leaf=is_leaf)
            updated = func(filtered, *args, **kwargs)
            # ? `other` comes first because leaves may have become subtrees in `updated`
            return eqx.combine(other, updated, is_leaf=is_leaf)

        return wrapper

    return decorator


def first_leaf(tree, is_leaf: Optional[Callable] = None):
    """Return the first leaf of a tree."""
    return jt.leaves(tree, is_leaf=is_leaf)[0]


def first_leaf_shape(tree):
    """Return the shape of the first leaf of a tree of arrays."""
    arrays = eqx.filter(tree, eqx.is_array)
    return first_leaf(arrays, is_leaf=None).shape


@filter_wrap(eqx.is_array)
def shapes(tree):
    """Returns a tree of the shapes of the leaves of `tree`."""
    return jt.map(lambda x: x.shape, tree)


# An alternative to partition-combine logic in `filter_wrap` is to define a custom `tree_map` function
# that only applies the function to leaves that satisfy the filter spec.
def filter_map(f, tree, filter_func):
    def map_func(x):
        return f(x) if filter_func(x) else x

    return jt.map(map_func, tree)


def filter_spec_leaves(
    tree: PyTree[Any, "T"],
    leaf_func: Callable,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[bool, "T"]:
    """Returns a filter specification for leaves selected by ``leaf_func``.

    If ``leaf_func`` selects a subtree rather than a single leaf, the selected
    subtree is expanded to matching ``True`` leaves. This produces masks that
    can be passed to filtering APIs which expect leaf-level boolean specs.
    """
    filter_spec = jt.map(lambda _: False, tree)
    filter_spec = eqx.tree_at(
        leaf_func,
        filter_spec,
        replace_fn=lambda x: jt.map(lambda _: True, x, is_leaf=is_leaf),
    )
    return filter_spec


def get_ensemble(
    func: Callable[..., PyTree[Any, "S"]],
    *args: Any,
    n: int,
    key: PRNGKeyArray,
    **kwargs: Any,
) -> PyTree[Any, "S"]:
    """Vmap a function over `n` random keys.

    Arguments:
        func: A function that returns a PyTree, and whose final keyword argument
            is `key: PRNGKeyArray`.
        n: The number of keys to split; i.e. the size of the batch
            dimensions in the array leaves of the returned PyTree.
        *args: The positional arguments to `func`.
        key: The key to split to perform the vmap.
        **kwargs: The keyword arguments to `func`.
    """
    keys = jr.split(key, n)

    def func_(key):
        return func(*args, **kwargs, key=key)

    return eqx.filter_vmap(func_)(keys)


@jax.named_scope("jax_cookbook.tree.take")
@filter_wrap(eqx.is_array)
def take(
    tree: PyTree[Array, "T"],
    indices: ArrayLike | Sequence[int] | Callable[[tuple[int, ...]], ArrayLike],
    axis: int = 0,
    **kwargs: Any,
) -> PyTree[Any, "T"]:
    """Indexes elements out of one axis of each array leaf of a PyTree.

    Any non-array leaves are returned unchanged.

    !!! Warning ""
        This function inherits the default indexing behaviour of JAX. If
        out-of-bounds indices are provided, no error will be raised.

    Arguments:
        tree: A PyTree whose array leaves are compatible with the indexing operations specified by
            `indices` and `axis`.
        indices: The indices of the values to take from each array leaf. Can also be supplied as
            a callable which takes the shape of each array leaf and returns the indices to take.
        axis: The axis of the array leaves over which to take their values.
            Defaults to 0.
        **kwargs: Additional arguments to [`jax.numpy.take`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.take.html).

    Returns:
        A PyTree with the same structure as `tree`, where array leaves from `tree` have been replaced by indexed-out elements.
    """
    if callable(indices):
        index_fn = indices
    else:
        idxs = jnp.array(indices)

        def index_fn(_, idxs=idxs):
            return idxs

    return jt.map(
        lambda xs: jnp.take(xs, index_fn(xs.shape), axis=axis, **kwargs),
        tree,
    )


# TODO: Assess performance of `take_multi`, then replace `take`
# (it is probably more performant to use `jax.lax.gather` here)
@filter_wrap(eqx.is_array)
def take_multi(
    tree: PyTree[Array, "T"],
    indices: Union[ArrayLike, Sequence[ArrayLike]],
    axes: Union[int, Sequence[int]],
    **kwargs: Any,
):
    """
    Indexes elements out of one or more axes of each array leaf of a PyTree.

    Any non-array leaves are returned unchanged.

    Singleton dimensions in the input `tree` are retained, while any singleton dimensions that result from
    indexing by a Python `int` passed in `indices` are squeezed out.

    !!! Warning ""
        This function inherits the default indexing behaviour of JAX. If
        out-of-bounds indices are provided, no error will be raised.

    Arguments:
        tree: A PyTree whose array leaves are compatible
              with the indexing operations specified by `indices` and `axes`.
        indices: Either a single array-like object or a sequence of array-like objects.
                 Each element specifies the indices to gather along the corresponding axis.
        axes: Either a single integer or a sequence of integers specifying the axes
              along which to gather elements. Must have the same length as `indices` if both are sequences.
        **kwargs: Additional arguments to [`jax.numpy.take`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.take.html).

    Returns:
        A PyTree with the same structure as `tree`, where array leaves from `tree` have been indexed according to `indices` and `axes`.
    """
    if isinstance(indices, ArrayLike):
        indices = [indices]
    if isinstance(axes, int):
        axes = [axes]

    assert len(indices) == len(axes), "Number of indices must match number of axes"

    for idxs, axis in zip(indices, axes):
        # Use `atleast_1d` to retain singleton dimensions until all `axes` have been indexed
        tree = jt.map(
            lambda xs: jnp.take(xs, jnp.atleast_1d(idxs), axis=axis, **kwargs),
            tree,
        )

    # Remove only those singleton dimensions created above by int indexing
    singleton_axes = [axis for idxs, axis in zip(indices, axes) if isinstance(idxs, int)]
    squeezed = jt.map(
        lambda xs: jnp.squeeze(xs, axis=singleton_axes),
        tree,
    )

    return squeezed


@jax.named_scope("jax_cookbook.tree.set")
def array_set(
    tree: PyTree[Union[Any, Shaped[Array, "batch *?dims"]], "T"],
    values: PyTree[Union[Any, Shaped[Array, "*?dims"]], "T"],
    idx: int,
) -> PyTree[Union[Any, Shaped[Array, "batch *?dims"]], "T"]:
    """Perform an out-of-place update of each array leaf of a PyTree.

    Non-array leaves are replaced by their matching leaves in `values`.

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

    def is_none_leaf(x):
        return x is None

    def _set_or_replace(current, value):
        if not eqx.is_array(current):
            return value
        if value is None:
            raise ValueError("values is missing an array leaf matching tree")
        return current.at[idx].set(value)

    return jt.map(_set_or_replace, tree, values, is_leaf=is_none_leaf)


@filter_wrap(eqx.is_array)
def array_set_scalar(
    tree: PyTree[Array, "T"],
    value: Any,
    idx: int,
    axis: int = 0,
) -> PyTree[Array, "T"]:
    """Do an out-of-place assignment to the same indices of each array in a PyTree."""

    def set_value(x):
        # Create the appropriate index tuple for the given axis
        idx_tuple = tuple(idx if i == axis else slice(None) for i in range(x.ndim))
        return x.at[idx_tuple].set(value)

    return jt.map(set_value, tree)


def random_split_like_tree(
    key: PRNGKeyArray,
    tree_or_treedef: Union[PyTree[Any, "T"], PyTreeDef],
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[Union[PRNGKeyArray, None], "T"]:
    """Returns a split of random keys, as the leaves of a target PyTree structure.

    Derived from [this](https://github.com/google/jax/discussions/9508#discussioncomment-2144076) comment
    on a discussion in the JAX GitHub repository.

    Arguments:
        key: The random key from which to split the tree of random keys.
        tree: Any PyTree.
        is_leaf: An optional function that decides whether each node in `tree`
            should be treated as a leaf, or traversed as a subtree.
    """
    if not isinstance(tree_or_treedef, PyTreeDef):
        treedef = jt.structure(tree_or_treedef, is_leaf=is_leaf)
    else:
        treedef = tree_or_treedef
    return _random_split_like_treedef(key, treedef)


def _random_split_like_treedef(
    key: PRNGKeyArray,
    treedef: PyTreeDef,
):
    keys = jr.split(key, treedef.num_leaves)
    return jt.unflatten(treedef, keys)


# @filter_wrap(eqx.is_array)
def stack(
    trees: Sequence[PyTree[Array, "T"]],
    axis: int = 0,
) -> PyTree[Array, "T"]:
    """Returns a PyTree whose array leaves stack those of the PyTrees in `trees`.

    !!! Example
        ```python
        a = [jnp.array([1, 2]), jnp.array([3, 4])]
        b = [jnp.array([5, 6]), jnp.array([7, 8])]

        stack([a, b], axis=0)
        # [jnp.array([[1, 2], [5, 6]]), jnp.array([[3, 4], [7, 8]])]
        ```

    Arguments:
        trees: A sequence of PyTrees with the same structure, and whose array
            leaves have the same shape.
        axis: The axis along which to stack the array leaves.
    """
    return jt.map(lambda *v: jnp.stack(v, axis=axis), *trees)


def stack_subtrees(
    tree: PyTree[Array],
    is_subtree: Callable[[Any], bool],
    axis: int = 0,
):
    """Map each node of `subtree_type` to the array of all its array leaves, stacked."""
    return jt.map(
        lambda subtree: jnp.stack(jt.leaves(subtree), axis=axis),
        tree,
        is_leaf=is_subtree,
    )


def concatenate(
    trees: Sequence[PyTree[Array, "T"]],
    axis: int = 0,
) -> PyTree[Any, "T"]:
    """Returns a PyTree whose array leaves concatenate those of the PyTrees in `trees`.

    Arguments:
        trees: A sequence of PyTrees with the same structure, and whose array
            leaves have the same shape.
        axis: The axis along which to stack the array leaves.
    """
    # TODO: Filter and combine non-array leaves
    return jt.map(lambda *v: jnp.concatenate(v, axis=axis), *trees)


def make_named_tuple_subclass(name):
    """Returns a trivial subclass of tuple with a different name.

    This is useful, for example, if we want a particular kind of tuple
    which we can select with `is_leaf`, but we don't want to define the fields
    of a `namedtuple`.
    """

    def __repr__(self):
        return f"{name}{tuple.__repr__(self)}"

    cls = type(name, (tuple,), dict(__repr__=__repr__))

    def tuple_flatten(x):
        return x, None

    def tuple_unflatten(aux_data, children):
        return cls(children)

    jtu.register_pytree_node(cls, tuple_flatten, tuple_unflatten)

    return cls


def make_named_dict_subclass(name):
    """Returns a trivial subclass of dict with a different name.

    This is useful if we want a particular kind of dict that we can select with `is_leaf`.
    """

    def __repr__(self):
        return f"{name}({dict.__repr__(self)})"

    cls = type(name, (dict,), dict(__repr__=__repr__))

    def dict_flatten_with_keys(obj):
        children = [(jtu.DictKey(k), v) for k, v in obj.items()]
        # return unzip2(sorted(x.items()))[::-1]
        return (children, obj.keys())

    def dict_unflatten(keys, children):
        return cls(zip(keys, children))

    jtu.register_pytree_with_keys(
        cls,
        dict_flatten_with_keys,
        dict_unflatten,
    )

    return cls


# TODO: There could be a variant of this for pure array trees, that collapses the
# tree into a single array and uses `moveaxis`
# TODO: There is also a way to modify this so that we don't need to assume
# the tree structure is totally consistent, I think. Only that all the nodes
# at a certain level are of `level_type`
def move_level_to_outside(tree, level_type):
    """Move a level with the given type to the outside of the tree.

    Assumes that all nodes at each level of the tree, up to the level
    moved outwards, have the same structure as their siblings.

    This can be used similarly to `tree_transpose`, in some cases.
    However it is particularly useful for trees with multiple nested
    levels, where we only want to move one of the levels outwards,
    similarly to how we might use `jnp.moveaxis` to move an array
    axis to the first position.

    ```
    a = ([(1,2), (3,4)], [(5, 6), (7, 8)])
    move_level_to_outside(a, list)
    >>> [((1,2), (5,6)), ((3,4), (7,8))]
    ```
    """
    _assert_uniform_tree(tree)
    leveldefs = ()
    subtree = tree
    children = [True]  # sentinel to enter loop
    leaf_type = None
    arity = None

    while children:
        children, sublevel_def = eqx.tree_flatten_one_level(subtree)
        if children == [subtree]:
            break
        node_data = sublevel_def.node_data()
        if node_data is None:
            raise ValueError(f"Subtree of structure {sublevel_def} has no node data")
        parent_type = node_data[0]
        if level_type is parent_type:
            leveldefs = (sublevel_def,) + leveldefs
            leaf_type = type(children[0])
            arity = sublevel_def.num_leaves
            break
        leveldefs = leveldefs + (sublevel_def,)
        subtree = children[0]

    if leaf_type is None:
        return tree

    assert arity is not None

    new_treedef = functools.reduce(lambda def1, def2: def1.compose(def2), leveldefs)

    def _is_leaf(x, root=tree):
        return isinstance(x, leaf_type) and x is not root

    leaves = jt.leaves(tree, is_leaf=_is_leaf)
    new_leaves = [x for xs in [leaves[i::arity] for i in range(arity)] for x in xs]

    return jt.unflatten(new_treedef, new_leaves)


_TmpTuple = make_named_tuple_subclass("_TmpTuple")


"""
There might be a simpler solution, here.
See https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
"""


def unstack(
    tree: PyTree[Any, "T"],
    axis: int = 0,
):
    """Returns a tuple of PyTrees by unstacking the array leaves of the input PyTree.

    Arguments:
        tree: A PyTree whose array leaves will be unstacked.
        axis: The axis along which to unstack the array leaves.

    Returns:
        A sequence of PyTrees, where each PyTree has the same structure as the input
        but contains slices of the original arrays.
    """
    array_tree, other = eqx.partition(tree, eqx.is_array)

    # Split each array into a tuple of arrays
    array_tuples_tree = jt.map(lambda x: _TmpTuple(jnp.moveaxis(x, axis, 0)), array_tree)

    # Transpose the tree to move the tuple to the outside
    tuple_of_array_trees = jt.transpose(
        jt.structure(array_tree, is_leaf=is_type(_TmpTuple)),
        jt.structure(jt.leaves(array_tuples_tree, is_leaf=is_type(_TmpTuple))[0]),
        array_tuples_tree,
    )

    # TODO: Maybe there's a way to modify `filter_wrap` to use `jt.map` -- then use it to wrap `unstack`
    return tuple(eqx.combine(subtree, other) for subtree in tuple_of_array_trees)


def _unstack_multi(
    tree: PyTree[Any, "T"],
    unstack_spec: Union[Sequence[int], dict[int, Sequence[Hashable]]],
):
    # TODO: Unstack one or more array dimensions into PyTree levels; either tuples or dicts (if keys are provided)
    # TODO: check that `Sequence[Hashable]` has the same length as the respective dimension
    ...


# TODO: Perform the stack at any specified level
def stack_inner(tree: PyTree, is_leaf: Optional[Callable] = None):
    """Stacks all the leaves of each first-level subtrees of a PyTree.

    This is particularly useful when we have a PyTree of results of an analysis (i.e. arrays),
    and we want to aggregate the results over all conditions except the condition indexed by
    the structure of the outermost level of the PyTree.
    """
    subtrees, structure = eqx.tree_flatten_one_level(tree)
    stacked = [stack(jt.leaves(subtree, is_leaf=is_leaf)) for subtree in subtrees]
    return jt.unflatten(structure, stacked)


def sum_squares(tree: PyTree[Array]) -> ArrayLike:
    """Sum the sums of squares of the leaves of a PyTree."""
    return jt.reduce(lambda x, y: x + y, jt.map(lambda x: jnp.sum(x**2), tree))


def sum_n_features(tree: PyTree[Array]) -> int:
    """Returns the sum the sizes of the last dimensions of all leaves."""
    return jt.reduce(lambda x, y: x + y, jt.map(lambda x: x.shape[-1], tree))


def _map(
    f: Callable[..., Any],
    tree: PyTree[Any, "T"],
    *rest,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[Any, "T"]:
    """Custom version of `jt.map`.

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
    return jt.map(f, tree, *rest, is_leaf=is_leaf)


# def map_module(
#     f: Callable[[Any], S],
#     tree: PyTree[Any, "T"],
#     *rest,
# ) -> PyTree[S, "T"]:
#     """Custom `tree_map` that treats `eqx.Module`s as leaves.

#     This is a convenience for performing analyses involving mapping
#     repeatedly over PyTrees of `eqx.Module`, where it would be repetitive
#     to write `is_leaf=lambda x: isinstance(x, eqx.Module)` every time.
#     """

#     return jt.map(f, tree, *rest, is_leaf=is_module)


def map_unzip(
    f: Callable[..., Tuple[Any, ...]],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Tuple[PyTree[Any, "T"], ...]:
    """Maps a tuple-valued function over a PyTree. Returns a tuple of PyTrees.

    For example, for a function `f(x) -> (y, z)`, we can do `ys, zs =
    map_unzip(f, xs)` where `ys`, `zs` are PyTrees, whereas with a normal
    `tree_map` we'd get a single PyTree of tuples `(y, z)`.
    """
    results = jt.map(f, tree, *rest, is_leaf=is_leaf)
    return unzip(results)


def unzip(
    tree: PyTree[Tuple[Any, ...], "T"],
    tuple_cls: type[TupleT] = tuple,
) -> TupleT:
    """Unzips a PyTree of tuples into a tuple of PyTrees.

    !!! Note
        Something similar could be done with `tree_transpose`, but `outer_treedef`
        would need to be specified.

        This version has `zip`-like behaviour, in that 1) the input tree should be
        flattenable to tuples, when tuples are treated as leaves; 2) the shortest
        of those tuples determines the length of the output.

    !!! Warning
        If the PyTree itself is a tuple, this returns the tree unchanged.

    !!! Dev
        We could partition into tuple and non-tuple elements, and only unzip the
        tuple elements.
    """
    tree_flat, treedef = jt.flatten(tree, is_leaf=lambda x: isinstance(x, tuple_cls))
    if any(not isinstance(x, tuple_cls) for x in tree_flat):
        raise ValueError("The input pytree is not flattenable to tuples")
    tree_flat_unzipped = zip(*tree_flat)
    cols = [jt.unflatten(treedef, x) for x in tree_flat_unzipped]
    return construct_tuple_like(tuple_cls, cols)


def zip_(
    *trees: PyTree[Any, "T"],
    is_leaf=None,
    zip_cls=tuple,
) -> PyTree[Tuple[Any, ...], "T"]:
    """Zips a sequence of PyTrees into a PyTree of tuples."""
    return jt.map(lambda *x: zip_cls(x), *trees, is_leaf=is_leaf)


def zip_named(
    is_leaf=None,
    **trees: PyTree[Any, "T"],
) -> PyTree[Tuple[Any, ...], "T"]:
    """Zips a sequence of PyTrees into a PyTree of namedtuples.

    This is more convenient than `tree.zip` when we want to manipulate the zipped tuples
    as leaves, without worrying whether tuples appear elsewhere in the PyTree structure.
    """
    LeafTuple = namedtuple("LeafTuple", trees.keys())
    zipped = jt.map(lambda *x: LeafTuple(*x), *trees.values(), is_leaf=is_leaf)
    return zipped, LeafTuple


def prefix_expand_simple(
    prefix: PyTree,
    tree: PyTree,
    is_leaf: Optional[Callable] = None,
    is_leaf_prefix: Optional[Callable] = None,
) -> PyTree:
    """Expands a prefix of a PyTree to have the same structure as the PyTree."""

    def expand_leaf(leaf, subtree):
        return jt.map(lambda _: leaf, subtree, is_leaf=is_leaf)

    return jt.map(expand_leaf, prefix, tree, is_leaf=is_leaf_prefix)


def _prefix_expand_inner(
    tree1: PyTree,
    tree2: PyTree,
    is_leaf: Optional[Callable] = None,
    is_leaf_prefix: Optional[Callable] = None,
) -> PyTree:
    """Expands a prefix of a PyTree to have the same structure as the PyTree."""

    def expand_leaf(leaf, subtree):
        return jt.map(lambda _: leaf, subtree, is_leaf=is_leaf)

    return jt.map(expand_leaf, tree1, tree2, is_leaf=is_leaf_prefix)


def prefix_expand(
    tree1: PyTree,
    tree2: PyTree,
    is_leaf: Optional[Callable] = None,
    is_leaf_prefix: Optional[PyTree[Callable]] = None,
) -> PyTree:
    """Expands a prefix of a PyTree to have the same structure as the PyTree.

    Handles cases where the outer structure of tree1 doesn't match tree2 by
    automatically descending through tree1 until finding nodes of the same type
    as tree2, then applying prefix expansion.

    Args:
        tree1: PyTree to expand (source)
        tree2: PyTree to match structure of (target)
        is_leaf: Leaf predicate for the expansion operation
        is_leaf_prefix: Leaf predicate for the prefix descent operation

    Returns:
        tree1 expanded to match the structure of tree2
    """
    ilp_tree = _prefix_expand_inner(
        is_leaf_prefix,
        tree1,
        is_leaf=is_type(type(tree2)),
        is_leaf_prefix=is_none,
    )
    return jt.map(
        lambda prefix, ilp: _prefix_expand_inner(
            prefix,
            tree2,
            is_leaf=is_leaf,
            is_leaf_prefix=ilp,
        ),
        tree1,
        ilp_tree,
        is_leaf=is_type(type(tree2)),
    )


def _character_generator():
    # Generates the infinite sequence of strings: a, b, c, ..., z, aa, ab, ac, ..., az, ba, ...
    for length in itertools.count(1):
        for combo in itertools.product(string.ascii_lowercase, repeat=length):
            yield "".join(combo)


def _n_unique_strs(n):
    return itertools.islice(_character_generator(), n)


def call(
    tree: PyTree[Any, "T"],
    *args: Any,
    exclude: Callable = lambda _: False,
    is_leaf: Optional[Callable] = None,
    **kwargs: Any,
) -> PyTree[Any, "T"]:
    """Returns a tree of the return values of a PyTree's callable leaves.

    !!! Note ""
        Every callable leaf is passed the same `*args, **kwargs`.

        Non-callable leaves, callable leaves that satisfy `exclude`, are passed through
        as-is.

    Arguments:
        tree: Any PyTree.
        *args: Positional arguments to pass to each callable leaf.
        exclude: A function that returns `True` for any callable leaf that
            should not be called.
        **kwargs: Keyword arguments to pass to each callable leaf.
    """
    callables, other_values = eqx.partition(
        tree,
        lambda x: isinstance(x, Callable) and not exclude(x),
        is_leaf=is_leaf,
    )
    callables_values = jt.map(
        lambda x: x(*args, **kwargs),
        callables,
        is_leaf=is_leaf,
    )
    return eqx.combine(callables_values, other_values, is_leaf=is_leaf)


def array_bytes(tree: PyTree, duplicates: bool = False) -> int:
    """Returns the total bytes of memory over all array leaves of a PyTree.

    Arguments:
        tree: The tree with arrays to measure.
        duplicates: If `False`, then leaves that refer to the same array in memory
            will only be counted once.
    """
    arrays = eqx.filter(tree, eqx.is_array)
    if not duplicates:
        flat, treedef = jt.flatten(arrays)
        arrays = jt.unflatten(treedef, list(unique_generator(flat, replace_duplicates=True)))
    array_bytes = jt.map(lambda x: x.nbytes, arrays)
    array_bytes_int_leaves = [x for x in jt.leaves(array_bytes) if x is not None]
    return sum(array_bytes_int_leaves)


def struct_bytes(tree: PyTree[jax.ShapeDtypeStruct]) -> int:
    """Returns the total bytes of memory implied by a PyTree of `ShapeDtypeStruct`s."""
    structs = eqx.filter(tree, lambda x: isinstance(x, jax.ShapeDtypeStruct))
    struct_bytes = jt.map(lambda x: x.size * x.dtype.itemsize, structs)
    return jt.reduce(lambda x, y: x + y, struct_bytes)


BuiltInKeyEntry = Union[jtu.DictKey, jtu.SequenceKey, jtu.GetAttrKey, jtu.FlattenedIndexKey]


def node_key_to_value(node_key: BuiltInKeyEntry) -> Any:
    """Given a node key, return its value."""
    if isinstance(node_key, jtu.DictKey):
        value = node_key.key
    elif isinstance(node_key, jtu.SequenceKey):
        value = node_key.idx
    elif isinstance(node_key, jtu.GetAttrKey):
        value = node_key.name
    elif isinstance(node_key, jtu.FlattenedIndexKey):
        value = node_key.idx
    else:
        raise ValueError(f"Unsupported node key type: {type(node_key)}")
    return value


def get_child_node_given_key(parent_node: Any, key: BuiltInKeyEntry):
    """Access a child from a parent node given an appropriate key path element."""
    if isinstance(key, jtu.DictKey):
        return parent_node[key.key]
    elif isinstance(key, jtu.SequenceKey):
        return parent_node[key.idx]
    elif isinstance(key, jtu.GetAttrKey):
        # For NamedTupleKey, getattr(node, key.name) is generally how JAX derives paths.
        return getattr(parent_node, key.name)
    elif isinstance(key, jtu.FlattenedIndexKey):
        leaves, _ = eqx.tree_flatten_one_level(parent_node)
        return leaves[key.idx]
    else:
        raise ValueError(f"Unsupported node key type: {type(key)}")


def leaves_with_annotated_path(
    tree: PyTree,
    annotation_func: Callable[[Any], T] = type,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> list[tuple[tuple[tuple[T, BuiltInKeyEntry], ...], Any]]:
    """Gets the leaves of a PyTree like `jax.tree.leaves_with_path`, also returning their node type paths."""
    leaves_with_path = jt.leaves_with_path(tree, is_leaf=is_leaf)
    if not leaves_with_path:
        return []

    key_paths, leaves = zip(*leaves_with_path)
    annotated_key_paths = []

    for key_path in key_paths:
        current_node = tree
        annotations_path = [annotation_func(current_node)]

        for key in key_path:
            current_node = get_child_node_given_key(current_node, key)
            annotations_path.append(annotation_func(current_node))

        annotated_key_paths.append(tuple(zip(annotations_path, key_path)))

    return list(zip(annotated_key_paths, leaves))


def _path_to_label(path: Sequence[BuiltInKeyEntry], join_with: str) -> str:
    # TODO: format based on key type; e.g. f"[{idx}]" for SequenceKey
    return join_with.join(map(lambda k: str(node_key_to_value(k)), path))


def labels(
    tree: PyTree[Any, "T"],
    join_with: str = "_",
    append_leaf: bool = False,
    path_idx: Union[int, slice] = slice(None),
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[str, "T"]:
    """Return a PyTree of labels based on each leaf's key path.

    !!! Note ""
        When `tree` is a flat dict:

        ```python
        labels(tree) == {k: str(k) for k in tree.keys()}
        ```

        When `tree` is a flat list:

        ```python
        labels(tree) == [str(i) for i in range(len(tree))]
        ```

    !!! Example "Verbose `tree_map`"
        This function is useful for creating descriptive labels when using `tree_map`
        to apply an expensive operation to a PyTree.

        ```python
        def expensive_op(x):
            # Something time-consuming
            ...

        def verbose_expensive_op(leaf, label):
            print(f"Processing leaf: {label}")
            return expensive_op(leaf)

        result = tree_map(
            verbose_expensive_op,
            tree,
            labels(tree),
        )
        ```

        A similar use case combines this function with
        [`map_tqdm`][jax_cookbook.tree.map_tqdm] to label a progress bar:

        ```python
        result = map_tqdm(
            expensive_op,
            tree,
            labels=labels(tree),
        )
        ```

    Arguments:
        tree: The PyTree for which to generate labels.
        join_with: The string with which to join a leaf's path keys, to form its label.
        append_leaf: Whether to append the string representation of the leaf to its label.
        path_idx: Index applied to each leaf's path before converting to a label. Useful to
            omit the beginning/end of the path from the label.
        is_leaf: An optional function that returns a boolean, which determines whether each
            node in `tree` should be treated as a leaf.
    """
    leaves_with_path, treedef = jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)
    paths, leaves = zip(*leaves_with_path)
    labels = [_path_to_label(path[path_idx], join_with) for path in paths]
    if append_leaf:
        labels = [join_with.join([label, str(leaf)]) for label, leaf in zip(labels, leaves)]
    return jt.unflatten(treedef, labels)


def key_tuples(
    tree: PyTree[Any, "T"],
    keys_to_strs: bool = False,
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[str, "T"]:
    leaves_with_path, treedef = jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)
    paths, leaves = zip(*leaves_with_path)
    if keys_to_strs:
        leaves = jt.map(lambda k: str(node_key_to_value(k)), paths)
    else:
        leaves = paths
    return jt.unflatten(treedef, leaves)


def _equal_or_allclose(a, b, rtol, atol):
    if isinstance(a, Array) and isinstance(b, Array):
        if not a.shape == b.shape:
            return False
        return jnp.allclose(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not a.shape == b.shape:
            return False
        return np.allclose(a, b, rtol=rtol, atol=atol)
    else:
        return a == b


def paths_of_equal_leaves(
    tree: PyTree[Any, "T"],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[set[tuple[BuiltInKeyEntry]], "T"]:
    """
    Returns a PyTree with the same structure, where leaves are sets of paths of other
    leaves that are equal.

    Does pairwise equality comparisons between all leaves, using `(j)np.allclose` in
    case of arrays.

    Note:
        This is inefficient and should only be used for small-ish PyTrees.
    """
    leaves_with_path, treedef = jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)

    paths, leaves = zip(*leaves_with_path)

    equal_paths = [
        set(
            paths[j]
            for j in range(len(leaves))
            if i != j and _equal_or_allclose(leaves[i], leaves[j], rtol, atol)
        )
        for i in range(len(leaves))
    ]

    return jt.unflatten(treedef, equal_paths)


def labels_of_equal_leaves(
    tree: PyTree[Any, "T"],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    join_with: str = "_",
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[set[str], "T"]:
    """Returns a PyTree with the same structure, where leaves are sets of labels of
    other leaves that are equal.

    Does pairwise equality comparisons between all leaves, using `(j)np.allclose` in
    case of arrays.
    """
    tree_equal_paths = paths_of_equal_leaves(tree, is_leaf=is_leaf, rtol=rtol, atol=atol)
    return jt.map(
        lambda xs: set(_path_to_label(x, join_with) for x in xs),
        tree_equal_paths,
        is_leaf=lambda x: isinstance(x, set),
    )


def infer_batch_size(tree: PyTree, exclude: Callable[..., bool] = lambda _: False) -> int:
    """Return the size of the first dimension of a tree's array leaves.

    Raise an error if any of the array leaves differ in the size of their first
    dimension.

    Arguments:
        tree: The PyTree to infer the batch size of.
        exclude: A function that returns `True` for nodes that should be treated
            as leaves and excluded from the check. This is useful when there are
            subtrees of a certain type, that contain array leaves which do not
            possess the batch dimension.
    """
    # TODO: Allow for `in_axes`-like control over which arrays will be checked

    leaves, treedef = jt.flatten(eqx.filter(tree, eqx.is_array), is_leaf=exclude)
    array_lens: list[Optional[int]] = [
        leaf.shape[0] if not exclude(leaf) else None for leaf in leaves
    ]
    array_lens_unique = set(x for x in array_lens if x is not None)
    if not len(array_lens_unique) == 1:
        tree_array_lens = jt.unflatten(treedef, array_lens)
        raise ValueError(
            "Not all array leaves have the same first dimension size\n\n"
            f"First dimension sizes:\n\n{tree_array_lens}"
        )
    return array_lens_unique.pop()


def leaves_of_type(leaf_type, tree):
    """Return the elements of a PyTree with type `leaf_type`.

    Note that (what would otherwise be) subtrees of tree are treated as leaves if they match
    `leaf_type`.
    """
    return jt.leaves(
        jt.map(
            lambda x: x if isinstance(x, leaf_type) else None,
            tree,
            is_leaf=is_type(leaf_type),
        ),
        is_leaf=is_type(leaf_type),
    )

    # return [x for x in jt.leaves(tree, is_leaf=is_type(leaf_type)) if isinstance(x, leaf_type)]


def _iter_aux_data(node_def, node_cls):
    """Yields the aux_data of all `node_cls` nodes in a PyTreeDef, depth first."""
    node_data = node_def.node_data()
    if node_data is not None:
        cls, aux = node_data
        if cls is node_cls:
            yield aux
    for child in node_def.children():
        yield from _iter_aux_data(child, node_cls)


def collect_aux_data(treedef: PyTreeDef, node_cls: type):
    """
    Returns a list of all the aux_data values that were attached
    to instances of `node_cls` in this pytree def, without side-effects.
    """
    return list(_iter_aux_data(treedef, node_cls))


#! TODO: Maintain node types, not just base dicts
def _rebuild_mapping(original: dict, data: dict, *, preserve_type: bool) -> dict:
    if not preserve_type or type(original) is dict:
        return data
    if isinstance(original, defaultdict):
        rebuilt = type(original)(original.default_factory)
        rebuilt.update(data)
        return rebuilt
    try:
        return type(original)(data)
    except Exception:
        return data


def expand_split_keys(tree: PyTree, key_sep: str = ".", *, preserve_type: bool = True) -> PyTree:
    """Expand dotted keys inside mappings into nested dicts, recursively.

    Args:
        tree: PyTree possibly containing mappings with dotted keys.
        key_sep: Separator to split keys on.
        preserve_type: Preserve mapping types when possible.
    """
    if isinstance(tree, dict):
        out = {}
        for key, val in tree.items():
            val = expand_split_keys(val, key_sep=key_sep, preserve_type=preserve_type)
            if isinstance(key, str) and key_sep in key:
                parts = key.split(key_sep)
                cur = out
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]
                last = parts[-1]
                if isinstance(val, dict) and isinstance(cur.get(last), dict):
                    cur[last] = deep_merge(cur[last], val)
                else:
                    cur[last] = val
            else:
                if isinstance(val, dict) and isinstance(out.get(key), dict):
                    out[key] = deep_merge(out[key], val)
                else:
                    out[key] = val
        return _rebuild_mapping(tree, out, preserve_type=preserve_type)
    else:
        leaves, treedef = jt.flatten(tree, is_leaf=is_type(dict))
        if jtu.treedef_is_leaf(treedef) and not isinstance(leaves, dict):
            return tree
        return jt.unflatten(
            treedef,
            [expand_split_keys(leaf, key_sep=key_sep, preserve_type=preserve_type) for leaf in leaves],
        )


def _annotation_func(node):
    #! For use with `jtree.leaves_with_annotated_path`
    if is_type(LDict)(node):
        return LDict.of(node.label)
    else:
        return type(node)


def tree_level_types(tree: PyTree, is_leaf=falsef) -> list[type | LDictConstructor]:
    """Given a PyTree, return a PyTree of the types of each node along the path to the first leaf."""
    # Get the path to the first leaf
    leaves_with_typed_paths = leaves_with_annotated_path(
        tree,
        is_leaf=is_leaf,
        annotation_func=_annotation_func,
    )
    if not leaves_with_typed_paths:
        return []
    first_path, _ = leaves_with_typed_paths[0]

    node_types = [node_type for node_type, _ in first_path]

    return node_types


LevelSpec: TypeAlias = str | type | LDictConstructor | EllipsisType


def _as_descriptor(x: LevelSpec) -> Any:
    if x is Ellipsis:
        return Ellipsis
    if isinstance(x, str):
        if LDict is None:
            raise TypeError("String level specs require LDict.of(...).")
        return LDict.of(x)
    if isinstance(x, (LDictConstructor, type)):  # type: ignore[arg-type]
        return x
    raise TypeError(f"Unsupported level specifier: {x!r}")


def _make_matcher(token: LevelSpec) -> Callable[[Any], bool]:
    if token is Ellipsis:
        raise ValueError("Internal: matcher requested for Ellipsis.")
    if isinstance(token, str):
        if LDict is None:
            raise TypeError("String level specs require LDict.")
        return LDict.is_of(token)
    if isinstance(token, LDictConstructor):
        return token.predicate
    if isinstance(token, type):
        T = token
        return lambda node: isinstance(node, T)
    raise TypeError(f"Unsupported spec token: {token!r}")


def _descriptor_key(x: Any) -> tuple[str, Any]:
    if isinstance(x, LDictConstructor):
        return ("ldict", x.label)
    if isinstance(x, type):
        return ("type", x)
    raise TypeError(f"Unsupported level descriptor: {x!r}")


def first_path_node_structure(
    tree: Any,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> tuple[list[Any], list[jtu.PyTreeDef]]:
    """Returns (nodes, node_treedefs) along a first-child path.

    Here, `node_treedefs` are the *flat* structures of each node along the path, i.e. *as if their
    children were leaves*. In the case that the tree is homogeneous (i.e. transposable) at every
    level, we can recover the treedef for the full tree using (for example):

    ```
    treedef = functools.reduce(lambda td1, td2: td1.compose(td2), node_treedefs)
    ```
    """
    nodes: list[Any] = []
    treedefs: list[jtu.PyTreeDef] = []

    node = tree
    while True:
        if is_leaf is not None and is_leaf(node):
            break
        children, td = eqx.tree_flatten_one_level(node)
        if jtu.treedef_is_leaf(td):
            break
        nodes.append(node)
        treedefs.append(td)
        if not children:
            break
        node = children[0]  # uniformity assumption

    return nodes, treedefs


def _assert_uniform_tree(
    tree: PyTree,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> None:
    """Raise if the tree is not uniform at each level (respecting is_leaf)."""
    level_nodes = [tree]
    depth = 0
    while level_nodes:
        next_nodes: list[Any] = []
        level_treedef: Optional[jtu.PyTreeDef] = None
        saw_leaf = False
        saw_nonleaf = False
        for node in level_nodes:
            if is_leaf is not None and is_leaf(node):
                saw_leaf = True
                continue
            children, td = eqx.tree_flatten_one_level(node)
            if jtu.treedef_is_leaf(td):
                saw_leaf = True
                continue
            saw_nonleaf = True
            if level_treedef is None:
                level_treedef = td
            elif td != level_treedef:
                raise ValueError(
                    f"Non-uniform tree at depth {depth}: differing node structures."
                )
            next_nodes.extend(children)
        if saw_leaf and saw_nonleaf:
            raise ValueError(
                f"Non-uniform tree at depth {depth}: mixture of leaves and non-leaves."
            )
        if not saw_nonleaf:
            break
        level_nodes = next_nodes
        depth += 1


def _expand_spec_to_target(spec: Sequence[LevelSpec], current: list[Any]) -> list[Any]:
    """Expand possibly-ellipsized `spec` into a full target list (no Ellipsis)."""
    if not current:
        return []
    spec_norm = [_as_descriptor(x) for x in spec]
    if sum(1 for x in spec_norm if x is Ellipsis) > 1:
        raise ValueError("`spec` may contain at most one Ellipsis (`...`).")

    cur_keys = [_descriptor_key(x) for x in current]
    specified = [x for x in spec_norm if x is not Ellipsis]

    def _consume_occurrences(tokens: list[Any]) -> list[bool]:
        used = [False] * len(current)
        for tok in tokens:
            key = _descriptor_key(tok)
            for i, cur_key in enumerate(cur_keys):
                if not used[i] and cur_key == key:
                    used[i] = True
                    break
            else:
                raise ValueError(
                    "Unknown level(s) in spec (not present in tree): " + repr(tok)
                )
        return used

    if Ellipsis in spec_norm:
        pos = spec_norm.index(Ellipsis)
        before = [x for x in spec_norm[:pos] if x is not Ellipsis]
        after = [x for x in spec_norm[pos + 1 :] if x is not Ellipsis]
        used = _consume_occurrences(before + after)
        middle = [x for i, x in enumerate(current) if not used[i]]
        return before + middle + after
    else:
        used = _consume_occurrences(specified)
        rest = [x for i, x in enumerate(current) if not used[i]]
        return list(spec_norm) + rest


def _order_from_target(probe_nodes: list[Any], target: list[Any]) -> list[int]:
    """Greedily match each target descriptor to the next unused axis in probe_nodes."""
    depth = len(probe_nodes)
    if len(target) != depth:
        raise AssertionError("Target length must equal number of probed levels.")
    used = [False] * depth
    order: list[int] = []
    for tok in target:
        matcher = _make_matcher(
            tok if tok is not Ellipsis else target[0]
        )  # tok is never Ellipsis here
        for i, node in enumerate(probe_nodes):
            if not used[i] and matcher(node):
                used[i] = True
                order.append(i)
                break
        else:
            raise ValueError(f"Level {tok!r} not found (or already used) in tree.")
    return order


def _compose_treedefs_in_order(treedefs: list[jtu.PyTreeDef], order: list[int]) -> jtu.PyTreeDef:
    td = treedefs[order[0]]
    for idx in order[1:]:
        td = td.compose(treedefs[idx])  # outer→inner
    return td


def _leaf_permutation_for_axes(sizes: list[int], order: list[int]) -> np.ndarray:
    """Index map idx s.t. new_leaves = [old_leaves[i] for i in idx]."""
    if not sizes:
        return np.array([], dtype=int)
    N = math.prod(sizes)
    if N == 0:
        return np.array([], dtype=int)
    return np.arange(N, dtype=int).reshape(sizes).transpose(order).ravel()


def rearrange_uniform_tree(
    tree: PyTree,
    spec: Sequence[LevelSpec],
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree:
    """Reorder levels of a uniform PyTree according to `spec`.

    Args:
        tree: A PyTree whose nodes at each level all have the same structure (treedef).
            Note that there is no requirement that all the leaves are the same type or shape,
            so long as `is_leaf` demarcates a uniform level at the bottom of the tree.
        spec: A sequence of level descriptors, specifying how to rearrange the tree. At most one
            Ellipsis (`...`) may be included, which keeps the remaining levels in their original
            relative order.
        is_leaf: An optional function that returns `True` for nodes which should be treated as
            leaves; i.e. we will only deal with levels above this boundary, and any transposition
            of a level to the "bottom" will result in it being placed just above this boundary.

    Note:
        Level descriptors in `spec` may be:
            • `str` (interpreted as `LDict.of(label)`)
            • `type` objects (e.g. `tuple`, `list`, custom classes)
            • `LDictConstructor` (e.g. `LDict.of("label")`).
            • at most one `Ellipsis`/`...` (keeps the remaining levels in original relative order;
            the order is defined by `tree_level_types(tree, is_leaf)`).

        If the tree contains repeated level types (e.g. two `list` levels), then each
        descriptor occurrence in `spec` matches the next unused level of that type
        from the outside in.
    """
    _assert_uniform_tree(tree, is_leaf=is_leaf)

    # Use your current ordering to expand spec → full target (no Ellipsis).
    current = tree_level_types(tree, is_leaf=is_leaf)
    target = _expand_spec_to_target(spec, current)

    # No-op if there is nothing to rearrange (empty or leaf-only).
    probe_nodes, _ = first_path_node_structure(tree, is_leaf=is_leaf)
    if not probe_nodes:
        return tree

    class _Wrap:
        __slots__ = ("x",)

        def __init__(self, x):
            self.x = x

    def _cut_to_front(t: Any, descriptor: Any) -> Any:
        outer_td = jt.structure(t, is_leaf=_make_matcher(descriptor))
        if is_leaf is None:
            return jt.transpose(outer_td, None, t)
        wrapped = jt.map(lambda x: _Wrap(x), t, is_leaf=is_leaf)
        transposed = jt.transpose(outer_td, None, wrapped)
        return jt.map(
            lambda x: x.x if isinstance(x, _Wrap) else x,
            transposed,
            is_leaf=lambda x: isinstance(x, _Wrap),
        )

    levels = list(current)
    placed: list[Any] = []

    for i, want in enumerate(target):
        if _descriptor_key(levels[i]) == _descriptor_key(want):
            placed.append(want)
            continue

        # Bring the desired level to the front.
        tree = _cut_to_front(tree, want)
        idx = next(j for j, d in enumerate(levels) if _descriptor_key(d) == _descriptor_key(want))
        levels = [levels[idx]] + levels[:idx] + levels[idx + 1 :]

        # Slide it behind the already-placed prefix.
        for prev in reversed(placed):
            tree = _cut_to_front(tree, prev)
            jdx = next(
                j for j, d in enumerate(levels) if _descriptor_key(d) == _descriptor_key(prev)
            )
            levels = [levels[jdx]] + levels[:jdx] + levels[jdx + 1 :]

        placed.append(want)

    return tree


# def swap_adjacent_levels(
#     is_outer: Callable[[Any], bool],
#     is_inner: Optional[Callable[[Any], bool]] = None,
#     tree: PyTree,
# ):
#     """Swap two adjacent levels of PyTree.

#     Arguments:
#         is_outer: A function that returns `True` for nodes in the outer level to swap.
#         is_inner: A function that returns `True` for nodes in the inner level to swap.
#         tree: The PyTree to swap levels in.
#     """

#     subtrees, outer_treedef = jt.flatten(tree, is_leaf=is_outer)
#     is_leaf = None
#     def transpose_subtree(subtree):

#     subtrees_transposed = [
#         jt.transpose(jt.structure(subtree)) for subtree in inner_wrapped
#     ]
