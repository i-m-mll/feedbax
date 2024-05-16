"""Tools for manipulation of PyTrees.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import Any, Optional, Tuple, TypeVar, TypeVarTuple, Union, get_type_hints

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, PyTreeDef, Shaped
import numpy as np

from feedbax._progress import _tqdm, _tqdm_write
from feedbax.misc import unique_generator, is_module


logger = logging.getLogger(__name__)


T = TypeVar("T")


def filter_spec_leaves(
    tree: PyTree[Any, "T"], leaf_func: Callable,
) -> PyTree[bool, "T"]:
    """Returns a filter specification for tree leaves matching `leaf_func`."""
    filter_spec = jax.tree_util.tree_map(lambda _: False, tree)
    filter_spec = eqx.tree_at(
        leaf_func,
        filter_spec,
        replace_fn=lambda x: True,
    )
    return filter_spec


def tree_index(tree: PyTree[Any, "T"], index: int) -> PyTree[Any, "T"]:
    """Returns the same PyTree, indexing out all of its array leaves."""
    models_arrays, models_other = eqx.partition(tree, eqx.is_array)
    return eqx.combine(
        jax.tree_map(lambda x: x[index], models_arrays),
        models_other,
    )


def get_ensemble(
    func: Callable[..., PyTree[Any, "S"]],
    *args: Any,
    n_ensemble: int,
    key: PRNGKeyArray,
    **kwargs: Any,
) -> PyTree[Any, "S"]:
    """Vmap a function over a set of random keys.

    Arguments:
        func: A function that returns a PyTree, and whose final keyword argument
            is `key: PRNGKeyArray`.
        n_ensemble: The number of keys to split; i.e. the size of the batch
            dimensions in the array leaves of the returned PyTree.
        *args: The positional arguments to `func`.
        key: The key to split to perform the vmap.
        **kwargs: The keyword arguments to `func`.
    """
    keys = jr.split(key, n_ensemble)
    func_ = lambda key: func(*args, **kwargs, key=key)
    return eqx.filter_vmap(func_)(keys)


@jax.named_scope("fbx.tree_take")
def tree_take(
    tree: PyTree[Any, "T"],
    indices: ArrayLike,
    axis: int = 0,
    **kwargs: Any,
) -> PyTree[Any, "T"]:
    """Indexes elements out of each array leaf of a PyTree.

    Any non-array leaves are returned unchanged.

    !!! Warning ""
        This function inherits the default indexing behaviour of JAX. If
        out-of-bounds indices are provided, no error will be raised.

    Arguments:
        tree: Any PyTree whose array leaves are equivalently indexable,
            according to the other arguments to this function. For example,
            `axis=0` could be used when the first dimension of every array leaf
            is a batch dimension, and `indices` specifies a subset of examples
            from the batch.
        indices: The indices of the values to take from each array leaf.
        axis: The axis of the array leaves over which to take their values.
            Defaults to 0.
        kwargs: Additional arguments to [`jax.numpy.take`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.take.html).

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
    tree: PyTree[Any | Shaped[Array, "batch *?dims"], "T"],
    items: PyTree[Any | Shaped[Array, "*?dims"], "T"],
    idx: int,
) -> PyTree[Any | Shaped[Array, "batch *?dims"], "T"]:
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
    arrays_update = jax.tree_map(lambda xs, x: xs.at[idx].set(x), arrays, vals_update)
    return eqx.combine(arrays_update, other_update)


def random_split_like_tree(
    key: PRNGKeyArray,
    tree: PyTree[Any, "T"],
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[PRNGKeyArray | None, "T"]:
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
    trees: Sequence[PyTree[Any, "T"]],
    axis: int = 0,
) -> PyTree[Any, "T"]:
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
    """Sum the sums of squares of the leaves of a PyTree."""
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: jnp.sum(x**2), tree)
    )


def tree_sum_n_features(tree: PyTree[Array]) -> int:
    """Returns the sum the sizes of the last dimensions of all leaves."""
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: x.shape[-1], tree)
    )


def _tree_map(
    f: Callable[..., Any],
    tree: PyTree[Any, "T"],
    *rest,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[Any, "T"]:
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


S = TypeVar("S")


def tree_map_module(
    f: Callable[[Any], S],
    tree: PyTree[Any, "T"],
    *rest,
) -> PyTree[S, "T"]:
    """Custom `tree_map` that treats `eqx.Module`s as leaves.

    This is a convenience for performing analyses involving mapping
    repeatedly over PyTrees of `eqx.Module`, where it would be repetitive
    to write `is_leaf=lambda x: isinstance(x, eqx.Module)` every time.
    """

    return jax.tree_map(f, tree, *rest, is_leaf=is_module)


# Horizontal rule
HR = u'\u2500' * 80


# TODO: Use a host callback so this can be wrapped in JAX transformations.
# See https://github.com/jeremiecoullon/jax-tqdm for a similar example.
# (Currently I only use this function when `f` is a `TaskTrainer`.)
def tree_map_tqdm(
    f: Callable[..., S],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    labels: Optional[PyTree[str, "T"]] = None,
    verbose: bool = False,
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[S, "T"]:
    """Adds a progress bar to `tree_map`.

    Arguments:
        f: The function to map over the tree.
        tree: The PyTree to map over.
        *rest: Additional arguments to `f`, as PyTrees with the same structure as `tree`.
        labels: A PyTree of labels for the leaves of `tree`, to be displayed on the
            progress bar.
        is_leaf: A function that returns `True` for leaves of `tree`.
    """
    n_leaves = len(jtu.tree_leaves(tree, is_leaf=is_leaf))
    pbar = _tqdm(total=n_leaves)
    def _f(leaf, label, *rest):
        if label is not None:
            pbar.set_description(f"Processing leaf: {label}")
        if verbose:
            _tqdm_write(f"Processing leaf: {label}\n\n")
        result = f(leaf, *rest)
        if verbose:
            _tqdm_write(f"\n{HR}\n")
        else:
            _tqdm_write(f"\n")
        pbar.update(1)
        return result
    if labels is None:
        pbar.set_description("Processing tree leaves")
        labels = jax.tree_map(lambda _: None, tree, is_leaf=is_leaf)
    return jax.tree_map(_f, tree, labels, *rest, is_leaf=is_leaf)


def tree_map_unzip(
    f: Callable[..., Tuple[Any, ...]],
    tree: PyTree[Any, "T"],
    *rest: PyTree[Any, "T"],
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Tuple[PyTree[Any, "T"], ...]:
    """Maps a tuple-valued function over a PyTree. Returns a tuple of PyTrees.

    For example, for a function `f(x) -> (y, z)`, we can do `ys, zs =
    tree_map_unzip(f, xs)` where `ys`, `zs` are PyTrees, whereas with a normal
    `tree_map` we'd get a single PyTree of tuples `(y, z)`.
    """
    results = jax.tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)
    return tree_unzip(results)


def tree_unzip(
    tree: PyTree[Tuple[Any, ...], "T"],
) -> Tuple[PyTree[Any, "T"], ...]:
    """Unzips a PyTree of tuples into a tuple of PyTrees.

    !!! Note
        Something similar could be done with `tree_transpose`, but `outer_treedef`
        would need to be specified.

        This version has `zip`-like behaviour, in that 1) the input tree should be
        flattenable to tuples, when tuples are treated as leaves; 2) the shortest
        of those tuples determines the length of the output.
    """
    tree_flat, treedef = jtu.tree_flatten(tree, is_leaf=lambda x: isinstance(x, tuple))
    if any(not isinstance(x, tuple) for x in tree_flat):
        raise ValueError("The input pytree is not flattenable to tuples")
    tree_flat_unzipped = zip(*tree_flat)
    return tuple(jtu.tree_unflatten(treedef, x) for x in tree_flat_unzipped)


def tree_call(
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
    callables_values = jax.tree_map(
        lambda x: x(*args, **kwargs),
        callables,
        is_leaf=is_leaf,
    )
    return eqx.combine(callables_values, other_values, is_leaf=is_leaf)


def tree_array_bytes(tree: PyTree, duplicates: bool = False) -> int:
    """Returns the total bytes of memory over all array leaves of a PyTree.

    Arguments:
        tree: The tree with arrays to measure.
        duplicates: If `False`, then leaves that refer to the same array in memory
            will only be counted once.
    """
    arrays = eqx.filter(tree, eqx.is_array)
    if not duplicates:
        flat, treedef = jtu.tree_flatten(arrays)
        arrays = jtu.tree_unflatten(
            treedef,
            list(unique_generator(flat, replace_duplicates=True))
        )
    array_bytes = jax.tree_map(lambda x: x.nbytes, arrays)
    array_bytes_int_leaves = [x for x in jtu.tree_leaves(array_bytes) if x is not None]
    return sum(array_bytes_int_leaves)



def tree_struct_bytes(tree: PyTree[jax.ShapeDtypeStruct]) -> int:
    """Returns the total bytes of memory implied by a PyTree of `ShapeDtypeStruct`s."""
    structs = eqx.filter(tree, lambda x: isinstance(x, jax.ShapeDtypeStruct))
    struct_bytes = jax.tree_map(lambda x: x.size * x.dtype.itemsize, structs)
    return jtu.tree_reduce(lambda x, y: x + y, struct_bytes)


BuiltInKeyEntry = Union[jtu.DictKey, jtu.SequenceKey, jtu.GetAttrKey, jtu.FlattenedIndexKey]


def _node_key_to_label(node_key: BuiltInKeyEntry) -> str:
    if isinstance(node_key, jtu.DictKey):
        label = str(node_key.key)
    elif isinstance(node_key, jtu.SequenceKey):
        label = str(node_key.idx)
    elif isinstance(node_key, jtu.GetAttrKey):
        label = str(node_key.name)
    elif isinstance(node_key, jtu.FlattenedIndexKey):
        label = str(node_key.key)
    else:
        raise ValueError(f"Unknown PyTree node key type: {type(node_key)}")
    return label


def _path_to_label(path: Sequence[BuiltInKeyEntry], join_with: str) -> str:
    # TODO: format based on key type; e.g. f"[{idx}]" for SequenceKey
    return join_with.join(map(_node_key_to_label, path))


def tree_labels(
    tree: PyTree[Any, 'T'],
    join_with: str = '_',
    append_leaf: bool = False,
    path_idx: int | slice = slice(None),
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[str, 'T']:
    """Return a PyTree of labels based on each leaf's key path.

    !!! Note ""
        When `tree` is a flat dict:

        ```python
        tree_keys(tree) == {k: str(k) for k in tree.keys()}
        ```

        When `tree` is a flat list:

        ```python
        tree_keys(tree) == [str(i) for i in range(len(tree))]
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
            tree_labels(tree),
        )
        ```

        A similar use case combines this function with
        [`tree_map_tqdm`][feedbax.tree_map_tqdm] to label a progress bar:

        ```python
        result = tree_map_tqdm(
            expensive_op,
            tree,
            labels=tree_labels(tree),
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
        labels = [
            join_with.join([label, str(leaf)]) 
            for label, leaf in zip(labels, leaves)
        ]
    return jtu.tree_unflatten(treedef, labels)


def _equal_or_allclose(a, b, rtol, atol):
    if isinstance(a, Array) and isinstance(b, Array):
        if not a.shape == b.shape:
            return False
        return jnp.allclose(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not a.shape == b.shape:
            return False
        return np.allclose(a, b, rtol=rtol, atol=atol)
    elif type(a) != type(b):
        return False
    else:
        return a == b


def tree_paths_of_equal_leaves(
    tree: PyTree[Any, 'T'],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[set[tuple[BuiltInKeyEntry]], 'T']:
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
            paths[j] for j in range(len(leaves))
            if i != j and _equal_or_allclose(leaves[i], leaves[j], rtol, atol)
        )
        for i in range(len(leaves))
    ]

    return jtu.tree_unflatten(treedef, equal_paths)


def tree_labels_of_equal_leaves(
    tree: PyTree[Any, 'T'],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    join_with: str = '_',
    is_leaf: Optional[Callable[..., bool]] = None,
) -> PyTree[set[str], 'T']:
    """Returns a PyTree with the same structure, where leaves are sets of labels of
    other leaves that are equal.

    Does pairwise equality comparisons between all leaves, using `(j)np.allclose` in
    case of arrays.
    """
    tree_equal_paths = tree_paths_of_equal_leaves(
        tree, is_leaf=is_leaf, rtol=rtol, atol=atol
    )
    return jtu.tree_map(
        lambda xs: set(_path_to_label(x, join_with) for x in xs),
        tree_equal_paths,
        is_leaf=lambda x: isinstance(x, set),
    )


def tree_infer_batch_size(
    tree: PyTree, exclude: Callable[..., bool] = lambda _ : False
) -> int:
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
    arrays, treedef = jtu.tree_flatten(eqx.filter(tree, eqx.is_array), is_leaf=exclude)
    array_lens: list[int | None] = [
        arr.shape[0] if not exclude(arr) else None for arr in arrays
    ]
    array_lens_unique = set(x for x in array_lens if x is not None)
    if not len(array_lens_unique) == 1:
        tree_array_lens = jtu.tree_unflatten(treedef, array_lens)
        raise ValueError(
            "Not all array leaves have the same first dimension size\n\n"
            f"First dimension sizes:\n\n{tree_array_lens}"
        )
    return array_lens_unique.pop()