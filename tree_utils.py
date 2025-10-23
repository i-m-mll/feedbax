import hashlib
import inspect
import json
import logging
import math
import re
import types
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, KeysView, Mapping, Sequence
from functools import partial
from types import EllipsisType, SimpleNamespace
from typing import Any, Optional, Sequence, TypeVar, overload

import equinox as eqx
import jax as jax
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from feedbax.intervene import AbstractIntervenor
from jax_cookbook import anyf, hash_callable, is_module, is_none, is_type
from jax_cookbook._func import falsef
from jaxtyping import Array, ArrayLike, PyTree, PyTreeDef

from feedbax_experiments.config import STRINGS
from feedbax_experiments.misc import deep_merge
from feedbax_experiments.types import LDict, LDictConstructor, TreeNamespace, _Wrapped

T = TypeVar("T")
M = TypeVar("M", bound=Mapping)
K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)


def swap_model_trainables(
    model: PyTree[..., "T"], trained: PyTree[..., "T"], where_train: Callable
):
    return eqx.tree_at(
        where_train,
        model,
        where_train(trained),
    )


def _get_mapping_constructor(d: Mapping[K, V]) -> Callable[[Mapping], Mapping]:
    if isinstance(d, LDict):
        # returns (Mapping[K, V]) -> LDict[K, V], which is fine (Mapping is a supertype of LDict)
        return LDict.of(d.label)
    return type(d)


@overload
def subdict(d: LDict[K, V], keys: Sequence[K]) -> LDict[K, V]: ...
@overload
def subdict(d: M, keys: Sequence[K]) -> M: ...
def subdict(d: Mapping[K, V], keys: Sequence[K]) -> Mapping[K, V]:
    ctor = _get_mapping_constructor(d)
    return ctor({k: d[k] for k in keys})


def dictmerge(*dicts: Mapping) -> Mapping:
    if len(set(type(d) for d in dicts)) == 1:
        constructor = _get_mapping_constructor(dicts[0])
    else:
        constructor = dict
    return constructor({k: v for d in dicts for k, v in d.items()})


# TODO: This exists because I was thinking of generalizing the way that
# the model PyTree is constructed in training notebook 2. If that doesn't get done,
# then it would make sense to just do a dict comprehension explicitly when
# constructing the `task_model_pairs` dict, instead of making an
# opaque call to this function.
def map_kwargs_to_dict(
    func: Callable[..., Any],  #! kwargs only
    keyword: str,
    values: Sequence[Any],
):
    """Given a function that takes optional kwargs, evaluate the function over
    a sequence of values of a single kwarg
    """
    return dict(
        zip(
            values,
            map(
                lambda value: func(**{keyword: value}),
                values,
            ),
        )
    )


def ldict_verbose_label_fn(x: Any) -> str:
    if isinstance(x, (LDict, LDictConstructor)):
        return f"LDict.of({x.label})"
    else:
        return x.__name__


def ldict_label_only_fn(x: Any) -> str:
    if isinstance(x, (LDict, LDictConstructor)):
        return x.label
    else:
        return x.__name__


def tree_level_types(tree: PyTree, is_leaf=falsef) -> list[Callable]:
    """Given a PyTree, return a PyTree of the types of each node along the path to the first leaf."""
    # Get the path to the first leaf
    leaves_with_typed_paths = jtree.leaves_with_annotated_path(
        tree,
        is_leaf=is_leaf,
        annotation_func=_annotation_fn,
    )
    if not leaves_with_typed_paths:
        return []
    first_path, _ = leaves_with_typed_paths[0]

    node_types = [node_type for node_type, _ in first_path]

    return node_types


def tree_level_labels(
    tree: PyTree,
    sep: Optional[str] = None,
    label_fn: Callable[..., str] = ldict_label_only_fn,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> list[str]:
    """
    Given a PyTree, return a list of labels, one for each level of the tree.

    This function assumes a consistent tree structure where all nodes at the same level
    have the same type/label. Traverses the tree from root to first leaf, collecting labels
    along the way.
    """
    node_types = tree_level_types(tree, is_leaf=is_leaf)

    # Collect the labels from all LDict nodes in the path
    labels = [label_fn(node_type) for node_type in node_types]

    if sep is not None:
        labels = [label.replace(STRINGS.hps_level_label_sep, sep) for label in labels]

    return labels


def _annotation_fn(node):
    #! For use with `jtree.leaves_with_annotated_path`
    if is_type(LDict)(node):
        return LDict.of(node.label)
    else:
        return type(node)


def print_ldict_tree_summary(tree):
    """Prints a summary of the labels and keys of a pure `LDict` tree."""
    while True:
        try:
            print(tree.label, "\t", list(tree.keys()))
            tree = next(iter(tree.values()))
        except AttributeError:
            break


def swap_adjacent_ldict_levels(
    outer_label: str,
    inner_label: str,
    tree,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """
    Swap an immediately-nested LDict pair `(..., outer_label, inner_label, ...)`
    so it becomes `(..., inner_label, outer_label, ...)`
    (and leave everything else untouched).

    Build the new sub-tree explicitly, keeping the original
    left-to-right order of the *other* levels.
    """

    is_outer = LDict.is_of(outer_label)
    is_inner = LDict.is_of(inner_label)

    def _swap_one(node: LDict):
        assert node.label == outer_label  # guaranteed by caller

        buckets: dict[Any, dict[Any, Any]] = defaultdict(dict)
        for outer_key, inner_ldict in node.items():
            if not is_inner(inner_ldict):
                if isinstance(inner_ldict, LDict):
                    label = f"LDict.of({inner_ldict.label})"
                else:
                    label = type(inner_ldict)
                raise ValueError(
                    f"{outer_label} was expected to hold only {inner_label} children, found {label}"
                )
            for inner_key, leaf in inner_ldict.items():
                buckets[inner_key][outer_key] = leaf

        # Re-wrap every bucket in an outer-label LDict,
        # then wrap the whole mapping in an inner-label LDict.
        swapped = LDict(
            inner_label,
            {k_in: LDict(outer_label, v_out) for k_in, v_out in buckets.items()},
        )
        return swapped

    # -------------------------------------------------------------------
    # walk the whole pytree but transform only outer-label nodes
    # -------------------------------------------------------------------
    def _maybe_swap(node):
        return _swap_one(node) if is_outer(node) else node

    def _stop_descent(node):
        # Do *not* look inside an outer-label node, and honour any
        # user-supplied `is_leaf`.
        return is_outer(node) or (is_leaf(node) if is_leaf is not None else False)

    return jt.map(_maybe_swap, tree, is_leaf=_stop_descent)


def _bubble_to_index(
    tree,
    levels: list[str],
    label: str,
    dst_idx: int,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """
    Move `label` to `dst_idx` by adjacent swaps, updating `levels` in lock-step.
    Returns (tree, levels).
    """
    i = levels.index(label)

    # Bubble inward (toward larger index) until we reach dst
    while i < dst_idx:
        outer = levels[i]  # current node
        inner = levels[i + 1]  # its child
        tree = swap_adjacent_ldict_levels(
            outer_label=outer, inner_label=inner, tree=tree, is_leaf=is_leaf
        )
        levels[i], levels[i + 1] = levels[i + 1], levels[i]
        i += 1

    # Bubble outward (toward smaller index) until we reach dst
    while i > dst_idx:
        outer = levels[i - 1]  # parent
        inner = levels[i]  # current node
        tree = swap_adjacent_ldict_levels(
            outer_label=outer, inner_label=inner, tree=tree, is_leaf=is_leaf
        )
        levels[i - 1], levels[i] = levels[i], levels[i - 1]
        i -= 1

    return tree, levels


def _normalize_label(s: str) -> str:
    # Accept both raw "label" and formatted "LDict.of(label)".
    if isinstance(s, str) and s.startswith("LDict.of(") and s.endswith(")"):
        return s[len("LDict.of(") : -1]
    return s


def _levels_raw(tree, *, is_leaf=None) -> list[str]:
    """Get current LDict level labels as raw strings (no 'LDict.of(...)')."""
    return [_normalize_label(s) for s in tree_level_labels(tree, is_leaf=is_leaf)]


def move_ldict_level_above(
    label_to_move: str,
    anchor_label: str,
    tree,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """
    Place `label_to_move` immediately *outside* (just above) `anchor_label`.

    - If mover is already outside the anchor: bubble inward to index (anchor_idx - 1).
    - If mover is inside the anchor: bubble outward to index (anchor_idx).
      (After the final swap with the anchor, mover sits right above it.)
    """
    levels = _levels_raw(tree, is_leaf=is_leaf)

    if label_to_move not in levels:
        raise ValueError(f"`label_to_move` not found among levels: {label_to_move!r}")
    if anchor_label not in levels:
        raise ValueError(f"`anchor_label` not found among levels: {anchor_label!r}")

    if label_to_move == anchor_label:
        return tree

    if not levels:
        return tree

    i_anchor = levels.index(anchor_label)
    i_move = levels.index(label_to_move)

    dst_idx = i_anchor - 1 if i_move < i_anchor else i_anchor
    tree, _ = _bubble_to_index(tree, levels, label_to_move, dst_idx, is_leaf=is_leaf)
    return tree


def ldict_level_to_top(
    label: str,
    tree,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """Move `label` to the outermost level, preserving other relative order."""
    levels = _levels_raw(tree, is_leaf=is_leaf)
    if not levels:
        return tree
    if label not in levels:
        raise ValueError(f"label not found among levels: {label!r}")
    tree, _ = _bubble_to_index(tree, levels, label, 0, is_leaf=is_leaf)
    return tree


def ldict_level_to_bottom(
    label: str,
    tree,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """Move `label` to the innermost level, preserving other relative order."""
    levels = _levels_raw(tree, is_leaf=is_leaf)
    if not levels:
        return tree
    if label not in levels:
        raise ValueError(f"label not found among levels: {label!r}")
    tree, _ = _bubble_to_index(tree, levels, label, len(levels) - 1, is_leaf=is_leaf)
    return tree


def getitem_at_level(level: str, key: Any, tree: PyTree) -> Any:
    """Get an item from a PyTree at a specific ldict level."""
    return jt.map(
        lambda x: x[key] if isinstance(x, LDict) else x,
        tree,
        is_leaf=LDict.is_of(level),
    )


def rearrange_ldict_levels(
    tree,
    spec: Sequence[str | EllipsisType],
    *,
    is_leaf=None,
):
    """
    Reorder LDict levels according to `spec`.

    `spec` may contain 0 or 1 Ellipsis (the literal `...`):
      - If Ellipsis is present: items before `...` become the top (outermost)
        in the given order; items after `...` become the bottom (innermost) in
        the given order; all other levels (not named in `spec`) remain in the
        middle, preserving their original relative order.
      - If Ellipsis is absent: bring the listed labels to the top (outermost)
        in the given order; all other levels follow in their original
        relative order.

    Assumes each level label is unique along the LDict stack.

    Raises:
      - ValueError for unknown labels, duplicates in `spec`, or more than one
        ellipsis.
    """
    # Current order of LDict level labels
    current_levels = tree_level_labels(tree, is_leaf=is_leaf)
    if not current_levels:
        return tree

    # Normalize spec & validate
    spec_list = list(spec)
    ell_count = sum(1 for x in spec_list if x is Ellipsis)
    if ell_count > 1:
        raise ValueError("`spec` may contain at most one Ellipsis (`...`).")

    spec_labels = [x for x in spec_list if x is not Ellipsis]
    # duplicate labels in spec?
    if len(set(spec_labels)) != len(spec_labels):
        dups = [lbl for lbl in spec_labels if spec_labels.count(lbl) > 1]
        raise ValueError(f"Duplicate label(s) in spec: {sorted(set(dups))}")

    # unknown labels?
    missing = [lbl for lbl in spec_labels if lbl not in current_levels]
    if missing:
        raise ValueError(f"Unknown level label(s): {missing}")

    if ell_count == 1:
        ell_ix = spec_list.index(Ellipsis)
        top_labels = list(spec_list[:ell_ix])  # before ...
        bottom_labels = list(spec_list[ell_ix + 1 :])  # after ...
        # safety: labels around ellipsis are strings
        top_labels = [x for x in top_labels if x is not Ellipsis]
        bottom_labels = [x for x in bottom_labels if x is not Ellipsis]
        if set(top_labels) & set(bottom_labels):
            both = sorted(set(top_labels) & set(bottom_labels))
            raise ValueError(f"Label(s) appear on both sides of Ellipsis: {both}")
    else:
        # No ellipsis: interpret as "bring these to the top"
        top_labels = spec_labels
        bottom_labels = []

    # Middle labels keep their original relative order
    pinned = set(top_labels) | set(bottom_labels)
    middle_labels = [lbl for lbl in current_levels if lbl not in pinned]

    target = list(top_labels) + middle_labels + list(bottom_labels)

    # Now transform the tree to match `target`, using adjacent swaps.
    # We mirror the same swaps in an in-memory `levels` list so we
    # don’t have to recompute labels each step.
    levels = list(current_levels)

    for i, want in enumerate(target):
        if levels[i] == want:
            continue
        # Find where the desired label currently sits
        j = levels.index(want)
        # Bubble it left to position i by swapping adjacent pairs
        while j > i:
            outer = levels[j - 1]
            inner = levels[j]
            tree = swap_adjacent_ldict_levels(
                outer_label=outer,
                inner_label=inner,
                tree=tree,
                is_leaf=is_leaf,
            )
            # Keep our local mirror in sync
            levels[j - 1], levels[j] = levels[j], levels[j - 1]
            j -= 1

    return tree


def ldict_level_keys(tree: PyTree, label: str) -> KeysView:
    """Returns the keys of the first `LDict` node with the given label."""
    return jtree.first_leaf(tree, is_leaf=LDict.is_of(label)).keys()


# def align_levels_and_map(
#     func, tree, *rest, is_leaf=None,
# ):
#     """
#     Map over the nodes within the lowest `LDict` level of `tree`.

#     Before mapping, rearrange the levels of the tree(s) in `*rest`
#     so that they are a prefix for `tree`, for the purpose of this mapping.

#     The argument `is_leaf` can be used to terminate the mapping at a higher
#     `LDict` level of `tree`.

#     NOTE: This assumes there are no non-LDict nodes in `tree` above the level
#     of the nodes to be mapped; otherwise `tree_level_labels` will return a
#     truncated list.
#     """
#     level_labels = tree_level_labels(tree, is_leaf=is_leaf)
#     other_trees = []
#     for tree in rest:
#         reordered = tree
#         # TODO: Move the levels of `tree` to the front of `reordered`
#     # TODO: Map over `tree`


def check_nan_in_pytree(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Checks for NaN values in the array leaves of a PyTree.

    Args:
        tree: A PyTree (e.g., nested dict, list, tuple) potentially
              containing JAX arrays as leaves.

    Returns:
        A tuple containing two PyTrees with the same structure as the input:
        1. has_nans_tree: A PyTree with boolean leaves. Each leaf is True if
           the corresponding leaf array in the input tree contains any NaN
           values, and False otherwise.
        2. where_nans_tree: A PyTree where each leaf contains the output of
           jnp.where(jnp.isnan(input_leaf)). This is typically a tuple of
           index arrays indicating the locations of NaNs in the corresponding
           input leaf array.
    """

    # Define a function that processes a single leaf (assumed to be an array)
    def process_leaf(leaf):
        # Check if the leaf is likely an array type JAX can handle
        if not hasattr(leaf, "dtype") or not hasattr(leaf, "shape"):
            # Handle non-array leaves if necessary.
            pass

        is_nan_mask = jnp.isnan(leaf)
        has_nans = jnp.any(is_nan_mask)
        # jnp.where called with only the condition returns a tuple of
        # arrays representing the indices where the condition is True.
        nan_indices = _Wrapped(jnp.where(is_nan_mask))
        return has_nans, nan_indices  # Return both results as a tuple

    processed_tree = jt.map(process_leaf, tree)

    return jt.map(
        lambda x: x.unwrap() if isinstance(x, _Wrapped) else x,
        jtree.unzip(processed_tree),
    )


def tree_map_with_keys(func, tree: PyTree, *rest, is_leaf=None, **kwargs):
    """Maps `func` over a PyTree, returning a PyTree of the results and the paths to the leaves.

    The first argument of `func` must be the path
    """
    return jt.map(
        func,
        tree,
        jtree.key_tuples(tree, is_leaf=is_leaf),
        *rest,
        is_leaf=is_leaf,
        **kwargs,
    )


K = TypeVar("K")
V = TypeVar("V")

LT = TypeVar("LT", bound=str)


def tree_subset_ldict_level(tree: PyTree[LDict[K, V]], keys: Sequence[K], label: str):
    """Maps `subdict` over LabeledDict nodes with a specific label in a PyTree."""
    ldicts, other = eqx.partition(tree, LDict.is_of(label), is_leaf=LDict.is_of(label))
    ldicts = [subdict(ld, keys) for ld in ldicts if ld is not None]
    return eqx.combine(ldicts, other)


def flatten_with_paths(tree, is_leaf=None):
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)


def index_multi(obj, *idxs):
    """Index zero or more times into a Python object."""
    if not idxs:
        return obj
    return index_multi(obj[idxs[0]], *idxs[1:])


_is_leaf = anyf(is_type(go.Figure, TreeNamespace))
first_leaf = partial(jtree.first_leaf, is_leaf=_is_leaf)


def pp(tree, truncate_leaf=_is_leaf):
    """Pretty-prints PyTrees, truncating objects commonly treated as leaves during data analysis."""
    eqx.tree_pprint(tree, truncate_leaf=truncate_leaf)


def pp2(tree, truncate_leaf=_is_leaf, **kwargs):
    """Substitute for `pp` given that `truncate_leaf` of `eqx.tree_pprint` appears to be broken atm."""
    tree = jt.map(
        lambda x: type(x).__name__ if truncate_leaf(x) else x,
        tree,
        is_leaf=truncate_leaf,
    )
    eqx.tree_pprint(tree, **kwargs)


def hash_callable_leaves(
    tree: PyTree,
    is_leaf: Optional[Callable] = None,
    ignore: tuple[Callable, ...] = (),
) -> PyTree:
    """Convert callable leaves in a PyTree to their source strings."""
    leaves, treedef = jt.flatten(tree, is_leaf=is_leaf)
    return jt.unflatten(
        treedef,
        [hash_callable(leaf, ignore=ignore) if callable(leaf) else leaf for leaf in leaves],
    )


def take_replicate(i, tree: PyTree[Array, "T"]) -> PyTree[Array, "T"]:
    """"""
    # TODO: Wrap non-batched array leaves in a `Module`?
    # e.g. `WithoutBatches[0]` means the wrapped array is missing axis 0 relative to the "full" state;
    # for ensembled models, this is the ensemble (or model replicate) axis. So in this function, we should
    # be able to check for `WithoutBatches[0]`, given that the model is ensembled.
    # Need to partition since there are non-vmapped *arrays* in the intervenors...
    intervenors, other = eqx.partition(
        tree,
        jt.map(
            lambda x: isinstance(x, AbstractIntervenor),
            tree,
            is_leaf=is_type(AbstractIntervenor),
        ),
    )
    return eqx.combine(intervenors, jtree.take(other, i))


def deep_update(d1, d2):
    """Updates a dict with another, recursively.

    ```
    deep_update(dict(a=dict(b=2, c=3)), dict(a=dict(b=4)))
    # Returns dict(a=dict(b=4, c=3)), not dict(a=dict(b=4)).
    ```
    """
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1


def at_path(path):
    def at_fn(obj):
        """Navigate to `path` in `obj` and return the value there.

        Supports JAX key-path entries (`jax.tree_util.GetAttrKey`, `DictKey`, `SequenceKey`)
        as well as simple keys (e.g., `"name"`, `0`).
        """
        for key in path:
            # jax.tree_util.GetAttrKey
            if isinstance(key, getattr(jtu, "GetAttrKey", ())):
                name = getattr(key, "name", None)
                if name is None:
                    name = getattr(key, "attr", None)
                if name is None:
                    name = getattr(key, "key", None)
                obj = getattr(obj, str(name))
                continue

            # jax.tree_util.DictKey
            if isinstance(key, getattr(jtu, "DictKey", ())):
                k = getattr(key, "key", key)
                obj = obj[k]
                continue

            # jax.tree_util.SequenceKey
            if isinstance(key, getattr(jtu, "SequenceKey", ())):
                idx = getattr(key, "idx", None)
                if idx is None:
                    idx = getattr(key, "index", None)
                if idx is None:
                    idx = getattr(key, "key", None)
                obj = obj[idx]
                continue

            # jax.tree_util.FlattenedIndexKey is not navigable on the structured object
            if isinstance(key, getattr(jtu, "FlattenedIndexKey", ())):
                raise TypeError(
                    "FlattenedIndexKey cannot be used to index into a structured object"
                )

            # Fallback: simple keys based on container type
            if isinstance(obj, (eqx.Module, TreeNamespace)):
                obj = getattr(obj, str(key))
            elif isinstance(obj, (dict, list, tuple)):
                obj = obj[key]
            else:
                # Try mapping-like indexing, then attribute access
                try:
                    obj = obj[key]  # type: ignore[index]
                except Exception:
                    obj = getattr(obj, str(key))

        return obj

    return at_fn


def _hash_pytree(tree) -> str:
    """Return a deterministic MD5 digest of a PyTree **content**.

    Strategy
    --------
    • Stream stable byte-representations of every leaf into a single ``hashlib.md5``
      object – no use of Python's salted ``hash``.
    • Arrays → ``arr.tobytes(order='C')``.
    • Primitive JSON scalars → their UTF-8 string.
    • Callables → qualified name ``<module>.<qualname>``.
    • Everything else → ``repr()`` with memory addresses (``0x…``) masked out
      so it stays identical across interpreter sessions.
    """

    md5 = hashlib.md5()

    # Pre-compiled regex to erase memory addresses like 0x7ffde82aaf80
    _ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

    def _bytes_from_leaf(leaf):
        """Convert *leaf* into a stable sequence of bytes."""
        # Arrays (JAX DeviceArray, numpy.ndarray, etc.)
        if eqx.is_array(leaf):
            return leaf.tobytes(order="C")

        # JSON primitives – cheap & deterministic
        if isinstance(leaf, (int, float, bool, str)) or leaf is None:
            return json.dumps(leaf, sort_keys=True).encode()

        # Bytes object – already bytes
        if isinstance(leaf, (bytes, bytearray)):
            return bytes(leaf)

        # Callables / functions
        if isinstance(leaf, types.FunctionType):
            qname = f"{leaf.__module__}.{leaf.__qualname__}"
            return qname.encode()

        # Equinox Module or arbitrary object – fall back to repr w/o memory addr
        rep = _ADDR_RE.sub("0x", repr(leaf))
        return rep.encode()

    try:
        leaves = jt.leaves(tree)
    except Exception as e:
        raise TypeError(f"Failed to flatten PyTree for hashing: {e}") from e

    for leaf in leaves:
        md5.update(_bytes_from_leaf(leaf))

    return md5.hexdigest()


def lohi(x: Iterable, **kwargs):
    """Returns a tuple containing the first and last values of a sequence, mapping, or other iterable."""
    if isinstance(x, Mapping):
        # TODO: Maybe should return first and last key-value pairs?
        return subdict(x, tuple(lohi(tuple(x.keys()))))

    elif isinstance(x, Iterator):
        first = last = next(x)
        for last in x:
            pass

    elif isinstance(x, Sequence):
        first = x[0]
        last = x[-1]

    elif isinstance(x, Array):
        return lohi(x.tolist())

    else:
        raise ValueError(f"Unsupported type: {type(x)}")

    return first, last


def lomidhi(x: Iterable):
    if isinstance(x, dict):
        keys: tuple = tuple(lomidhi(x.keys()))
        return subdict(x, keys)

    elif isinstance(x, Iterator):
        x = tuple(x)
        first, last = lohi(x)
        mid = x[len(x) // 2]
        return first, mid, last

    elif isinstance(x, Array):
        return lomidhi(x.tolist())

    else:
        raise ValueError(f"Unsupported type: {type(x)}")


class DoNotHashTree: ...


def filter_varying_leaves(
    pytrees: Sequence[PyTree],
) -> Sequence[dict[str, Any]]:
    """
    Extract only the varying leaves from a sequence of PyTrees with the same structure.

    Args:
        pytrees: Sequence of PyTrees with identical structure

    Returns:
        List of flat dicts (one per input tree) with keys as "path.to.leaf" strings
        and values as the leaf values. Only leaves that vary across the input trees
        are included.

    Notes:
        - Returns flat dicts rather than nested structures, since removing leaves
          would break the original tree structure anyway
        - For YAML dumping, flat keys like "model.network.n_hidden" are more readable
        - All trees must have the same structure (will raise if they don't)

    Example:
        >>> tree1 = {"a": 1, "b": 2, "c": 3}
        >>> tree2 = {"a": 1, "b": 5, "c": 3}
        >>> tree3 = {"a": 1, "b": 7, "c": 3}
        >>> filter_varying_leaves([tree1, tree2, tree3])
        [{"b": 2}, {"b": 5}, {"b": 7}]  # Only "b" varies
    """
    if not pytrees:
        return []

    if len(pytrees) == 1:
        # Single tree: all leaves are "varying" (nothing to compare against)
        paths_and_leaves, _ = jt.flatten_with_path(pytrees[0])
        return [dict((_path_to_str(path), leaf) for path, leaf in paths_and_leaves)]

    # Flatten all trees with paths
    all_paths_and_leaves = [jt.flatten_with_path(tree)[0] for tree in pytrees]

    # Build dicts mapping path strings to values for each tree
    all_flat_dicts = [
        {_path_to_str(path): leaf for path, leaf in paths_and_leaves}
        for paths_and_leaves in all_paths_and_leaves
    ]

    # Get all unique paths (should be same across all trees)
    all_paths = set()
    for flat_dict in all_flat_dicts:
        all_paths.update(flat_dict.keys())

    # For each path, check if values vary across trees
    varying_dicts = []
    for flat_dict in all_flat_dicts:
        varying_dict = {}
        for path in sorted(all_paths):  # Sort for deterministic order
            # Get values at this path across all trees
            path_values = [d.get(path) for d in all_flat_dicts]

            # Check if values vary
            if not _all_equal(path_values):
                varying_dict[path] = flat_dict.get(path)

        varying_dicts.append(varying_dict)

    return varying_dicts


def _path_to_str(path: tuple) -> str:
    """Convert a JAX tree path to a dot-separated string."""
    parts = []
    for key in path:
        # Handle different JAX key types
        if isinstance(key, getattr(jtu, "GetAttrKey", ())):
            name = getattr(key, "name", None)
            if name is None:
                name = getattr(key, "attr", None)
            parts.append(str(name))
        elif isinstance(key, getattr(jtu, "DictKey", ())):
            k = getattr(key, "key", key)
            parts.append(str(k))
        elif isinstance(key, getattr(jtu, "SequenceKey", ())):
            idx = getattr(key, "idx", None)
            if idx is None:
                idx = getattr(key, "index", None)
            parts.append(str(idx))
        else:
            # Fallback for simple keys
            parts.append(str(key))

    return ".".join(parts)


def _all_equal(values: list[Any]) -> bool:
    """Check if all values in a list are equal, handling arrays properly."""
    if not values:
        return True

    first = values[0]

    for val in values[1:]:
        try:
            # For arrays, use array_equal
            if eqx.is_array(first) and eqx.is_array(val):
                if not jnp.array_equal(first, val):
                    return False
            # For other types, use regular equality
            elif first != val:
                return False
        except Exception:
            # If comparison fails, assume they're different
            return False

    return True


def _expand_missing_levels(
    tree: PyTree,
    target_levels: list[str],
    current_levels: list[str],
    reference_trees: Optional[dict[str, PyTree]] = None,
) -> PyTree:
    """
    Expand missing LDict levels in tree to match target structure.

    Args:
        tree: The tree to expand
        target_levels: The desired level structure
        current_levels: The current levels in tree
        reference_trees: Other trees to use as reference for expansion keys

    Returns:
        Tree with missing levels expanded
    """
    missing_levels = [level for level in target_levels if level not in current_levels]

    for level in missing_levels:
        # Find a reference tree that has this level to get the keys
        keys_to_use = None
        if reference_trees:
            for ref_name, ref_tree in reference_trees.items():
                if ref_tree is None:
                    continue
                # Check if this reference has the level we need
                try:
                    # Navigate to find an LDict with this label
                    sample = jt.leaves(ref_tree, is_leaf=LDict.is_of(level))
                    if sample:
                        keys_to_use = list(sample[0].keys())
                        logger.debug(
                            f"Using keys {keys_to_use} from {ref_name} "
                            f"for expanding level '{level}'"
                        )
                        break
                except:
                    continue

        # If no reference found, use a default single key
        if keys_to_use is None:
            logger.warning(
                f"No reference found for LDict level '{level}', "
                f"expanding with single default key '_'"
            )
            keys_to_use = ["_"]

        # Expand tree by wrapping in LDict with discovered keys
        if len(keys_to_use) == 1:
            # Single key - simple case
            tree = LDict.of(level)({keys_to_use[0]: tree})
        else:
            # Multiple keys - replicate tree for each key
            # Note: This creates separate references to the same tree,
            # which is fine for immutable structures
            logger.debug(f"Replicating tree for {len(keys_to_use)} keys at level '{level}'")
            tree = LDict.of(level)({key: tree for key in keys_to_use})

    return tree


#! TODO: This might be generalizable with the helpers for `jtree.rearrange_uniform_tree`...
def _align_trees_to_structure(
    trees: dict[str, PyTree],
    target_structure: list[str],
    is_leaf: Optional[Callable] = None,
    reference_trees: Optional[dict[str, PyTree]] = None,
) -> dict[str, PyTree]:
    """
    Align multiple trees to have the same LDict structure.

    Args:
        trees: Dictionary of trees to align
        target_structure: The target level structure
        is_leaf: Optional leaf predicate for tree_level_labels
        reference_trees: Additional reference trees for key expansion

    Returns:
        Dictionary of aligned trees
    """
    # Combine trees and reference_trees for expansion references
    all_trees = {}
    all_trees.update(trees)
    if reference_trees:
        all_trees.update(reference_trees)

    aligned = {}

    for name, tree in trees.items():
        # Get current structure
        current_levels = tree_level_labels(tree, is_leaf=is_leaf)

        # Expand missing levels using all available trees as reference
        other_trees = {k: v for k, v in all_trees.items() if k != name}
        tree = _expand_missing_levels(tree, target_structure, current_levels, other_trees)

        # Rearrange to match target structure
        if target_structure:
            #! use `jtree.rearrange_uniform_tree`
            tree = rearrange_ldict_levels(tree, target_structure)

        aligned[name] = tree

    return aligned
