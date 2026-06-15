"""Task-specific PyTree callable evaluation helpers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
from jaxtyping import PRNGKeyArray, PyTree


def tree_call(
    tree: PyTree[Any, "T"],
    *args: Any,
    exclude: Callable[[Any], bool] = lambda _: False,
    is_leaf: Optional[Callable[[Any], bool]] = None,
    **kwargs: Any,
) -> PyTree[Any, "T"]:
    """Evaluate callable leaves, passing every callable leaf the same arguments."""
    return jtree.call(tree, *args, exclude=exclude, is_leaf=is_leaf, **kwargs)


def tree_call_with_keys(
    tree: PyTree[Any, "T"],
    *args: Any,
    key: PRNGKeyArray,
    exclude: Callable[[Any], bool] = lambda _: False,
    is_leaf: Optional[Callable[[Any], bool]] = None,
    shared_key: PRNGKeyArray | None = None,
    use_shared: Optional[Callable[[Any], bool]] = None,
    **kwargs: Any,
) -> PyTree[Any, "T"]:
    """Evaluate callable leaves with per-leaf keys and optional Feedbax shared-key markers."""

    def choose_key(fn: Callable[..., Any], per_leaf_key: PRNGKeyArray) -> PRNGKeyArray:
        if shared_key is not None and use_shared is not None and use_shared(fn):
            return shared_key
        return per_leaf_key

    callables, other_values = eqx.partition(
        tree,
        lambda leaf: (isinstance(leaf, Callable) or is_marked_shared(leaf)) and not exclude(leaf),
        is_leaf=is_leaf,
    )
    keys = jtree.random_split_like_tree(key, callables, is_leaf=is_leaf)
    if shared_key is not None and use_shared is not None:
        keys = jt.map(choose_key, callables, keys, is_leaf=is_leaf)

    unwrapped = unwrap_shared_markers(callables)
    values = jt.map(
        lambda fn, fn_key: fn(*args, **kwargs, key=fn_key),
        unwrapped,
        keys,
        is_leaf=is_leaf,
    )
    return eqx.combine(values, other_values, is_leaf=is_leaf)


@dataclass(frozen=True)
class _SharedKeyWrapper:
    fn: Callable[..., Any]


def mark_shared(tree: PyTree[Any, "T"], *wheres: Callable[[Any], Any]) -> PyTree[Any, "T"]:
    """Wrap callable leaves selected by ``wheres`` so they can receive a shared key."""

    def _wrap(fn: Callable[..., Any]) -> _SharedKeyWrapper:
        return _SharedKeyWrapper(fn)

    out = tree
    for where in wheres:
        out = eqx.tree_at(where, out, replace_fn=_wrap)
    return out


def is_marked_shared(value: Any) -> bool:
    return isinstance(value, _SharedKeyWrapper)


def unwrap_shared_markers(tree: PyTree[Any, "T"]) -> PyTree[Any, "T"]:
    return jt.map(
        lambda value: value.fn if isinstance(value, _SharedKeyWrapper) else value,
        tree,
    )
