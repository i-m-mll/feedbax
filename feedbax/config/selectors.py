"""Selector and where-function conversion helpers."""

from collections.abc import Callable, Iterable, Mapping, MutableSequence
import functools
import re
from typing import Any, Tuple, TypeVar

import equinox as eqx
import jax.tree as jt
import jax.tree_util as jtu
from jaxtyping import PyTree

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def unzip2(xys: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]:
    """Unzip sequence of length-2 tuples into two tuples."""
    xs: MutableSequence[T1] = []
    ys: MutableSequence[T2] = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


class _NodeWrapper:
    def __init__(self, value):
        self.value = value


class NodePath:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        return iter(self.path)


def where_func_to_paths(where: Callable, tree: PyTree):
    """Return PyTree node paths selected by a where-function."""
    tree = eqx.tree_at(where, tree, replace_fn=lambda x: _NodeWrapper(x))
    id_tree = jt.map(id, tree, is_leaf=lambda x: isinstance(x, _NodeWrapper))
    node_ids = where(id_tree)

    paths_by_id = {
        node_id: path
        for path, node_id in jtu.tree_leaves_with_path(
            jt.map(
                lambda x: x if x in jt.leaves(node_ids) else None,
                id_tree,
            )
        )
    }

    return jt.map(lambda node_id: NodePath(paths_by_id[node_id]), node_ids)


class _WhereStrConstructor:
    def __init__(self, label: str = ""):
        self.label = label

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            key = f"'{key}'"
        elif isinstance(key, type):
            key = key.__name__
        return _WhereStrConstructor("".join([self.label, f"[{key}]"]))

    def __getattr__(self, name: str):
        sep = "." if self.label else ""
        return _WhereStrConstructor(sep.join([self.label, name]))


def _get_where_str_constructor_label(x: _WhereStrConstructor) -> str:
    return x.label


def where_func_to_attr_str_tree(where: Callable) -> PyTree[str]:
    """Convert a where-function into a PyTree of compact attribute strings."""
    try:
        return jt.map(_get_where_str_constructor_label, where(_WhereStrConstructor()))
    except TypeError:
        raise TypeError("`where` must return a PyTree of node references")


def _resolve_attr_or_graph_node_path(obj: Any, path: str) -> Any:
    """Resolve a compact where path, preferring graph nodes when unambiguous."""
    value = obj
    parts = path.split(".") if path else []

    for i, part in enumerate(parts):
        node_match = re.fullmatch(r"""nodes\[['"]([^'"]+)['"]\]""", part)
        if node_match is not None:
            node_name = node_match.group(1)
            value = value.nodes[node_name]
            continue

        nodes = getattr(value, "nodes", None)
        if i == 0 and isinstance(nodes, Mapping) and part in nodes:
            value = nodes[part]
            continue

        value = getattr(value, part)

    return value


def attr_str_tree_to_where_func(tree: PyTree[str]) -> Callable:
    """Convert compact attribute strings into an executable where-function."""
    getters = jt.map(
        lambda s: functools.partial(_resolve_attr_or_graph_node_path, path=s),
        tree,
    )

    def where_func(obj):
        return jt.map(lambda g: g(obj), getters)

    return where_func
