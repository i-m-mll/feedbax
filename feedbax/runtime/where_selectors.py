"""Where-function conversion helpers for the scientific runtime."""

from collections.abc import Callable, Iterable, Mapping, MutableSequence
from dataclasses import dataclass
import dis
import functools
import inspect
from operator import attrgetter, itemgetter
import re
from types import CodeType, FunctionType
from typing import Any, Tuple, TypeVar

import equinox as eqx
import jax.tree as jt
import jax.tree_util as jtu
from jaxtyping import PyTree

T1 = TypeVar("T1")
T2 = TypeVar("T2")


@dataclass(frozen=True)
class WherePath:
    """An executable, immutable path into a PyTree."""

    steps: tuple[jtu.GetAttrKey | jtu.DictKey, ...]

    @functools.cached_property
    def address(self) -> str:
        return jtu.keystr(self.steps).removeprefix(".").replace("#", r"\x23")

    def __call__(self, tree: PyTree) -> Any:
        for step in self.steps:
            if isinstance(step, jtu.GetAttrKey):
                tree = getattr(tree, step.name)
            else:
                tree = tree[step.key]
        return tree

    def __repr__(self) -> str:
        return f"(lambda state: state{jtu.keystr(self.steps)})"


_WHERE_PATH_ERROR = (
    "A WhereDict selector must return its single argument or a path of attributes "
    "and literal string or integer indices on that argument"
)


def where_path(where: Callable) -> WherePath:
    """Compile a selector without executing arbitrary code on a proxy."""
    if isinstance(where, WherePath):
        return where
    if isinstance(where, FunctionType):
        return _compile_where_path(where.__code__)
    if isinstance(where, attrgetter):
        _, attrs = where.__reduce__()
        if len(attrs) == 1:
            names = attrs[0].split(".")
            if all(name.isidentifier() for name in names):
                return WherePath(tuple(jtu.GetAttrKey(name) for name in names))
    if isinstance(where, itemgetter):
        _, keys = where.__reduce__()
        if len(keys) == 1 and type(keys[0]) in (int, str):
            return WherePath((jtu.DictKey(keys[0]),))
    raise ValueError(_WHERE_PATH_ERROR)


@functools.lru_cache(maxsize=512)
def _compile_where_path(code: CodeType) -> WherePath:
    if (
        code.co_argcount != 1
        or code.co_kwonlyargcount
        or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
    ):
        raise ValueError(_WHERE_PATH_ERROR)

    instructions = iter(dis.get_instructions(code))
    start = next(instructions)
    root = next(instructions, None)
    if (
        start.opname != "RESUME"
        or root is None
        or root.opname not in {"LOAD_FAST", "LOAD_FAST_BORROW"}
        or root.arg != 0
    ):
        raise ValueError(_WHERE_PATH_ERROR)

    steps = []
    for instruction in instructions:
        match instruction.opname:
            case "LOAD_ATTR":
                steps.append(jtu.GetAttrKey(instruction.argval))
            case "LOAD_CONST" | "LOAD_SMALL_INT" if type(instruction.argval) in (int, str):
                subscript = next(instructions, None)
                if subscript is None or (
                    subscript.opname != "BINARY_SUBSCR"
                    and (subscript.opname, subscript.argrepr) != ("BINARY_OP", "[]")
                ):
                    raise ValueError(_WHERE_PATH_ERROR)
                steps.append(jtu.DictKey(instruction.argval))
            case "RETURN_VALUE" if next(instructions, None) is None:
                return WherePath(tuple(steps))
            case _:
                raise ValueError(_WHERE_PATH_ERROR)
    raise ValueError(_WHERE_PATH_ERROR)


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
