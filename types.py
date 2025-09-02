from collections import namedtuple
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from types import SimpleNamespace
from typing import (
    Any,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import equinox as eqx
import jax.tree as jt
import jax.tree_util as jtu
from equinox import Module
from feedbax.task import AbstractTask
from jax_cookbook import LDict, LDictConstructor, anyf, is_type
from jaxtyping import Array, ArrayLike, PyTree

__all__ = [
    "LDict",
    "LDictConstructor",
    "LDictTree",
    "TreeNamespace",
    "TaskModelPair",
    "ResponseVar",
    "Direction",
    "Polar",
    "VarSpec",
    "Labels",
    "AnalysisInputData",
    "dict_to_namespace",
    "namespace_to_dict",
    "is_dict_with_int_keys",
    "unflatten_dict_keys",
]


TaskModelPair = namedtuple("TaskModelPair", ["task", "model"])


TNS_REPR_INDENT_STR = "  "
LDICT_REPR_INDENT_STR = "    "


NT = TypeVar("NT", bound=SimpleNamespace)
DT = TypeVar("DT", bound=dict)


def convert_kwargy_node_type(
    x, to_type: type, from_type: type, exclude: Callable = lambda x: False
):
    """Convert a nested dictionary to a nested SimpleNamespace.

    !!! dev
        This should convert all the dicts to namespaces, even if the dicts are not contiguous all
        the way down (e.g. a dict in a list in a list in a dict)
    """
    return _convert_value(x, to_type, from_type, exclude)


def dict_to_namespace(
    d: dict,
    to_type: type[NT] = SimpleNamespace,
    exclude: Callable = lambda x: False,
) -> NT:
    """Convert a nested dictionary to a nested SimpleNamespace.

    This is the inverse operation of namespace_to_dict.
    """
    return convert_kwargy_node_type(d, to_type=to_type, from_type=dict, exclude=exclude)


def namespace_to_dict(
    ns: SimpleNamespace,
    to_type: type[DT] = dict,
    exclude: Callable = lambda x: False,
) -> DT:
    """Convert a nested SimpleNamespace to a nested dictionary.

    This is the inverse operation of dict_to_namespace.
    """
    # TODO: Now that `TreeNamespace` implements the mapping protocol, we might be able to simplify this
    return convert_kwargy_node_type(ns, to_type=to_type, from_type=SimpleNamespace, exclude=exclude)


def is_dict_with_int_keys(d: dict) -> bool:
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(k, int) for k in d.keys())


@jtu.register_pytree_with_keys_class
class TreeNamespace(SimpleNamespace):
    """A simple namespace that's a PyTree.

    This is useful when we want to attribute-like access to the data in
    a nested dict. For example, `hyperparameters['train']['n_batches']`
    becomes `TreeNamespace(**hyperparameters).train.n_batches`.

    NOTE:
        If it weren't for `update_none_leaves`, `__or__`, and perhaps `__repr__`,
        we could simply register `SimpleNamespace` as a PyTree. Consider whether
        these methods can be replaced by e.g. functions.
    """

    def tree_flatten_with_keys(self):
        children_with_keys = [(jtu.GetAttrKey(k), v) for k, v in self.__dict__.items()]
        aux_data = self.__dict__.keys()
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))

    def __repr__(self):
        return self._repr_with_indent(0)

    def _repr_with_indent(self, level):
        cls_name = self.__class__.__name__
        if not any(self.__dict__):
            return f"{cls_name}()"

        attr_strs = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, TreeNamespace):
                attr_repr = attr._repr_with_indent(level + 1)
            else:
                attr_repr = repr(attr)
            attr_strs.append(f"{name}={attr_repr},")

        current_indent = TNS_REPR_INDENT_STR * level
        inner_str = "\n".join(current_indent + TNS_REPR_INDENT_STR + s for s in attr_strs)

        return f"{cls_name}(\n" + inner_str + f"\n{current_indent})"

    def update_none_leaves(self, other):
        # I would just use `jt.map` or `eqx.combine` to do this, however I don't want to assume
        # that `other` will have identical PyTree structure to `self` -- only that it contains at
        # least the keys whose values are `None` in `self`.
        # ? Could work on flattened trees.
        def _update_none_leaves(target: TreeNamespace, source: TreeNamespace) -> TreeNamespace:
            result = deepcopy(target)
            source = deepcopy(source)

            for attr_name in vars(result):
                if attr_name == "load":
                    continue

                result_value = getattr(result, attr_name)
                source_value = getattr(source, attr_name, None)

                if result_value is None:
                    if source_value is None:
                        raise ValueError(
                            f"Cannot replace `None` value of key {attr_name}; no matching key available in source"
                        )
                    setattr(result, attr_name, source_value)

                elif isinstance(result_value, TreeNamespace):
                    if source_value is None:
                        continue
                    if not isinstance(source_value, TreeNamespace):
                        raise ValueError(
                            "Source must contain all the parent keys (but not necessarily all the leaves) of the target"
                        )
                    setattr(result, attr_name, _update_none_leaves(result_value, source_value))

            return result

        return _update_none_leaves(self, other)

    def __or__(self, other: "TreeNamespace | dict") -> "TreeNamespace":
        """Merge two TreeNamespaces, or a TreeNamespace and a dict, with values from `other` taking precedence.

        Handles nested inputs recursively.
        """
        result = deepcopy(self)

        if isinstance(other, dict):
            other = dict_to_namespace(other, to_type=type(self), exclude=is_type(LDict))

        for attr_name, other_value in vars(other).items():
            self_value = getattr(result, attr_name, None)

            if isinstance(self_value, TreeNamespace):
                if isinstance(other_value, dict):
                    other_value = dict_to_namespace(
                        other_value,
                        to_type=type(self_value),
                        exclude=is_type(LDict),
                    )
                if isinstance(other_value, TreeNamespace):
                    # Recursively merge nested TreeNamespaces
                    setattr(result, attr_name, self_value | other_value)
            else:
                setattr(result, attr_name, other_value)

        return result

    def __ror__(self, other: dict) -> dict:
        return other | namespace_to_dict(self)

    # Implement the mapping protocol so we can treat the namespace as a dict sometimes
    def __iter__(self):
        """Return an iterator over the keys of the namespace."""
        return iter(self.__dict__)

    def __getitem__(self, key):
        """Get an item using dictionary-style access."""
        return self.__dict__[key]

    def keys(self):
        """Return the keys of the namespace, enabling dict(**tree_namespace)."""
        return self.__dict__.keys()

    def items(self):
        """Return the items of the namespace."""
        return self.__dict__.items()

    def values(self):
        """Return the values of the namespace."""
        return self.__dict__.values()


def unflatten_dict_keys(flat_dict: dict, sep: str = "__") -> dict:
    """Unflatten a dictionary by splitting keys on the separator.

    Supports multiple levels of nesting.
    """
    result = {}

    for key, value in flat_dict.items():
        current = result

        if sep in key:
            parts = key.split(sep)

            for part in parts[:-1]:
                current = current.setdefault(part, {})

            current[parts[-1]] = value
        else:
            result[key] = value

    return result


class _Wrapped:
    """Simple wrapper, e.g. for turning PyTree nodes into leaves when `is_leaf` fails."""

    def __init__(self, value: Any):
        self.value = value

    def unwrap(self):
        return self.value


@runtime_checkable
class _ReprIndentable(Protocol):
    def _repr_with_indent(self, level: int) -> str: ...


# Use to statically type trees whose non-leaf nodes are LDicts only.
# If `T` is parameterized, leaves must be of type `T`.
type LDictTree[T] = T | LDict[Any, LDictTree[T]]


def pprint_ldict_structure(
    tree: LDict,
    indent: int = 0,
    indent_str: str = "  ",
    homogeneous: bool = True,
):
    """Pretty print the structure of a nested LDict PyTree.

    Args:
        tree: An LDict or nested structure of LDicts
        indent: Current indentation level (used recursively)
        indent_str: String used for each level of indentation
        homogeneous: If True, assumes all nodes at each level have the same label and keys,
                    so only prints the first occurrence at each level
    """
    if not isinstance(tree, LDict):
        return

    # Print current level's label and keys
    current_indent = indent_str * indent
    print(f"{current_indent}LDict('{tree.label}') with keys: {list(tree.keys())}")

    # Process LDict values, breaking after first one if homogeneous
    for value in tree.values():
        if isinstance(value, LDict):
            pprint_ldict_structure(value, indent + 2, indent_str, homogeneous)
            if homogeneous:
                break


def _convert_value(value: Any, to_type: type, from_type: type, exclude: Callable) -> Any:
    def _recurse_fn(value: Any) -> Any:
        return _convert_value(value, to_type, from_type, exclude)

    def _map_recurse_fn(tree: Any) -> Any:
        return jt.map(_recurse_fn, tree, is_leaf=is_type(from_type))

    if exclude(value):
        subtrees, treedef = eqx.tree_flatten_one_level(value)
        if jtu.treedef_is_leaf(treedef):
            return value
        subtrees = [_map_recurse_fn(subtree) for subtree in subtrees]
        return jt.unflatten(treedef, subtrees)

    elif isinstance(value, from_type):
        if isinstance(value, SimpleNamespace):
            value = vars(value)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a dict or namespace, got {type(value)}")

        return to_type(**{str(k): _recurse_fn(v) for k, v in value.items()})

    elif isinstance(value, (str, type(None))) or isinstance(value, ArrayLike):
        return value

    # Map over any remaining PyTrees
    elif isinstance(value, PyTree):
        # `object` is an atomic PyTree, so without this check we'll get infinite recursion
        is_leaf = anyf(is_type(from_type), exclude)
        leaves, treedef = jt.flatten(value, is_leaf=is_leaf)
        if not any(is_leaf(leaf) for leaf in leaves):
            return value
        leaves = [_map_recurse_fn(leaf) for leaf in leaves]
        return jt.unflatten(treedef, leaves)

    return value


class AnalysisInputData(Module):
    models: PyTree[Module]
    tasks: PyTree[Module]
    states: PyTree[Module]
    hps: PyTree[TreeNamespace]
    extras: PyTree[TreeNamespace]


class Labels(NamedTuple):
    full: PyTree[str]
    medium: PyTree[str]
    short: PyTree[str]


class VarSpec(eqx.Module):
    where: Callable[[AnalysisInputData], Array]
    labels: Labels
    time_axis: int = -2
    vec_axis: int = -1
    origin: Optional[ArrayLike | Callable[[AbstractTask], ArrayLike]] = None


class Polar(NamedTuple):
    angle: Array
    radius: Array


class Direction(str, Enum):
    """Available directions for vector components."""

    PARALLEL = "parallel"
    LATERAL = "lateral"


class ResponseVar(str, Enum):
    """Variables available in response state."""

    POSITION = "pos"
    VELOCITY = "vel"
    COMMAND = "command"
    FORCE = "force"
