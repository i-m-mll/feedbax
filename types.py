from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from token import COMMA
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import equinox as eqx
import jax
import jax.tree as jt
import jax.tree_util as jtu
import jax_cookbook.tree as jtree
from equinox import Module, field
from feedbax.task import AbstractTask
from jax_cookbook import anyf, is_type
from jaxtyping import Array, ArrayLike, PyTree
from ruamel.yaml import YAML, nodes
from sklearn import tree

yaml = YAML(typ="safe")


TaskModelPair = namedtuple("TaskModelPair", ["task", "model"])


TNS_REPR_INDENT_STR = "  "
LDICT_REPR_INDENT_STR = "    "


K = TypeVar("K")
V = TypeVar("V")
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


U = TypeVar("U")


@jax.tree_util.register_pytree_with_keys_class
class LDict(Mapping[K, V], Generic[K, V]):
    """Immutable dictionary with a distinguishingstring label.

    Our PyTrees will contain levels corresponding to training conditions (standard deviation
    of disturbance amplitude during training), evaluation conditions (disturbance
    amplitudes during analysis), and so on. Associating a label with a mapping will allow us
    to identify and map over specific levels of these PyTrees, as well as to keep track of the
    names of hyperparameters stored in the PyTree, e.g. so we can automatically determine
    which columns to store those hyperparameters in, in the DB.
    """

    def __init__(self, label: str, data: Mapping[K, V]):
        self._label = label
        self._data = dict(data)

    @property
    def label(self) -> str:
        return self._label

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # def __repr__(self) -> str:
    #     #! TODO: Proper line breaks when nested
    #     return f"LDict({repr(self.label)}, {self._data})"

    def __repr__(self) -> str:
        return self._repr_with_indent(0)

    def _repr_with_indent(self, level: int) -> str:
        cls_name = self.__class__.__name__
        label_repr = f"{self.label!r}"

        # Indentation for the LDict's own structure (e.g., its closing '})')
        current_level_indent = LDICT_REPR_INDENT_STR * level
        # Indentation for items (key: value pairs) within this LDict's data
        item_level_indent = LDICT_REPR_INDENT_STR * (level + 1)

        if not self._data:
            return f"{cls_name}.of({label_repr})({{}})"

        item_strings = []
        for key, value in self._data.items():
            key_as_repr = repr(key)
            value_as_repr: str

            if isinstance(value, _ReprIndentable):
                # Recursive call for nested LDicts (or similar)
                # Pass level + 1 for the nested structure's own indentation
                value_as_repr = value._repr_with_indent(level + 1)
            else:
                # Handle primitive types and their potential multi-line representations
                raw_value_lines = repr(value).splitlines()
                if not raw_value_lines:  # Should not happen for standard reprs
                    raw_value_lines = [""]

                if len(raw_value_lines) > 1:
                    # For multi-line primitives, indent subsequent lines.
                    # The first line is already positioned by "key: ".
                    # Subsequent lines are indented to the item_level_indent.
                    # This provides basic readability for multi-line strings within the LDict.
                    indented_value_lines = [raw_value_lines[0]]
                    for i in range(1, len(raw_value_lines)):
                        indented_value_lines.append(f"{item_level_indent}{raw_value_lines[i]}")
                    value_as_repr = "\n".join(indented_value_lines)
                else:
                    value_as_repr = raw_value_lines[0]

            # Each item string is fully formed, including its indentation and trailing comma.
            # If value_as_repr is multi-line (e.g., from a nested LDict), its existing
            # newlines and internal indentation are preserved.
            item_strings.append(f"{item_level_indent}{key_as_repr}: {value_as_repr},")

        # Assemble the LDict representation
        # Start with: LDict.of('label')({
        # Followed by each item on a new line (if items exist)
        # Ended by:
        # current_level_indent})

        dict_body_content = "\n".join(item_strings)

        return f"{cls_name}.of({label_repr})({{\n{dict_body_content}\n{current_level_indent}}})"

    def tree_flatten_with_keys(self):
        # Avoids `FlattenedIndexKey` appearing in key paths
        children_with_keys = [(jtu.DictKey(k), v) for k, v in self.items()]
        return children_with_keys, (self._label, self.keys())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        label, keys = aux_data
        return cls(label, dict(zip(keys, children)))

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    # def get(self, key, default: U = ...) -> V | U:
    #     return self._data.get(key, default)

    @staticmethod
    def of(label: str):
        """Returns a constructor function for the given label."""
        return LDictConstructor(label)

    @staticmethod
    def is_of(label: str) -> Callable[[Any], bool]:
        """Return a predicate checking if an object is an `LDict` with a given label."""

        def is_ldict_of(node: Any) -> bool:
            """Check if the node is an LDict with the specified label."""
            return isinstance(node, LDict) and node.label == label

        return is_ldict_of

    @staticmethod
    @overload
    def fromkeys(label: str, keys: Iterable[K]) -> "LDict[K, None]": ...

    @staticmethod
    @overload
    def fromkeys(label: str, keys: Iterable[K], value: V) -> "LDict[K, V]": ...

    @staticmethod
    def fromkeys(label: str, keys: Iterable[Any], value: Any = None) -> "LDict[Any, Any]":
        """Create a new LDict with the given label and keys, each with value set to value."""
        return LDict(label, dict.fromkeys(keys, value))

    def __or__(self, other: "LDict | Mapping[K, V]") -> "LDict[K, V]":
        """Merge with another LDict or Mapping, keeping self's label. Values from `other` take precedence."""
        new_data = self._data.copy()
        # if isinstance(other, LDict):
        #     new_data.update(other._data)
        if isinstance(other, Mapping):
            new_data.update(other)
        else:
            return NotImplemented  # Indicate that the operation is not supported for this type
        return LDict(self._label, new_data)

    def __ror__(self, other: Mapping[K, V]) -> "LDict[K, V] | Mapping[K, V]":
        """Merge with a Mapping from the left. The result type depends on `other`."""
        if isinstance(other, Mapping):
            new_data = dict(other)  # Start with a copy of the left operand
            new_data.update(self._data)  # Update with self's data
            # If the left operand was also an LDict, preserve its label
            if isinstance(other, LDict):
                return LDict(other.label, new_data)
            else:
                # Otherwise return a standard dict
                return new_data
        else:
            return NotImplemented


class LDictConstructor(Generic[K, V]):
    """Constructor for an `LDict` with a particular label."""

    def __init__(self, label: str):
        self.label = label

    @overload
    def __call__(self, __mapping: Mapping[K, V], /) -> LDict[K, V]: ...
    @overload
    def __call__(self, /, **kwargs: V) -> LDict[str, V]: ...

    def __call__(self, __mapping: Optional[Mapping[Any, V]] = None, /, **kwargs: V):
        """Call with either a single mapping positional arg or keyword args (not both)."""
        if __mapping is not None and kwargs:
            raise TypeError("Pass either a mapping positional argument or keyword args, not both.")
        data: Mapping[Any, V]
        if __mapping is not None:
            data = __mapping
        else:
            data = dict(kwargs)
        return LDict(self.label, data)

    def __repr__(self) -> str:
        return f"LDict.of({self.label})"

    def fromkeys(self, keys: Iterable[K], value: Optional[V] = None):
        return LDict.fromkeys(self.label, keys, value)

    def from_ns(self, namespace: SimpleNamespace):
        """Convert the top level of `namespace` to an `LDict`."""
        return LDict(self.label, namespace.__dict__)

    @property
    def predicate(self) -> Callable[[Any], bool]:
        """A predicate that checks if an object is an `LDict` with this constructor's label."""

        def is_ldict_of(node: Any) -> bool:
            """Check if the node is an LDict with the specified label."""
            if isinstance(node, LDict):
                return node.label == self.label
            return False

        return is_ldict_of


# YAML serialisation/deserialisation for LDict objects
def _ldict_representer(representer, data: LDict):
    return representer.represent_mapping(f"!LDict:{data.label}", data._data)


yaml.representer.add_representer(LDict, _ldict_representer)


def _ldict_multi_constructor(constructor, tag_suffix, node):
    # e.g. !LDict:mylabel {k: v}
    if not isinstance(node, nodes.MappingNode):
        raise TypeError("!LDict:* must tag a mapping")
    mapping = constructor.construct_mapping(node, deep=True)
    return LDict(tag_suffix, mapping)


yaml.constructor.add_multi_constructor("!LDict:", _ldict_multi_constructor)


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
    def _recurse_func(value: Any) -> Any:
        return _convert_value(value, to_type, from_type, exclude)

    def _map_recurse_func(tree: Any) -> Any:
        return jt.map(_recurse_func, tree, is_leaf=is_type(from_type))

    if exclude(value):
        subtrees, treedef = eqx.tree_flatten_one_level(value)
        if jtu.treedef_is_leaf(treedef):
            return value
        subtrees = [_map_recurse_func(subtree) for subtree in subtrees]
        return jt.unflatten(treedef, subtrees)

    elif isinstance(value, from_type):
        if isinstance(value, SimpleNamespace):
            value = vars(value)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a dict or namespace, got {type(value)}")

        return to_type(**{str(k): _recurse_func(v) for k, v in value.items()})

    elif isinstance(value, (str, type(None))) or isinstance(value, ArrayLike):
        return value

    # Map over any remaining PyTrees
    elif isinstance(value, PyTree):
        # `object` is an atomic PyTree, so without this check we'll get infinite recursion
        is_leaf = anyf(is_type(from_type), exclude)
        leaves, treedef = jt.flatten(value, is_leaf=is_leaf)
        if not any(is_leaf(leaf) for leaf in leaves):
            return value
        leaves = [_map_recurse_func(leaf) for leaf in leaves]
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
    # is_spatial: bool = True
