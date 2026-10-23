from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace
from typing import (
    Any,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import jax
import jax.tree_util as jtu

LDICT_REPR_INDENT_STR = "    "


@runtime_checkable
class _ReprIndentable(Protocol):
    def _repr_with_indent(self, level: int) -> str: ...


K = TypeVar("K")
V_co = TypeVar("V_co", covariant=True)

# Invariant typevars for methods
_K = TypeVar("_K")
_V = TypeVar("_V")


@jax.tree_util.register_pytree_with_keys_class
class LDict(Mapping[K, V_co], Generic[K, V_co]):
    """Immutable dictionary with a distinguishingstring label.

    Our PyTrees will contain levels corresponding to training conditions (standard deviation
    of disturbance amplitude during training), evaluation conditions (disturbance
    amplitudes during analysis), and so on. Associating a label with a mapping will allow us
    to identify and map over specific levels of these PyTrees, as well as to keep track of the
    names of hyperparameters stored in the PyTree, e.g. so we can automatically determine
    which columns to store those hyperparameters in, in the DB.
    """

    def __init__(
        self, label: str, data: Mapping[K, V_co], *, metadata: Optional[Mapping[str, Any]] = None
    ):
        self._label = label
        self._data = MappingProxyType(dict(data))
        self.metadata = MappingProxyType(dict(metadata)) if metadata is not None else None

    @property
    def label(self) -> str:
        return self._label

    def __getitem__(self, key: K) -> V_co:
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
    def fromkeys(label: str, keys: Iterable[_K]) -> "LDict[_K, None]": ...

    @staticmethod
    @overload
    def fromkeys(label: str, keys: Iterable[_K], value: _V) -> "LDict[_K, _V]": ...

    @staticmethod
    def fromkeys(label: str, keys: Iterable[_K], value: Any = None) -> "LDict[_K, Any]":
        """Create a new LDict with the given label and keys, each with value set to value."""
        return LDict(label, dict.fromkeys(keys, value))

    def __or__(self, other: "LDict | Mapping[K, V_co]") -> "LDict[K, V_co]":
        """Merge with another LDict or Mapping, keeping self's label. Values from `other` take precedence."""
        new_data = dict(self._data)
        if isinstance(other, Mapping):
            new_data.update(other)
        else:
            return NotImplemented  # Indicate that the operation is not supported for this type
        return LDict(self._label, new_data)

    def __ror__(self, other: Mapping[K, V_co]) -> "LDict[K, V_co] | Mapping[K, V_co]":
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

    # For pickling/deepcopying: bypass the MappingProxyType
    def __deepcopy__(self, memo):
        base = dict(self._data)
        base_dc = deepcopy(base, memo)
        return LDict(self._label, base_dc)

    def __reduce_ex__(self, protocol):
        return (LDict, (self._label, dict(self._data)))


class LDictConstructor:
    """Constructor for an `LDict` with a particular label."""

    def __init__(self, label: str):
        self.label = label

    @overload
    def __call__(self, __mapping: Mapping[_K, _V], /) -> LDict[_K, _V]: ...
    @overload
    def __call__(self, __pairs: Iterable[tuple[_K, _V]], /) -> LDict[_K, _V]: ...
    @overload
    def __call__(self, /, **kwargs: _V) -> LDict[str, _V]: ...
    def __call__(
        self,
        __arg: Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None = None,
        /,
        **kwargs: Any,
    ):
        """Call with either a single mapping positional arg or keyword args (not both)."""
        if __arg is not None and kwargs:
            raise TypeError("Pass either a mapping positional argument or keyword args, not both.")
        if __arg is None:
            data: Mapping[Any, Any] = dict(kwargs)
        elif isinstance(__arg, Mapping):
            data = __arg
        else:
            data = dict(__arg)
        return LDict(self.label, data)

    def __repr__(self) -> str:
        return f"LDict.of({self.label})"

    @property
    def predicate(self) -> Callable[[Any], bool]:
        """Predicate that matches LDicts with this constructor's label."""
        return LDict.is_of(self.label)

    @overload
    def fromkeys(self, keys: Iterable[_K]) -> LDict[_K, None]: ...
    @overload
    def fromkeys(self, keys: Iterable[_K], value: _V) -> LDict[_K, _V]: ...
    def fromkeys(self, keys: Iterable[Any], value: Any = None):
        return LDict.fromkeys(self.label, keys, value)

    def from_ns(self, namespace: SimpleNamespace):
        """Convert the top level of `namespace` to an `LDict`."""
        return LDict(self.label, namespace.__dict__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LDictConstructor):
            return False
        return self.label == other.label

    @property
    def predicate(self) -> Callable[[Any], bool]:
        """A predicate that checks if an object is an `LDict` with this constructor's label."""

        def is_ldict_of(node: Any) -> bool:
            """Check if the node is an LDict with the specified label."""
            if isinstance(node, LDict):
                return node.label == self.label
            return False

        return is_ldict_of


def is_ldict_of(label: str) -> Callable[[Any], bool]:
    """Return a predicate checking if an object is an `LDict` with a given label."""
    return LDict.is_of(label)


#! This might not be doable until update to Python 3.12
# Use to statically type trees whose non-leaf nodes are LDicts only.
# If `T` is parameterized, leaves must be of type `T`.
# type LDictTree[T] = T | LDict[Any, LDictTree[T]]
