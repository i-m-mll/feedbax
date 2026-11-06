"""Scientific-runtime selector mapping classes.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, MutableMapping

from typing import Any, Generic, TypeVar, overload

import jax.tree_util as jtu
from jaxtyping import PyTree

from feedbax.runtime.where_selectors import WherePath, where_path

T = TypeVar("T")
KT1 = TypeVar("KT1")
KT2 = TypeVar("KT2")
VT = TypeVar("VT")


class AbstractTransformedOrderedDict(MutableMapping[KT2, VT], Generic[KT1, KT2, VT]):
    """Base for `OrderedDict`s which transform keys when getting and setting items.

    It stores the original keys, and otherwise behaves (e.g. when iterating)
    as though they are the true keys.

    This is useful when we want to use a certain type of object as a key, but
    it would not be hashed properly by `OrderedDict`, so we need to transform
    it into something else. In particular, by replacing `_key_transform` with
    a code parsing function, a subclass of this class can take lambdas as keys.

    See [`feedbax.WhereDict`][feedbax.WhereDict] for an example.

    Based on https://stackoverflow.com/a/3387975
    """
    store: OrderedDict[KT1, tuple[KT2, VT]]

    def __init__(self, *args, **kwargs):
        self.store = OrderedDict()
        self.update(OrderedDict(*args, **kwargs))

    def __getitem__(self, key: KT1 | KT2) -> VT:
        k = self._key_transform(key)
        return self.store[k][1]

    @overload
    def get(self, key: KT1 | KT2) -> VT | None: ...

    @overload
    def get(self, key: KT1 | KT2, default: T) -> VT | T: ...

    def get(self, key, default=None):
        k = self._key_transform(key)
        if k in self.store:
            return self.store[k][1]
        else:
            return default

    def __setitem__(self, key: KT2, value: VT):
        self.store[self._key_transform(key)] = (key, value)

    def __delitem__(self, key: KT2):
        del self.store[self._key_transform(key)]

    def __iter__(self):
        for key in self.store:
            yield self.store[key][0]

    def __len__(self) -> int:
        return len(self.store)

    @abstractmethod
    def _key_transform(self, key: KT1 | KT2) -> KT1:
        ...

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


WhereCallable = Callable[[PyTree], Any]
WhereKey = str | WhereCallable | tuple[WhereCallable, str]
_CanonicalKey = str | WherePath | tuple[WherePath, str]


def _canonical_key(key: WhereKey) -> _CanonicalKey:
    if isinstance(key, str):
        return key
    if callable(key):
        return where_path(key)
    if isinstance(key, tuple) and len(key) == 2 and callable(key[0]) and isinstance(key[1], str):
        return where_path(key[0]), key[1]
    raise ValueError("Each WhereDict key must be a string, a callable, or a (callable, string) pair")


@jtu.register_pytree_node_class
class WhereDict(AbstractTransformedOrderedDict[str, WhereKey, T]):
    """An `OrderedDict` that allows use of where-functions as keys.

    Selector functions take one argument and return that argument or a path
    of attributes and literal string or integer indices on it. PyTree
    transformations reconstruct executable keys from those paths. Keys
    supplied as strings are retained as strings.

    Functions are parsed to equivalent strings, which can be used
    interchangeably as keys. For example, the following are equivalent when
    `init_spec` is a `WhereDict`:

    ```python
    init_spec[lambda state: state.mechanics.effector]
    ```
    ```python
    init_spec['mechanics.effector']
    ```

    Finally, a `tuple[Callable, str]` may also be provided as a key, for cases where
    different unique entries must be included for the same `Callable`. For example,
    the following are equivalent:

    ```python
    init_spec[(lambda state: state.mechanics.effector, "first")]
    ```
    ```python
    init_spec['mechanics.effector#first']
    ```

    Note that the hash symbol `#` is used to concatenate the usual string representation
    for the callable with the paired string. Literal indices use Python repr
    spelling, with `#` in indices escaped as `\\x23`.
    """

    def _key_transform(self, key: WhereKey) -> str:
        return self.key_transform(key)

    @staticmethod
    def key_transform(key: WhereKey) -> str:
        canonical = _canonical_key(key)
        if isinstance(canonical, str):
            return canonical
        if isinstance(canonical, WherePath):
            return canonical.address
        path, tag = canonical
        return f"{path.address}#{tag}"

    def tree_flatten(self):
        keys = tuple(_canonical_key(key) for key, _ in self.store.values())
        values = tuple(value for _, value in self.store.values())
        return values, keys

    def __setstate__(self, state):
        """Restore mappings whose persisted addresses predate escaped indices."""
        restored = OrderedDict()
        for key, value in state["store"].values():
            address = self.key_transform(key)
            if address in restored:
                raise ValueError(f"Persisted WhereDict contains conflicting keys for {address!r}")
            restored[address] = (key, value)
        self.__dict__.update(state)
        self.store = restored

    def _repr_with_indent(self, level: int) -> str:
        cls_name       = self.__class__.__name__
        current_indent = "  " * level
        item_indent    = "  " * (level + 1)

        # empty case
        if not self.store:
            return f"{cls_name}({{}})"

        item_strings = []
        for key, value in self.store.values():
            key_repr = repr(_canonical_key(key))

            # capture repr(value) and split into lines
            raw_lines = repr(value).splitlines() or [""]
            if len(raw_lines) > 1:
                # first line as‐is, indent subsequent
                first, *rest = raw_lines
                rest_indented = "\n".join(f"{item_indent}{ln}" for ln in rest)
                value_repr = f"{first}\n{rest_indented}"
            else:
                value_repr = raw_lines[0]

            item_strings.append(f"{item_indent}{key_repr}: {value_repr},")

        body = "\n".join(item_strings)
        return (
            f"{cls_name}({{\n"
            f"{body}\n"
            f"{current_indent}}})"
        )

    def __repr__(self) -> str:
        # now just forward to your own indent‐printer
        return self._repr_with_indent(0)
