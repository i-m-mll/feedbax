"""Custom mapping classes.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping

import logging
from typing import Generic, TypeVar

from feedbax.misc import unzip2


logger = logging.getLogger(__name__)


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

    See `feedbax.task.WhereDict` for an example.

    Based on https://stackoverflow.com/a/3387975
    """
    store: OrderedDict[KT1, tuple[KT2, VT]]

    def __init__(self, *args, **kwargs):
        self.store = OrderedDict()
        self.update(OrderedDict(*args, **kwargs))

    def __getitem__(self, key: KT1 | KT2) -> VT:
        k = self._key_transform(key)
        return self.store[k][1]

    def __setitem__(self, key: KT2, value: VT):
        self.store[self._key_transform(key)] = (key, value)

    def __delitem__(self, key: KT2):
        del self.store[self._key_transform(key)]

    def __iter__(self):
        # Apparently you're supposed to only yield the key
        for key in self.store:
            yield self.store[key][0]

    def __len__(self) -> int:
        return len(self.store)

    @abstractmethod
    def _key_transform(self, key: KT1 | KT2) -> KT1:
        ...

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))
