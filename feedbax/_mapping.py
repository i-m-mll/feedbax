"""Custom mapping classes.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping

import logging 
from typing import TypeVar

from feedbax.misc import unzip2


logger = logging.getLogger(__name__)


KT = TypeVar("KT")
VT = TypeVar("VT")


class AbstractTransformedOrderedDict(MutableMapping[KT, VT]):
    """Base for `OrderedDict`s which transform keys when getting and setting items.
    
    It stores the original keys, and otherwise behaves (e.g. when iterating)
    as though they are the true keys.
    
    This is useful when we want to use a certain type of object as a key, but 
    it would not be hashed properly by `OrderedDict`, so we need to transform
    it into something else. In particular, by replacing `_key_transform` with 
    a code parsing function, a subclass of this class can take lambdas as keys.
    
    See `feedbax.task.InitSpecDict` for an example.
       
    Based on https://stackoverflow.com/a/3387975
    
    TODO: 
    - I'm not sure how the typing should work. I guess `VT` might need to correspond
      to a tuple of the original key and the value.
    """
    
    def __init__(self, *args, **kwargs):
        self.store = OrderedDict()
        self.update(OrderedDict(*args, **kwargs))
            
    def __getitem__(self, key):
        return self.store[self._key_transform(key)][1]

    def __setitem__(self, key, value):
        self.store[self._key_transform(key)] = (key, value)
    
    def __delitem__(self, key):
        del self.store[self._key_transform(key)]
        
    def __iter__(self):
        # Apparently you're supposed to only yield the key
        for key in self.store:
            yield self.store[key][0]

    def __len__(self):
        return len(self.store)

    @abstractmethod
    def _key_transform(self, key):
        ...

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))

