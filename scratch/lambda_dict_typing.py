"""This is for testing how the typing works with `AbstractTransformedOrderedDict`,
in particular whether the key-value types refer to those that appear at the interface  
of `__getitem__` and `__setitem__`, or those that are actually stored in `self.store`.
"""

from abc import abstractmethod
from collections import OrderedDict
import dis
from typing import Callable, MutableMapping, TypeVar
from jax import Array
from jaxtyping import PyTree 

from feedbax.model import AbstractModelState

KT = TypeVar('KT')
VT = TypeVar('VT')


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
    
    def __repr__(self):
        # TODO: Shouldn't print as `OrderedDict` but `TransformedDict`
        return repr(OrderedDict(self.items()))
    
    @abstractmethod
    def _key_transform(self, key):
        ...


def get_where_str(where_func):
    bytecode = dis.Bytecode(where_func)
    return '.'.join(instr.argrepr for instr in bytecode
                    if instr.opname == "LOAD_ATTR")


class InitSpecDict(AbstractTransformedOrderedDict[
    Callable[["AbstractModelState"], PyTree[Array]],
    PyTree[Array]
]):
    def _key_transform(self, key):
        if isinstance(key, Callable):
            return get_where_str(key)
        return key
    
    
init_spec = InitSpecDict({
    lambda y: y: object,
    lambda x: x.mechanics.effector: 3,
    lambda state: state.mechanics.plant.skeleton: None,
})