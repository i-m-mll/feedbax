"""Dataclass PyTree containers.

The classes in this module are more or less reductions of classes from
    `eqx._module`, that retain a subset of features.

NOTE: 
    This is currently unused, since it doesn't offer an advantage over
    `eqx.Module`, according to a simple benchmark of 3 runs as performed in
    notebook 15, in one case monkey patching `AbstractState = eqx.Module` from
    inside this module.
    
    However, in the future we might want to use `AbstractState` as a subclass
    of `eqx.Module` to add certain features.

    The original concern was that `eqx.Module` has some (relatively)
    significant overhead related to verification of occupancy of dataclass
    fields. This takes up a lot of the instantiation time for `eqx.Module`
    but in practice it doesn't seem to make a difference to `feedbax`, 
    probably because we don't instantiate enough module objects for it to 
    matter.    
"""

from abc import ABCMeta
from dataclasses import dataclass
import dataclasses
import functools as ft
from typing import TYPE_CHECKING, dataclass_transform

import equinox as eqx
import jax 

from feedbax.utils import datacls_flatten, datacls_unflatten


@dataclass_transform(field_specifiers=(dataclasses.field, eqx.field))
class _StateMeta(ABCMeta): 
    """Based on `eqx._module._ModuleMeta`."""
    
    def __new__(mcs, name, bases, dict_, **kwargs):
        #? I think this works as long as no default values are supplied
        dict_['__slots__'] = tuple(dict_['__annotations__'].keys())
        
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)
        
        cls = dataclass(eq=False, repr=False)(
            cls
        )
        
        jax.tree_util.register_pytree_node(
            cls, 
            flatten_func=datacls_flatten,
            unflatten_func=ft.partial(datacls_unflatten, cls),
        )
        
        return cls
        

#! this was the patch used to test performance of `eqx.Module` vs. `AbstractState`
# AbstractState = eqx.Module


class AbstractState(metaclass=_StateMeta):
    """Base class for state dataclasses.
    
    Automatically instantiated as a dataclass with slots.
    
    Based on `eqx.Module`.
    """
    # 
    __annotations__ = dict()
    
    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))
    
    def __repr__(self):
        return eqx.tree_pformat(self)



    
    
