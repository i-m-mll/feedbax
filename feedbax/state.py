"""Dataclass PyTree containers.

We could just use `eqx.Module` for state dataclasses, but it has some
significant overhead related to verification of occupancy of dataclass fields.
The classes in this module are more or less reductions of classes from 
`eqx._module` that retain a subset of their features.
"""

from abc import ABCMeta
from dataclasses import dataclass
import functools as ft

import equinox as eqx
import jax 

from feedbax.utils import datacls_flatten, datacls_unflatten


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
        

class AbstractState(metaclass=_StateMeta):
    """Base class for state dataclasses.
    
    Automatically instantiated as a dataclass with slots.
    
    Based on `eqx.Module`. I could probably use `eqx.Module` instead; 
    I wonder if there is a performance difference for instantiations?
    """
    # 
    __annotations__ = dict()
    
    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))
    
    def __repr__(self):
        return eqx.tree_pformat(self)
    
    
