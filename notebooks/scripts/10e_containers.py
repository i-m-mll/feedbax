# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
import dataclasses
from dataclasses import dataclass
from abc import ABCMeta
import functools

import equinox as eqx
import jax
import jax.numpy as jnp


# %% [markdown]
# Writing a pared-down version of `eqx.Module` with the features I need for an abstract state class, and seeing how much faster concrete instantiation is.

# %%
class _StateMeta(ABCMeta): 
    """Based on `eqx._module._ModuleMeta`."""
    
    def __new__(mcs, name, bases, dict_, **kwargs):
        # #? I think this works to add slots for dataclass fields as long as no default values supplied
        # TODO: test performance difference
        dict_['__slots__'] = tuple(dict_['__annotations__'].keys())
        
        cls = super().__new__(mcs, name, bases, dict_, **kwargs)
        
        cls = dataclass(eq=False, repr=False)(
            cls
        )
        
        def datacls_flatten(datacls):
            field_names, field_values = zip(*[
                #(field_.name, datacls.__dict__[field_.name])
                (field_.name, getattr(datacls, field_.name))
                for field_ in dataclasses.fields(datacls)
            ])
            
            # for field_ in dataclasses.fields(datacls):
            #     field_names.append(field_.name)
            #     field_values.append(datacls.__dict__[field_.name])
                              
            children, aux = tuple(field_values), tuple(field_names)
            return children, aux
        
        def datacls_unflatten(cls, aux, children):
            field_names, field_values = aux, children
            datacls = object.__new__(cls)
            for name, value in zip(field_names, field_values):
                object.__setattr__(datacls, name, value)
            return datacls 
        
        jax.tree_util.register_pytree_node(
            cls, 
            flatten_func=datacls_flatten,
            unflatten_func=functools.partial(datacls_unflatten, cls),
        )
        
        return cls
        
class AbstractState(metaclass=_StateMeta):
    """Base class for state dataclasses.
    
    Automatically instantiated as a dataclass with slots.
    
    Based on `eqx.Module`. I could probably use `eqx.Module` instead; 
    I wonder if there is a performance difference for instantiations?
    """
    __annotations__ = dict()
    
    def __hash__(self):
        return hash(tuple(jax.tree_util.tree_leaves(self)))
    
    # def __repr(self):
    #     return 


# %%
class TestEqx(eqx.Module):
    a: int 
    b: float 
    c: jnp.ndarray
    d: jnp.ndarray 
    
class TestNestedEqx(eqx.Module):
    x: jnp.ndarray
    y: TestEqx 


# %%
class TestState(AbstractState):
    a: int 
    b: float 
    c: jnp.ndarray
    d: jnp.ndarray
    
class TestNestedState(AbstractState):
    x: jnp.ndarray
    y: TestState


# %%
data = (1, 2.0, jnp.ones((10,)), jnp.empty((1000,)))
x = jnp.zeros((500,))

# %%
# %timeit TestEqx(*data)

# %%
# %timeit TestState(*data)

# %%
# %timeit TestNestedEqx(x, TestEqx(*data))

# %%
# %timeit TestNestedState(x, TestState(*data))

# %% [markdown]
# Why is `eqx.Module` so much slower?

# %%
import cProfile 
import pstats
from pstats import SortKey

eqx_filename = 'eqx_module_profile_stats'
state_filename = 'abstractstate_profile_stats'

n_eqx = 100_000
n_state = 100_000

def test_eqx(): 
    for _ in range(n_eqx): 
        TestEqx(*data) 

def test_state(): 
    for _ in range(n_state):
        TestState(*data)

cProfile.run('test_eqx()', eqx_filename)
cProfile.run('test_state()', state_filename)

# %%
p_eqx = pstats.Stats(eqx_filename)
p_eqx.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)

p_state = pstats.Stats(state_filename)
p_state.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)

# %% [markdown]
# We can also use `line_profiler` to get a line-by-line profile of certain functions.

# %%
# %load_ext line_profiler

# %%
from line_profiler import LineProfiler

profile = LineProfiler(test_eqx)
profile.add_function(eqx._module._ModuleMeta.__new__)
profile.add_function(eqx._module._ModuleMeta.__call__)

profile.runcall(test_eqx)

# %%
profile.print_stats()


# %% [markdown]
# I don't know why the profiling didn't work for `__new__`. Maybe has something to do with the way object construction happens.

# %% [markdown]
# I'm not sure what the point of the `missing_names` code is. 
#
# I'm not sure how a name could appear in `fields(cls)` but not appear in `dir(self)`. 
#
# I assumed that dataclass instantiation would raise an error if data were not supplied to occupy all the fields of the instance. 
#
# I suppose this might have to do with subclassing or special field descriptors or...

# %%
@dataclass 
class Test:
    a: int 
    b: float 
    c: jnp.ndarray
    d: jnp.ndarray
    
aa = Test(*data)

print(dataclasses.fields(Test))

print(dir(aa))
