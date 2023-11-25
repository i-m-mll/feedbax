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
from collections import UserDict
from functools import cached_property
from types import MappingProxyType

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

# %%
from feedbax.loss import LossDict 

# %%
import jax.tree_util as jtu 

a = dict(a=1, b=2)
b = dict(c=3, d=4)
c = dict(e=5, d=6)
d = dict()

jtu.tree_reduce(
    lambda x, y: x | y,
    (a,b,c,d),
    is_leaf=lambda x: isinstance(x, dict)
)

# %%
LossDict_ = LossDict

test = LossDict_(
    test1=jnp.ones(10),
    test2=jr.normal(jr.PRNGKey(0), (10,)),
    test3=jr.normal(jr.PRNGKey(1), (10,)),
)

loss_terms = (
    LossDict_(test1=jnp.ones(10)),
    LossDict_(test2=jr.normal(jr.PRNGKey(0), (10,))),
    LossDict_(test3=jr.normal(jr.PRNGKey(1), (10,))),
)

losses = jtu.tree_reduce(
    lambda x, y: x | y,
    loss_terms,
    is_leaf=lambda x: isinstance(x, LossDict_)
)

type(losses)

# %%
type(test)

# %%
jtu.tree_leaves(test)

# %%
jax.tree_map(jnp.sum, test)

# %%
losses.total

jax.tree_leaves(losses)


# %% [markdown]
# Basic test that we can make a dict that, when used in float operations, behaves as the total loss

# %%
class LossDict(UserDict):
    @cached_property
    def total(self):
        loss_term_values = list(self.values())
        return jnp.sum(jtu.tree_map(
            lambda *args: sum(args), 
            *loss_term_values
        ))
    
    def __add__(self, other):
        return self.total + other 
    
    def __radd__(self, other):
        return self.__add__(other)


# %%
n = 1_000_000

losses_large = LossDict(zip(
    tuple('abcdef'),
    [jr.uniform(jr.PRNGKey(i), (n,)) for i in range(6)],
))

# %%
losses_large.total.shape

# %%
# %time losses_large.total

# %%
loss = LossDict({
    'test_term_1': jnp.array(0.56),
    'test_term_2': jnp.array(0.97),
})

# %%
# %%timeit
loss = LossDict({
    'test_term_1': jnp.array(0.56),
    'test_term_2': jnp.array(0.97),
})

# %%
loss.total

# %%
loss + 1

# %%
type(loss)


# %% [markdown]
# Try doing the same thing by subclassing `dict` rather than `UserDict`. Also override `__setitem__` and `update` to make it "immutable". However, the values need to be immutable for it to be truly immutable. Assuming the values will be JAX arrays, I guess the assumption of immutability is as strong as it is anywhere else in JAX code.

# %%
class LossDict(dict[str, Array]):
    # @cached_property
    @property
    def total(self):
        loss_term_values = list(self.values())
        return jnp.sum(jtu.tree_map(
            lambda *args: sum(args), 
            *loss_term_values
        ))
    
    def __add__(self, other):
        return self.total + other 
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __setitem__(self, key, value):
        raise TypeError("LossDict does not support item assignment")
        
    def update(self, dict_: dict):
        raise TypeError("LossDict does not support update")
    
    

# %%
loss = LossDict({
    'test_term_1': jnp.array(0.56),
    'test_term_2': jnp.array(0.97),
})

# %%
# %%timeit
loss = LossDict({
    'test_term_1': jnp.array(0.56),
    'test_term_2': jnp.array(0.97),
})

# %%
loss.total

# %%
loss + 1

# %%
type(loss)

# %%
loss.update({'test_term_3': jnp.array(0.97)})

# %%
loss.total

# %% [markdown]
# Try converting to an immutable view:

# %%
frozen_loss = MappingProxyType(loss)

# %%
# doesn't work
frozen_loss.total


# %% [markdown]
# I'm not sure this is going to work with JAX... it might require that the return value of a `grad`ed function is literally a JAX scalar

# %%
def loss_func(x):
    return LossDict({
        'test_term_1': jnp.array(x),
        'test_term_2': jnp.array(x),
    })

jax.grad(loss_func)(1.)


# %% [markdown]
# Apparently there's a dunder for JAX types to return their array value... also we probably need to expose the shape and dtype

# %%
class LossDict(dict[str, Array]):
    # @cached_property
    @property
    def total(self):
        loss_term_values = list(self.values())
        return jnp.sum(jtu.tree_map(
            lambda *args: sum(args), 
            *loss_term_values
        ))
    
    def __jax_array__(self):
        return self.total
    
    def __add__(self, other):
        return self.total + other 
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __setitem__(self, key, value):
        raise TypeError("LossDict does not support item assignment")
        
    def update(self, dict_: dict):
        raise TypeError("LossDict does not support update")
    
    dtype = property(lambda self: self.total.dtype)
    ndim = property(lambda self: self.total.ndim)
    shape = property(lambda self: self.total.shape)
    
    

# %%
test = LossDict({
        'test_term_1': jnp.array(0.3),
        'test_term_2': jnp.array(0.4),
    })

# %%
test.shape


# %%
def loss_func(x):
    return LossDict({
        'test_term_1': jnp.array(0.3),
        'test_term_2': jnp.array(0.4),
    })

jax.grad(loss_func)(jnp.array(1.))

# %%
tt= LossDict({
        'test_term_1': jnp.array(0.3),
        'test_term_2': jnp.array(0.4),
    })

issubclass(type(tt), dict)


# %% [markdown]
# It's still angry because... the value is a tracer?
#
# Let's see what happens if we use a test case from the JAX API unit testsc

# %%
class AlexArray:
      def __init__(self, jax_val):
        self.jax_val = jax_val
      def __jax_array__(self):
        return self.jax_val
      dtype = property(lambda self: self.jax_val.dtype)
      shape = property(lambda self: self.jax_val.shape)


# %%
def func(x):
    return AlexArray(jnp.array(0.33))

x = jnp.array(1.)

jax.grad(func)(x)

# %%
