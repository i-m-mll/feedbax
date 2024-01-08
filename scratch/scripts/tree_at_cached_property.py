# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %% [markdown]
# Making sure a `cached_property` in an Equinox module gets re-calculated when we do model surgery

# %%
from functools import cached_property
import equinox as eqx 
import jax
import jax.numpy as jnp


# %%
class Foo(eqx.Module):
    
    a: jax.Array
    
    @cached_property
    def test(self):
        return self.a + 2


f = Foo(jnp.ones(5))

f.test

# %%
eqx.tree_at(
    lambda foo: foo.a, 
    f, 
    jnp.zeros(5),
).test


# %% [markdown]
# Does `tree_at` re-initialize modules?

# %%
class Bar(eqx.Module):
    
    a: jax.Array
    b: bool 
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
        print("Smee")
        
    def __check_init__(self):
        print("Glue")
        
b = Bar(jnp.ones(5), True)

# %%
eqx.tree_at(
    lambda bar: bar.a, 
    b, 
    jnp.zeros(5),
).a

# %% [markdown]
# So it doesn't call `__init__` again, looks like. Nor Equinox's `__check_init__`.

# %% [markdown]
#
