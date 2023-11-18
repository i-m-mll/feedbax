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
from functools import wraps

import equinox as eqx


# %% [markdown]
# How do decorators interact with `__call__`, and in particular with equinox modules?

# %%
class Test(eqx.Module):
    "Smoo"
    
    def __call__(self, a, b):
        print("Smoopathy")
        return a + b 


def test_decorator(module):
    @wraps(module)
    def wrapper(*args, **kwargs):
        print("Before")
        result = module(*args, **kwargs)
        print("After")
        return result
    return wrapper 
        


# %%
test = Test()

test = test_decorator(test)

test.__doc__

# %%
test(2, 3)

# %% [markdown]
# `wraps` seems to work fine, and importantly, note that it carries through the class docstring, not that of `__call__`.

# %%
