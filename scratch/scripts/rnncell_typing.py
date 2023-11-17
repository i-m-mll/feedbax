# ---
# jupyter:
#   jupytext:
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
from typing import Protocol, Type

import equinox as eqx 
import jax
import jax.numpy as jnp


# %%
class RNNCell(Protocol):
    hidden_size: int
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        use_bias: bool, 
        *, 
        key: jax.Array,
        **kwargs, 
    ):
        ...
        

class Test(eqx.Module):
    cell: eqx.Module
    
    def __init__(
        self, 
        cell_type: Type[RNNCell], 
        input_size: int, 
        hidden_size: int, 
        **kwargs
    ):
        self.cell = cell_type(input_size, hidden_size, **kwargs)

# %%
