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
from dataclasses import dataclass, make_dataclass
from typing import Protocol 

import equinox as eqx 
import jax
from typeguard import typechecked


# %%
class Test(eqx.Module):
    a: jax.Array


# %%
make_dataclass(
    'TestDynamic', 
    [
        ('b', jax.Array),
    ],
    bases=(Test,),
    frozen=True,
    eq=False,
    repr=False,
)

# %%
