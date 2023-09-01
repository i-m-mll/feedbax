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
import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm

from feedbax.mechanics.linear import point_mass
from feedbax.networks import SimpleMultiLayerNet 

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a point mass to reach from a starting position to a target position. 
#
# Components:
#
# - Point mass module
# - RNN module
# - Feedback loop 
#     - RNN call plus a single step of diffrax integration
# - Loss function
#     - quadratic in position near final state
#     - quadratic in controls
# - Generate reach endpoints
#     - uniformly sampled in a rectangular workspace
#     - i.i.d. start and end (variable magnitude)

# %%
N_DIM = 2

mass = 1.

mechanics = point_mass(mass=mass, n_dim=N_DIM)

# %%
n_input = 6  # 2D pos, vel feedback & target pos
n_hidden = 20
net_key = 5566

net = SimpleMultiLayerNet(
    (n_input, n_hidden, N_DIM),
    layer_type=eqx.nn.GRUCell,
    use_bias=True,
    key=jrandom.PRNGKey(net_key),
)


# %%
def generate_endpoints():
    pass 


# %%
def loss_fn():
    pass 


# %%
def train():
    pass
