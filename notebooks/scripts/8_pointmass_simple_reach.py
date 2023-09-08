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
from functools import partial
import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Array, Float
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

# %%
mass = 1.

n_input = 6  # 2D pos, vel feedback & target pos
n_hidden = 20
net_key = 5566

# feedback loop 
dt0 = 0.01
y = y0 = None 
args = None 


# %%
def get_mechanics_step(mechanics):
    term = dfx.ODETerm(mechanics)
    solver = dfx.Tsit5()
    return partial(solver.step, term)


# %%
# #! mean over batch after vmap

def loss_fn(states, controls, activities, target_state):
    pos_cost = (states[:, :2] - target_state[:2]) ** 2
    vel_cost = (states[-1, 2:] - target_state[2:]) ** 2
    control_cost = controls ** 2
    activity_cost = activities ** 2
    # #! weights; sum over time
    return pos_cost + vel_cost + control_cost + activity_cost


# %%
def main(
    n_steps=100,
    dt=0.1,
    dataset_size=50000,
    workspace = jnp.array([[-1., 1.], [-1., 1.]]),
    batch_size=100,
    epochs=1,
    learning_rate=3e-4,
    hidden_size=20,
    seed=5566,
):
    key = jrandom.PRNGKey(seed)

    endpoints_dataset = jrandom.uniform(
            key, 
            (dataset_size, N_DIM, 2),
            minval=workspace[:, 0], 
            maxval=workspace[:, 1]
        )
    
    net = SimpleMultiLayerNet(
        (n_input, n_hidden, N_DIM),
        layer_type=eqx.nn.GRUCell,
        use_bias=True,
        key=jrandom.PRNGKey(net_key),
    )
    
    mechanics = point_mass(mass=mass, n_dim=N_DIM)
    mechanics_step = get_mechanics_step(mechanics)
    
    def body_step(dt, state, target_state, solver_state):
        net_inputs = (state, target_state)
        control, activity = net(net_inputs)
        state, _, _, solver_state, _ = mechanics_step(0, dt, state, control, solver_state)
        return state, control, activity, solver_state
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(net)
    
    # TODO: get_batch: generate and store endpoints randomly on each batch
    # TODO: `loss, model, opt_state = train_batch(model, opt_state, init_state, target_state)`
    for epoch in range(epochs):
        for batch in tqdm(range(dataset_size // batch_size)):
            endpoints = endpoints_dataset[batch * batch_size : (batch + 1) * batch_size]
            endpoint_state = jnp.pad(endpoints, ((0, 0), (0, 2), (0, 0)))  # 0 start-end velocity
            state = endpoint_state[:, :, 0]
            target_state = endpoint_state[:, :, 1]
            
            states, controls, activities = [], [], []
            
            for step in range(n_steps):
                state, control, activity, solver_state = body_step(dt, state, target_state, solver_state)
                states.append(state)
                controls.append(control)
                activities.append(activity)
            
            # TODO: does model need to be called from inside loss function for grad to work?
            loss, grads = eqx.filter_value_and_grad(loss_fn)(states, controls, activities, target_state)
            loss = loss.item()
            updates, opt_state = optim.update(grads, opt_state, net)
            net = eqx.apply_updates(net, updates)
            
            if step % 10 == 0:
                print(f"step: {step}, loss: {loss:.4f}")
                
            
