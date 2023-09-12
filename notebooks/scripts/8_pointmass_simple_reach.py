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

from feedbax.mechanics.linear import point_mass, System
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
mm = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=8, depth=1, key=jrandom.PRNGKey(0)
)

filter_spec = jax.tree_util.tree_map(lambda _: False, mm)
filter_spec = eqx.tree_at(
    lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias), 
    filter_spec,
    replace=(True, True)
)

filter_spec


# %%
class Mechanics(eqx.Module):
    system: System
    term: dfx.AbstractTerm
    solver: dfx.AbstractSolver
    dt: float
    
    def __init__(self, system, dt):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        self.solver = dfx.Tsit5()
        self.dt = dt
        
        # TODO: not sure if I can actually use (0, dt) for (tprev, tnext) every time
        # #! self.solver_state = self.solver.init(self.term, 0, dt, init_state, controls[0])
    
    def __call__(self, state, control):
        state, _, _, self.solver_state, _ = self.solver.step(
            self.term, 0, self.dt, state, control, self.solver_state
        )
        return state
    

class SimpleFeedback(eqx.Module):
    net: eqx.Module  
    mechanics: Mechanics
    
    def __init__(self, net, mechanics, dt):
        self.net = net
        self.mechanics = mechanics
        
    def __call__(self, state, target_state, solver_state):
        net_inputs = jnp.concatenate([state.reshape(-1), target_state])
        control, activity = self.net(net_inputs)
        state = self.mechanics(state, control)
        return state, control, activity
    
    
class Recursion(eqx.Module):
    step: eqx.Module
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self._step = step
        self.n_steps = n_steps
            
        # TODO: define based on output pytree (trace?)
        # could use eqx.filters to control which are stored
        self.states = jnp.zeros((n_steps, N_DIM, 2))
        self.controls = jnp.zeros((n_steps, N_DIM))
        self.activities = jnp.zeros((n_steps, n_hidden))
        
    def step(self, i, x):
        states, controls, activities, target_state = x 
        states[i+1], controls[i], activities[i] = self._step(
            states[i], target_state #, solver_state
        )
        return states, controls, activities, target_state
    
    def __call__(self, state, control, activity, target_state):
        self.states[0] = state
        return lax.fori_loop(
            0, 
            self.n_steps, 
            self.step,
            (self.states, self.controls, self.activities, target_state)
        )
        state, control, activity, solver_state = self.model(state, target_state, solver_state)
        return state, control, activity, solver_state

# %%
# def body_loop(i, trajectory):
#     states, controls, activities, target_state, solver_state = trajectory
#     states[i+1], controls[i], activities[i], solver_state = body(states[i], target_state, solver_state)
#     return states, controls, activities, target_state, solver_state

# def model(init_state, target_state, solver_state):
#     states = jnp.zeros((n_steps, N_DIM, 2))
#     states[0] = init_state
#     controls = jnp.zeros((n_steps, N_DIM))
#     activities = jnp.zeros((n_steps, n_hidden))
    
#     
#     solver_state = solver.init(term, 0, dt, init_state, controls[0])
    
#     states, controls, activities, solver_state = lax.fori_loop(
#         0, n_steps, body_loop, (states, controls, activities, target_state, solver_state)
#     )
    
#     return states, controls, activities
    

# %%
net = SimpleMultiLayerNet(
    (6, 20, N_DIM),
    layer_type=eqx.nn.GRUCell,
    use_bias=True,
    key=jrandom.PRNGKey(0),
) 
mechanics = point_mass(mass=1., n_dim=N_DIM)

sf = SimpleFeedback(net, mechanics, dt=0.01)

sf


# %%
def get_model(dt, mass, n_input, n_hidden, n_steps, key):
    mechanics = Mechanics(point_mass(mass=mass, n_dim=N_DIM), dt)
  
    net = SimpleMultiLayerNet(
        (n_input, n_hidden, N_DIM),
        layer_type=eqx.nn.GRUCell,
        use_bias=True,
        key=key,
    ) 
    
    body = SimpleFeedback(net, mechanics)
    
    model = Recursion(body, n_steps)
    
    return model



# %%
def loss_fn(diff_model, static_model, init_state, target_state, weights=(1., 1., 1e-5, 1e-5)):  
    model = eqx.combine(diff_model, static_model)  
    states, controls, activities = jax.vmap(model)(init_state, target_state)
    
    pos_cost = jnp.sum((states[..., :2] - target_state[..., :2]) ** 2, axis=-1)  # sum over xyz
    vel_cost = jnp.sum((states[..., -1, 2:] - target_state[..., 2:]) ** 2, axis=-1)  # over xyz
    control_cost = jnp.sum(controls ** 2, axis=-1)  # over control variables
    activity_cost = activities ** 2
    
    return jnp.sum(w * jnp.mean(cost, axis=0)  # mean over batch 
                   for w, cost in zip(weights, 
                                      [pos_cost, vel_cost, control_cost, activity_cost]))


# %%
def main(
    mass=1.0,
    n_steps=100,
    dt=0.1,
    workspace = jnp.array([[-1., 1.], [-1., 1.]]),
    batch_size=100,
    n_batches=1000,
    epochs=1,
    learning_rate=3e-4,
    hidden_size=20,
    seed=5566,
):
    key = jrandom.PRNGKey(seed)

    def get_batch(batch_size, key):
        pos_endpoints = jrandom.uniform(
            key, 
            (batch_size, N_DIM, 2),
            minval=workspace[:, 0], 
            maxval=workspace[:, 1]
        )
        # add 0 velocity to init and target state
        state_endpoints = jnp.pad(pos_endpoints, ((0, 0), (0, 2), (0, 0)))
        return state_endpoints
    
    n_input = N_DIM * 2 * 2  # 2D state (pos, vel) feedback & target state
    
    model = get_model(dt, mass, n_input, hidden_size, n_steps, key)
    
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (model.layers[0].weight, model.layers[0].bias),
        filter_spec,
        replace=(True, True)
    )     
    
    optim = optax.adam(learning_rate)
    # TODO: only update net parameters
    opt_state = optim.init(model)

    def train_step(loss_fn, init_state, target_state, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model, static_model, init_state, target_state)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for _ in range(epochs):
        for batch in tqdm(range(n_batches)):
            key = jrandom.split(key)[0]
            state_endpoints = get_batch(batch_size, key)
            state = init_state = state_endpoints[:, :, 0]
            target_state = state_endpoints[:, :, 1]
            
            loss, model, opt_state = train_step(model, init_state, target_state, opt_state)
            
            if batch % 10 == 0:
                print(f"step: {batch}, loss: {loss:.4f}")


# %%
main(dt=0.01, seed=5566)

# %%
