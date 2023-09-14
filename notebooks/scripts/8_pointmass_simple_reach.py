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
import sys

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Array, Float, PyTree
import matplotlib.pyplot as plt
import numpy as np
import optax 
import tqdm
from tqdm import tqdm

from feedbax.mechanics.linear import point_mass, System
from feedbax.networks import SimpleMultiLayerNet, RNN
from feedbax.plot import plot_loglog_losses, plot_states_forces_2d
from feedbax.task import centreout_endpoints, uniform_endpoints

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
DEBUG = False

# %%
N_DIM = 2


# %%
class Mechanics(eqx.Module):
    system: System 
    dt: float = eqx.field(static=True)
    term: dfx.AbstractTerm = eqx.field(static=True)
    solver: dfx.AbstractSolver #= eqx.field(static=True)
    
    def __init__(self, system, dt):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        self.solver = dfx.Tsit5()
        self.dt = dt        
    
    def __call__(self, state, args):
        # TODO: option for multiple timesteps per call
        inputs, solver_state = args 
        # using (0, dt) for (tprev, tnext) seems to work if there's no t dependency in the system
        state, _, _, solver_state, _ = self.solver.step(
            self.term, 0, self.dt, state, inputs, solver_state, made_jump=False
        )
        return state
    
    def init_solver_state(self, state):
        args = inputs_empty = jnp.zeros((self.system.B.shape[1],))
        return self.solver.init(self.term, 0, self.dt, state, args)
    

class SimpleFeedback(eqx.Module):
    net: eqx.Module  
    mechanics: Mechanics 
    
    def __init__(self, net, mechanics):
        self.net = net
        self.mechanics = mechanics
    
    def __call__(self, state, args):
        mechanics_state, _, hidden, solver_state = state
        inputs = args
        
        # mechanics state feedback plus task inputs (e.g. target state)
        net_inputs = jnp.concatenate([mechanics_state.reshape(-1), inputs]) 
        control, hidden = self.net(net_inputs, hidden)
        
        mechanics_state = self.mechanics(mechanics_state, (control, solver_state))
        
        return mechanics_state, control, hidden, solver_state
    
    def init_state(self, mechanics_state):
        return (
            mechanics_state, 
            jnp.zeros((self.net.out_size,)),
            jnp.zeros((self.net.hidden_size,)),
            self.mechanics.init_solver_state(mechanics_state),   
        )
    
    
class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        states, args = x 
        # #! this might break on non-array leaves
        state = jax.tree_util.tree_map(lambda x: x[i], states)
        state = self.step(state, args)
        states = jax.tree_util.tree_map(lambda x, y: x.at[i+1].set(y), states, state)
        return states, args
    
    def __call__(self, state, args):
        init_state = self.step.init_state(state)
        init_states = self._init_zero_arrays(init_state, args)
        # #! self.states[0] = state
        
        if DEBUG: #! jax.debug doesn't work inside of lax loops  
            states = init_states
                
            for i in range(self.n_steps):
                state = jax.tree_util.tree_map(lambda x: x[i], states)
                state = self.step(state, args)
                states = jax.tree_util.tree_map(lambda x, y: x.at[i+1].set(y), states, state)
                
            return states, args    
        
        return lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (init_states, args),
        )
    
    def _init_zero_arrays(self, state, args):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            eqx.filter_eval_shape(self.step, state, args)
        )


# %%
# jax.debug.print(''.join([f"{s.shape}\t{p}\n" 
#                             for p, s in jax.tree_util.tree_leaves_with_path(state)]))

# %%
def get_model(dt, mass, n_input, n_hidden, n_steps, key):
    
    mechanics = Mechanics(point_mass(mass=mass, n_dim=N_DIM), dt)
    net = RNN(n_input, N_DIM, n_hidden, key=key)
    body = SimpleFeedback(net, mechanics)

    return Recursion(body, n_steps)


# %%
LOSS_TERMS = ('final_position', 'position', 'final_velocity', 'control', 'hidden')

def loss_fn(
    diff_model, 
    static_model, 
    init_state, 
    target_state, 
    weights=jnp.array((10., 0.1, 10., 1e-3, 1e-5)),
    discount=1.,
):  
    model = eqx.combine(diff_model, static_model)  
    model = jax.vmap(model)

    states, _ = model(init_state, target_state)
    states, controls, activities, solver_state = states
    
    loss_terms = dict(
        final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
        position=jnp.sum((states[..., :2] - target_state[:, None, :2]) ** 2, axis=(-1, -2)),  # sum over xyz and time
        final_velocity=jnp.sum((states[..., -1, 2:] - target_state[..., 2:]) ** 2, axis=-1).squeeze(),  # over xyz
        control=jnp.sum(controls ** 2, axis=(-1, -2)),  # over control variables and time
        hidden=jnp.sum(activities ** 2, axis=(-1, -2)),  # over network units and time
    )
    
    loss_terms = jnp.mean(jnp.stack(list(loss_terms.values()), axis=1), axis=0)  # mean over batch
    
    loss = jnp.sum(weights * loss_terms)  # sum over terms
    return loss, loss_terms


# %%
def train(
    mass=1.0,
    n_steps=100,
    dt=0.1,
    workspace = jnp.array([[-1., 1.], [-1., 1.]]),
    batch_size=100,
    n_batches=50,
    epochs=1,
    learning_rate=3e-4,
    hidden_size=20,
    seed=5566,
):
    key = jrandom.PRNGKey(seed)

    def get_batch(batch_size, key):
        """Segment endpoints uniformly distributed in a rectangular workspace."""
        return uniform_endpoints(key, batch_size, N_DIM, workspace)
    
    n_input = N_DIM * 2 * 2  # 2D state (pos, vel) feedback & target state
    
    model = get_model(dt, mass, n_input, hidden_size, n_steps, key)

    # only train the RNN layer (input weights and hidden weights and biases)
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.step.net.cell.weight_hh, 
                      tree.step.net.cell.weight_ih, 
                      tree.step.net.cell.bias),
        filter_spec,
        replace=(True, True, True)
    )     
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @jax.jit
    def train_step(model, init_state, target_state, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state

    losses = []
    losses_terms = [] 

    for _ in range(epochs):
        for batch in tqdm(range(n_batches)):
            key = jrandom.split(key)[0]
            init_state, target_state = get_batch(batch_size, key)
            
            loss, loss_terms, model, opt_state = train_step(model, init_state, target_state, opt_state)
            
            losses.append(loss)
            losses_terms.append(loss_terms)
            
            if batch % 50 == 0:
                tqdm.write(f"step: {batch}, loss: {loss:.4f}", file=sys.stderr)
    
    losses_terms = jnp.vstack(losses_terms)
    
    return model, losses, losses_terms


# %%
trained, losses, losses_terms = train(batch_size=500, dt=0.1, n_batches=1000, n_steps=100, seed=5566)

# %%
plot_loglog_losses(losses, losses_terms)

# %% [markdown]
# Evaluate on a centre-out task

# %%
n_directions = 8
reach_length = 1.
state_endpoints = centreout_endpoints(jnp.array([0., 0.]), n_directions, 0, reach_length)
(states, controls, activities, _), _ = jax.vmap(trained)(*state_endpoints)

# %% [markdown]
# The network activities are constant after the 0th step. Probably because I keep overriding the hidden state with zeros whenever GRUCell is called...

# %%
plot_states_forces_2d(states, controls, endpoints=state_endpoints[...,:2])


# %% [markdown]
# ### Single module
#
# Trying a single module that uses `dfx.diffeqsolve`...

# %%
class SimpleFeedback(eqx.Module):
    net: eqx.Module
    system: System 
    dt: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    term: dfx.AbstractTerm = eqx.field(static=True)
    solver: dfx.AbstractSolver = eqx.field(static=True)
    
    def __init__(self, net, system, dt, t1):
        self.net = net
        self.system = system
        self.dt = dt
        self.t1 = t1
        self.term = dfx.ODETerm(self.system.vector_field)
        self.solver = dfx.Tsit5()
    
    def __call__(self, state, args):
        inputs = args
        
        # #! how can we track the control and activity trajectory?
        def control_callback(t, state, inputs):
            net_inputs = jnp.concatenate([state.reshape(-1), inputs])
            control, activity = self.net(net_inputs)
            return control
        
        sol = dfx.diffeqsolve(
            self.term, 
            self.solver, 
            0, 
            self.t1, 
            self.dt, 
            state, 
            args=(control_callback, inputs), 
            saveat=dfx.SaveAt(dense=True)
        )
        
        return sol


# %%
def get_model(dt, mass, n_input, n_hidden, t1, key):
    system = point_mass(mass=mass, n_dim=N_DIM)
  
    net = SimpleMultiLayerNet(
        (n_input, n_hidden, N_DIM),
        layer_type=eqx.nn.GRUCell,
        use_bias=True,
        linear_final_layer=True,
        key=key,
    ) 
        
    return SimpleFeedback(net, system, dt, t1)



# %%
def loss_fn(diff_model, static_model, init_state, target_state, weights=(1., 1., 1e-5, 1e-5)):  
    model = eqx.combine(diff_model, static_model)  
    model = jax.vmap(model)

    states, _ = model(init_state, target_state)
    states, controls = states
    
    pos_cost = jnp.sum((states[..., :2] - target_state[..., :2]) ** 2, axis=-1)  # sum over xyz
    vel_cost = jnp.sum((states[..., -1, 2:] - target_state[..., 2:]) ** 2, axis=-1)  # over xyz
    control_cost = jnp.sum(controls ** 2, axis=-1)  # over control variables
    #activity_cost = activities ** 2
    
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
    
    # only train the hidden RNN layer 
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.net.layers[0].weight_hh, 
                      tree.net.layers[0].weight_ih, 
                      tree.net.layers[0].bias),
        filter_spec,
        replace=(True, True, True)
    )     
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def train_step(model, init_state, target_state, opt_state):
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

# %%
