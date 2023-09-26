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
import sys
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
import tqdm
from tqdm import tqdm

from feedbax.mechanics.linear import point_mass
from feedbax.mechanics.system import System
from feedbax.networks import SimpleMultiLayerNet, RNN
from feedbax.plot import plot_loglog_losses, plot_states_forces_2d
from feedbax.task import centreout_endpoints, uniform_endpoints
from feedbax.utils import tree_set_idx, tree_sum_squares

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
    solver: Optional[dfx.AbstractSolver] #= eqx.field(static=True)
    
    def __init__(self, system, dt, solver=None):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        if solver is None:
            self.solver = dfx.Tsit5()
        else:
            self.solver = solver
        self.dt = dt        
    
    def __call__(self, state, args):
        # TODO: optional multiple timesteps per call
        inputs, solver_state = args 
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        state, _, _, solver_state, _ = self.solver.step(
            self.term, 0, self.dt, state, inputs, solver_state, made_jump=False
        )
        # #! I don't even return solver state, so apparently it's not important
        return state
    
    def init_solver_state(self, state):
        args = inputs_empty = jnp.zeros((self.system.control_size,))
        return self.solver.init(self.term, 0, self.dt, state, args)
    

class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    delay: int = eqx.field(static=True)
    
    def __init__(self, net, mechanics, delay=0):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(self, state, args):
        mechanics_states, _, hidden, solver_state = state
        inputs = args
        
        # mechanics state feedback plus task inputs (e.g. target state)
        feedback_state = mechanics_states[-self.delay]
        net_inputs = jnp.concatenate([feedback_state.reshape(-1), inputs]) 
        control, hidden = self.net(net_inputs, hidden)
        
        net_force = control  # TODO: noise, perturbations
        mechanics_state = self.mechanics(mechanics_states[-1], (net_force, solver_state))
        mechanics_states = jnp.roll(mechanics_states, -1, 0).at[-1].set(mechanics_state)
        
        return mechanics_states, control, hidden, solver_state
    
    def _recursion(self, state, args):
        """Thinking about how to create an interface for `Recursion` that doesn't require `__call__` arguments to always be the same."""
        # init_state, _, hidden, solver_state = state
        # inputs = args
        # state = self(inputs, init_state)
        # return state 
    
    def init_state(self, mechanics_state):
        return (
            jnp.zeros((self.delay, *mechanics_state.shape)).at[-1].set(mechanics_state), 
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
        # this seems to work, but I'm worried it will break on non-array leaves later
        state = jax.tree_util.tree_map(lambda xs: xs[i], states)
        state = self.step(state, args)
        states = jax.tree_util.tree_map(lambda xs, x: xs.at[i+1].set(x), states, state)
        return states, args
    
    def __call__(self, state, args):
        init_state = self.step.init_state(state)
        init_states = self._init_zero_arrays(init_state, args)
        init_states = jax.tree_util.tree_map(lambda xs, x: xs.at[0].set(x), 
                                             init_states, init_state)
        
        if DEBUG: #! jax.debug doesn't work inside of lax loops  
            states = init_states
  
            for i in range(self.n_steps):
                states, args = self._body_func(i, (states, args))
                
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


# %% [markdown]
# Notice the delay machinery in `SimpleFeedback`. This isn't ideal because returning `delay` timesteps of mechanics state, causes `Recursion._init_zero_arrays` to memorize the entire delay period at *each* timestep; so that e.g. if `delay=5` then 6x as much data then necessary will be memorized by `Recursion`.
#
# Ideally we will generalize `Body` (of which `SimpleFeedback` would be a subclass) so that it can be constructed by wiring together different modules according to their arguments and returns, and specifying which returns should be memorized and how. In particular this would need to deal with cases like this one where the wiring occurs between one call and the next...
#
# But for now I just want to deal with the memory issue, so I'm implementing a different hack that indexes out feedback from the memory of the state that `Recursion` already keeps.
#
# (One more option might be to keep the delay states as an aux array that is returned from `SimpleFeedback` calls, so that it is passed from one call to the next without being memorized.)

# %%
class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    delay: int = eqx.field(static=True)
    
    def __init__(self, net, mechanics, delay=0):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(self, state, args):
        mechanics_state, _, hidden, solver_state = state
        inputs, feedback_state = args
        
        # mechanics state feedback plus task inputs (e.g. target state)
        net_inputs = jnp.concatenate([feedback_state.reshape(-1), inputs]) 
        control, hidden = self.net(net_inputs, hidden)
        
        mechanics_state = self.mechanics(mechanics_state, (control, solver_state))
        
        return mechanics_state, control, hidden, solver_state
    
    def _recursion(self, state, args):
        """Thinking about how to create an interface for `Recursion` that doesn't require `__call__` arguments to always be the same."""
        # init_state, _, hidden, solver_state = state
        # inputs = args
        # state = self(inputs, init_state)
        # return state 
    
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
        # this seems to work, but I'm worried it will break on non-array leaves later
        state = jax.tree_util.tree_map(lambda xs: xs[i], states)
        
        # #! this ultimately shouldn't be here, but costs less memory than a `SimpleFeedback`-based storage hack:
        feedback_state = states[0][i - self.step.delay]  
        args = (args[0], feedback_state)  
        
        state = self.step(state, args)
        states = jax.tree_util.tree_map(lambda xs, x: xs.at[i+1].set(x), states, state)
        return states, args
    
    def __call__(self, state, args):
        init_state = self.step.init_state(state)
        args_init = (args, jnp.zeros_like(args))  #! part of the feedback_state hack: stand in target_state for feedback_state
        init_states = self._init_zero_arrays(init_state, args_init)
        init_states = jax.tree_util.tree_map(lambda xs, x: xs.at[0].set(x), 
                                             init_states, init_state)
        
        if DEBUG: #! jax.debug doesn't work inside of lax loops  
            states = init_states
            args = args_init
  
            for i in range(self.n_steps):
                states, args = self._body_func(i, (states, args))
                
            return states, args    
        
        return lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (init_states, args_init),
        )
    
    def _init_zero_arrays(self, state, args):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            eqx.filter_eval_shape(self.step, state, args)
        )


# %%
def get_model(dt, mass, n_hidden, n_steps, key, feedback_delay=0):
    
    system = point_mass(mass=mass, n_dim=N_DIM)
    mechanics = Mechanics(system, dt)
    # #! in principle n_input is a function of mechanics state and task inputs
    n_input = system.state_size * 2  # feedback & target states
    net = RNN(n_input, system.control_size, n_hidden, key=key)
    body = SimpleFeedback(net, mechanics, delay=feedback_delay)

    return Recursion(body, n_steps)


# %%
LOSS_TERMS = ('position', 'final_velocity', 'control', 'hidden')

def loss_fn(
    diff_model, 
    static_model, 
    init_state, 
    target_state, 
    term_weights=dict(
        position=1., 
        final_velocity=1., 
        control=1e-5, 
        hidden=1e-5
    ),
    discount=1.,
    weight_decay=None,
):  
    """Quadratic in states, controls, and hidden activities.
    
    Assumes the `target_state` is fixed; i.e. this is not a tracking task.
    
    User can apply a temporal discount broadcastable by elementwise multiplication with `(n_batch, n_steps)`.
    """
    model = eqx.combine(diff_model, static_model)  
    batched_model = jax.vmap(model)

    states, _ = batched_model(init_state, target_state)
    states, controls, activities, solver_state = states
    # states = states[:, :, -1]  #! this was for the old delay solution with `SimpleFeedback`
    
    # sum over xyz, apply temporal discount, sum over time
    position_loss = jnp.sum(discount * jnp.sum((states[..., :2] - target_state[:, None, :2]) ** 2, axis=-1), axis=-1)
    
    loss_terms = dict(
        #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
        position=position_loss,  
        final_velocity=jnp.sum((states[..., -1, 2:] - target_state[..., 2:]) ** 2, axis=-1).squeeze(),  # over xyz
        control=jnp.sum(controls ** 2, axis=(-1, -2)),  # over control variables and time
        hidden=jnp.sum(activities ** 2, axis=(-1, -2)),  # over network units and time
    )
    # mean over batch
    loss_terms = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_terms)
    # term scaling
    loss_terms = jax.tree_map(lambda term, weight: term * weight, loss_terms, term_weights) 
    
    # NOTE: optax also gives optimizers that implement weight decay
    if weight_decay is not None:
        # this is separate because the tree map of `jnp.mean` doesn't like floats
        # and it doesn't make sense to batch-mean the model parameters anyway
        loss_terms['weight_decay'] = weight_decay * tree_sum_squares(diff_model)
        
    # sum over terms
    loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
    
    return loss, loss_terms


# %%
def train(
    mass=1.0,
    n_steps=100,
    dt=0.1,
    hidden_size=20,
    feedback_delay_steps=0,
    workspace = jnp.array([[-1., 1.], 
                           [-1., 1.]]),
    batch_size=100,
    n_batches=50,
    epochs=1,
    learning_rate=3e-4,
    term_weights=dict(
        position=1., 
        final_velocity=1., 
        control=1e-5, 
        hidden=1e-5
    ),
    seed=5566,
):
    key = jrandom.PRNGKey(seed)

    def get_batch(batch_size, key):
        """Segment endpoints uniformly distributed in a rectangular workspace."""
        return uniform_endpoints(key, batch_size, N_DIM, workspace)
    
    model = get_model(dt, mass, hidden_size, n_steps, key, 
                      feedback_delay=feedback_delay_steps)
    
    # only train the RNN layer (input weights & hidden weights and biases)
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
    position_error_discount = jnp.linspace(1./n_steps, 1., n_steps) ** 6
    
    @eqx.filter_jit
    def train_step(model, init_state, target_state, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state, 
            term_weights=term_weights, discount=position_error_discount
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state

    losses = jnp.empty((n_batches,))
    losses_terms = dict(zip(
        term_weights.keys(), 
        [jnp.empty((n_batches,)) for _ in term_weights]
    ))

    for _ in range(epochs):
        for batch in tqdm(range(n_batches)):
            key = jrandom.split(key)[0]
            init_state, target_state = get_batch(batch_size, key)
            
            loss, loss_terms, model, opt_state = train_step(model, init_state, target_state, opt_state)
            
            losses = losses.at[batch].set(loss)
            #losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
            
            if batch % 50 == 0:
                tqdm.write(f"step: {batch}, loss: {loss:.4f}", file=sys.stderr)
    
    return model, losses, losses_terms


# %%
trained, losses, losses_terms = train(
    batch_size=500, 
    dt=0.1, 
    feedback_delay_steps=5,
    n_batches=2000, 
    n_steps=100, 
    hidden_size=50, 
    seed=5566,
    learning_rate=0.01,
)

# %%
plot_loglog_losses(losses, losses_terms)

# %% [markdown]
# Evaluate on a centre-out task

# %%
n_directions = 8
reach_length = 1.
state_endpoints = centreout_endpoints(jnp.array([0., 0.]), n_directions, 0, reach_length)
(states, controls, activities, _), _ = jax.vmap(trained)(*state_endpoints)
#states = states[:, :, -1]  # TODO: this is because of the delay stuff in SimpleFeedback...

# %%
plot_states_forces_2d(states[...,:2], states[..., 2:], controls, endpoints=state_endpoints[...,:2])

# %%
plt.plot(jnp.sum(states[...,2:] ** 2, -1).T, '-')
plt.show()


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
