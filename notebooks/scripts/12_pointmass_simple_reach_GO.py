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
from datetime import datetime 
import json
from pathlib import Path 
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
from tqdm.auto import tqdm

from feedbax.mechanics.linear import point_mass
from feedbax.mechanics.system import System
from feedbax.networks import RNN, RNNCell
from feedbax.plot import (
    animate_3D_rotate,
    plot_activity_heatmap,
    plot_activity_sample_units,
    plot_loglog_losses, 
    plot_states_forces_2d,
    plot_task_and_speed_profiles,
)
from feedbax.task import (
    centreout_endpoints, 
    uniform_endpoints,
    gen_epoch_lengths,
    get_masked_seqs,
    get_scalar_epoch_seq,
    get_masks, 
)
from feedbax.utils import (
    tree_set_idx, 
    tree_sum_squares, 
    tree_get_idx,
    filter_spec_leaves,
)

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a point mass to reach from a starting position to a target position. The network should hold at the start position until a hold signal is switched off.

# %%
# %matplotlib widget

# %%
NB_PREFIX = '12_GO'

DEBUG = False

# gets set to True after a training run
# to prevent accidentally loading a model over the trained one
trained = False  

model_dir = Path('../models')

# %%
N_DIM = 2


# %% [markdown]
# Constructing task input signals. 
#
# Inputs: 
#
# - a pytree `x` giving the static (single-timestep) task input arrays
# - an ndarray `l` giving the integer ranges for uniform sampling of epoch lengths
#     - shape `(epochs, 2)`; typically should omit the final epoch and constrain the sequence length to be constant
#     
# Outputs:
#
# - a pytree `xs` with the same structure as `x`, where each array has an additional sequence dimension and has zeros (or some other fill value) everywhere, except for epochs filled with the static task input values from `x`
# - additional pytrees for other signals (e.g. hold signal) that require the same epoch structure but are not dependent on `x`
#
# Comments: 
#
# - Presumably a distinct function should do the epoch length sampling, and the returned values may be re-used.
#     - `jrandom.randint` can take arrays for `min` and `max` that broadcast with `shape`
# - Use `jnp.cumsum` on the epoch lengths to get the indices 
# - Tree map `jnp.zeros` onto `x` to give whole sequences (shape `(n_steps, *leaf.shape)`)
#     - don't concatenate epochs since this requires dynamic array sizing and doesn't play well with jit/vmap
# - Insert values from `x` into `xs` during the appropriate epochs
# - Construct other signals (e.g. fixation/hold) as full sequences and insert values in appropriate epochs
#     - Possibly append these signals to `xs`
#

# %%
# for experimenting

def get_batch(
    workspace = jnp.array([[-1., 1.], 
                           [-1., 1.]]),
    batch_size=5, 
    key=jrandom.PRNGKey(0),
):
    """Segment endpoints uniformly distributed in a rectangular workspace."""
    pos_endpoints = uniform_endpoints(key, batch_size, N_DIM, workspace)  # (start, end)
    vel_endpoints = jnp.zeros_like(pos_endpoints)
    init_states, target_states = tuple(zip(pos_endpoints, vel_endpoints))  # ((pos, vel), (pos, vel))
    
    return init_states, target_states



# %% [markdown]
# Generate an example batch:

# %%
seed = 0
batch_size = 500
n_steps = 100
epoch_len_ranges = ((5, 15),   # start
                    (10, 15),  # stim
                    (10, 30))  # hold

key = jrandom.PRNGKey(seed)
init_states, target_states = get_batch(batch_size=batch_size)

# %% [markdown]
# For a single trial, no batching:

# %%
# epoch_idxs = [0, 3, 4, None]
# idxs = jnp.r_[*[(i, slice(idx0, idx1)) 
#                 for i, (idx0, idx1) in enumerate(zip(epoch_idxs[:-1], epoch_idxs[1:]))]]
# jnp.ones((5,4), dtype=bool).at[idxs].set(False)

# %%
init_state = tree_get_idx(init_states, 0)
target_state = tree_get_idx(target_states, 0)

def get_sequences(key, n_steps, epoch_len_ranges, init, target):
    """Convert static task inputs to sequences, and make hold signal.
    
    TODO: 
    - this could be part of a `Task` subclass
    """
    stim_epochs = jnp.array((1,))
    hold_epochs = jnp.array((0, 1, 2))
    
    epoch_lengths = gen_epoch_lengths(key, epoch_len_ranges)
    epoch_idxs = jnp.pad(jnp.cumsum(epoch_lengths), (1, 0), constant_values=(0, -1))
    epoch_masks = get_masks(n_steps, epoch_idxs)
    
    move_epoch_mask = jnp.logical_not(jnp.prod(epoch_masks, axis=0))[None, :]
    
    stim_seqs = get_masked_seqs(target, epoch_masks[stim_epochs])
    target_seqs = jax.tree_map(
        lambda x, y: x + y, 
        get_masked_seqs(target, move_epoch_mask),
        get_masked_seqs(init, epoch_masks[hold_epochs]),
    )
    stim_on_seq = get_scalar_epoch_seq(epoch_idxs, n_steps, 1., stim_epochs)
    hold_seq = get_scalar_epoch_seq(epoch_idxs, n_steps, 1., hold_epochs)
    
    task_input = stim_seqs + (hold_seq, stim_on_seq)
    target = target_seqs
    
    return task_input, target, epoch_idxs

#get_sequences = jax.jit(get_sequences, static_argnums=(1,2,))

task_input, target, epoch_idxs = get_sequences(key, n_steps, epoch_len_ranges, init_state, target_state)
target

# %% [markdown]
# Try batching:

# %%
key = jrandom.PRNGKey(131254)
keys = jrandom.split(key, batch_size)

jax.vmap(get_sequences, in_axes=(0, None, None, 0, 0))(
    keys, n_steps, epoch_len_ranges, init_states, target_states
)


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
    
    def __call__(self, input, state):
        # TODO: optional multiple timesteps per call
        system_state, solver_state = state 
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        system_state, _, _, solver_state, _ = self.solver.step(
            self.term, 0, self.dt, system_state, input, solver_state, made_jump=False
        )
        state = system_state, solver_state
        return state
    
    def init(self, system_state, input=None, key=None):
        args = inputs_empty = jnp.zeros((self.system.control_size,))
        return (
            system_state,
            self.solver.init(self.term, 0, self.dt, system_state, args),
        )


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
    
    def __call__(self, input, state, args, key):
        mechanics_state, _, hidden = state
        feedback_state = args
        
        key1, key2 = jrandom.split(key)
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((input, feedback_state), hidden, key1)
        
        mechanics_state = self.mechanics(control, mechanics_state)
        
        return mechanics_state, control, hidden
    
    def init(self, system_state):
        return (
            self.mechanics.init(system_state),
            jnp.zeros((self.net.out_size,)),
            self.net.init(),
        )
    
    
class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        input, states, key = x 
        
        key1, key2 = jrandom.split(key)
        
        # #! this ultimately shouldn't be here, but costs less memory than a `SimpleFeedback`-based storage hack:
        feedback = tree_get_idx(states[0][0], i - self.step.delay)
        args = feedback 
        
        # this seems to work, but I'm worried it will break on non-array leaves later
        state = tree_get_idx(states, i)     
        input_i = tree_get_idx(input, i)  
        state = self.step(input_i, state, args, key1)
        states = tree_set_idx(states, state, i + 1)
        
        return input, states, key2
    
    def __call__(self, input, system_state, key):
        
        # #! vestigial; part of the feedback_state hack: stand in target_state for feedback_state
        args = jax.tree_map(jnp.zeros_like, system_state)
        
        key1, key2, key3 = jrandom.split(key, 3)
        
        state = self.step.init(system_state)
        states = self.init(input, state, args, key2)
        
        if DEBUG: #! jax.debug doesn't work inside of lax loops  
            for i in range(self.n_steps):
                states, args = self._body_func(i, (states, args))
                
            return states, args    
        
        _, states, _ = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (input, states, key3),
        )
        
        return states
    
    def init(self, input, state, args, key):
        # 1. generate empty trajectories of states 
        input = tree_get_idx(input, 0)  # sequence of inputs
        outputs = eqx.filter_eval_shape(self.step, input, state, args, key)
        # `eqx.is_array_like` is False for jax.ShapeDtypeSty
        scalars, array_structs = eqx.partition(outputs, eqx.is_array_like)
        asarrays = eqx.combine(jax.tree_map(jnp.asarray, scalars), array_structs)
        states = jax.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            asarrays,
        )
        # 2. initialize the first state
        states = tree_set_idx(states, state, 0)
        return states


# %%
def get_model(
    dt=0.1, 
    mass=1., 
    n_hidden=50, 
    tau_leak=5, 
    n_steps=100, 
    key=None, 
    feedback_delay=0
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jrandom.PRNGKey(0)  
        
    keyc, keyn = jrandom.split(key)
    
    system = point_mass(mass=mass, n_dim=N_DIM)
    mechanics = Mechanics(system, dt)
    n_input = 1 + 1 + system.state_size * 2  # hold, target-on signals; feedback & target states
    #cell = RNNCell(n_input, n_hidden, dt=dt, tau=tau_leak, key=keyc)
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=keyc)
    net = RNN(cell, system.control_size, key=keyn)
    body = SimpleFeedback(net, mechanics, delay=feedback_delay)

    return Recursion(body, n_steps)


# %%
def loss_fn(
    diff_model, 
    static_model, 
    init_state, 
    target_state,
    task_input, 
    key,
    term_weights=dict(
        fixation=1.,
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
    batched_model = jax.vmap(model, in_axes=(0, 0, None))  # don't batch random key

    state = batched_model(task_input, init_state, key)
    (system_state, _), controls, activity = state
    state = system_state

    # add fixation signal to the discount vector to make sure fixation is penalized
    #discount = discount + jnp.squeeze(task_input[2])

    # sum over xyz, apply temporal discount, sum over time
    position_error = jnp.sum((state[0] - target_state[0]) ** 2, axis=-1)
    
    #fixation_control_loss = jnp.sum(jnp.squeeze(task_input[2]) * jnp.sum(controls ** 2, axis=-1), axis=-1)
    
    loss_terms = dict(
        #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
        fixation=jnp.sum(position_error * jnp.squeeze(task_input[2]), axis=-1),
        position=jnp.sum(position_error * discount, axis=-1),  # apply temporal discount, sum over time
        final_velocity=jnp.sum((state[1][:, -1] - target_state[1][:, -1]) ** 2, axis=-1).squeeze(),  # over xyz
        control=jnp.sum(controls ** 2, axis=(-1, -2)),  # over control variables and time
        hidden=jnp.sum(activity ** 2, axis=(-1, -2)),  # over network units and time
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
def get_sequences(key, n_steps, epoch_len_ranges, init, target):
    """Convert static task inputs to sequences, and make hold signal.
    
    TODO: 
    - this could be part of a `Task` subclass
    """
    stim_epochs = jnp.array((1,))
    hold_epochs = jnp.array((0, 1, 2)) 
    
    epoch_lengths = gen_epoch_lengths(key, epoch_len_ranges)
    epoch_idxs = jnp.pad(jnp.cumsum(epoch_lengths), (1, 0), constant_values=(0, -1))

    epoch_masks = get_masks(n_steps, epoch_idxs)
    
    move_epoch_mask = jnp.logical_not(jnp.prod(epoch_masks, axis=0))[None, :]
    
    stim_seqs = get_masked_seqs(target, epoch_masks[stim_epochs])
    target_seqs = jax.tree_map(
        lambda x, y: x + y, 
        get_masked_seqs(target, move_epoch_mask),
        get_masked_seqs(init, epoch_masks[hold_epochs]),
    )
    stim_on_seq = get_scalar_epoch_seq(epoch_idxs, n_steps, 1., stim_epochs)
    hold_seq = get_scalar_epoch_seq(epoch_idxs, n_steps, 1., hold_epochs)
    
    # TODO: catch trials: modify a proportion of hold_seq and target_seqs
    
    task_input = stim_seqs + (hold_seq, stim_on_seq)
    target = target_seqs
    
    return task_input, target, epoch_idxs


# %%
def train(
    mass=1.0,
    n_steps=100,
    dt=0.1,
    hidden_size=20,
    tau_leak=10,
    feedback_delay_steps=0,
    workspace = jnp.array([[-1., 1.], 
                           [-1., 1.]]),
    batch_size=100,
    n_batches=50,
    task_epoch_len_ranges = ((5, 15),   # start
                             (10, 15),  # stim
                             (5, 10)),  # hold
    train_epochs=1,
    learning_rate=3e-4,
    term_weights=dict(
        fixation=0.1,
        position=1., 
        final_velocity=1., 
        control=1e-5, 
        hidden=1e-5
    ),
    seed=5566,
    log_step=50,
):
    key = jrandom.PRNGKey(seed)

    def get_batch(batch_size, key):
        """Segment endpoints uniformly distributed in a rectangular workspace."""
        pos_endpoints = uniform_endpoints(key, batch_size, N_DIM, workspace)  # (start, end)
        vel_endpoints = jnp.zeros_like(pos_endpoints)
        init_states, target_states = tuple(zip(pos_endpoints, vel_endpoints))  # ((pos, vel), (pos, vel))
        keys = jrandom.split(key, batch_size)
        task_inputs, target_states, _ = jax.vmap(get_sequences, in_axes=(0, None, None, 0, 0))(
            keys, n_steps, task_epoch_len_ranges, init_states, target_states
        )
        return init_states, target_states, task_inputs
    
    model = get_model(dt, mass, hidden_size, tau_leak, n_steps, key=key, 
                      feedback_delay=feedback_delay_steps)
    
    # only train the RNN layer (input weights & hidden weights and biases)
    filter_spec = filter_spec_leaves(
        model, 
        lambda tree: (tree.step.net.cell.weight_hh, 
                      tree.step.net.cell.weight_ih, 
                      tree.step.net.cell.bias),
    )
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    position_error_discount = jnp.linspace(1./n_steps, 1., n_steps) ** 6
    
    @eqx.filter_jit
    def train_step(model, init_state, target_state, task_input, opt_state, key):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state, task_input, key, 
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

    for _ in range(train_epochs):
        for batch in tqdm(range(n_batches)):
            key, key_train = jrandom.split(key)
            init_state, target_state, task_input = get_batch(batch_size, key)
            
            loss, loss_terms, model, opt_state = train_step(
                model, init_state, target_state, task_input, opt_state, key_train
            )
            
            # #! slow
            losses = losses.at[batch].set(loss)
            losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
            
            if batch % log_step == 0:
                tqdm.write(f"step: {batch}, loss: {loss:.4f}", file=sys.stderr)
    
    return model, losses, losses_terms


# %%
n_steps = 100
task_epoch_len_ranges = ((5, 15),   # start
                         (10, 20),  # stim
                         (10, 25))  # hold

# %%
model, losses, losses_terms = train(
    batch_size=500, 
    dt=0.1, 
    tau_leak=5,
    feedback_delay_steps=5,
    n_batches=10_000, 
    n_steps=n_steps,
    task_epoch_len_ranges=task_epoch_len_ranges, 
    hidden_size=50, 
    seed=5566,
    learning_rate=0.01,
    log_step=250,
    term_weights=dict(
        fixation=1.,
        position=1., 
        final_velocity=1., 
        control=1e-4, 
        hidden=1e-5,
    ),
)

trained = True 

plot_loglog_losses(losses, losses_terms);


# %%
def save_model(
        model, 
        hyperparams: Optional[dict] = None, 
        fig_funcs: Optional[dict]=None
):
    
    timestr = datetime.today().strftime("%Y%m%d-%H%M%S") 
    name = f"model_{timestr}_{NB_PREFIX}"
    eqx.tree_serialise_leaves(model_dir / f'{name}.eqx', model)
    if hyperparams is not None:
        with open(model_dir / f'{name}.json', 'w') as f:
            hyperparams_str = json.dumps(hyperparams, indent=4)
            f.write(hyperparams_str)
    if fig_funcs is not None:
        for label, fig_func in fig_funcs.items():
            fig = fig_func()
            fig.savefig(model_dir / f'{name}_{label}.png')
            plt.close(fig)

save_model(model)

# %%
model_ = get_model() 
model__ = eqx.tree_deserialise_leaves(
    model_dir / 'model_20231018-150134_12_GO.eqx', 
    model_
)

# %%
model.step.delay

# %%
jax.tree_map(
    lambda x, y: None,
    model, 
    model__,
)

# %% [markdown]
# Optional: load pretrained model (if restarting notebook)

# %%
model_path = "../models/model_20231018-131339_12_GO.eqx"

if not trained: 
    model = get_model()
    model = eqx.tree_deserialise_leaves(model_path, model)

# %% [markdown]
# Evaluate on a centre-out task

# %%
n_directions = 8
reach_length = 1.

key = jrandom.PRNGKey(5566)
pos_endpoints = centreout_endpoints(jnp.array([0., 0.]), n_directions, 0, reach_length)
vel_endpoints = jnp.zeros_like(pos_endpoints)   
init_states, target_states = tuple(zip(pos_endpoints, vel_endpoints))

keys = jrandom.split(key, n_directions)
task_inputs, target_states, epoch_idxs = jax.vmap(get_sequences, in_axes=(0, None, None, 0, 0))(
    keys, n_steps, task_epoch_len_ranges, init_states, target_states
)
states = jax.vmap(model, in_axes=(0, 0, None))(
    task_inputs, init_states, key
)
(system_states, _), controls, activities = states
states = system_states

# %% [markdown]
# Plot speeds along with a line indicating the first availability of target information.

# %%
plot_task_and_speed_profiles(
    velocity=states[1], 
    task_variables={
        'target X': target_states[0][..., 0],
        'target Y': target_states[0][..., 1],
        'fixation signal': task_inputs[2],
        'target ON signal': task_inputs[3],
    }, 
    epoch_idxs=epoch_idxs
)

# %%
plot_states_forces_2d(states[0], states[1], controls, endpoints=pos_endpoints)
plt.show()

# %% [markdown]
# Plot network activity. Heatmap of all units, and a sample of six units.

# %%
plot_activity_heatmap(activities[0])
plt.show()

# %%
seed = 5566
n_samples = 6
key = jrandom.PRNGKey(seed)

plot_activity_sample_units(activities, n_samples, key=key)


# %% [markdown]
# ## PCA

# %%
def pca(x):
    X = x.reshape(-1, x.shape[-1])
    X -= X.mean(axis=0)
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    L = S * 2 / (X.shape[0] - 1)
    PCs = (U @ jnp.diag(S)).reshape(*x.shape)
    return L, Vt, PCs 


# %%
L, Vt, PCs = pca(activities)

# %%
ts = slice(5, 17)

cmap = 'tab10'
cmap = plt.get_cmap(cmap)
colors = [cmap(i) for i in np.linspace(0, 1, PCs.shape[0])]

for i in range(PCs.shape[0]):
    # TODO: time ranges based on epoch indices
    plt.plot(PCs[i, ts, 0], PCs[i, ts, 1], color=colors[i])


# %%
epoch_idxs

# %%
from itertools import zip_longest
from typing import Tuple
from jaxtyping import Array, Float, Int

pc_idxs = (0, 3, 4)

def plot_3D_paths(
    paths: Float[Array, "batch steps 3"], 
    epoch_start_idxs: Int[Array, "batch epochs"],  
    epoch_linestyles: Tuple[str, ...]  # epochs
):

    if not np.max(epoch_idxs) < paths.shape[1]:
        raise IndexError("epoch indices out of bounds")
    if not epoch_start_idxs.shape[1] == len(epoch_linestyles):
        raise ValueError("TODO")

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    for i in range(paths.shape[0]):
        for idxs, ls in zip(
            zip_longest(
                epoch_start_idxs[i], 
                epoch_start_idxs[i, 1:] + 1, 
                fillvalue=None
            ), 
            epoch_linestyles
        ):
            ts = slice(*idxs)
            ax.plot(*paths[i, ts, :].T, color=colors[i], lw=2, linestyle=ls)

    return fig, ax 
    
fig, ax = plot_3D_paths(
    PCs[:, :, pc_idxs], 
    epoch_idxs, 
    epoch_linestyles=('-', ':', '--', '-')
)

ax.set_xlabel(f'PC {pc_idxs[0]}')
ax.set_ylabel(f'PC {pc_idxs[1]}')
ax.set_zlabel(f'PC {pc_idxs[2]}')

# %%
PCs[:, :, pc_idxs].T.shape


# %%
def tt():
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    for i in range(PCs.shape[0]):
        for idxs, ls in zip(
            zip_longest(epoch_idxs[i], epoch_idxs[i, 1:] + 1, fillvalue=None), 
            epoch_linestyles
        ):
            ts = slice(*idxs)
            ax.plot(PCs[i, ts, pc_idxs[0]], PCs[i, ts, pc_idxs[1]], PCs[i, ts, pc_idxs[2]], 
                    color=colors[i], lw=2, linestyle=ls)
    return fig, ax 

fig, ax = tt()
anim = animate_3D_rotate(fig, ax, azim_range=(0, 36))
#anim.to_html5_video()
anim.save('test2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


# %%
ax.azim
