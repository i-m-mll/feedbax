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

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a point mass to reach from a starting position to a target position. The network should hold at the start position until a hold signal is switched off.

# %%
LOG_LEVEL = "INFO"
NB_PREFIX = "nb12"
DEBUG = False
ENABLE_X64 = False
N_DIM = 2  # TODO: not here

# %%
# %load_ext autoreload
# %autoreload 2

# #%matplotlib widget

# %%
import os
import logging
import sys

from IPython import get_ipython

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# redirect stderr (e.g. warnings) to file
stderr_log = sys.stderr = open(f'log/stderr_{NB_PREFIX}.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(LOG_LEVEL)

# %%
from pathlib import Path 

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.channel import ChannelState
from feedbax.context import SimpleFeedback, SimpleFeedbackState
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.linear import point_mass
from feedbax.networks import RNN
from feedbax.recursion import Recursion
from feedbax.task import RandomReachesDelayed
from feedbax.trainer import TaskTrainer

from feedbax.plot import (
    animate_3D_rotate,
    plot_planes,
    plot_3D_paths,
    plot_activity_heatmap,
    plot_activity_sample_units,
    plot_loglog_losses, 
    plot_states_forces_2d,
    plot_task_and_speed_profiles,
)

# %%
os.environ["FEEDBAX_DEBUG"] = str(DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update("jax_enable_x64", ENABLE_X64)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

# %%
# paths
# training checkpoints
chkpt_dir = Path("/tmp/feedbax-checkpoints")
chkpt_dir.mkdir(exist_ok=True)

# tensorboard
tb_logdir = Path("runs")

model_dir = Path("../models/")


# %%
def get_model(
    key=None, 
    dt=0.1, 
    mass=1., 
    n_hidden=50, 
    # tau_leak=5, 
    n_steps=100, 
    feedback_delay=0,
    out_nonlinearity=lambda x: x,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jrandom.PRNGKey(0)  
        
    key1, key2 = jrandom.split(key)
    
    system = point_mass(mass=mass, n_dim=N_DIM)
    mechanics = Mechanics(system, dt)
    
    feedback_leaves_func = lambda mechanics_state: mechanics_state.system
    
    # hold, target-on signals; feedback & target states
    n_input = 1 + 1 + system.state_size * 2 
    #cell = RNNCell(n_input, n_hidden, dt=dt, tau=tau_leak, key=keyc)t
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(
        cell, 
        system.control_size, 
        out_nonlinearity=out_nonlinearity,
        persistence=True, 
        key=key2
    )
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,
        feedback_leaves_func=feedback_leaves_func,
    )
    
    states_includes = SimpleFeedbackState(
        mechanics=True, 
        control=True, 
        hidden=True, 
        feedback=ChannelState(output=True, queue=False)
    )

    return Recursion(body, n_steps, states_includes=states_includes)


# %%
seed = 5566
key = jrandom.PRNGKey(seed)

n_steps = 100
dt = 0.1
feedback_delay_steps = 5
workspace = jnp.array([[-1., 1.], 
                       [-1., 1.]])
# epoch lengths (in steps) will be sampled uniformly from these ranges
# the movement epoch will last the remainder until `n_steps`
task_epoch_len_ranges = ((5, 15),   # start
                         (10, 20),  # stim
                         (10, 25))  # hold
n_hidden = 50

n_batches = 10_000
batch_size = 500
learning_rate = 0.01
log_step = 100

model = get_model(
    key, 
    dt=dt,
    n_hidden=n_hidden,
    n_steps=n_steps,
    feedback_delay=feedback_delay_steps,
)

discount = jnp.linspace(1. / n_steps, 1., n_steps) ** 6
loss_func = fbl.CompositeLoss(
    (
        # these assume a particular PyTree structure to the states returned by the model
        # which is why we simply instantiate them 
        fbl.EffectorFixationLoss(),
        fbl.EffectorPositionLoss(discount=discount),
        fbl.EffectorFinalVelocityLoss(),
        fbl.ControlLoss(),
        fbl.NetworkActivityLoss(),
    ),
    weights=(1., 1., 1., 1e-4, 1e-5)
)

task = RandomReachesDelayed(
    loss_func=loss_func,
    workspace=workspace, 
    n_steps=n_steps,
    epoch_len_ranges=task_epoch_len_ranges,
    eval_grid_n=1,
    eval_n_directions=8,
    eval_reach_length=0.5,
)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=optax.linear_schedule(
            init_value=0.025, 
            end_value=1e-4,
            transition_steps=n_batches - 200,
            transition_begin=200,
        )
    ),
    chkpt_dir=chkpt_dir,
    checkpointing=True,
)

# %%
model, losses, losses_terms = trainer(
    task=task,
    model=model,
    batch_size=batch_size, 
    n_batches=n_batches, 
    log_step=log_step,
    key=key,
)

trained = True 

plot_loglog_losses(losses, losses_terms);

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
task = RandomReachesDelayed(
    loss_func=loss_func,
    workspace=workspace, 
    n_steps=n_steps,
    epoch_len_ranges=task_epoch_len_ranges,
    eval_grid_n=1,
    eval_n_directions=8,
    eval_reach_length=0.5,
)

# %%
loss, loss_terms, states = task.eval(model, key=jrandom.PRNGKey(0))

# %% [markdown]
# Plot speeds along with a line indicating the first availability of target information.

# %%
init_states, target_states, task_inputs = task.trials_eval
# assume the goal is the target state at the last time step
goal_states = jax.tree_map(lambda x: x[:, -1], target_states)
plot_task_and_speed_profiles(
    velocity=states.mechanics.effector.vel, 
    task_variables={
        'target X': task_inputs.stim.pos[..., 0],
        'target Y': task_inputs.stim.pos[..., 1],
        'fixation signal': task_inputs.hold,
        'target ON signal': task_inputs.stim_on,
    }, 
    #epoch_idxs=epoch_idxs
)

# %%
plot_states_forces_2d(
    states.mechanics.system.pos, 
    states.mechanics.system.vel, 
    states.control, 
    endpoints=(init_states.pos, goal_states.pos),
)
plt.show()

# %% [markdown]
# Plot network activity. Heatmap of all units, and a sample of six units.

# %%
plot_activity_heatmap(states.hidden[0])
plt.show()

# %%
seed = 5566
n_samples = 6
key = jrandom.PRNGKey(seed)

plot_activity_sample_units(states.hidden, n_samples, key=key)


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
L, Vt, PCs = pca(states.hidden)

# %% [markdown]
# Plot three of the PCs in 3D.

# %%
pc_idxs = (0, 3, 4)

fig, ax = plot_3D_paths(
    PCs[:, :, pc_idxs], 
    epoch_idxs, 
    epoch_linestyles=('-', ':', '--', '-')
)

ax.update(dict(zip(
    ('xlabel', 'ylabel', 'zlabel'), 
    [f'PC {i}' for i in pc_idxs]
)))

# %% [markdown]
# Save an animation of the plot, rotating.

# %%
anim = animate_3D_rotate(fig, ax, azim_range=(0, 360))
#anim.to_html5_video()
anim.save('test.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


# %% [markdown]
# Plot multiple PC planes in 2D

# %%
plot_planes(
    PCs[..., :6],
    epoch_start_idxs=epoch_idxs,
    epoch_linestyles=('-', ':', '--', '-'),    
    lw=2,
)

# %%
