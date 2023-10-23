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
# %load_ext autoreload
# %autoreload 2

# %%
import os
import logging
from pathlib import Path
import sys

from IPython import get_ipython

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# redirect stderr (e.g. warnings) to file
stderr_log = sys.stderr = open('log/stderr.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(logging.INFO)

# %%
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.context import SimpleFeedback
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.muscle import (
    ActivationFilter,
    TodorovLiVirtualMuscle, 
) 
from feedbax.mechanics.muscled_arm import TwoLinkMuscled 
from feedbax.networks import RNN
from feedbax.recursion import Recursion
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer


from feedbax.plot import (
    plot_loglog_losses, 
    plot_2D_joint_positions,
    plot_states_forces_2d,
    plot_activity_heatmap,
)

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a two-link arm to reach from a starting position to a target position. 

# %%
NB_PREFIX = "nb10"
N_DIM = 2  # everything is 2D
DEBUG = False
ENABLE_X64 = False

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


# %% [markdown]
# Define the model.

# %%
def get_model(
        key=None,
        dt=0.05, 
        n_hidden=50, 
        n_steps=50, 
        feedback_delay=0, 
        tau=0.01, 
        out_nonlinearity=jax.nn.sigmoid,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jrandom.PRNGKey(0)
    key1, key2 = jrandom.split(key)
    
    system = TwoLinkMuscled(
        muscle_model=TodorovLiVirtualMuscle(), 
        activator=ActivationFilter(
            tau_act=tau,  
            tau_deact=tau,
        )
    )
    mechanics = Mechanics(system, dt)
    # joint state feedback + effector state + target state
    n_input = system.twolink.state_size + 2 * N_DIM + 2 * N_DIM
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(cell, system.control_size, out_nonlinearity=out_nonlinearity, key=key2)
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,
    )

    return Recursion(body, n_steps)


# %% [markdown]
# Train the model.

# %%
seed = 5566
key = jrandom.PRNGKey(seed)

n_steps = 50
dt = 0.05 
workspace = jnp.array([[-0.15, 0.15], 
                       [0.20, 0.50]])
n_hidden  = 50
learning_rate = 0.05

model = get_model(
    key, 
    dt=dt,
    n_hidden=n_hidden,
    n_steps=n_steps,
    feedback_delay=0,
    tau=0.01,
    out_nonlinearity=jax.nn.sigmoid,
)

# #! these assume a particular PyTree structure to the states returned by the model
# #! which is why we simply instantiate them 
discount = jnp.linspace(1. / n_steps, 1., n_steps) ** 6
loss_func = fbl.CompositeLoss(
    (
        fbl.EffectorPositionLoss(discount=discount),
        fbl.EffectorFinalVelocityLoss(),
        fbl.ControlLoss(),
        fbl.NetworkActivityLoss(),
    ),
    weights=(1, 0.1, 1e-4, 0.)
)

task = RandomReaches(
    loss_func=loss_func,
    workspace=workspace, 
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.05,
)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    chkpt_dir=chkpt_dir,
    checkpointing=True,
)

# %%
model, losses, loss_terms = trainer(
    task=task, 
    model=model,
    n_batches=10_000, 
    batch_size=500, 
    log_step=250,
    key=key,
)

# %%
losses[-1]

# %%
plt.style.use('dark_background')
plot_loglog_losses(losses, loss_terms)
plt.show()

# %% [markdown]
# Optionally, load an existing model

# %%
model = get_model()
model = eqx.tree_deserialise_leaves("../models/model_20230926-093821_nb10.eqx", model)

# %% [markdown]
# Evaluate on a centre-out task

# %%
task = RandomReaches(
    loss_func=loss_func,
    workspace=workspace, 
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.05,
)   

# %%
loss, loss_terms, states = task.eval(model, key=jrandom.PRNGKey(0))

# %%
# fig = make_eval_plot(states[1], states[2], workspace)
init_states, target_states, _ = task.trials_eval
pos_endpoints = tuple(zip(init_states, target_states))[0]
plot_states_forces_2d(
    states.effector[0], 
    states.effector[1], 
    states.control[:, 2:, -2:], 
    pos_endpoints, 
    force_labels=('Biarticular controls', 'Flexor', 'Extensor'), 
    cmap='plasma', 
    workspace=workspace,
);

# %% [markdown]
# Plot entire arm trajectory for an example direction

# %%
idx = 0

# %%
# convert all joints to Cartesian since I only saved the EE state
xy_pos = eqx.filter_vmap(nlink_angular_to_cartesian)(
    model.step.mechanics.system.twolink, states[0][0].reshape(-1, 2), states[0][1].reshape(-1, 2)
)[0].reshape(states[0][0].shape[0], -1, 2, 2)

# %%
ax = plot_2D_joint_positions(xy_pos[idx], add_root=True)
plt.show()

# %% [markdown]
# Network hidden activities over time for the same example reach direction

# %%
# semilogx is interesting in this case without a GO cue
# ...helps to visualize the initial activity
fig, ax = plt.subplots(1, 1)
ax.semilogx(states.hidden[idx])
ax.set_xlabel('Time step')
ax.set_ylabel('Hidden unit activity')
plt.show()

# %% [markdown]
# Heatmap of network activity over time for an example direction

# %%
plot_activity_heatmap(states.hidden[2], cmap='viridis')

# %%
