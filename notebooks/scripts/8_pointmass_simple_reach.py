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
LOG_LEVEL = "INFO"
NB_PREFIX = "nb8"
DEBUG = False
ENABLE_X64 = False
N_DIM = 2  # TODO: not here

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
stderr_log = sys.stderr = open(f'log/stderr_{NB_PREFIX}.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(LOG_LEVEL)

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
from feedbax.mechanics.linear import point_mass
from feedbax.networks import RNN
from feedbax.plot import plot_loglog_losses, plot_states_forces_2d
from feedbax.recursion import Recursion
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer

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
from feedbax.utils import tree_sum_n_features


def get_model(
    task,
    key=None,
    dt=0.05, 
    mass=1., 
    n_hidden=50, 
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
     
    # TODO: automatically calculate `n_input` from task and mechanics
    # unfortunately the following doesn't work because we don't have access 
    # to a `mechanics_state` instance here; but `Mechanics` could provide this info
    # based on `feedback_leaves_func`
    # n_feedback = tree_sum_n_features(feedback_leaves_func(mechanics_state))
    # n_task_inputs = tree_sum_n_features(task.get_trial(key)[2])
    # n_input = n_task_inputs + n_feedback
    
    n_input = system.state_size * 2 # feedback & target states
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(cell, system.control_size, out_nonlinearity=out_nonlinearity, key=key2)
    body = SimpleFeedback(net, mechanics, delay=feedback_delay)

    model = Recursion(
        body, 
        n_steps,
        feedback_leaves_func,
    )
    
    return model


# %%
seed = 5566
key = jrandom.PRNGKey(seed)

n_steps = 100
dt = 0.1
feedback_delay = 0
workspace = jnp.array([[-1., 1.], 
                       [-1., 1.]])
n_hidden  = 50
learning_rate = 0.01

# these assume a particular PyTree structure to the states returned by the model
# which is why we simply instantiate them 
discount = jnp.linspace(1. / n_steps, 1., n_steps) ** 6
loss_func = fbl.CompositeLoss(
    (
        fbl.EffectorPositionLoss(discount=discount),
        fbl.EffectorFinalVelocityLoss(),
        fbl.ControlLoss(),
        fbl.NetworkActivityLoss(),
    ),
    weights=(1., 1., 1e-5, 1e-5)
)

task = RandomReaches(
    loss_func=loss_func,
    workspace=workspace, 
    n_steps=n_steps,
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.5,
)

model = get_model(
    task,
    key, 
    dt=dt,
    n_hidden=n_hidden,
    n_steps=n_steps,
    feedback_delay=feedback_delay,
)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    chkpt_dir=chkpt_dir,
    checkpointing=True,
)

# %%
trainable_leaves_func = lambda model: (
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias
)

model, losses, losses_terms = trainer(
    task=task, 
    model=model,
    n_batches=2000, 
    batch_size=500, 
    log_step=100,
    trainable_leaves_func=trainable_leaves_func,
    key=key,
)

# %%
plot_loglog_losses(losses, losses_terms)

# %% [markdown]
# Evaluate on a centre-out task

# %%
loss, loss_terms, states = task.eval(model, key=jrandom.PRNGKey(0))

# %%
init_states, target_states, _ = task.trials_eval
goal_states = jax.tree_map(lambda x: x[:, -1], target_states)
pos_endpoints = tuple(zip(init_states, goal_states))[0]
plot_states_forces_2d(
    states.mechanics.system[0], 
    states.mechanics.system[1], 
    states.control, 
    endpoints=pos_endpoints
)

# %%
plt.plot(jnp.sum(states.mechanics.system[1] ** 2, -1).T, '-')
plt.show()

# %%
