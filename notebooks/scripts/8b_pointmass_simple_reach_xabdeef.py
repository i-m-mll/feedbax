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

# %%
LOG_LEVEL = "INFO"
NB_PREFIX = "nb8"
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
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import optax 

from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_RNN
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer

from feedbax.plot import plot_loss, plot_pos_vel_force_2D

# %%
logging.getLogger("jax").setLevel(logging.INFO)

plt.style.use('dark_background')

# %%
seed = 5566
key_model = jr.PRNGKey(seed)

mass = 1.0
n_steps = 100
dt = 0.1
feedback_delay_steps = 5
workspace = ((-1., 1.),
             (-1., 1.))
hidden_size  = 50
learning_rate = 0.01

n_batches = 10_000
batch_size = 500

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    checkpointing=True,
)

task = RandomReaches(
    loss_func=simple_reach_loss(n_steps),
    workspace=workspace, 
    n_steps=n_steps,
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.5,    
)

model = point_mass_RNN(
    task,
    key=key_model,
    dt=dt,
    mass=mass,
    hidden_size=hidden_size, 
    n_steps=n_steps,
    feedback_delay_steps=feedback_delay_steps,
)

trainable_leaves_func = lambda model: (
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias
)

# %%
key_train = jr.PRNGKey(seed + 1)

model, losses, losses_terms, learning_rates = trainer(
    task=task, 
    model=model,
    n_batches=n_batches, 
    batch_size=batch_size, 
    log_step=200,
    trainable_leaves_func=trainable_leaves_func,
    key=key_train,
)

plot_loss(losses, losses_terms)

# %% [markdown]
# Evaluate on a centre-out task

# %%
loss, loss_terms, states = task.eval(model, key=jr.PRNGKey(0))

# %%
trial_specs, _ = task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)
plot_pos_vel_force_2D(
    states,
    endpoints=(trial_specs.init.pos, goal_states.pos),
)
plt.show()

# %%
(loss, loss_terms, states), trials, aux = task.eval_train_batch(
    model, 
    batch_size=10,
    key=jr.PRNGKey(0), 
)

# %%
goal_states = jax.tree_map(lambda x: x[:, -1], trials.target)
plot_pos_vel_force_2D(
    states,
    endpoints=(trials.init.pos, goal_states.pos),
)
plt.show()

# %%
plt.plot(jnp.sum(states.mechanics.system.vel ** 2, -1).T, '-')
plt.show()

# %%
