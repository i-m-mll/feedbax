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
# In this notebook, we use the highest-level API from `feedbax.xabdeef`, which allows users to easily train common models from default parameters.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# This isn't strictly necessary to train a model,
# but it may be necessary for reproducibility.
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# %%
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt

from feedbax.intervene import EffectorCurlForceField
from feedbax.model import add_intervenors
from feedbax.xabdeef import point_mass_RNN_simple_reaches

from feedbax.plot import plot_losses, plot_pos_vel_force_2D

# %%
# changes matplotlib style, to match dark notebook themes
plt.style.use('dark_background')

# %% [markdown]
# ### Absolutely minimal example, using defaults

# %%
seed = 5566

context = point_mass_RNN_simple_reaches(key=jr.PRNGKey(seed))

model, losses, _ = context.train(
    n_batches=10_000, 
    batch_size=500, 
    key=jr.PRNGKey(seed + 1),
)

plot_losses(losses)

# %% [markdown]
# What do the task and model PyTrees look like?

# %%
context.task

# %%
context.model

# %% [markdown]
# ### Slightly less minimal example

# %%
seed = 5566
key_model = jr.PRNGKey(seed)
key_train = jr.PRNGKey(seed + 1)

context = point_mass_RNN_simple_reaches(
    n_steps=100,
    dt=0.1,
    mass=1.0,
    workspace=((-1., -1.),
               (1., 1.)),
    hidden_size=50,
    feedback_delay_steps=5,
    key=key_model,
)

model, losses, _ = context.train(
    n_batches=10_000, 
    batch_size=500, 
    log_step=200,
    learning_rate=0.01,
    key=key_train,
)

plot_losses(losses)


# %% [markdown]
# Evaluate the model on the task---in this case, center-out reaches:

# %%
key_eval = jr.PRNGKey(seed + 2)
losses, states = context.task.eval(model, key=key_eval)

# %%
trial_specs, _ = context.task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)
plot_pos_vel_force_2D(
    states,
    endpoints=(trial_specs.init.pos, goal_states.pos),
)
plt.show()

# %% [markdown]
# Test the response to perturbation.
#
# Use a single center-out set for easier debugging

# %%
from feedbax.task import RandomReaches


task2 = RandomReaches(
    loss_func = context.task.loss_func,
    workspace = context.task.workspace,
    n_steps = context.task.n_steps,
    eval_n_directions=7,
    eval_reach_length=0.5,
    eval_grid_n=1,
)

# %%
model_ = eqx.tree_at(
    lambda model: model.step,
    model,
    add_intervenors(model.step, [EffectorCurlForceField(1)]),
)

# %%
losses, states = task2.eval(model_, key=key_eval)

# %%
trial_specs, _ = task2.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)
plot_pos_vel_force_2D(
    states,
    endpoints=(trial_specs.init.pos, goal_states.pos),
)
plt.show()
