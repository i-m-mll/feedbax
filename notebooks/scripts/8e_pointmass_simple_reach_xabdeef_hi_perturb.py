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

from feedbax.plot import (
    plot_losses, 
    plot_pos_vel_force_2D,
    plot_activity_heatmap, 
    plot_activity_sample_units,
)

# %%
# changes matplotlib style, to match dark notebook themes
plt.style.use('dark_background')

# %% [markdown]
# ### Absolutely minimal example, using defaults

# %%
seed = 5566

context = point_mass_RNN_simple_reaches(key=jr.PRNGKey(seed))

model, losses, _ = context.train(
    n_batches=1_000, 
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
# Evaluate the model on the task---in this case, center-out reaches:

# %%
from feedbax.task import RandomReaches


task = RandomReaches(
    loss_func = context.task.loss_func,
    workspace = context.task.workspace,
    n_steps = context.task.n_steps,
    eval_n_directions=7,
    eval_reach_length=0.5,
    eval_grid_n=1,
)

# %%
key_eval = jr.PRNGKey(seed + 2)
losses, states = task.eval(model, key=key_eval)

# %%
trial_specs, _ = task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)
plot_pos_vel_force_2D(
    states,
    endpoints=(trial_specs.init.pos, goal_states.pos),
)
plt.show()

# %%
plot_activity_heatmap(states.network.activity[0])
plt.show()

# %%
seed = 5566
n_samples = 6
key = jr.PRNGKey(seed)

plot_activity_sample_units(states.network.activity, n_samples, key=key)

# %% [markdown]
# Test the response to perturbation.

# %%
unit_spec = jax.tree_map(
    lambda x: jnp.zeros(x.shape[-1]),
    states.network,
)

activity = unit_spec.activity.at[4].set(1)

unit_spec = eqx.tree_at(
    lambda tree: tree.activity,
    unit_spec,
    activity,
)

# %%
from feedbax.intervene import NetworkConstantPerturbation
import jax.random as jr

model_ = eqx.tree_at(
    lambda model: model.step,
    model,
    add_intervenors(
        model.step, 
        {'nn_readout': [NetworkConstantPerturbation(unit_spec)]},
        key=jr.PRNGKey(seed + 3),
    ),
)

# %%
losses, states = task.eval(model_, key=key_eval)

# %%
trial_specs, _ = task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], trial_specs.target)
plot_pos_vel_force_2D(
    states,
    endpoints=(trial_specs.init.pos, goal_states.pos),
)
plt.show()

# %%

# %%
seed = 5566
n_samples = 6
key = jr.PRNGKey(seed)

plot_activity_sample_units(states.network.activity, n_samples, key=key)

# %%
