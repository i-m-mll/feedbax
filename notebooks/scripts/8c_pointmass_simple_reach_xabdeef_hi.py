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
# This isn't strictly necessary to train a model,
# but it may be necessary for reproducibility.
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# %%
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt

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
# What does the PyTree of states look like?

# %%
states

# %% [markdown]
# In this case, `states.mechanics.system` and `states.mechanics.effector` provide the same data, since the "effector" of a point mass is the point mass itself.

# %%
states.mechanics.system == states.mechanics.effector

# %% [markdown]
# A little technical detail: notice that all of the states have a middle dimension of `n_steps` (which is 100, unless you changed it earlier) except for the members of `states.feedback.queue`. 
#
# That's because the history of state variables is tracked by the outermost `Iterator` module of the model. However, feedback needs to be implemented *as part of each time step of the sensorimotor loop* implemented in the `SimpleFeedback` module, which doesn't have access to the entire state history. Instead, `SimpleFeedback` locally implements a feedback queue that remembers only as many time steps of feedback information as needed. However, the states stored in the queue are states that are anyway stored by default in `states.mechanics`, by `Iterator`. Thus, to save memory, only the current time step of the feedback queue is remembered by `Iterator`, to be fed back to `SimpleFeedback` on the next iteration, and only the last timestep of the queue is returned with the rest of the states upon evaluation.

# %% [markdown]
# We can easily refer to specific state variables in the PyTree of states that are returned. For example, calculate the speed profile for each trial from the effector velocities:

# %%
plt.plot(jnp.sqrt(jnp.sum(states.mechanics.effector.vel ** 2, -1)).T, '-')
plt.show()

# %% [markdown]
# We can also evaluate the model on an example training batch, to see what the training trials look like. The method `eval_train_batch` also returns information on the trials in the batch, since unlike the validation task, these aren't constant and accessible from `context.task`.

# %%
(losses, states), trials, _ = context.task.eval_train_batch(
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
