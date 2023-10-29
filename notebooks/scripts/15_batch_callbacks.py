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
NB_PREFIX = "nb8"
N_DIM = 2  # TODO: not here

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# %%
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.loss import simple_reach_loss
from feedbax.plot import plot_loss, plot_pos_vel_force_2D
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer
from feedbax.xabdeef import point_mass_RNN_loop
from feedbax.utils import Timer

# %%
plt.style.use('dark_background')

# %%
seed = 5566

mass = 1.0
n_steps = 100
dt = 0.1
feedback_delay_steps = 5
workspace = ((-1., 1.),
             (-1., 1.))
n_hidden  = 50
learning_rate = 0.01

# %%
key = jr.PRNGKey(seed)

task = RandomReaches(
    loss_func=simple_reach_loss(n_steps),
    workspace=workspace, 
    n_steps=n_steps,
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.5,
)

model = point_mass_RNN_loop(
    task,
    dt=dt,
    mass=mass,
    n_hidden=n_hidden,
    n_steps=n_steps,
    feedback_delay=feedback_delay_steps,
    key=key, 
)
    
trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    checkpointing=True,
)

# %% [markdown]
# Define our training hyperparameters like usual

# %%
n_batches = 200
batch_size = 500
key_train = jr.PRNGKey(seed + 1)

trainable_leaves_func = lambda model: (
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias
)

# %% [markdown]
# However, this time we'll also define some functions for `TaskTrainer` to call on certain iterations of the training loop. 
#
# In particular, we'll ask it to profile iterations 100-103, and to start a timer on iteration 100 and end it on 200. 
#
# To do this, we pass a dictionary where a key gives the index of the iteration, and the respective value is a sequence (e.g. tuple) of functions to be called.
#
# TODO: don't let the profiling and the timer overlap, otherwise the time is longer than it would be due to the profiling overhead

# %%
timer = Timer()

batch_callbacks = {
    50: (lambda: timer.start(), 
          lambda: jax.profiler.start_trace("/tmp/tensorboard")),
    52: (lambda: jax.profiler.stop_trace(),),
    100: (lambda: timer.stop(),),
}

model, losses, losses_terms, learning_rates = trainer(
    task=task, 
    model=model,
    n_batches=n_batches, 
    batch_size=batch_size, 
    log_step=200,
    trainable_leaves_func=trainable_leaves_func,
    batch_callbacks=batch_callbacks,
    key=key_train,
)

avg_rate = (100 - 50) / timer.time
print(f"\n The timed iterations took {timer.time:.2f} s, "
      + f"at an average rate of {avg_rate:.2f} it/s.")


plot_loss(losses, losses_terms)

# %% [markdown]
# Evaluate on a centre-out task

# %%
loss, loss_terms, states = task.eval(model, key=jr.PRNGKey(0))

# %%
init_states, target_states, _ = task.trials_validation
goal_states = jax.tree_map(lambda x: x[:, -1], target_states)
plot_pos_vel_force_2D(
    states,
    endpoints=(init_states.pos, goal_states.pos),
)
plt.show()
