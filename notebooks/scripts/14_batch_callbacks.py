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
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 

from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_RNN

from feedbax.plot import plot_loss, plot_pos_vel_force_2D
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer
from feedbax.utils import Timer

# %%
plt.style.use('dark_background')

# %%
seed = 5566

mass = 1.0
n_steps = 1000
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

model = point_mass_RNN(
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
n_batches = 410
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
# In particular, we'll ask it to profile iterations 50-52, and to start a timer on iteration 300 and end it on 400. Note that the functions will be run immediately after the call to `train_step` on the respective iteration.
#
# To do this, we pass a dictionary where a key gives the index of the iteration, and the respective value is a sequence (e.g. tuple) of functions to be called.

# %%
timer = Timer()

batch_timer_start = 300
batch_timer_stop = 400

batch_callbacks = {
    50: (lambda: jax.profiler.start_trace("/tmp/tensorboard"),),
    52: (lambda: jax.profiler.stop_trace(),),
    batch_timer_start: (lambda: timer.start(),),
    batch_timer_stop: (lambda: timer.stop(),),
}

model, losses, losses_terms, learning_rates = trainer(
    task=task, 
    model=model,
    n_batches=n_batches, 
    batch_size=batch_size, 
    key=key_train,
    log_step=200,
    trainable_leaves_func=trainable_leaves_func,
    batch_callbacks=batch_callbacks,
)

avg_rate = (batch_timer_stop - batch_timer_start) / timer.time
print(f"\n The timed iterations took {timer.time:.2f} s, "
      + f"at an average rate of {avg_rate:.2f} it/s.")


plot_loss(losses, losses_terms)

# %% [markdown]
# Time multiple runs to get some statistics

# %%
n_runs = 3
n_batches = 510

batch_callbacks = lambda timer: {
    200: (lambda: timer.start(),),
    300: (lambda: timer.stop(), lambda: timer.start()),
    400: (lambda: timer.stop(), lambda: timer.start()),
    500: (lambda: timer.stop(),),
}

timers = [Timer() for _ in range(n_runs)]

for i in range(n_runs):
    model, losses, losses_terms, learning_rates = trainer(
        task=task, 
        model=model,
        n_batches=n_batches, 
        batch_size=batch_size, 
        key=key_train,
        log_step=200,
        trainable_leaves_func=trainable_leaves_func,
        batch_callbacks=batch_callbacks(timers[i]),
    )

# %% [markdown]
# Calculate two timing statistics:
#
# 1) Within-run mean and std
# 2) Between-run mean and std
#
# TODO: Not sure if there is another way to do the between-run mean and std. Here I do a mean-of-means.

# %%
within = [dict(mean=np.mean(timer.times),
               std=np.std(timer.times)) 
          for timer in timers]
within_means = np.array([d["mean"] for d in within])
between = dict(mean=np.mean(within_means),
               std=np.std(within_means))
    
eqx.tree_pprint(within)
print(between)

# %%
# average rate, given we're timing intervals of 100 iterations
100 / between['mean']
