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
from functools import partial
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

from feedbax.plot import plot_loss
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer
from feedbax.utils import Timer, get_model_ensemble

# %%
plt.style.use('dark_background')

# %%
seed = 5566

mass = 1.0
n_steps = 10
dt = 0.1
feedback_delay_steps = 5
workspace = ((-1., 1.),
             (-1., 1.))
n_hidden  = 50
learning_rate = 0.01

n_batches = 410
batch_size = 500

task = RandomReaches(
    loss_func=simple_reach_loss(n_steps),
    workspace=workspace, 
    n_steps=n_steps,
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.5,
)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    checkpointing=True,
)

trainable_leaves_func = lambda model: (
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias
)


# %%
def get_models(n_replicates, key):
    return get_model_ensemble(
        partial(
            point_mass_RNN,
            task,
            dt=dt,
            mass=mass,
            n_hidden=n_hidden,
            n_steps=n_steps,
            feedback_delay=feedback_delay_steps,
        ),
        n_replicates,
        key=key, 
    )


# %% [markdown]
# We'll use batch callbacks to time three runs for each of several values of `n_replicates`, to see how training rate depends on it.

# %%
batch_callbacks = lambda timer: {
    n_batches - 110: (lambda: timer.start(),),
    n_batches - 10: (lambda: timer.stop(),),
}

# %%
from typing import List, Tuple

n_runs = 3
n_replicates = [1, 2, 4, 8, 16, 32, 64, 128]
times_mean_std: List[Tuple[float, float]] = []

key = jr.PRNGKey(seed)

for n in n_replicates:
    timer = Timer()
    for i in range(n_runs):
        models = get_models(n, key)
        trainer.train_ensemble(
            task=task, 
            models=models,
            n_replicates=n,
            n_batches=n_batches, 
            batch_size=batch_size, 
            key=jr.PRNGKey(seed + i),
            log_step=200,
            trainable_leaves_func=trainable_leaves_func,
            batch_callbacks=batch_callbacks(timer),
        )
    times_mean_std.append((
        np.mean(timer.times),
        np.std(timer.times)
    ))
    

# avg_rate = (batch_timer_stop - batch_timer_start) / timer.time
# print(f"\n The timed iterations took {timer.time:.2f} s, "
#       + f"at an average rate of {avg_rate:.2f} it/s.")

# %%
means, stds = list(zip(*times_mean_std))

means = np.array(means) * 1000 / 100
stds = np.array(stds) * 1000 / 100

fig, ax = plt.subplots()
ax.errorbar(n_replicates, means, stds, fmt='.', capsize=3)
ax.set_xscale('log')
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Time per iteration (ms)')

# %% [markdown]
# Time multiple runs to get some statistics

# %%
n_runs = 3
n_batches = 610

batch_callbacks = lambda timer: {
    300: (lambda: timer.start(),),
    400: (lambda: timer.stop(), lambda: timer.start()),
    500: (lambda: timer.stop(), lambda: timer.start()),
    600: (lambda: timer.stop(),),
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
