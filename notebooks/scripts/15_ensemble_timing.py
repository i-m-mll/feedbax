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
from tqdm.auto import tqdm

from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_RNN

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

n_batches = 201
batch_size = 500

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

# %% [markdown]
# We'll use batch callbacks to time three runs for each of several values of `n_replicates`, to see how training rate depends on it.

# %%
batch_callbacks = lambda timer: {
    n_batches - 101: (lambda: timer.start(),),
    n_batches - 1: (lambda: timer.stop(),),
}

# %%
n_runs = 1
n_steps = [10, 25, 50, 75, 100]
n_replicates = [1, 2, 4, 8, 16, 32, 64]

times_means = np.zeros((len(n_steps), len(n_replicates)))
times_stds = np.zeros((len(n_steps), len(n_replicates)))

key = jr.PRNGKey(seed)

for idx0 in tqdm(range(len(n_steps)), desc="n_steps"):
    s = n_steps[idx0]
    
    for idx1 in tqdm(range(len(n_replicates)), desc="n_replicates"):
        n = n_replicates[idx1]
        
        timer = Timer()
        
        task = RandomReaches(
            loss_func=simple_reach_loss(s),
            workspace=workspace, 
            n_steps=s,
            eval_grid_n=2,
            eval_n_directions=8,
            eval_reach_length=0.5,
        )
        
        # get `n` models with `s` steps each
        models = get_model_ensemble(
            partial(
                point_mass_RNN,
                task,
                dt=dt,
                mass=mass,
                n_hidden=n_hidden,
                n_steps=s,
                feedback_delay=feedback_delay_steps,
            ),
            n,
            key=key, 
        )
        
        for i in range(n_runs):
            models = models
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
                disable_tqdm=True,
            )
        
        times_means[idx0, idx1] = np.mean(timer.times)
        times_stds[idx0, idx1] = np.std(timer.times)
    


# %% [markdown]
# Save the results

# %%
os.makedirs("./data", exist_ok=True)
np.save(f"./data/{NB_PREFIX}_times_means.npy", times_means)
np.save(f"./data/{NB_PREFIX}_times_stds.npy", times_stds)

# %% [markdown]
# Plot the training rate in iterations/second

# %%
fig, ax = plt.subplots()

means = np.array(times_means) / 100

cmap_func = plt.get_cmap('viridis')
colors = [cmap_func(i) for i in np.linspace(0, 1, len(n_steps))]

rate = 1 / means 

for i in range(len(n_steps)):
    ax.plot(n_replicates, rate[i], c=colors[i], lw=1.5)

# harcoded estimate I made earlier for a single 1,000-step model
ax.plot(1, 1.75, 'w*', ms=7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Training rate (it/s)')
ax.set_xticks([1, 8, 16, 32, 64], [1, 8, 16, 32, 64])
ax.set_yticks([1, 10, 30], [1, 10, 30])

plt.legend(n_steps, title="Time steps")

# %% [markdown]
# Plot the time taken for a 10,000 iteration training run

# %%
fig, ax = plt.subplots()

means = 10_000 * (np.array(times_means) / 100) / 60
stds = np.array(times_stds) / 100

cmap_func = plt.get_cmap('viridis')
colors = [cmap_func(i) for i in np.linspace(0, 1, len(n_steps))]

for i in range(len(n_steps)):
    ax.plot(n_replicates, means[i], c=colors[i], lw=1.5)

# harcoded estimate I made earlier for a single 1,000-step model
ax.plot(1, 95, 'w*', ms=7)

ax.plot()    
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Training time (min)')
ax.set_xticks([1, 8, 16, 32, 64], [1, 8, 16, 32, 64])
ax.set_yticks([5, 10, 100], [5, 10, 100])
# ax.tick_params(axis='y', which='minor', left=False)

plt.legend(n_steps, title="Time steps")
