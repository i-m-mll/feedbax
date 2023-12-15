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
NB_PREFIX = "nb15"
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
from feedbax.xabdeef.models import point_mass_NN

from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer
from feedbax.utils import Timer, get_model_ensemble

# %%
plt.style.use('dark_background')

# %%
seed = 5566

mass = 1.0
# n_steps = 10
dt = 0.05
feedback_delay_steps = 0
workspace = ((-1., -1.),
             (1., 1.))
hidden_size  = 50
learning_rate = 0.01

n_batches = 201
batch_size = 500

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    checkpointing=True,
)

get_task = lambda n_steps: RandomReaches(
    loss_func=simple_reach_loss(n_steps),
    workspace=workspace, 
    n_steps=n_steps,
    eval_grid_n=2,
    eval_n_directions=8,
    eval_reach_length=0.5,
)

get_models = lambda task, n_steps, n_replicates, key: get_model_ensemble(
    partial(
        point_mass_NN,
        task,
        dt=dt,
        mass=mass,
        hidden_size=hidden_size,
        n_steps=n_steps,
        feedback_delay_steps=feedback_delay_steps,
    ),
    n_replicates,
    key=key, 
)

trainable_leaves_func = lambda model: (
    model.step.net.hidden.weight_hh, 
    model.step.net.hidden.weight_ih, 
    model.step.net.hidden.bias
)

# %% [markdown]
# We'll use batch callbacks to time some training runs for each of several values of `n_replicates`, to see how training rate depends on it.

# %%
n_batches_timer = 100

batch_callbacks = lambda timer: {
    n_batches - n_batches_timer - 1: (lambda: timer.start(),),
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
        
        task = get_task(s)
        
        # get `n` models with `s` steps each
        models = get_models(task, s, n, key)
        
        for i in range(n_runs):
            trainer(
                task=task, 
                model=models,
                n_batches=n_batches, 
                batch_size=batch_size, 
                ensembled=True,
                key=jr.PRNGKey(seed + i),
                log_step=200,
                trainable_leaves_func=trainable_leaves_func,
                batch_callbacks=batch_callbacks(timer),
                disable_tqdm=True,
            )
        
        times_means[idx0, idx1] = np.mean(timer.times) / n_batches_timer 
        times_stds[idx0, idx1] = np.std(timer.times) / n_batches_timer

# %% [markdown]
# Save the results

# %%
os.makedirs("../data", exist_ok=True)
np.save(f"../data/{NB_PREFIX}_train_times_means_0delay.npy", times_means)
np.save(f"../data/{NB_PREFIX}_train_times_stds_0delay.npy", times_stds)

# %%
times_means = np.load(f"../data/{NB_PREFIX}_train_times_means.npy")
times_stds = np.load(f"../data/{NB_PREFIX}_train_times_stds.npy")

# %% [markdown]
# Plot the training rate in iterations/second

# %%
fig, ax = plt.subplots()

means = np.array(times_means) 

cmap_func = plt.get_cmap('viridis')
colors = [cmap_func(i) for i in np.linspace(0, 1, len(n_steps))]

rate = 1 / means 

for i in range(len(n_steps)):
    ax.plot(n_replicates, rate[i], c=colors[i], lw=1.5)

# harcoded estimate I made earlier for a single 1,000-step model
#ax.plot(1, 1.75, 'w*', ms=7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Training rate (it/s)')
ax.set_xticks([1, 8, 16, 32, 64], 
              [1, 8, 16, 32, 64])
ax.set_ylim(0.1, 1000)
# ax.set_yticks([1, 10, 30], 
#               [1, 10, 30])

plt.legend(n_steps, title="Time steps")

# %% [markdown]
# Plot the time taken for a 10,000 iteration training run

# %%
n_batches_example = 10_000

fig, ax = plt.subplots()

means = n_batches_example * np.array(times_means) / 60  # minutes
stds = np.array(times_stds) 

cmap_func = plt.get_cmap('viridis')
colors = [cmap_func(i) for i in np.linspace(0, 1, len(n_steps))]

for i in range(len(n_steps)):
    ax.plot(n_replicates, means[i], c=colors[i], lw=1.5)

# harcoded estimate I made earlier for a single 1,000-step model
#ax.plot(1, 95, 'w*', ms=7)

ax.plot()    
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Training time (min)')
ax.set_xticks([1, 8, 16, 32, 64], 
              [1, 8, 16, 32, 64])
ax.set_ylim(0.1, 300)
# ax.set_yticks([5, 10, 100], 
#               [5, 10, 100])
# ax.tick_params(axis='y', which='minor', left=False)

plt.legend(n_steps, title="Time steps")

# %%
jnp.max(means)

# %% [markdown]
# ### Evaluation rate

# %% [markdown]
# Let's do the same thing, but for model evaluation. 
#
# First, let's evaluate on untrained models.

# %%
n_eval = 50
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
        
        task = get_task(s)
        
        # get `n` models with `s` steps each
        # don't need a new key since the splits are changing
        models = get_models(task, s, n, key)
        
        models_arrays, models_other = eqx.partition(models, eqx.is_array)
        
        def evaluate_single(model_arrays, model_other, batch_size, key):
            model = eqx.combine(model_arrays, model_other)
            return task.eval_train_batch(model, batch_size, key)         
        
        @eqx.filter_jit
        def evaluate(keys_eval):
            return eqx.filter_vmap(evaluate_single, in_axes=(0, None, None, 0))(
                models_arrays, models_other, batch_size, keys_eval
            )
        
        keys_eval = jr.split(key, n)
        
        # make sure the model is compiled
        evaluate(keys_eval)
        
        # same eval keys but different models
        for _ in range(n_eval):    
            with timer:            
                evaluate(keys_eval)
        
        times_means[idx0, idx1] = np.mean(timer.times) 
        times_stds[idx0, idx1] = np.std(timer.times)


# %%
n_eval = 50
n_steps = [10, 25, 50, 75, 100]
n_replicates = [1, 2, 4, 8, 16, 32, 64]
n_warmup = 0

times_means = np.zeros((len(n_steps), len(n_replicates)))
times_stds = np.zeros((len(n_steps), len(n_replicates)))

key = jr.PRNGKey(seed)

for idx0 in tqdm(range(len(n_steps)), desc="n_steps"):
    s = n_steps[idx0]
    
    for idx1 in tqdm(range(len(n_replicates)), desc="n_replicates"):
        n = n_replicates[idx1]
        
        timer = Timer()
                
        task = get_task(s)
        
        # get `n` models with `s` steps each
        # don't need a new key since the splits are changing
        models = get_models(task, s, n, key)
        
        # make sure the model is compiled
        task.eval_ensemble_train_batch(models, n, batch_size, key)
        
        # warmup before first run
        if s == n_steps[0] and n == n_replicates[0]:
            for _ in range(n_warmup):
                task.eval_ensemble_train_batch(models, n, batch_size, key)
        
        # same eval keys but different models
        for _ in range(n_eval):    
            key, _ = jr.split(key)
            with timer:            
                task.eval_ensemble_train_batch(models, n, batch_size, key)
        
        times_means[idx0, idx1] = np.mean(timer.times) 
        times_stds[idx0, idx1] = np.std(timer.times)


# %%
os.makedirs("../data", exist_ok=True)
np.save(f"../data/{NB_PREFIX}_eval_times_means_0delay.npy", times_means)
np.save(f"../data/{NB_PREFIX}_eval_times_stds_0delay.npy", times_stds)

# %%
fig, ax = plt.subplots()

means = np.array(times_means) 

cmap_func = plt.get_cmap('viridis')
colors = [cmap_func(i) for i in np.linspace(0, 1, len(n_steps))]

rate = 1 / means 

for i in range(len(n_steps)):
    ax.plot(n_replicates, rate[i], c=colors[i], lw=1.5)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Evaluation rate ($s^{-1}$)')
ax.set_xticks([1, 2, 4, 8, 16, 32, 64], 
              [1, 2, 4, 8, 16, 32, 64])
ax.set_ylim(0.1, 1000)
# ax.set_yticks([1, 10, 30], 
#               [1, 10, 30])
# ax.set_ylim(1, 1500)

plt.legend(n_steps[::-1], title="Time steps")

# %% [markdown]
# Time taken for 10,000 evaluations of 500 trials

# %%
n_batches_example = 10_000

fig, ax = plt.subplots()

means = n_batches_example * np.array(times_means) / 60  # minutes
stds = np.array(times_stds) 

cmap_func = plt.get_cmap('viridis')
colors = [cmap_func(i) for i in np.linspace(0, 1, len(n_steps))]

for i in range(len(n_steps)):
    ax.plot(n_replicates, means[i], c=colors[i], lw=1.5)

# # harcoded estimate I made earlier for a single 1,000-step model
# ax.plot(1, 95, 'w*', ms=7)

ax.plot()    
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('Number of replicates')
ax.set_ylabel('Evaluation time (min)')
ax.set_xticks([1, 8, 16, 32, 64], 
              [1, 8, 16, 32, 64])
ax.set_ylim(0.1, 300)
# ax.tick_params(axis='y', which='minor', left=False)

plt.legend(n_steps[::-1], title="Time steps")

# %%
jnp.max(means)
