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

# %matplotlib inline

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
import numpy as np

from feedbax.intervene import (
    EffectorCurlForceField,
    NetworkClamp,
    NetworkConstantInputPerturbation,
)
from feedbax.model import add_intervenors
from feedbax.task import RandomReaches
from feedbax.xabdeef import point_mass_RNN_simple_reaches

from feedbax.plot import (
    plot_losses, 
    plot_pos_vel_force_2D,
    plot_activity_heatmap, 
    plot_activity_sample_units,
    animate_3D_rotate,
    circular_hist,
)

# %%
# changes matplotlib style, to match dark notebook themes
# plt.style.use('dark_background')

# %% [markdown]
# ### Absolutely minimal example, using defaults

# %%
seed = 5566

# context = point_mass_RNN_simple_reaches(key=jr.PRNGKey(seed))

context = point_mass_RNN_simple_reaches(eval_grid_n=1, key=jr.PRNGKey(seed))

model, losses, _ = context.train(
    n_batches=2500, 
    batch_size=500, 
    key=jr.PRNGKey(seed + 1),
)

plot_losses(losses)

# %% [markdown]
# Evaluate the model on a set of center-out reaches:

# %%
n_directions = 20

task = RandomReaches(
    loss_func = context.task.loss_func,
    workspace = context.task.workspace,
    n_steps = context.task.n_steps,
    eval_n_directions=n_directions,
    eval_reach_length=0.5,
    eval_grid_n=1,
)

# %%
key_eval = jr.PRNGKey(seed + 2)
losses, states = task.eval(model, key=key_eval)
trial_specs, _ = task.trials_validation

# %%
plot_pos_vel_force_2D(
    states,
    endpoints=(
        trial_specs.init['mechanics']['effector'].pos, 
        trial_specs.goal.pos
    ),
)

# %%
seed = 5566
n_samples = 6
key = jr.PRNGKey(seed + 1)

plot_activity_sample_units(states.network.hidden, 3, key=key, unit_includes=(6,))

# %% [markdown]
# Regress, for each unit, its activity against the target position. The result is, for each unit, the parameters for a 3D plane whose slope points in the unit's preferred movement direction.
#
# First we set up the linear regression problem numerically in JAX. 

# %%
import optax

def loss_func(model, X, y):
    mse = jnp.mean((model(X) - y) ** 2)
    return mse 

optimizer = optax.sgd(learning_rate=0.01)
    
def fit(model, X, y, n_iter=50):
    """This is easily vmappable over multiple datasets."""
    opt_state = optimizer.init(model)

    for _ in range(n_iter):
        loss, grads = jax.value_and_grad(loss_func)(model, X, y)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)    
        #model = update(model, grads)
    
    return model

def fit_linear(X, y, n_iter=50):
    model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(seed))
    model = jax.tree_map(jnp.zeros_like, model)
    
    return fit(model, X.T, y, n_iter=n_iter)


# %%
def prep_data(hidden, target):

    # not worrying about train-test splits here
    ys = jnp.reshape(hidden, (-1, hidden.shape[-1]))  # (conditions, time, units)
    X = jnp.reshape(target, (-1, 2))

    # center and normalize
    X = X - jnp.mean(X, axis=0)
    X = X / jnp.max(X, axis=0)
    # TODO: should probably rescale (or renormalize) the preference vector after fitting
    
    return X, ys


# %% [markdown]
# Now we set up our specific problem: the hidden rates are dependent, and the target positions are independent. 
#
# We only fit the first 20 time steps, which is approximately the first "half-wavelength" of the neural oscillation.

# %%
from functools import partial

def get_unit_preferred_angles(model, task, *, key, n_iter=50, ts=slice(None)):
    _, states = task.eval(model, key=key)
    trial_specs, _ = task.trials_validation
    
    hidden = states.network.hidden[:, ts]
    target = trial_specs.target.pos[:, ts]
    
    X, ys = prep_data(hidden, target)
    
    def fit_linear(X, y, n_iter=50):
        lin_model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(seed))
        lin_model = jax.tree_map(jnp.zeros_like, lin_model)
        
        return fit(lin_model, X.T, y, n_iter=n_iter)
    
    batch_fit_linear = jax.vmap(
        partial(fit_linear, n_iter=n_iter),
        in_axes=(None, 1)
    )
    
    fits = batch_fit_linear(X, ys)
    
    pds = jnp.squeeze(fits.weight)  # preferred directions

    # Vector length is irrelevant to angles
    preferred_angles = jnp.arctan2(pds[:, 1], pds[:, 0])
    
    pds = pds / jnp.linalg.norm(pds, axis=1, keepdims=True)
    
    return preferred_angles, pds, states, trial_specs
    


# %%
ts = slice(0, 20)
preferred_angles, preferred_vectors_start, states, trial_specs = get_unit_preferred_angles(
    model, 
    task, 
    ts=ts,
    key=key_eval,
)

# %%
_ = circular_hist(preferred_angles)

# %% [markdown]
# Visualize one of the regressions:

# %%
# %matplotlib inline

unit = 3

hidden = states.network.hidden[:, ts]
target = trial_specs.target.pos[:, ts]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = 'tab10'
cmap_func = plt.get_cmap(cmap)
colors = [cmap_func(i) for i in np.linspace(0, 1, hidden.shape[0])]

for X_, y_, c in zip(target, hidden, colors):
    ax.scatter(X_[:, 0], X_[:, 1], y_[..., unit], c=c, marker='o')
ax.quiver(0, 0, 0, *preferred_vectors_start[unit], 0, length=0.5, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Unit activity')

# %%
anim = animate_3D_rotate(fig, ax)
anim.save(f'unit_{unit}_pd.mp4', fps=20)

# %% [markdown]
# Do the preference analysis for a different time period: steady state at the end of the trial.

# %%
ts = slice(60, 100)

preferred_angles, preferred_vectors_end, _, _ = get_unit_preferred_angles(model, task, ts=ts, key=key_eval)

# %%
_ = circular_hist(preferred_angles)

# %%
unit = 44

hidden = states.network.hidden[:, ts]
target = trial_specs.target.pos[:, ts]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = 'tab10'
cmap_func = plt.get_cmap(cmap)
colors = [cmap_func(i) for i in np.linspace(0, 1, hidden.shape[0])]

for X_, y_, c in zip(target, hidden, colors):
    ax.scatter(X_[:, 0], X_[:, 1], y_[..., unit], c=c, marker='o')
ax.quiver(0, 0, 0, *preferred_vectors_end[unit], 0, length=0.5, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Unit activity')

# %% [markdown]
# Test the response to perturbation.
#
# Apply a clockwise curl field:

# %%
model_curl = eqx.tree_at(
    lambda model: model.step,
    model,
    add_intervenors(
        model.step, 
        [EffectorCurlForceField(2, direction='cw')],
        key=jr.PRNGKey(seed + 3),
    ),
)

# %%
ts = slice(0, 20)

preferred_angles, pds, states, trial_specs = get_unit_preferred_angles(model_curl, task, ts=ts, key=key_eval)

# %%
plot_pos_vel_force_2D(
    states,
    endpoints=(
        trial_specs.init['mechanics']['effector'].pos, 
        trial_specs.goal.pos
    ),
)
plt.show()

# %%
_ = circular_hist(preferred_angles)


# %%
def angle_between_vectors(v1, v2):
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0], 
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )   


# %%
_ = circular_hist(angle_between_vectors(pds, preferred_vectors_start), plot_mean=True)

# %% [markdown]
# Add a constant input to a single unit:

# %%
unit = 3
input_current = 2.

unit_spec = jnp.full(states.network.hidden.shape[-1], jnp.nan)

unit_spec = unit_spec.at[unit].set(input_current)

# %%
model_ = eqx.tree_at(
    lambda model: model.step.net,
    model,
    add_intervenors(
        model.step.net, 
        {'readout': [NetworkConstantInputPerturbation(unit_spec)]},
        key=jr.PRNGKey(seed + 3),
    ),
)

# %%
ts = slice(0, 20)

preferred_angles, pds, states, trial_specs = get_unit_preferred_angles(model_, task, ts=ts, key=key_eval)

# %%
plot_pos_vel_force_2D(
    states,
    endpoints=(
        trial_specs.init['mechanics']['effector'].pos, 
        trial_specs.goal.pos
    ),
)
plt.show()

# %%
seed = 5566
n_samples = 3
key = jr.PRNGKey(seed + 1)

plot_activity_sample_units(states.network.hidden, n_samples, unit_includes=(6,), key=key)

# %%
_ = circular_hist(preferred_angles)

# %%
preferred_angle_change = angle_between_vectors(pds, preferred_vectors_start)
ax, _, _, _ = circular_hist(preferred_angle_change, plot_mean=True)

stim_unit_original_angle = jnp.arctan2(*preferred_vectors_start[unit])
suoa = stim_unit_original_angle
ax.plot([suoa, suoa], [0, 0.5], 'b-', lw=2)

# %% [markdown]
# Interesting that the unit additive perturbation doesn't alter the distribution of preferred directions, but the curl field does.

# %% [markdown]
# Does this alter the distribution of preferences?

# %% [markdown]
# Clamp the unit rather than adding the input to its existing activity

# %%
model_ = eqx.tree_at(
    lambda model: model.step.net,
    model,
    add_intervenors(
        model.step.net, 
        {'readout': [NetworkClamp(unit_spec)]},
        key=jr.PRNGKey(seed + 3),
    ),
)

# %%
losses, states = task.eval(model_, key=key_eval)

# %%
trial_specs, _ = task.trials_validation

plot_pos_vel_force_2D(
    states,
    endpoints=(
        trial_specs.init['mechanics']['effector'].pos, 
        trial_specs.goal.pos
    ),
)
plt.show()

# %%
seed = 5566
n_samples = 6
key = jr.PRNGKey(seed)

plot_activity_sample_units(states.network.hidden, n_samples, key=key)

# %%
