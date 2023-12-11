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
import pandas as pd
import seaborn as sns

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

from feedbax.utils import angle_between_vectors, vector_angle

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

def get_unit_pref_angles(model, task, *, key, n_iter=50, ts=slice(None)):
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
    pref_angles = jnp.arctan2(pds[:, 1], pds[:, 0])
    
    pds = pds / jnp.linalg.norm(pds, axis=1, keepdims=True)
    
    return pref_angles, pds, states, trial_specs
    


# %%
ts = slice(0, 20)
pref_angles, pref_vectors_start, states, trial_specs = get_unit_pref_angles(
    model, 
    task, 
    ts=ts,
    key=key_eval,
)

# %%
_ = circular_hist(pref_angles)

# %% [markdown]
# Visualize one of the regressions:

# %%
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
ax.quiver(0, 0, 0, *pref_vectors_start[unit], 0, length=0.5, color='k')

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

pref_angles, pref_vectors_end, _, _ = get_unit_pref_angles(
    model, task, ts=ts, key=key_eval
)

# %%
_ = circular_hist(pref_angles)

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
ax.quiver(0, 0, 0, *pref_vectors_end[unit], 0, length=0.5, color='k')

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

pref_angles, pds, states, trial_specs = get_unit_pref_angles(model_curl, task, ts=ts, key=key_eval)

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
_ = circular_hist(pref_angles)

# %% [markdown]
# What is the distribution of changes in preferred direction over units?

# %%
_ = circular_hist(angle_between_vectors(pds, pref_vectors_start), plot_mean=True)


# %% [markdown]
# Add a constant input to a single unit:

# %%
def add_unit_perturbation(model, unit, input_current, perturbation_type=NetworkConstantInputPerturbation, *, key):
    unit_spec = jnp.full(model.step.net.hidden_size, jnp.nan)
    unit_spec = unit_spec.at[unit].set(input_current)

    model = eqx.tree_at(
        lambda model: model.step.net,
        model,
        add_intervenors(
            model.step.net, 
            {'readout': [perturbation_type(unit_spec)]},
            key=key,
        ),
    )
    
    return model


# %%
unit = 3
input_current = 2.

model_ = add_unit_perturbation(model, unit, input_current, key=jr.PRNGKey(seed + 3))

# %%
ts = slice(0, 20)

pref_angles, pds, states, trial_specs = get_unit_pref_angles(
    model_, task, ts=ts, key=key_eval
)

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
_ = circular_hist(pref_angles)

# %%
pref_angle_change = angle_between_vectors(pds, pref_vectors_start)
ax, _, _, _ = circular_hist(pref_angle_change, plot_mean=True)

stim_unit_original_angle = jnp.arctan2(*pref_vectors_start[unit])
suoa = stim_unit_original_angle
ax.plot([suoa, suoa], [0, 0.5], 'b-', lw=2)


# %% [markdown]
# Now, apply the same perturbation to each unit in turn

# %%
def pref_vectors_pre_post_pert(model, task, unit, input_current, pert_type=NetworkConstantInputPerturbation, *, key, ts=slice(None)):
    key1, key2 = jr.split(key)
    
    # no perturbation
    _, pref_vectors, _, _ = get_unit_pref_angles(
        model, task, ts=ts, key=key2
    )
    
    model = add_unit_perturbation(model, unit, input_current, perturbation_type=pert_type, key=key1)
    
    # with perturbation
    _, pref_vectors_pert, _, _ = get_unit_pref_angles(
        model, task, ts=ts, key=key2
    )
    
    return pref_vectors, pref_vectors_pert


pref_vectors, pref_vectors_pert = \
    eqx.filter_vmap(
        partial(
            pref_vectors_pre_post_pert,
            key=key_eval, ts=ts,
        ),
        in_axes=(None, None, 0, None),
    )(
        model, task, jnp.arange(model.step.net.hidden_size), input_current
    )

# %% [markdown]
# Plot the distribution of mean angle change over perturbation of different individual units

# %%
pref_angle_changes = angle_between_vectors(
    pref_vectors_pert, 
    pref_vectors,
)

_ = circular_hist(jnp.mean(pref_angle_changes, axis=1), plot_mean=True)

# %% [markdown]
# Note that the red line (due to `plot_mean=True`) is the mean of means.
#
# Now plot the mean angle change against the preferred angle of the stimulated unit

# %%
stim_unit_original_pds = np.diagonal(pref_vectors, axis1=0, axis2=1).T
stim_unit_original_angles = vector_angle(stim_unit_original_pds)

fig, ax = plt.subplots(1,1)
ax.plot(stim_unit_original_angles, jnp.mean(pref_angle_changes, axis=1), 'o')
ax.set_xlabel("Original pref. angle of stim. unit")
ax.set_ylabel("Mean change in pref. angles of non-stim. units")
plt.show()

# %% [markdown]
# These analyses don't really demonstrate that individual units PDs aren't biased with respect to the PD of the stimulated unit---more that there aren't any global systematic biases in PD. 
#
# For example, the polar histogram of the mean `pref_angle_changes` is centered on 0, but each of the means is with respect to a particular perturbed unit with a particular PD. But within that distribution, are the changes in PD biased toward/away from the original PD of the stimulated unit?
#
# We can investigate this by taking the angle of the preferred direction of each unit with respect to that of the stimulated unit, with and without the perturbation. The ratio of the latter to the former will then indicate if the angle between a unit's PD and the stim unit's PD grew (>1) or shrank (<1) due to the perturbation. 

# %%
change_ratio_pref_angle_wrt_stim = angle_between_vectors(
    pref_vectors_pert, 
    stim_unit_original_pds[:, None, :],
) / angle_between_vectors(
    pref_vectors, 
    stim_unit_original_pds[:, None, :],
)

# %% [markdown]
# Now plot the distribution of the ratio, in particular the mean over the non-stimulated units, for perturbation of each of the stimulated units.

# %%
mean_change_ratio_pref_angle_wrt_stim = jnp.nanmean(
    jnp.where(jnp.isinf(change_ratio_pref_angle_wrt_stim), jnp.nan, change_ratio_pref_angle_wrt_stim),
    axis=1,
)

fig, ax = plt.subplots(1, 1)
ax.hist(mean_change_ratio_pref_angle_wrt_stim, bins=20, density=True)
ax.set_xlabel("Pre-post pert. ratio of mean non-stim. PD to (pre) stim. PD")

# %% [markdown]
# Clearly most of these values are slightly less than 1, indicating that the PD of the non-stimulated units in a network tend to be biased toward the PD of the stimulated unit. 

# %%
jnp.mean(mean_change_ratio_pref_angle_wrt_stim)

# %% [markdown]
# Mechanistically, if the unit PDs are biased towards that of the stimulated unit, then their activities are relatively higher than pre-perturbation for those reach directions which are closer to the PD of the stimulated unit. But the perturbation is present for all reaches, including those that are in the opposite direction to the PD of the stimulated unit. This may suggest that a unit's causal influence on the other units in the network is greater during reaches in its preferred direction.
#
# However, this hypothesis might be trivially true because the perturbation is additive in this case, so the stimulated units remain more active during reaches in their PD, and less active in reaches away from their PD. 
#
# What happens if we repeat the analysis for single-unit perturbations that aren't additive, but instead clamp each unit to a particular value? Then we should still expect to see the above relationship only if the direction-dependent influence of each unit is encoded by the rest of the network, and not merely because that unit's output is itself direction-dependent.

# %%
pref_vectors, pref_vectors_pert = \
    eqx.filter_vmap(
        partial(
            pref_vectors_pre_post_pert,
            pert_type=NetworkClamp, key=key_eval, ts=ts,
        ),
        in_axes=(None, None, 0, None),
    )(
        model, task, jnp.arange(model.step.net.hidden_size), input_current
    )

# %%
stim_unit_original_pds = np.diagonal(pref_vectors, axis1=0, axis2=1).T

change_ratio_pref_angle_wrt_stim = angle_between_vectors(
    pref_vectors_pert, 
    stim_unit_original_pds[:, None, :],
) / angle_between_vectors(
    pref_vectors, 
    stim_unit_original_pds[:, None, :],
)

# %%
mean_change_ratio_pref_angle_wrt_stim = jnp.nanmean(
    jnp.where(jnp.isinf(change_ratio_pref_angle_wrt_stim), jnp.nan, change_ratio_pref_angle_wrt_stim),
    axis=1,
)

fig, ax = plt.subplots(1, 1)
ax.hist(mean_change_ratio_pref_angle_wrt_stim, bins=20, density=True)
ax.set_xlabel("Pre-post pert. ratio of mean non-stim. PD to (pre) stim. PD")

# %%
jnp.mean(mean_change_ratio_pref_angle_wrt_stim)

# %% [markdown]
# How does the effect scale with the size of the perturbation?

# %%
input_currents = jnp.linspace(-5, 5, 20)

def mean_PD_bias(model, task, unit, input_current, key):
    pref_vectors, pref_vectors_pert = eqx.filter_vmap(
            partial(
                pref_vectors_pre_post_pert,
                pert_type=NetworkClamp, key=key, ts=ts,
            ),
            in_axes=(None, None, 0, None),
        )(
            model, task, unit, input_current,
    )

    stim_unit_original_pds = np.diagonal(pref_vectors, axis1=0, axis2=1).T

    change_ratio_pref_angle_wrt_stim = angle_between_vectors(
        pref_vectors_pert, 
        stim_unit_original_pds[:, None, :],
    ) / angle_between_vectors(
        pref_vectors, 
        stim_unit_original_pds[:, None, :],
    )
    
    return jnp.where(
        jnp.isinf(change_ratio_pref_angle_wrt_stim), 
        jnp.nan, 
        change_ratio_pref_angle_wrt_stim
    )

change_ratio_pref_angle_wrt_stim = \
    eqx.filter_vmap(
        mean_PD_bias,
        in_axes=(None, None, None, 0, None),
    )(
        model, task, jnp.arange(model.step.net.hidden_size), input_currents, key_eval
    )
    
mean_change_ratio_by_stim = jnp.nanmean(change_ratio_pref_angle_wrt_stim, axis=2)
mcr = mean_change_ratio_by_stim

# %% [markdown]
# Here's an example of the the distribution for a negative (-5) value of the perturbation strength:

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(mcr[10], bins=20, density=True)
ax.set_xlabel("Pre-post pert. ratio of mean non-stim. PD to (pre) stim. PD")

# %% [markdown]
# I'm not sure about the point near 14, which appears in all the distributions for perturbation strengths <0. We're going to exclude it from the following analysis so that it doesn't distort the estimate of the mean and variability of the bulk of the distribution, which is near 1. You can comment out the following cell and run the following ones, to see the effect of including the "outlier".
#
# #### TODO: 
# What do the points at >10 and <1 mean? It can't be as simple as what happened with the 180 degree flip with one of the PDs above---the datapoints in these histograms are means over the entire non-stimulated population of units, for stimulation of each unit in turn. So these "outlier" points should correspond to particular units which were perturbed, and which had an unusual effect on the rest of the network. I suppose it might have something to do with the ratios being calculated between angles, so that if the PD of the stimulated unit is very near 0 or $2\pi$ and we shift over the boundary, we might get a very large ratio, 6 / 0.4 or something? But that would involve shifts that are larger than what we'd expect on average...

# %%
mcr = jnp.where(mcr > 10, jnp.nan, mcr)

# %%
df = pd.DataFrame(
    mcr, 
    index=input_currents, 
    #columns=jnp.arange(model.step.net.hidden_size),
).reset_index().melt(id_vars="index", var_name="stim_unit", value_name="shift")

df

# %%
fig, ax = plt.subplots(1, 1)
sns.lineplot(
    data=df,
    x="index",
    y="shift",
    errorbar='sd',
    ax=ax,
    n_boot=1000,
)
ax.hlines(1, xmin=input_currents[0], xmax=input_currents[-1], linestyles='dashed')
ax.set_xlabel("Input to stim. unit")
ax.set_ylabel("PD shift ratio of non-stim units")


# %%
