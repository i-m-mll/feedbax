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

# %matplotlib widget

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
from feedbax.xabdeef import point_mass_RNN_simple_reaches

from feedbax.plot import (
    plot_losses, 
    plot_pos_vel_force_2D,
    plot_activity_heatmap, 
    plot_activity_sample_units,
    animate_3D_rotate,
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

n_directions = 50

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


# %% [markdown]
# Regress, for each unit, its activity against the target position. The result is, for each unit, the parameters for a 3D plane whose slope points in the unit's preferred movement direction.
#
# First we set up the linear regression problem numerically in JAX. 

# %%
def linear(X, params):
    return jnp.dot(X, params['w']) + params['b']

def loss_func(params, X, y):
    mse = jnp.mean((linear(X, params) - y) ** 2)
    return mse 

def update(params, grads, lr=0.01):
    return jax.tree_map(
        lambda param, grad: param - lr * grad, 
        params, 
        grads
    )

def init_params(X):
    return dict(
        w=jnp.zeros(X.shape[1:]),
        b=0.,
    )
    
def fit_linear(X, y, n_iter=50):
    """This is easily vmappable over multiple datasets."""
    params = init_params(X)

    for _ in range(n_iter):
        loss, grads = jax.value_and_grad(loss_func)(params, X, y)
        #print(loss)
        
        params = update(params, grads)    
    
    return params 


# %% [markdown]
# Now we set up our specific problem: the hidden rates are dependent, and the target positions are independent. 
#
# We only fit the first 20 time steps, which is approximately the first "half-wavelength" of the neural oscillation.

# %%
ts = slice(0, 20)

hidden = states.network.hidden[:, ts]
target_pos = trial_specs.target.pos[:, ts]

# not worrying about train-test splits here
ys = jnp.reshape(hidden, (-1, model.step.net.hidden_size))  # (conditions, time, units)
X = jnp.reshape(target_pos, (-1, 2))

# center and normalize
X = X - jnp.mean(X, axis=0)
X = X / jnp.max(X, axis=0)
# TODO: should probably rescale (or renormalize) the preference vector after fitting

params = jax.vmap(fit_linear, in_axes=(None, 1))(X, ys)

# normalize the vectors
pds = params['w']  # preferred directions
pds = pds / jnp.sqrt(jnp.sum(pds ** 2, axis=1))[:, None]

# %%
# Note that normalization is irrelevant to angles.
preferred_angles = jnp.arctan2(pds[:, 1], pds[:, 0])


# %%
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
        
    From https://stackoverflow.com/a/55067613
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


# %%
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))

circular_hist(ax, preferred_angles)

# %%

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(*pds.T, '.')

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = 'tab10'
cmap_func = plt.get_cmap(cmap)
colors = [cmap_func(i) for i in np.linspace(0, 1, hidden.shape[0])]

for X_, y_, c in zip(target_pos, hidden, colors):
    ax.scatter(X_[:, 0], X_[:, 1], y_[..., unit], c=c, marker='o')
ax.quiver(0, 0, 0, params['w'][0], params['w'][1], params['b'], length=5, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Unit activity')

anim = animate_3D_rotate(fig, ax)

# %%
anim.save('unit_3.mp4', fps=20)

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
        [EffectorCurlForceField(0.25, direction='cw')],
        key=jr.PRNGKey(seed + 3),
    ),
)

# %%
losses, states = task.eval(model_curl, key=key_eval)

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

# %% [markdown]
# Add a constant input to a single unit:

# %%
unit = 3
input_current = 0.5

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
plot_activity_heatmap(states.network.hidden[0])
plt.show()

# %%
seed = 5566
n_samples = 6
key = jr.PRNGKey(seed)

plot_activity_sample_units(states.network.hidden, n_samples, key=key)

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
