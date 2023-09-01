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

# %%
import math
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import jaxopt
from jaxtyping import Float, Array
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax 
import seaborn as sns
from tqdm import tqdm

from feedbax.networks import SimpleNet
from feedbax.utils import normalize

# %%
SAVE_PATH = Path("../models/muscle_flv")


# %% [markdown]
# ### Lumped models for two-link arm
#
# This notebook is a sketch of the implementation of the kind of non-linear functions used for the force-length-velocity relationship of lumped muscles (e.g. Lillicrap & Scott 2013, Todorov & Li 2004).
#
# Joint torques are given by $\tau(t) = M\cdot\mathbf{h}\left(\mathbf{x}(t), \mathbf{u}(t)\right)$, where $\mathbf{h}\left(\mathbf{x}(t),\mathbf{u}(t)\right)=\mathbf{u}(t)\odot\mathbf{f}_{flv}(\mathbf{l},\dot{\mathbf{l}})$. 
#
# $M$ are muscle moment arms (fixed by geometry) and $\mathbf{u}$ are muscle activations (inputs). 
#
# We need to define the function $\mathbf{f}_{flv}(\mathbf{l},\dot{\mathbf{l}})$.

# %%
class MuscleGroup:  # TODO rename
    M: jnp.array = jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0), 
                              (0.0, 0.0, 2.0, -2.0, 2.0, -1.50)))  # [cm], apparently
    theta0: jnp.array = jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), 
                                   (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) # [deg] TODO: radians
    l0: jnp.array = jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04))  # [m]


# %% [markdown]
# #### Define FLV function

# %% [markdown]
# $\mathbf{f}_{flv}$ depends on $l$ and $\dot{l}$. 
#
# At each step of a biomechanical simulation, we typically have access to joint angles ($\theta$ or `theta`) and angular velocities ($\dot\theta$ or `dtheta`), so we define functions for $l(\theta)$ and $\dot l(\dot\theta)$:

# %%
def muscle_l(theta, muscles):
    M, theta0, l0 = muscles.M, muscles.theta0, muscles.l0
    l = 1 + (M[0] * (theta0[0] - theta[0]) + M[1] * (theta0[1] - theta[1])) / l0
    return l

def muscle_v(dtheta, muscles):
    # muscle velocity
    M, l0 = muscles.M, muscles.l0
    v = (M[0] * dtheta[0] + M[1] * dtheta[1]) / l0
    return v


# %%
# a little test
muscles = MuscleGroup()
theta = jnp.array((45, 90))

muscle_l(theta, muscles)

# %% [markdown]
# Now we define the FLV functions themselves, which depend only on $l$ and $\dot l$ (`l` and `v`).

# %%
# Lillicrap & Scott 2013

# TODO these parameters should be in a dataclass; perhaps the FLV function should be a method 

beta = 1.55
omega = 0.81
vmax = -7.39
cv0 = -3.21
cv1 = 4.17
av0 = -3.12
av1 = 4.21
av2 = -2.67
bv = 0.62

def tension_from_lv_ls2013(l, v):
    """FLV function from Lillicrap & Scott 2013."""
    # #! why is the inner sign different from Todorov & Li 2004
    f_l = jnp.exp(jnp.abs((l ** beta - 1) / omega))
    f_fv_rhs = (bv - v * (av0 + av1 * l + av2 * l**2)) / (bv + v)
    f_fv_lhs = (vmax - v) / (vmax + v * (cv0 + cv1 * l))
    rhs_cond = v > 0
    f_fv = rhs_cond * f_fv_rhs + ~rhs_cond * f_fv_lhs
    flv = f_l * f_fv
    return flv

def tension_ls2013(theta, dtheta):
    """Simple helper to take joint configuration as input."""
    l = muscle_l(theta)
    v = muscle_v(dtheta)
    return tension_from_lv_ls2013(l, v)


# %%
# same thing but for Li & Todorov 2004

beta = 1.93
omega = 1.03
rho = 1.87
vmax = -5.72
cv0 = 1.38
cv1 = 2.09
av0 = -3.12
av1 = 4.21
av2 = -2.67
bv = 0.62
nf0 = 2.11
nf1 = 4.16
a_f = 0.56
c2 = -0.02
k2 = -18.7
l_r2 = 0.79
tmp1 = -k2 * l_r2

def tension_from_lv_lt2004(l, v, a=1):
    """FLV function from Li & Todorov 2004."""
    f_l = jnp.exp(-jnp.abs((l ** beta - 1) / omega) ** rho)
    f_fv_rhs = (bv - v * (av0 + av1 * l + av2 * l ** 2)) / (bv + v)
    f_fv_lhs = (vmax - v) / (vmax + v * (cv0 + cv1 * l))
    rhs_cond = v > 0
    f_fv = rhs_cond * f_fv_rhs + ~rhs_cond * f_fv_lhs  # FV = 1 for isometric condition
    f_p = c2 * jnp.exp(tmp1 + k2 * l)  # PE; elastic muscle fascicles
    # (this f_p accounts for only the compressive spring component F_PE2 from Brown 1999)
    n_f = nf0 + nf1 * (1 / l - 1)
    A_f = 1 - jnp.exp(-(a / (a_f * n_f)) ** n_f)
    tension = A_f * (f_l * f_fv + f_p)
    return tension

def tension_lt2004(theta, dtheta):
    """Simple helper to take joint configuration as input."""
    l = muscle_l(theta)
    v = muscle_v(dtheta)
    return tension_from_lv_lt2004(l, v)


# %% [markdown]
# #### Examine FLV function

# %% [markdown]
# Plot the two functions, for the range of (normalized!) muscle length and velocity given in Fig. 1C of Todorov & Li 2004.

# %%
n_plot = 75

l_range = (0.7, 1.3)
v_range = (-2, 2)

l_outer_range = (0., 2.)
v_outer_range = (-4, 4)

l = jnp.linspace(*l_range, n_plot)
v = jnp.linspace(*v_range, n_plot)
grid_l, grid_v = jnp.meshgrid(l, v)

# %%
CMAP_LABEL = 'viridis'

fig = plt.figure(constrained_layout=True, figsize=(10,5))
axs = [fig.add_subplot(121, projection='3d'), 
       fig.add_subplot(122, projection='3d')]

tension_ls2013 = tension_from_lv_ls2013(grid_l, grid_v)  # TODO don't overwrite tension_ls2013
tension_lt2004 = tension_from_lv_lt2004(grid_l, grid_v)

max_tension_plot = max(jnp.max(tension_ls2013), jnp.max(tension_lt2004))

for i, tension in enumerate([tension_ls2013, tension_lt2004]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension_plot)

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, max_tension_plot)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()

axs[0].set_title('Lillicrap & Scott 2013')
axs[1].set_title('Todorov & Li 2004')

# %% [markdown]
# They are different, especially for low lengths + high positive velocities.     
# The scale is obviously different—the Lillicrap & Scott tensions are not normalized. Could this be due to the scaling of the optimal lengths/angles? The values of given earlier are for L&S 2013, whereas T&L 2004 seems to use normalized lengths...
# Let's normalize them and compare again.

# %%
fig = plt.figure(constrained_layout=True, figsize=(10,5))
axs = [fig.add_subplot(121, projection='3d'), 
       fig.add_subplot(122, projection='3d')]

tension_ls2013 = tension_from_lv_ls2013(grid_l, grid_v)
tension_ls2013 = tension_ls2013 / jnp.max(tension_ls2013)
tension_lt2004 = tension_from_lv_lt2004(grid_l, grid_v)
tension_lt2004 = tension_lt2004 / jnp.max(tension_lt2004)

for i, tension in enumerate([tension_ls2013, tension_lt2004]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=1)

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, 1)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()


# %% [markdown]
# ### Approximating the FLV function with a neural network
#
# To try to improve performance, we can train a neural network to approximate these functions, as in Lillicrap & Scott 2013.

# %% [markdown]
# #### Data generation for training
#
# We'll optimize over the plotting range, though ultimately it may be better to optimize over the desired range of joint angles and angular velocities.

# %%
# data generation
tension_fn = tension_from_lv_lt2004

tension_max = jnp.max(tension_fn(grid_l, grid_v))
flv = lambda l, v: tension_fn(l, v) / tension_max
tension_exact = flv(grid_l, grid_v).squeeze()

eval_input = jnp.stack([grid_l, grid_v], axis=-1).reshape((-1, 2))
eval_input_norm = normalize(eval_input, min=-1, max=1)[0]

def rand_range(lo, hi, size, key):
    """Helper to generate random numbers in a range using"""
    return (hi - lo) * jrandom.uniform(key, size) + lo

def get_batch(tension_fn, batch_size, key):
    """This is specific to the type of function we're fitting"""""
    l_key, v_key = jrandom.split(key)
    inputs = jnp.stack([
        rand_range(*l_range, (batch_size,), l_key),
        rand_range(*v_range, (batch_size,), v_key),
    ], axis=1)
    targets = tension_fn(*inputs.T)
    return inputs, targets

def get_batch1D(tension_fn, batch_size, key):
    """This is specific to the type of function we're fitting"""""
    inputs = rand_range(*l_range, (batch_size,), key)
    targets = tension_fn(inputs, jnp.full_like(inputs, 2.0))
    return inputs[:,None], targets


# %%
# plot example batch of data samples

batch_size = 1000
x, y = get_batch(tension_fn, batch_size, jrandom.PRNGKey(0))

fig = plt.figure(constrained_layout=True, figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*x.T, y, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=1)

ax.set_xlabel('l [L]')
ax.set_ylabel('v [L/T]')
ax.set_zlabel('FLV [F]')
ax.set_zlim(0, 1)
ax.set_box_aspect(None, zoom=0.8)
ax.invert_xaxis()

# %% [markdown]
# #### Simple network in plain JAX

# %%
layer_sizes = [2, 5, 1]
learning_rate = 1e-2
n_batches = 15000
batch_size = 2000


# %%
# functions to generate network parameters

def random_layer_params(m, n, key, scale=1.):
    w_key, b_key = jrandom.split(key)
    stdv = 1. / jnp.sqrt(m)
    w = jrandom.uniform(w_key, (n, m), minval=-stdv, maxval=stdv)
    b = jrandom.uniform(b_key, (n,), minval=-stdv, maxval=stdv)
    return scale * w, scale * b
    #return scale * jrandom.normal(w_key, (n, m)), scale * jrandom.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = jrandom.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


# %%
# network algorithm

def model(params, input):
    activations = input
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = jax.nn.sigmoid(outputs)
        #activations = jnp.tanh(outputs)
        #activations = jnp.maximum(0, outputs)  # relu
    
    final_w, final_b = params[-1]
    outputs = jnp.dot(final_w, activations) + final_b
    return outputs  

batched_model = jax.vmap(model, in_axes=(None, 0))


# %%
# loss and learning rule

def loss_fn(params, inputs, targets):
    preds = batched_model(params, inputs)
    return jnp.mean((preds.squeeze() - targets) ** 2)

# #! was using this prior to switching to optax
# @jax.jit
# def step(params, opt_state, x, y):
#     loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
#     return loss, [(w - learning_rate * dw, b - learning_rate * db) 
#                   for (w, b), (dw, db) in zip(params, grads)], opt_state

# schedule = optax.linear_schedule(1e-5, 1e-2, 1000)

optimizer = optax.chain(
    #optax.clip(1.0),  # gradient clipping
    optax.radam(learning_rate=learning_rate),  # ADAM with weight decay
)

@jax.jit
def step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


# %%
# initialize network and train on randomly-generated batches

key = jrandom.PRNGKey(0)

params = init_network_params(layer_sizes, jrandom.PRNGKey(0))
params_init = params

opt_state = optimizer.init(params)
losses = []

for batch in tqdm(range(n_batches)):
    x, y = get_batch(flv, batch_size, key)
    x, = normalize(x, min=-1, max=1)
    loss, params, opt_state = step(params, opt_state, x, y)
    losses.append(loss)
    _, key = jrandom.split(key)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(losses)
ax.set_ylim(bottom=1e-5);


# %%
print('\nfinal params:\n', params, )
print('\ninitial params:\n', params_init) 
print('\ndifference:\n', jax.tree_map(lambda x, y: x - y, params, params_init))

# %%
# plot the approximation versus the exact function

# in case of (e.g.) an input batch norm layer, need to collapse into a single batch dimension
tension_model = batched_model(params, eval_input_norm).reshape((n_plot, n_plot))

max_abs_error = jnp.max(jnp.abs(tension_model - tension_exact))
mean_abs_error = jnp.mean(jnp.abs(tension_model - tension_exact))
print("Max. abs. error: ", max_abs_error.item())
print("Mean abs. error: ", mean_abs_error.item())

error = tension_model - tension_exact

fig = plt.figure(constrained_layout=True, figsize=(10,5))
axs = [fig.add_subplot(131, projection='3d'), 
       fig.add_subplot(132, projection='3d'),
       fig.add_subplot(133, projection='3d')]

for i, tension in enumerate([tension_model, tension_exact, error]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension_plot)

max_tension_plot = max(jnp.max(tension_model), jnp.max(tension_exact)).item()

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, max_tension_plot)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()

axs[0].set_title("1. Model")
axs[1].set_title("2. Exact")    
axs[2].set_title("1 - 2");

# %% [markdown]
# #### Second-order methods
#
# These need to be trained on the entire dataset at once. But our problem is otherwise rather small so probably no big deal.

# %%
data_key = jrandom.PRNGKey(5678)

batch_size = 50000
X, y = get_batch(tension_fn, batch_size, data_key)
X, = normalize(X, min=-1, max=1)

params = init_network_params(layer_sizes, jrandom.PRNGKey(0))
params_init = params

# %%
solver = jaxopt.LBFGS(fun=loss_fn, maxiter=1000, tol=1e-8)
res = solver.run(params, X, y)

# %%
solver = jaxopt.BFGS(fun=loss_fn, maxiter=1000, tol=1e-8)
res = solver.run(params, X, y)

# %%
solver = jaxopt.NonlinearCG(fun=loss_fn, maxiter=100000, tol=1e-8)
res = solver.run(params, X, y)

# %%
# plot the approximation versus the exact function

# in case of (e.g.) an input batch norm layer, need to collapse into a single batch dimension
tension_model = batched_model(res[0], eval_input_norm).reshape((n_plot, n_plot))

max_abs_error = jnp.max(jnp.abs(tension_model - tension_exact))
mean_abs_error = jnp.mean(jnp.abs(tension_model - tension_exact))
print("Max. abs. error: ", max_abs_error.item())
print("Mean abs. error: ", mean_abs_error.item())

error = tension_model - tension_exact

fig = plt.figure(constrained_layout=True, figsize=(10,5))
axs = [fig.add_subplot(131, projection='3d'), 
       fig.add_subplot(132, projection='3d'),
       fig.add_subplot(133, projection='3d')]

for i, tension in enumerate([tension_model, tension_exact, error]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension_plot)

max_tension_plot = max(jnp.max(tension_model), jnp.max(tension_exact)).item()

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, max_tension_plot)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()

axs[0].set_title("1. Model")
axs[1].set_title("2. Exact")    
axs[2].set_title("1 - 2");

# %% [markdown]
# #### Equinox
#
# I can't seem to get it to converge... tried normalization, a range of learning rates, some different weight initializations, and it always outputs more or less a plane.

# %%
layer_sizes = [2, 5, 1]
use_bias = (True, False)
nonlinearity = jax.nn.sigmoid
output_nonlinearity = jax.nn.sigmoid
model_key = jrandom.PRNGKey(0)
learning_rate = 1.e-2
n_batches = 20000
batch_size = 2000

# %%
model = SimpleNet(
    layer_sizes, 
    use_bias=use_bias, 
    nonlinearity=nonlinearity,
    output_nonlinearity=output_nonlinearity,
    key=model_key,
)
print(model)


# %%
# custom weight initialization (see https://docs.kidger.site/equinox/tricks/)

def weight_init_fn(param: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    stddev = math.sqrt(1 / param.shape[-1])
    return stddev * jrandom.uniform(key, param.shape, minval=-1, maxval=1) #lower=-2, upper=2)

def init_linear_weight(model, init_fn, key):
    """Re-initialize the weights of all linear layers in a model."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    for param_type in ["weight", "bias"]:
        get_params = lambda m: [getattr(x, param_type)
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
        params = get_params(model)
        new_params = [init_fn(param, subkey) if param is not None else None
                      for param, subkey in zip(params, jax.random.split(key, len(params)))]
        model = eqx.tree_at(get_params, model, new_params)
    return model

model = init_linear_weight(model, weight_init_fn, jrandom.PRNGKey(567))


# %%
def loss_fn(model, inputs, targets):
    preds = jax.vmap(model)(inputs).squeeze()
    return jnp.mean(optax.l2_loss(preds, targets))

schedule = optax.linear_schedule(learning_rate, 1.e-4, 10000, transition_begin=10000)

optimizer = optax.chain(
    #optax.clip(1.0),
    optax.radam(learning_rate=schedule),
)

@eqx.filter_jit
def step(model, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# %%
# initialize network and train on randomly-generated batches

key = jrandom.PRNGKey(0)

opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
losses = []

for batch in tqdm(range(n_batches)):
    x, y = get_batch(flv, batch_size, key)
    x, = normalize(x, min=-1, max=1)
    loss, model, opt_state = step(model, opt_state, x, y)
    losses.append(loss)
    _, key = jrandom.split(key)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(losses)
ax.set_ylim(bottom=1e-5);


# %%
# plot the approximation versus the exact function

# in case of (e.g.) an input batch norm layer, need to collapse into a single batch dimension
tension_model = jax.vmap(model)(eval_input_norm).reshape((n_plot, n_plot))

max_abs_error = jnp.max(jnp.abs(tension_model - tension_exact))
mean_abs_error = jnp.mean(jnp.abs(tension_model - tension_exact))
print("Max. abs. error: ", max_abs_error.item())
print("Mean abs. error: ", mean_abs_error.item())

error = tension_model - tension_exact

fig = plt.figure(constrained_layout=True, figsize=(10,5))
axs = [fig.add_subplot(131, projection='3d'), 
       fig.add_subplot(132, projection='3d'),
       fig.add_subplot(133, projection='3d')]

max_tension_plot = max(jnp.max(tension_model), jnp.max(tension_exact)).item()

for i, tension in enumerate([tension_model, tension_exact, error]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension_plot)

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, max_tension_plot)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()

axs[0].set_title("1. Model")
axs[1].set_title("2. Exact")    
axs[2].set_title("1 - 2");

# %% [markdown]
# Serialize the model to disk

# %%
# #! this doesn't save the model architecture; when reloading need a similar model already instantiated
model_save_path = SAVE_PATH / 'todorov_li_2004.eqx'
eqx.tree_serialise_leaves(model_save_path, model)

# %%
# example reloading the model; requires a similar model instance
eqx.tree_deserialise_leaves(model_save_path, model)
