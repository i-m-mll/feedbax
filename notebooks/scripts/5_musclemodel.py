# ---
# jupyter:
#   jupytext:
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

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Float, Array
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax 
import seaborn as sns
from tqdm import tqdm

from feedbax.utils import exp_taylor, sincos_derivative_signs


# %% [markdown]
# ### Lumped model for two-link arm
#
# Lillicrap & Scott 2013, Todorov & Li 2004
#
# Joint torques are given by $\tau(t) = M\cdot\mathbf{h}\left(\mathbf{x}(t), \mathbf{u}(t)\right)$, where $\mathbf{h}\left(\mathbf{x}(t),\mathbf{u}(t)\right)=\mathbf{u}(t)\odot\mathbf{f}_{flv}(\mathbf{l},\dot{\mathbf{l}})$. $M$ are muscle moment arms (fixed by geometry) and $\mathbf{u}$ are muscle activations (inputs). We need to define the function $\mathbf{f}_{flv}(\mathbf{l},\dot{\mathbf{l}})$, which depends on functions for $l$ and $\dot{l}$:

# %%
class MuscleGroup:
    M: jnp.array = jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0), 
                              (0.0, 0.0, 2.0, -2.0, 2.0, -1.50)))  # [cm], apparently
    theta0: jnp.array = jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), 
                                   (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) # [deg] TODO: radians
    l0: jnp.array = jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04))  # [m]

# def muscle_l0(theta):
#     l = 1 + (M[0][None, :] * (theta0[0][None, :] - theta[..., 0, None])
#              + M[1][None, :] * (theta0[1][None, :] - theta[..., 1, None])) / l0[None, :]
#     return l

def muscle_l(theta, muscles):
    M, theta0, l0 = muscles.M, muscles.theta0, muscles.l0
    l = 1 + (M[0] * (theta0[0] - theta[0]) + M[1] * (theta0[1] - theta[1])) / l0
    return l

# def muscle_v0(dtheta):
#     # muscle velocity
#     v = (M[0][None, :] * dtheta[..., 0, None] + M[1][None, :] * dtheta[..., 1, None]) / l0[None, :]
#     return v

def muscle_v(dtheta, muscles):
    # muscle velocity
    M, l0 = muscles.M, muscles.l0
    v = (M[0] * dtheta[0] + M[1] * dtheta[1]) / l0
    return v


# %%
muscles = MuscleGroup()
theta = jnp.array((45, 90))

muscle_l(theta, muscles)

# %%
# Lillicrap & Scott 2013

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
    f_l = jnp.exp(jnp.abs((l ** beta - 1) / omega))
    f_fv_rhs = (bv - v * (av0 + av1 * l + av2 * l**2)) / (bv + v)
    f_fv_lhs = (vmax - v) / (vmax + v * (cv0 + cv1 * l))
    rhs_cond = v > 0
    f_fv = rhs_cond * f_fv_rhs + ~rhs_cond * f_fv_lhs
    flv = f_l * f_fv
    return flv

def tension_ls2013(theta, dtheta):
    l = muscle_l(theta)
    v = muscle_v(dtheta)
    return tension_from_lv_ls2013(l, v)


# %%
# Li & Todorov 2004

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
    l = muscle_l(theta)
    v = muscle_v(dtheta)
    return tension_from_lv_lt2004(l, v)


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

max_tension = max(jnp.max(tension_ls2013), jnp.max(tension_lt2004))

for i, tension in enumerate([tension_ls2013, tension_lt2004]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension)

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, max_tension)
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

max_tension = max(jnp.max(tension_ls2013), jnp.max(tension_lt2004))

for i, tension in enumerate([tension_ls2013, tension_lt2004]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension)

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, 1)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()


# %% [markdown]
# #### Approximating the tension function with a small neural network

# %% [markdown]
# To try to improve performance, we can train a neural network to approximate these functions, as in Lillicrap & Scott 2013.
#
# We'll optimize over the plotting range, though ultimately it may be better to optimize over the desired range of joint angles and angular velocities.

# %%
tension_fn = tension_from_lv_lt2004
layer_sizes = [2, 5, 1]
learning_rate = 1.e-3
n_batches = 10000
batch_size = 2000


# %%
# functions to generate network parameters

def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = jrandom.split(key)
    return scale * jrandom.normal(w_key, (n, m)), scale * jrandom.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = jrandom.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


# %%
# neural network algorithm

def predict(params, input):
    activations = input
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = jnp.tanh(outputs)
        #activations = jnp.maximum(0, outputs)
    
    final_w, final_b = params[-1]
    outputs = jnp.dot(final_w, activations) + final_b
    return outputs  # TODO: sigmoid?

batched_predict = jax.vmap(predict, in_axes=(None, 0))


# %%
# loss and learning rule

def loss_fn(params, inputs, targets):
    preds = batched_predict(params, inputs)
    return jnp.mean((preds - targets) ** 2)

@jax.jit
def update(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return loss, [(w - learning_rate * dw, b - learning_rate * db) 
                  for (w, b), (dw, db) in zip(params, grads)]


# %%
# data generation

tension_max = jnp.max(tension_fn(grid_l, grid_v))
flv = lambda l, v: tension_fn(l, v) / tension_max

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


# %%
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


# %%
# initialize network and train on randomly-generated batches

losses = []
key = jrandom.PRNGKey(0)

params = init_network_params(layer_sizes, jrandom.PRNGKey(0))
params_init = params

for batch in tqdm(range(n_batches)):
    x, y = get_batch(flv, batch_size, key)
    loss, params, opt_state = step(params, opt_state, x, y)
    #loss, params = update(params, x, y)
    losses.append(loss)
    _, key = jrandom.split(key)
    
plt.loglog(losses)

# %%
# plot last batch of data samples (inputs and targets) 

fig = plt.figure(constrained_layout=True, figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*x.T, y, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension)

ax.set_xlabel('l [L]')
ax.set_ylabel('v [L/T]')
ax.set_zlabel('FLV [F]')
ax.set_zlim(0, max_tension)
ax.set_box_aspect(None, zoom=0.8)
ax.invert_xaxis()

# %%
fig = plt.figure(constrained_layout=True, figsize=(10,5))
axs = [fig.add_subplot(131, projection='3d'), 
       fig.add_subplot(132, projection='3d'),
       fig.add_subplot(133, projection='3d')]

# in case of (e.g.) an input batch norm layer, need to collapse into a single batch dimension
eval_input = jnp.stack([grid_l, grid_v], axis=-1).reshape((-1, 2))
tension_model = batched_predict(params, eval_input).reshape((n_plot, n_plot))
tension_exact = flv(grid_l, grid_v).squeeze()

max_tension = max(jnp.max(tension_model), jnp.max(tension_exact)).item()
max_abs_error = jnp.max(jnp.abs(tension_model - tension_exact))
mean_abs_error = jnp.mean(jnp.abs(tension_model - tension_exact))

error = (tension_model - tension_exact)
tension_model = tension_model.squeeze()

for i, tension in enumerate([tension_model, tension_exact, error]):
    axs[i].plot_surface(grid_l, grid_v, tension, cmap=plt.get_cmap(CMAP_LABEL), vmin=0, vmax=max_tension)

for ax in axs:
    ax.set_xlabel('l [L]')
    ax.set_ylabel('v [L/T]')
    ax.set_zlabel('FLV [F]')
    ax.set_zlim(0, max_tension)
    ax.set_box_aspect(None, zoom=0.8)
    ax.invert_xaxis()

axs[0].set_title("1. Model")
axs[1].set_title("2. Exact")    
axs[2].set_title("1 - 2")

print("Max. abs. error: ", max_abs_error.item())
print("Mean abs. error: ", mean_abs_error.item())


# %%
