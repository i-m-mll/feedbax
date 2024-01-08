# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm

from feedbax.networks import SimpleMultiLayerNet

# %% [markdown]
# Try a very simple function approximation, since I was having trouble getting it to converge for the muscle model FLV function.

# %%
batch_size = 1000
layer_sizes = [1, 10, 1]
learning_rate = 1.e-2
n_batches = 5000

model = SimpleMultiLayerNet(layer_sizes, key=jr.PRNGKey(0))
print(model)


# %%
def loss_fn(model, inputs, targets):
    preds = jax.vmap(model)(inputs)
    return jnp.mean((preds - targets) ** 2)

optimizer = optax.chain(
    # optax.clip(1.0),
    optax.adamw(learning_rate=learning_rate),
)

@eqx.filter_jit
def step(model, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# %%

key = jr.PRNGKey(678)

opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

losses = []

for t in tqdm(range(n_batches)):
    key1, key2 = jr.split(key)
    train_x = jr.uniform(key1, (batch_size, 1), minval=-1, maxval=1)
    train_y = train_x ** 4 + 0.1 * train_x * jr.normal(key2, (batch_size, 1))
    
    loss, model, opt_state = step(model, opt_state, train_x, train_y)
    losses.append(loss)
    _, key = jr.split(key)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(losses)
ax.set_ylim(bottom=1e-5);

# %%
fig = plt.figure()
train_x_sort = train_x.sort(axis=0)
ax = fig.add_subplot(111)
ax.scatter(train_x, train_y)
ax.plot(train_x_sort, jax.vmap(model)(train_x_sort), 'k-', lw=3)

# %%
