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
#     display_name: fbx
#     language: python
#     name: python3
# ---

# %%
import math

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm


# %% [markdown]
# Reproduction of [this example](http://docs.kidger.site/equinox/examples/train_rnn) from the equinox docs.
#
# Classify spirals as CW or CCW.

# %%
def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, *, key):
    t = jnp.linspace(0, 2 * math.pi, 16)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y


# %%
# plot example data sample
data_key, model_key = jrandom.split(jrandom.PRNGKey(5678), 2)
xs, ys = get_data(100, key=data_key)
plt.plot(*xs[0].T)


# %%
class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array
    
    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)
        
    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))
        
        def f(carry, inp):
            return self.cell(inp, carry), None
        
        out, _ = lax.scan(f, hidden, input)
        
        return jax.nn.sigmoid(self.linear(out) + self.bias)


# %%
def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    hidden_size=16,
    depth=1,
    seed=5678,
):
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size)
    
    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)
    
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        y_pred = jax.vmap(model)(x)
        # binary cross-entropy
        return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(tqdm(range(steps)), iter_data):
        loss, model, opt_state = make_step(model, opt_state, x, y)
        loss = loss.item()
        if step % 10 == 0:
            print(f"step: {step}, loss: {loss:.4f}")
    
    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final accuracy: {final_accuracy:.4f}")


# %%
main()
