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
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# %% [markdown]
# ### `jax.grad` examples

# %% [markdown]
# simple example

# %%
def func(x):
    return x ** 2 

grad_func = jax.grad(func)

# we know that the gradient of x**2 is 2*x
grad_func(3.0)


# %% [markdown]
# pytree example

# %%
class State(eqx.Module):
    x: float 
    y: float

def f(state):
    return state.x ** 2 + state.y ** 3

# we know df/dx = 2 * x and df/dy = 3 * y**2
grad_f = jax.grad(f)  

# %%
state = State(x=3.0, y=4.0)

grad_f(state)

# %%
eqx.tree_pprint(grad_f(state), short_arrays=False)

# %% [markdown]
# ### `jax.vmap` examples

# %%
func = lambda x: x**2
grad_func = jax.grad(func)

xs = jnp.arange(10, dtype=float)
ys = jax.vmap(func)(xs)
dy_dxs = jax.vmap(grad_func)(xs)  

# %%
xs

# %%
ys

# %%
dy_dxs


# %% [markdown]
# with PyTrees

# %%

# %%
class State(eqx.Module):
    x: float 
    y: float


def get_state(key):
    
    key1, key2 = jr.split(key)
    
    return State(
        x=jr.uniform(key1, ()), 
        y=jr.normal(key2, ()),
    )  

seed = 1234
key = jr.PRNGKey(seed)

state = get_state(key)

# %%
eqx.tree_pprint(state, short_arrays=False, width=50)

# %%
n_states = 10
keys = jr.split(key, n_states)
states = jax.vmap(get_state)(keys)

# %%
eqx.tree_pprint(states, short_arrays=False, width=50)


# %%
def f(state):
    return state.x ** 2 + state.y ** 3

# we know df/dx = 2 * x and df/dy = 3 * y**2
grad_f = jax.grad(f)  

jax.vmap(grad_f)(states)

# %%
eqx.tree_pprint(jax.vmap(grad_f)(states), short_arrays=False)

# %% [markdown]
# ### Using `jax.eval_shape`

# %% [markdown]
# Recall:

# %%
jax.vmap(grad_f)(states)

# %% [markdown]
# However these are actual return values. Sometimes a computation is expensive, and before start to compute anything, we want to infer what the shapes of the return values will be. This is what `jax.eval_shape` is for:

# %%
jax.eval_shape(jax.vmap(grad_f), states)


# %% [markdown]
# The time difference between these is not super obvious here, because this is not a very expensive computation:

# %%
# %timeit jax.vmap(grad_f)(states)
# %timeit jax.eval_shape(jax.vmap(grad_f), states)

# %%
class RNN(eqx.Module):
    cell: eqx.nn.GRUCell

    def __init__(self, **kwargs):
        self.cell = eqx.nn.GRUCell(**kwargs)

    def __call__(self, xs):
        scan_fn = lambda state, input: (self.cell(input, state), None)
        init_state = jnp.zeros(self.cell.hidden_size)
        final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
        return final_state

rnn = RNN(input_size=50, hidden_size=1000, key=key)

n_examples = 50000
n_steps = 100
examples = jr.uniform(key, (n_examples, n_steps, 50))
    
# %time jax.vmap(rnn)(examples).block_until_ready()
# %timeit jax.eval_shape(jax.vmap(rnn), examples)

# %% [markdown]
# ### `jax.jit` and performance

# %%
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# %%
def multiply_matrices(key, shape1, shape2):
    key1, key2 = jr.split(key)
    mat1 = jr.normal(key1, shape1)
    mat2 = jr.normal(key2, shape2)
    return jnp.matmul(mat1, mat2)

multiply_many_matrices = jax.vmap(multiply_matrices, in_axes=(0, None, None))

seed = 1234
n_matrices = 100
shape1 = (10, 500)
shape2 = shape1[::-1]
keys = jr.split(jr.PRNGKey(seed), n_matrices)

# %timeit multiply_many_matrices(keys, shape1, shape2).block_until_ready()


# %%
multiply_many_matrices_jit = jax.jit(multiply_many_matrices, static_argnums=(1, 2))

# %time multiply_many_matrices_jit(keys, shape1, shape2).block_until_ready()

print()

# %timeit multiply_many_matrices_jit(keys, shape1, shape2).block_until_ready()
