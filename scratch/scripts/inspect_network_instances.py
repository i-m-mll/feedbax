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

# %% [markdown]
# Sometimes we want to tell the difference between neural network instances without pre-specifying what their types should be. Then it may be appropriate to inspect the parameters of their `__call__` method.

# %% [markdown]
#

# %%
import inspect 
import equinox as eqx 
import jax.random as jr

gru = eqx.nn.GRUCell(2, 3, key=jr.PRNGKey(0))

sig_gru = inspect.signature(gru)

lin = eqx.nn.Linear(2, 3, key=jr.PRNGKey(0))

sig_lin = inspect.signature(lin)

# %%
sig_gru.parameters 

# %%
'hidden' in sig_gru.parameters

# %%
sig_lin.parameters

# %%
'hidden' in sig_gru.parameters


# %%
def n_positional(sig):
    return sum(1 for param in sig.parameters.values() 
               if param.kind == param.POSITIONAL_OR_KEYWORD)

n_positional(sig_gru)

# %%
n_positional(sig_lin)
