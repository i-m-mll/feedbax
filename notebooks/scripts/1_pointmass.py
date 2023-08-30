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

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm

from feedbax.utils import exp_taylor

# %%
N_DIM = 2
ORDER = 2


# %%
# #! this is the discrete-time version, we want the continuous-time one  
# def point_mass(order: int = 2, n_dim: int = 2, dt: float = 0.1) -> LTISystem:
#     A = sum([jnp.diagflat(jnp.array([t] * (order - i) * n_dim), i * n_dim)
#              for i, t in enumerate(exp_taylor(dt, order))])
#     B_terms = exp_taylor(dt, 3)[::-1] + [0] * (order - 3)
#     B = jnp.concatenate([term * jnp.eye(n_dim) for term in B_terms[:order]], 
#                         axis=0)
#     C = jnp.array(1)
#     return LTISystem(A, B, C)

# def step_lti_system(sys: LTISystem, y: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
#    return sys.A @ y + sys.B @ u

# %%
# https://docs.kidger.site/diffrax/examples/kalman_filter/

class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray

def point_mass(mass=1):
    A = sum([jnp.diagflat(jnp.ones((ORDER - i) * N_DIM), i * N_DIM)
             for i in range(1, ORDER)])
    B = jnp.concatenate([jnp.zeros((N_DIM, N_DIM)), jnp.eye(N_DIM) / mass], axis=0)
    C = jnp.array(1)
    return LTISystem(A, B, C)

sys = point_mass(mass=2)


# %% [markdown]
# Simulate the system with a time-dependent force

# %%
def force(t, y, args):
    return jnp.array([t, -t])

def lti_field(sys):
    def field(t, y, args):
        return sys.A @ y + sys.B @ force(t, y, args)
    
    return field

def solve(y0, dt0, args):
    term = dfx.ODETerm(lti_field(sys))
    solver = dfx.Tsit5()
    t0 = 0
    t1 = 1
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

y0 = jnp.array([0., 0., 0.1, 0.1])  
dt0 = 0.01  
args = None
sol = solve(y0, dt0, args)   

# %%
# simulation performance with and without JIT for different step sizes
# %timeit solve(y0, 0.1, args)  
# %timeit jax.jit(solve)(y0, 0.1, args) 
# %timeit solve(y0, 0.01, args)  
# %timeit jax.jit(solve)(y0, 0.01, args) 
# %timeit solve(y0, 0.001, args)  
# %timeit jax.jit(solve)(y0, 0.001, args)  

# %%
# plot the simulated position over time
plt.plot(*sol.ys[:, :2].T)

# %% [markdown]
# We can also [step through](https://docs.kidger.site/diffrax/usage/manual-stepping/) the solution:

# %%
term = dfx.ODETerm(lti_field(sys))
solver = dfx.Tsit5()

t0 = 0
dt0 = 0.01  
t1 = 1
y0 = jnp.array([0.1, 0.1, 0.1, 0.1])  
args = None

tprev = t0
tnext = t0 + dt0
y = y0
state = solver.init(term, tprev, tnext, y, args)

while tprev < t1:
    y, _, _, state, _ = solver.step(term, tprev, tnext, y, args, state, made_jump=False)
    print(f"At t={tnext:0.2f}, y={y}")
    tprev = tnext
    tnext = min(tprev + dt0, t1)  

# %%
