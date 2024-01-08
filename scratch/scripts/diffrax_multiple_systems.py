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

# %% [markdown]
# This notebook is to examine how diffrax behaves when multiple systems are solved at once.

# %%
import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm

from feedbax.mechanics.skeleton import PointMass
from feedbax.state import CartesianState2D
from feedbax.utils import exp_taylor

# %%
N_DIM = 2
ORDER = 2

# %%
# #! use the version from feedbax
sys1 = PointMass(mass=1)

# %%
sys2 = PointMass(mass=2)

# %% [markdown]
# Simulate the system with a time-dependent force

# %%
from copy import deepcopy

init_state1 = CartesianState2D(
    pos=jnp.array([0.0, 0.0]),
    vel=jnp.array([1.0, 0.0]),
)

init_state2 = init_state1.copy()

init_state = (init_state1, init_state2)


# %%
def solve(y0, t0, t1, dt0, args):
    term = (dfx.ODETerm(sys1), dfx.ODETerm(sys2))
    solver = dfx.SemiImplicitEuler()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 100))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

t0 = 0
dt0 = 0.01  
t1 = 1
args = jnp.array([-0.5, 0.5])
sol = solve(init_state, t0, t1, dt0, args)   

# %%
# plot the simulated position over time
cmap = plt.get_cmap('viridis')
cs = [cmap(i) for i in sol.ts * 0.9]  # 0.9 cuts off yellows

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].scatter(*sol.ys[0].pos.T, c=cs, s=1)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_aspect('equal')
axs[1].scatter(*sol.ys[0].vel.T, c=cs, s=1) 
axs[1].set_xlabel('$v_x$')
axs[1].set_ylabel('$v_y$')
axs[1].set_aspect('equal')


# %% [markdown]
# This... is incorrect, I think. And `dfx.Euler` doesn't work with a PyTree of `ODETerm`. So I think in general, we can't use a single solver with multiple `ODETerm`s. We need to write the entire system as a single vector field, and wrap it in a single `ODETerm`.

# %%
# %timeit solve(y0, t0, t1, dt0, args)

# %%
# simulation performance with and without JIT for different step sizes
# %timeit solve(y0, 0.1, args)  
# %timeit jax.jit(solve)(y0, 0.1, args) 
# %timeit solve(y0, 0.01, args)  
# %timeit jax.jit(solve)(y0, 0.01, args) 
# %timeit solve(y0, 0.001, args)  
# %timeit jax.jit(solve)(y0, 0.001, args)  

# %% [markdown]
# We can also [step through](https://docs.kidger.site/diffrax/usage/manual-stepping/) the solution:

# %%
# #! this takes much longer, ~17 s for 100 steps 
def diffeqsolve_loop_old(term, solver, t0, t1, dt0, y0, args):
    
    tprev = t0
    tnext = t0 + dt0
    state = solver.init(term, tprev, tnext, y0, args)
    ys = [y0]
    ts = [t0]
    
    while tprev < t1:
        y, _, _, state, _ = solver.step(term, tprev, tnext, ys[-1], args, state, made_jump=False)
        ys.append(y)
        #print(f"At t={tnext:0.2f}, y={y}")
        tprev = tnext
        ts.append(tprev)
        tnext = min(tprev + dt0, t1)  
    
    return ys, ts


# #! JIT-compatible version, ~0.2 s for 100 steps
@eqx.filter_jit
def diffeqsolve_loop(term, solver, t0, t1, dt0, y0, args):
    
    steps = int(t1 // dt0) + 1
    ys = jnp.zeros((steps, y0.shape[0]))
    ys = ys.at[0].set(y0)
    
    state = solver.init(term, t0, t1 + dt0, y0, args)
    init_val = ys, state

    def body_fn(i, x):
        ys, state = x
        y, _, _, state, _ = solver.step(term, 0, dt0, ys[i], args, state, made_jump=False)
        ys = ys.at[i+1].set(y)
        return ys, state
    
    ys, state = jax.lax.fori_loop(0, steps, body_fn, init_val)
    
    return ys


    

# %%
def solve_loop(y0, t0, t1, dt0, args):
    term = dfx.ODETerm(sys.vector_field)
    solver = dfx.Tsit5()
    sol = diffeqsolve_loop(term, solver, t0, t1, dt0, y0, args=args)
    return sol


# %%
t0 = 0
dt0 = 0.01  
t1 = 1
y0 = jnp.array([0., 0., 0.5, 0.])  
args = jnp.array([-0.5, 0.5])

ys = solve_loop(y0, t0, t1, dt0, args)

# %%
# %timeit solve_loop(y0, t0, t1, dt0, args)

# %%
ys = jnp.vstack(ys)
# plot the simulated position over time
fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].scatter(*ys[:, :2].T, s=1)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_aspect('equal')
axs[1].scatter(*ys[:, 2:].T, s=1) 
axs[1].set_xlabel('$v_x$')
axs[1].set_ylabel('$v_y$')
axs[1].set_aspect('equal')

# %%
