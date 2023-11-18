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

from feedbax.mechanics.linear import point_mass
from feedbax.types import CartesianState2D
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
    
    def vector_field(self, t, state, input_):
        d_state = self.A @ state + self.B @ input_
        return d_state

def point_mass_old(mass=1):
    A = sum([jnp.diagflat(jnp.ones((ORDER - i) * N_DIM), i * N_DIM)
             for i in range(1, ORDER)])
    B = jnp.concatenate([jnp.zeros((N_DIM, N_DIM)), jnp.eye(N_DIM) / mass], axis=0)
    C = jnp.array(1)
    return LTISystem(A, B, C)

sys = point_mass_old(mass=1)

# %%
# #! use the version from feedbax
sys = point_mass(mass=1)


# %% [markdown]
# Simulate the system with a time-dependent force

# %%
def solve(y0, t0, t1, dt0, args):
    term = dfx.ODETerm(sys.vector_field)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 100))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

y0 = CartesianState2D(
    pos=jnp.array([-20., -2.]),
    vel=jnp.array([0.5, 0.]),
)  
t0 = 0
dt0 = 0.01  
t1 = 1
args = jnp.array([-0.5, 0.5])
sol = solve(y0, t0, t1, dt0, args)   

# %%
# plot the simulated position over time
cmap = plt.get_cmap('viridis')
cs = [cmap(i) for i in sol.ts * 0.9]  # 0.9 cuts off yellows

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].scatter(*sol.ys.pos.T, c=cs, s=1)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_aspect('equal')
axs[1].scatter(*sol.ys.vel.T, c=cs, s=1) 
axs[1].set_xlabel('$v_x$')
axs[1].set_ylabel('$v_y$')
axs[1].set_aspect('equal')


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
