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
from typing import Any

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
from tqdm import tqdm

from feedbax.mechanics.skeleton import TwoLink
from feedbax.mechanics.muscle import (
    LillicrapScottVirtualMuscle, 
    TodorovLiVirtualMuscle, 
    ActivationFilter,
)
from feedbax.mechanics.muscled_arm import TwoLinkMuscled
from feedbax.plot import plot_2D_joint_positions

# %%
jax.config.update("jax_debug_nans", False)


# %%
def solve(field, y0, dt0, t0, t1, args, **kwargs):
    term = dfx.ODETerm(field)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat, **kwargs)
    return sol


# %%
arm2M = TwoLinkMuscled(
    muscle_model=TodorovLiVirtualMuscle(),
    activator=ActivationFilter(),
)

y0 = (jnp.array([np.pi / 5, np.pi / 3]), 
      jnp.array([0., 0.]),
      jnp.zeros(6))
u = jnp.array([0., 0., 0., 0., 0.0, 0.00015])
t0 = 0
dt0 = 1  # [ms]
t1 = 1000
with jax.default_device(jax.devices('cpu')[0]):
    sol = solve(arm2M.vector_field, y0, dt0, t0, t1, u)   

# %%
# TODO: use the forward kinematics built into TwoLink
xy_pos, xy_vel = eqx.filter_vmap(nlink_angular_to_cartesian)(TwoLink(), sol.ys[0], sol.ys[1])
#xy_pos = np.pad(xy_pos.squeeze(), ((0,0), (0,0), (1,0)))

# %%
ax = plot_2D_joint_positions(xy_pos, lw_arm=4, add_root=True)
plt.show()

# %% [markdown]
# Testing the activation dynamics. See Fig 1D in Todorov & Li 2004.

# %%
tau_act = 50 # [ms]
tau_deact = 66 # [ms]

def muscle_activation_field(t, y, args):
    """Approximation of muscle activation (calcium) dynamics from Todorov & Li 2004.
    
    Just a simple filter.
    """
    activation = y
    u_t = args  # switching from before to use an interpolated step up-down
    u = u_t.evaluate(t)
    
    # TODO: assuming tau_act and tau_deact aren't passed as args; e.g. if this is a method of a dataclass
    tau = tau_deact + jnp.where(u < activation, u, jnp.zeros(1)) * (tau_act - tau_deact)
    d_activation = (u - activation) / tau
    
    return d_activation


# %%
t0 = 0
dt0 = 0.1
t1 = 600
ts = jnp.array([t0, t1/100, t1/100 + dt0, t1/2, t1/2 + dt0, t1])

us = jnp.array([0,0,1,1,0,0])
us = us * jnp.array([0.1, 0.25, 0.5, 1.0])[:,None]
u_t = dfx.LinearInterpolation(ts, us.T)

y0 = jnp.array([0., 0., 0., 0.]) 

field = muscle_activation_field
args = u_t
sol = solve(field, y0, dt0, t0, t1, args, max_steps=10000)  

# %%
plt.plot(sol.ts, sol.ys)
plt.ylabel("Muscle activation")
plt.xlabel("Time [ms]");

# %% [markdown]
# The shape doesn't quite match the figure from Todorov & Li 2004... the lower curves look similar, but the upper curve plateaus more quickly in their figure.
#
# I chose the magnitude of the inputs `u` arbitrarily. If I choose a value higher than 1, the muscle activation will exceed 1. But perhaps it should saturate at 1. In that case, maybe their plateauing curve is due to a larger input, plus saturation? But they never mentioned anything like that. 
#
# It's probably not a big deal, overall the dynamics are very similar and it could just be a matter of the exact values I chose and how I plotted it.

# %% [markdown]
# ### Use FLV approximation
#
# TODO

# %%
