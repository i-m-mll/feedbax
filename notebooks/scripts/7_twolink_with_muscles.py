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
from jaxtyping import Float, Array
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax 
from tqdm import tqdm

from feedbax.mechanics.arm import TwoLink, nlink_angular_to_cartesian
from feedbax.mechanics.muscle import MuscleGroup
from feedbax.plot import plot_2D_positions

# %%
from jax import config

config.update("jax_debug_nans", False)

# %%
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

def muscle_l(theta, muscles):
    M, theta0, l0 = muscles.M, muscles.theta0, muscles.l0
    l = 1 + (M[0] * (theta0[0] - theta[0]) + M[1] * (theta0[1] - theta[1])) / l0
    return l

def muscle_v(d_theta, muscles):
    # muscle velocity
    M, l0 = muscles.M, muscles.l0
    v = (M[0] * d_theta[0] + M[1] * d_theta[1]) / l0
    return v

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

def tension_lt2004(theta, d_theta, muscles):
    """Simple helper to take joint configuration as input."""
    l = muscle_l(theta, muscles)
    v = muscle_v(d_theta, muscles)
    return tension_from_lv_lt2004(l, v)


# %%
tau_act = 50  # [ms]
tau_deact = 66  # [ms]

def torque(t, y, args):
    muscles, _ = args
    theta, d_theta, activation = y
    torque = muscles.M @ (activation * tension_lt2004(theta, d_theta, muscles))
    return torque


def muscle_activation_field(t, y, args):
    """Approximation of muscle activation (calcium) dynamics from Todorov & Li 2004.
    
    Just a simple filter.
    """
    activation = y
    u = args  
    
    # TODO: assuming tau_act and tau_deact aren't passed as args; e.g. if this is a method of a dataclass
    tau = tau_deact + jnp.where(u < activation, u, jnp.zeros(1)) * (tau_act - tau_deact)
    d_activation = (u - activation) / tau
    
    return d_activation


# TODO: separate dd_dtheta calculation from composite field
# def twolink_field(t, y, args):
#     theta, d_theta = y


def twolink_field(twolink):
    def field(t, y, args):
        theta, d_theta, activation = y 
        _, u = args  # TODO: what is the best way to pass the input? (if interactively iterating the solver, can pass `args` at each step)
        
        d_activation = muscle_activation_field(t, activation, u)
        
        # centripetal and coriolis forces 
        c_vec = jnp.array((
            -d_theta[1] * (2 * d_theta[0] + d_theta[1]),
            d_theta[0] ** 2
        )) * twolink.a2 * jnp.sin(theta[1])  
        
        cs1 = jnp.cos(theta[1])
        tmp = twolink.a3 + twolink.a2 * cs1
        inertia_mat = jnp.array(((twolink.a1 + 2 * twolink.a2 * cs1, tmp),
                                 (tmp, twolink.a3 * jnp.ones_like(cs1))))
        
        net_torque = torque(t, y, args) - c_vec.T - jnp.matmul(d_theta, twolink.B.T) # - viscosity * state.d_theta
        
        dd_theta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return d_theta, dd_theta, d_activation
    
    return field


def solve(field, y0, dt0, t0, t1, args):
    term = dfx.ODETerm(field)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol


# %%
y0 = (jnp.array([np.pi / 5, np.pi / 3]), 
      jnp.array([0., 0.]),
      jnp.zeros(6))
u = jnp.array([0., 0., 0., 0., 1e-4, 0.])
t0 = 0
dt0 = 1  # [ms]
t1 = 1000
field = twolink_field(TwoLink())
args = (MuscleGroup(), u)
sol = solve(field, y0, dt0, t0, t1, args)   

# %%
xy_pos, xy_vel = jax.vmap(nlink_angular_to_cartesian, in_axes=[0, 0, None])(sol.ys[0], sol.ys[1], TwoLink())
xy_pos = np.pad(xy_pos.squeeze(), ((0,0), (0,0), (1,0)))

# %%
ax = plot_2D_positions(xy_pos, add_root=False)
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
dt0 = 1
t1 = 600
ts = jnp.array([t0, t1/100, t1/100 + dt0, t1/2, t1/2 + dt0, t1])

us = jnp.array([0,0,1,1,0,0])
us = us * jnp.array([0.1, 0.25, 0.5, 1.0])[:,None]
u_t = dfx.LinearInterpolation(ts, us.T)

y0 = jnp.array([0., 0., 0., 0.]) 

field = muscle_activation_field
args = u_t
sol = solve(field, y0, dt0, t0, t1, args)  

# %%
plt.plot(sol.ts, sol.ys)
plt.ylabel("Muscle activation")
plt.xlabel("Time [ms]")

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
