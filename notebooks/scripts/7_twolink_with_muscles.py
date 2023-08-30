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

config.update("jax_debug_nans", True)

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

def muscle_v(dtheta, muscles):
    # muscle velocity
    M, l0 = muscles.M, muscles.l0
    v = (M[0] * dtheta[0] + M[1] * dtheta[1]) / l0
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

def tension_lt2004(theta, dtheta, muscles):
    """Simple helper to take joint configuration as input."""
    l = muscle_l(theta, muscles)
    v = muscle_v(dtheta, muscles)
    return tension_from_lv_lt2004(l, v)


# %%
l = muscle_l(y0[0], muscles)
v = muscle_v(y0[1], muscles)
tension_lt2004(y0[0], y0[1], muscles)

# %%
twolink = TwoLink()
muscles = MuscleGroup()
u = jnp.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.1])

def torque(t, y, args):
    theta, dtheta = y
    torque = muscles.M @ (u * tension_lt2004(theta, dtheta, muscles))
    return torque


def twolink_field(twolink):
    def field(t, y, args):
        theta, dtheta = y 
        
        # centripetal and coriolis forces 
        c_vec = jnp.array((
            -dtheta[1] * (2 * dtheta[0] + dtheta[1]),
            dtheta[0] ** 2
        )) * twolink.a2 * jnp.sin(theta[1])  
        
        cs1 = jnp.cos(theta[1])
        tmp = twolink.a3 + twolink.a2 * cs1
        inertia_mat = jnp.array(((twolink.a1 + 2 * twolink.a2 * cs1, tmp),
                                 (tmp, twolink.a3 * jnp.ones_like(cs1))))
        
        net_torque = torque(t, y, args) - c_vec.T - jnp.matmul(dtheta, twolink.B.T) # - viscosity * state.dtheta
        
        ddtheta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return dtheta, ddtheta
    
    return field

def solve(y0, dt0, args):
    term = dfx.ODETerm(twolink_field(twolink))
    solver = dfx.Tsit5()
    t0 = 0
    t1 = 1
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

y0 = (jnp.array([np.pi / 5, np.pi / 3]), jnp.array([0., 0.]))
dt0 = 0.01  
args = None
sol = solve(y0, dt0, args)   

# %%
xy_pos, xy_vel = jax.vmap(nlink_angular_to_cartesian, in_axes=[0, 0, None])(sol.ys[0], sol.ys[1], TwoLink())
xy_pos = np.pad(xy_pos.squeeze(), ((0,0), (0,0), (1,0)))

# %%
ax = plot_2D_positions(xy_pos, add_root=False)
plt.show()

# %%
