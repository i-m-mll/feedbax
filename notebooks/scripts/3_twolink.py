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
import seaborn as sns
from tqdm import tqdm

from feedbax.mechanics.arm import nlink_angular_to_cartesian
from feedbax.plot import plot_2D_joint_positions, plot_states_forces_2d
from feedbax.utils import SINCOS_GRAD_SIGNS


# %%
def torque(t, y, args):
    return jnp.array([0.1, 0.])

class TwoLink:
    l: Float[Array, "2"] = jnp.array((0.30, 0.33))  # [m] lengths of arm segments
    m: Float[Array, "2"] = jnp.array((1.4, 1.0))  # [kg] masses of segments
    I: Float[Array, "2"] = jnp.array((0.025, 0.045))  # [kg m^2] moments of inertia of segments
    s: Float[Array, "2"] = jnp.array((0.11, 0.16))  # [m] distance from joint center to segment COM
    B: Float[Array, "2 2"] = jnp.array(((0.05, 0.025),
                                         (0.025, 0.05))) # [kg m^2 s^-1] joint friction matrix
    inertia_gain: float = 1.    
    
    @property
    def a1(self):
        return self.I[0] + self.I[1] + self.m[1] * self.l[0] ** 2 # + m[1]*s[1]**2 + m[0]*s[0]**2
    
    @property
    def a2(self):
        return self.m[1] * self.l[0] * self.s[1]
    
    @property
    def a3(self):
        return self.I[1]  # + m[1] * s[1] ** 2


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
    term = dfx.ODETerm(twolink_field(TwoLink()))
    solver = dfx.Tsit5()
    t0 = 0
    t1 = 1
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    state = solver.init(term, t0, t1 + dt0, y0, args)
    return state
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

y0 = (jnp.array([np.pi / 5, np.pi / 3]), jnp.array([0., 0.]))
dt0 = 0.01  
args = None

with jax.default_device(jax.devices('cpu')[0]):
    sol = solve(y0, dt0, args)      

# %%

# %%
xy_pos, xy_vel = eqx.filter_vmap(nlink_angular_to_cartesian)(
    TwoLink(), sol.ys[0], sol.ys[1]
)

# %% [markdown]
# Plot the trajectory of the end of the arm, plus the (constant) control torques:

# %%
controls = jnp.tile(jnp.array([-0.1, 0.1]), (1000, 1))
position = xy_pos[..., -1]
velocity = xy_vel[..., -1]
data = (position, velocity, controls)
plot_states_forces_2d(*jax.tree_map(lambda x: x[None, :], data), force_label_type='torques')
plt.show()

# %% [markdown]
# Plot the position of the entire arm over time

# %%
ax = plot_2D_joint_positions(xy_pos, add_root=True)
plt.show()


# %% [markdown]
# Iterative solution

# %%
@eqx.filter_jit
def diffeqsolve_loop(term, solver, t0, t1, dt0, y0, args):
    
    steps = int(t1 // dt0) + 1
    ys = jnp.zeros((steps, 4))
    ys = ys.at[0, :2].set(y0[0])
    ys = ys.at[0, 2:].set(y0[1])
    
    state = solver.init(term, t0, t1 + dt0, y0, args)
    init_val = ys, state

    def body_fn(i, x):
        ys, state = x
        y, _, _, state, _ = solver.step(term, 0, dt0, (ys[i, :2], ys[i, 2:]), args, state, made_jump=False)
        ys = ys.at[i+1, :2].set(y[0])
        ys = ys.at[i+1, 2:].set(y[1])
        return ys, state
    
    ys, state = jax.lax.fori_loop(0, steps, body_fn, init_val)
    
    return ys


# %%
def solve_loop(y0, t0, t1, dt0, args):
    term = dfx.ODETerm(twolink_field(TwoLink()))
    solver = dfx.Tsit5()
    sol = diffeqsolve_loop(term, solver, t0, t1, dt0, y0, args=args)
    return sol


# %%
y0 = (jnp.array([np.pi / 5, np.pi / 3]), jnp.array([0., 0.]))
dt0 = 0.01  
t0 = 0
t1 = 1 

ys = solve_loop(y0, t0, t1, dt0, args)

# %% [markdown]
# Verifying that the segments stay constant length
#
# TODO: this should be a unit test for angular_to_cartesian

# %%
twolink = TwoLink()

distances = np.sqrt(np.sum(np.diff(xy_pos, axis=2)**2, axis=1)) - twolink.l
print(distances, '\n')
print("Mean difference from actual length: ", np.mean(distances, axis=0))
print("% difference from actual length: ", 100 * np.mean(distances, axis=0) / twolink.l)
print("St. dev. difference from actual length: ", np.std(distances, axis=0))

# %% [markdown]
# Inverse kinematics:

# %%
from feedbax.mechanics.arm import TwoLink, twolink_effector_pos_to_angles

# %%
batch_size = 5
pos = jnp.tile(jnp.array([0., 0.5]), (batch_size, 1))
twolink = TwoLink()
# %timeit jax.vmap(twolink_effector_pos_to_angles, 
                 in_axes=(None, 0))(twolink, pos)
