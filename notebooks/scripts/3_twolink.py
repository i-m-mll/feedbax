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
# %load_ext autoreload
# %autoreload 2

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

from feedbax.mechanics.skeleton.arm import TwoLink, TwoLinkState
from feedbax.mechanics.plant import SimplePlant, PlantState
from feedbax.plot import plot_2D_joint_positions, plot_pos_vel_force_2D
from feedbax.utils import SINCOS_GRAD_SIGNS


# %%
def torque(t, y, args):
    return jnp.array([0.1, 0.])

class TwoLink_:
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


class State(eqx.Module):
    angle: Float[Array, "2"]
    d_angle: Float[Array, "2"]


def twolink_field(twolink):
    def field(t, state, args):
        angle, dtheta = state.angle, state.d_angle 
        
        # centripetal and coriolis forces 
        c_vec = jnp.array((
            -dtheta[1] * (2 * dtheta[0] + dtheta[1]),
            dtheta[0] ** 2
        )) * twolink.a2 * jnp.sin(angle[1])  
        
        cs1 = jnp.cos(angle[1])
        tmp = twolink.a3 + twolink.a2 * cs1
        inertia_mat = jnp.array(((twolink.a1 + 2 * twolink.a2 * cs1, tmp),
                                 (tmp, twolink.a3 * jnp.ones_like(cs1))))
        
        net_torque = torque(t, state, args) - c_vec.T - jnp.matmul(dtheta, twolink.B.T) # - viscosity * state.dtheta
        
        ddtheta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return State(dtheta, ddtheta)
    
    return field


# %%
def solve(vf, y0, t0, t1, dt0, args, n_save_steps=None):
    if n_save_steps is None:
        n_save_steps = int((t1 - t0) / dt0)
    
    term = dfx.ODETerm(vf)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, n_save_steps))
    # solver_state = solver.init(term, t0, t1 + dt0, y0, args)
    sol = dfx.diffeqsolve(
        term, 
        solver, 
        t0, 
        t1, 
        dt0, 
        y0, 
        args=args, 
        saveat=saveat,
    )
    return sol


# %%
arm = TwoLink()

t0 = 0
t1 = 1
dt0 = 0.01  
n_steps = int((t1 - t0) / dt0)
n_save_steps = n_steps
y0 = TwoLinkState(
    angle=jnp.array([np.pi / 5, np.pi / 3]), 
    d_angle=jnp.array([0., 0.]),
)
args = input_torque = jnp.array([0.1, 0.])

with jax.default_device(jax.devices('cpu')[0]):
    sol = solve(arm, y0, t0, t1, dt0, args, n_save_steps=n_save_steps)      

# %%
xy = eqx.filter_vmap(arm.forward_kinematics)(sol.ys)

# %% [markdown]
# Plot the trajectory of the end of the arm, plus the (constant) control torques:

# %%
plot_pos_vel_force_2D(
    xy, 
    leaf_func=lambda states: (
        states.pos[None, :, -1],
        states.vel[None, :, -1],
        jnp.tile(input_torque, (1, 1000, 1)),
    ),
    force_label_type='torques',
)
plt.show()

# %% [markdown]
# Plot the position of the entire arm over time

# %%
ax = plot_2D_joint_positions(xy.pos, add_root=True)
plt.show()

# %% [markdown]
# Repeat the solution, but wrapping the arm in `SimplePlant`. The `vector_field` of the resulting instance should behave identically to the original arm.

# %%
plant = SimplePlant(arm)

# %%
arm = TwoLink()

t0 = 0
t1 = 1
dt0 = 0.01  
n_steps = int((t1 - t0) / dt0)
n_save_steps = n_steps
y0 = PlantState(
    skeleton=TwoLinkState(
        angle=jnp.array([np.pi / 5, np.pi / 3]), 
        d_angle=jnp.array([0., 0.]),
    ),
)
args = input_torque = jnp.array([0.1, 0.])

with jax.default_device(jax.devices('cpu')[0]):
    sol = solve(plant.vector_field, y0, t0, t1, dt0, args, n_save_steps=n_save_steps)      

# %%
xy = eqx.filter_vmap(arm.forward_kinematics)(
    sol.ys.skeleton
)

ax = plot_2D_joint_positions(xy.pos, add_root=True)
plt.show()


# %% [markdown]
# Iterative solution
#
# TODO: update this to work with `TwoLinkState`

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
# add root joint at (0, 0)
pos = np.pad(xy.pos, ((0, 0), (1, 0), (0, 0)))

distances = np.sqrt(np.sum(np.diff(pos, axis=1) ** 2, axis=2)) - arm.l
print(distances, '\n')
print("Mean difference from actual length: ", np.mean(distances, axis=0))
print("% difference from actual length: ", 100 * np.mean(distances, axis=0) / arm.l)
print("St. dev. difference from actual length: ", np.std(distances, axis=0))

# %% [markdown]
# Inverse kinematics:

# %%
batch_size = 5
pos = jnp.tile(jnp.array([0., 0.5]), (batch_size, 1))
twolink = TwoLink()
# %timeit jax.vmap(twolink_effector_pos_to_angles, 
                 in_axes=(None, 0))(twolink, pos)
