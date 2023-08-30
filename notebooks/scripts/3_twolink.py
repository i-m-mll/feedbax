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

from feedbax.utils import exp_taylor, sincos_derivative_signs


# %%
def torque(t, y, args):
    return jnp.array([-0.1, 0.1])

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
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

y0 = (jnp.array([np.pi / 5, np.pi / 3]), jnp.array([0., 0.]))
dt0 = 0.01  
args = None
sol = solve(y0, dt0, args)   


# %%
def nlink_angular_to_cartesian(theta, dtheta, nlink):
    angle_sum = jnp.cumsum(theta)  # links
    length_components = nlink.l * jnp.array([jnp.cos(angle_sum),
                                             jnp.sin(angle_sum)])  # xy, links
    xy_position = jnp.cumsum(length_components, axis=1)  # xy, links
    
    ang_vel_sum = jnp.cumsum(dtheta)  # links
    xy_velocity = jnp.cumsum(jnp.flip(length_components, (0,)) * ang_vel_sum
                             * sincos_derivative_signs(1),
                             axis=1)
    return xy_position, xy_velocity


# %%
xy_pos, xy_vel = jax.vmap(nlink_angular_to_cartesian, in_axes=[0, 0, None])(sol.ys[0], sol.ys[1], TwoLink())
xy_pos = np.pad(xy_pos.squeeze(), ((0,0), (0,0), (1,0)))


# %%
def plot_2D_positions(xy, links=True, cmap_func=mpl.cm.viridis,
                      unit='m', ax=None, add_root=True):
    """Plot 2D position trajectories.
    Could also plot the controls on the joints.
    Args:
        xy ():
        links (bool):
        cmap_func ():
    """
    if ax is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_subplot()

    if add_root:
        xy = np.pad(xy, ((0,0), (1,0), (0,0)))

    cmap = cmap_func(np.linspace(0, 0.66, num=xy.shape[0], endpoint=True))
    cmap = mpl.colors.ListedColormap(cmap)

    ax.plot(*xy[0], c=cmap(0.), lw=2, marker="o")
    ax.plot(*xy[len(xy)//2], c=cmap(0.5), lw=2, marker='o')
    ax.plot(*xy[-1], c=cmap(1.), lw=2, marker='o')

    for j in range(xy.shape[2]):
        ax.scatter(*xy[..., j].T, marker='.', s=4, linewidth=0, c=cmap.colors)

    ax.margins(0.1, 0.2)
    ax.set_aspect('equal')
    return ax


# %%
ax = plot_2D_positions(xy_pos, add_root=False)
plt.show()

# %% [markdown]
# Verifying that the segments stay constant length
#
# TODO: this should be a unit test for angular_to_cartesian

# %%
distances = np.sqrt(np.sum(np.diff(xy_pos, axis=2)**2, axis=1)) - twolink.l
print(distances, '\n')
print("Mean difference from actual length: ", np.mean(distances, axis=0))
print("% difference from actual length: ", 100 * np.mean(distances, axis=0) / twolink.l)
print("St. dev. difference from actual length: ", np.std(distances, axis=0))


# %% [markdown]
# ### Linearization

# %%
class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
