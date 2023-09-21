"""Plotting utilities."""

from typing import Optional

from jaxtyping import Float, Array
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def plot_2D_joint_positions(
        xy, 
        t0t1=(0, 1),  # (t0, t1)
        cmap_func=mpl.cm.viridis,
        length_unit=None, 
        ax=None, 
        add_root=True,
        colorbar=True,
        ms_trace=6,
        lw_arm=4,
):
    """Plot paths of joint position for an n-link arm. 
    
    TODO: 
    - Plot the controls on the joints?
    
    Args:
        xy ():
        links (bool):
        cmap_func ():
    """
    if ax is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_subplot()

    if add_root:
        xy = np.pad(xy, ((0,0), (0,0), (1,0)))

    cmap = cmap_func(np.linspace(0, 0.66, num=xy.shape[0], endpoint=True))
    cmap = mpl.colors.ListedColormap(cmap)
    
    # arms: beginning, midpoint, and end of trajectory
    for i in (len(xy), 2, 1):
        idx = len(xy) // i - 1
        c = cmap(idx / len(xy))
        # segment lines
        ax.plot(*xy[idx, :], c=c, lw=lw_arm, ms=0)
        # mobile joints
        ax.plot(*xy[idx, :, 1:], c=c, lw=0, marker='o', ms=5)
        # root joint
        ax.plot(*xy[idx, :, 0], c=c, lw=lw_arm, marker='s', ms=7)

    # full joint traces along trajectory
    for j in range(xy.shape[2]):
        ax.scatter(*xy[..., j].T, 
                   marker='.', s=ms_trace, linewidth=0, c=cmap.colors)

    if colorbar:
        fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(*t0t1), cmap),
                     ax=ax, label='Time', location='bottom', ticks=[],
                     shrink=0.7, pad=0.1)  
    
    if length_unit is not None:
        ax.set_xlabel(f' [{length_unit}]')
        ax.set_ylabel(f' [{length_unit}]')

    ax.margins(0.1, 0.2)
    ax.set_aspect('equal')
    return ax


def plot_states_forces_2d(
        positions: Float[Array, "batch time xy"], 
        velocities: Float[Array, "batch time xy"],
        forces: Float[Array, "batch time control"],
        endpoints: Optional[Float[Array, "startend batch xy"]] = None, 
        straight_guides=False,
        force_label_type='linear',
        fig=None, 
        ms=3, 
        ms_source=6, 
        ms_target=7,
):
    """Plot trajectories of position, velocity, force in 2D subplots.
    
    Intended for systems with Cartesian force control (e.g. point mass).
    
    - [x, y, v_x, v_y] in last dim of `states`; [f_x, f_y] in last dim of `forces`.
    - First dim is batch, second dim is time step.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, positions.shape[0])]
   
    for i in range(positions.shape[0]):
        # position and 
        axs[0].plot(positions[i, :, 0], positions[i, :, 1], '.', color=colors[i], ms=ms)
        if endpoints is not None:
            if straight_guides:
                axs[0].plot(*endpoints[:, i].T, linestyle='dashed', color=colors[i])
            axs[0].plot(*endpoints[0, i], linestyle='none', marker='s', fillstyle='none',
                        color=colors[i], ms=ms_source)
            axs[0].plot(*endpoints[1, i], linestyle='none', marker='o', fillstyle='none',
                        color=colors[i], ms=ms_target)
        
        # velocity
        axs[1].plot(velocities[i, :, 0], velocities[i, :, 1], '-o', color=colors[i], ms=ms)
        
        # force 
        axs[2].plot(forces[i, :, 0], forces[i, :, 1], '-o', color=colors[i], ms=ms)

    if force_label_type == 'linear':
        force_labels = ("Control force", r"$\mathrm{f}_x$", r"$\mathrm{f}_y$")
    elif force_label_type == 'torques':
        force_labels = ("Control torques", r"$\tau_1$", r"$\tau_2$")
        
    labels = [("Position", "$x$", "$y$"),
              ("Velocity", "$\dot x$", "$\dot y$"),
              force_labels]

    for i, (title, xlabel, ylabel) in enumerate(labels):
        axs[i].set_title(title)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_aspect('equal')
        
    plt.tight_layout()

    return fig, axs


def plot_loglog_losses(losses, losses_terms, loss_term_labels=[]):
    """Log-log plot of losses and component loss terms."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.loglog(losses, 'white', lw=3)
    
    if losses_terms is not None:
        ax.loglog(losses_terms, lw=0.75)
        
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    
    ax.legend(['Total', *loss_term_labels])
    
    return fig, ax


def animate_arm2(xy):
    """xy: (n_seq, n_links, n_dim)"""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    # TODO: set limits based on arm geometry and angle limits
    ax.set_xlim([-0.4, 0.75])
    ax.set_ylim([-0.35, 0.65])

    arm_line, = ax.plot(*xy[0].T, 'k-')
    traj_line, = ax.plot(*xy[0, :, 2].T, 'g-')

    def animate(i):
        arm_line.set_data(*xy[i].T)
        traj_line.set_data(*xy[:i+1, :, 2].T)
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=len(xy),
                                interval=1, blit=True)
    return anim