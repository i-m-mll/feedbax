"""Plotting utilities."""

from typing import Optional

from jaxtyping import Float, Array
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def plot_2D_joint_positions(
        xy, 
        links=True, 
        cmap_func=mpl.cm.viridis,
        unit='m', 
        ax=None, 
        add_root=True
):
    """Plot paths of joint position for an n-link arm. 
    
    TODO: Could also plot the controls on the joints.
    
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

    ax.plot(*xy[0], c=cmap(0.), lw=2, marker="o")
    ax.plot(*xy[len(xy)//2], c=cmap(0.5), lw=2, marker='o')
    ax.plot(*xy[-1], c=cmap(1.), lw=2, marker='o')

    for j in range(xy.shape[2]):
        ax.scatter(*xy[..., j].T, marker='.', s=4, linewidth=0, c=cmap.colors)

    ax.margins(0.1, 0.2)
    ax.set_aspect('equal')
    return ax


def plot_states_forces_2d(
        states: Float[Array, "batch time state"], 
        forces: Float[Array, "batch time control"],
        endpoints: Optional[Float[Array, "batch xy startend"]] = None, 
        straight_guides=False,
        fig=None, 
        ms=3, 
        ms_source=6, 
        ms_target=7
):
    """Plot trajectories of position, velocity, force in 2D subplots.
    
    Intended for systems with Cartesian force control (e.g. point mass).
    
    - [x, y, v_x, v_y] in last dim of `states`; [f_x, f_y] in last dim of `forces`.
    - First dim is batch, second dim is time step.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, states.shape[0])]
   
    for i in range(states.shape[0]):
        # position and 
        axs[0].plot(states[i, :, 0], states[i, :, 1], '.', color=colors[i], ms=ms)
        if endpoints is not None:
            if straight_guides:
                axs[0].plot(*endpoints[:, i].T, linestyle='dashed', color=colors[i])
            axs[0].plot(*endpoints[0, i], linestyle='none', marker='s', fillstyle='none',
                        color=colors[i], ms=ms_source)
            axs[0].plot(*endpoints[1, i], linestyle='none', marker='o', fillstyle='none',
                        color=colors[i], ms=ms_target)
        
        # velocity
        axs[1].plot(states[i, :, 2], states[i, :, 3], '-o', color=colors[i], ms=ms)
        
        # force 
        axs[2].plot(forces[i, :, 0], forces[i, :, 1], '-o', color=colors[i], ms=ms)

    labels = [("Position", "$x$", "$y$"),
              ("Velocity", "$\dot x$", "$\dot y$"),
              ("Control force", "$\mathrm{f}_x$", "$\mathrm{f}_y$")]

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