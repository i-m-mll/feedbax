"""Plotting utilities.

TODO:
- Some vmap-like option for plotting batches of trials on a single axis?

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import io
from itertools import zip_longest
import logging 
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Float, Array, Int
import matplotlib as mpl
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from feedbax import utils


logger = logging.getLogger(__name__)


def plot_2D_joint_positions(
        xy, 
        t0t1=(0, 1),  # (t0, t1)
        cmap="viridis",
        length_unit=None, 
        ax=None, 
        add_root=True,
        colorbar=True,
        ms_trace=6,
        lw_arm=4,
        workspace=None,
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

    cmap_func = plt.get_cmap(cmap)
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

    if workspace is not None:
        corners = utils.corners_2d(workspace)[:, jnp.array([0, 1, 3, 2, 0])]
        ax.plot(*corners, 'w--', lw=0.8)

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


def plot_3D_paths(
    paths: Float[Array, "batch steps 3"], 
    epoch_start_idxs: Int[Array, "batch epochs"],  
    epoch_linestyles: Tuple[str, ...],  # epochs
    cmap: str = 'tab10',
):
    """Plot a set/batch of 3D trajectories over time.
    
    Linestyle can be specified for each epoch, where the starting
    indices for the epochs are specified separately for each trajectory.
    """
    if not np.max(epoch_start_idxs) < paths.shape[1]:
        # TODO: what if it's one of those cases where the delay period goes until trial end?
        raise IndexError("epoch indices out of bounds")
    if not epoch_start_idxs.shape[1] == len(epoch_linestyles):
        raise ValueError("TODO")

    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, paths.shape[0])]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    for i in range(paths.shape[0]):
        for idxs, ls in zip(
            zip_longest(
                epoch_start_idxs[i], 
                epoch_start_idxs[i, 1:] + 1, 
                fillvalue=None
            ), 
            epoch_linestyles
        ):
            ts = slice(*idxs)
            ax.plot(*paths[i, ts, :].T, color=colors[i], lw=2, linestyle=ls)

    return fig, ax 


def plot_planes(
    x, 
    epoch_start_idxs,  
    epoch_linestyles,  # epochs
    # marker='-o', 
    lw=0.5, 
    ms=1, 
    cmap='tab10',
    init_color='black',
    label="PC",
):
    """Subplot for every consecutive pair of features (columns).
    
    TODO:
    - Subplot columns (with argument)
    """
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, x.shape[0])]
    
    n_planes = x.shape[-1] // 2
    fig, axs = plt.subplots(1, n_planes, figsize=(12, 5)) 
    
    for i, p in enumerate(range(0, x.shape[-1], 2)):
        for j in range(x.shape[0]):
            for k, (idxs, ls) in enumerate(zip(
                zip_longest(
                    epoch_start_idxs[j], 
                    epoch_start_idxs[j, 1:], 
                    fillvalue=None
                ), 
                epoch_linestyles
            )):                   
                if init_color is not None and k == 0:
                    color = init_color
                else:
                    color = colors[j]
                ts = slice(*idxs)
                axs[i].plot(x[j, ts, p].T, x[j, ts, p+1].T, 
                            lw=lw, ms=ms, linestyle=ls, color=color)
                axs[i].set_xlabel(f"{label:}{p:}")
                axs[i].set_ylabel(f"{label:}{p+1:}")

    plt.tight_layout()
    return axs,


def plot_states_forces_2d(
        positions: Float[Array, "batch time xy"], 
        velocities: Float[Array, "batch time xy"],
        forces: Float[Array, "batch time control"],
        endpoints: Optional[Float[Array, "startend batch xy"]] = None, 
        straight_guides=False,
        force_labels=None,
        force_label_type='linear',
        cmap='tab10',
        workspace=None,
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
    endpoints = jnp.asarray(endpoints)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    cmap_func = plt.get_cmap(cmap)
    colors = [cmap_func(i) for i in np.linspace(0, 1, positions.shape[0])]
   
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

    if force_labels is None:
        if force_label_type == 'linear':
            force_labels = ("Control force", r"$\mathrm{f}_x$", r"$\mathrm{f}_y$")
        elif force_label_type == 'torques':
            force_labels = ("Control torques", r"$\tau_1$", r"$\tau_2$")
        
    labels = [("Position", "$x$", "$y$"),
              ("Velocity", "$\dot x$", "$\dot y$"),
              force_labels]
    
    if workspace is not None:
        corners = utils.corners_2d(workspace)[:, jnp.array([0, 1, 3, 2, 0])]
        axs[0].plot(*corners, 'w--', lw=0.8)

    for i, (title, xlabel, ylabel) in enumerate(labels):
        axs[i].set_title(title)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_aspect('equal')
        
    plt.tight_layout()

    return fig, axs


def plot_activity_heatmap(
    activity: Float[Array, "time unit"],
    cmap: str = 'viridis',
):
    """Plot activity of network units over time."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    im = ax.imshow(activity.T, aspect='auto', cmap=cmap)
    ax.set_xlabel('Time')
    ax.set_ylabel('Unit')
    fig.colorbar(im)
    return fig, ax


def plot_activity_sample_units(
    activities: Float[Array, "batch time unit"],
    n_samples: int, 
    cols: int = 2, 
    cmap: str = 'tab10', 
    *, 
    key: jrandom.PRNGKeyArray
):
    """Plot activity of a random sample of units over time.
    
    TODO:
    - optional vlines for epoch boundaries
    - Could generalize this to sampling any kind of time series. Depends
      on if there is a standard way we shape our state arrays.
    """
    xlabel = 'time step'
    
    unit_idxs = jrandom.choice(
        key, jnp.arange(activities.shape[-1]), (n_samples,), replace=False
    )
    x = activities[..., unit_idxs]

    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, x.shape[0])]

    #if len(x.shape) == 3:
    rows = int(np.ceil(x.shape[-1] / cols))
    fig, axs = plt.subplots(rows, cols, sharey=True, sharex=True)
    # else:
    #     rows, cols = 1, 1
    #     fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    #     axs = [[axs]]
    #     x = [x]

    for i, unit_idx in enumerate(range(x.shape[-1])):
        row = i // cols
        col = i % cols
        unit = x[..., unit_idx]
        for j in range(unit.shape[0]):
            axs[row][col].plot(unit[j], color=colors[j], lw=1.5)
        axs[row][col].set_ylabel(
            "Unit {} output".format(unit_idxs[i])
        )
        axs[row][col].hlines(
            0, xmin=0, xmax=unit.shape[1], linewidths=1, linestyles='dotted'
        )

    for j in range(cols):
        axs[-1][-j-1].set_xlabel(xlabel)
        

def plot_loglog_losses(losses, losses_terms: dict = None):
    """Log-log plot of losses and component loss terms."""
    if losses_terms is None:
        losses_terms = dict()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.loglog(losses, 'white', lw=3)
    
    if losses_terms is not None:
        for loss_term in losses_terms.values():
            ax.loglog(loss_term, lw=0.75)
        
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    
    ax.legend(['Total', *losses_terms.keys()])
    
    return fig, ax


def plot_task_and_speed_profiles(
    velocity: Float[Array, "batch time xy"], 
    task_variables=None, 
    epoch_idxs=None,
    cmap='tab10',
):
    """For visualizing learned movements versus task structure.
    
    For example: does the network start moving before the go cue is given?
    """
    speeds = jnp.sqrt(jnp.sum(velocity**2, axis=-1))

    task_rows = len(task_variables)
    height_ratios = (1,) * task_rows + (task_rows,)

    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, speeds.shape[0])]

    fig, axs = plt.subplots(1 + task_rows, 1, height_ratios=height_ratios, 
                            sharex=True, constrained_layout=True)

    axs[-1].set_title('speed profiles')
    for i in range(speeds.shape[0]):
        axs[-1].plot(speeds[i].T, color=colors[i])

    ymax = 1.5 * jnp.max(speeds)
    if epoch_idxs is not None:
        axs[-1].vlines(epoch_idxs[:, 1], ymin=0, ymax=ymax, colors=colors, 
                       linestyles='dotted', linewidths=1, label='target ON')
        axs[-1].vlines(epoch_idxs[:, 3], ymin=0, ymax=ymax, colors=colors, 
                       linestyles='dashed', linewidths=1, label='fixation OFF')
        plt.legend()
    axs[-1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for i, (label, arr) in enumerate(task_variables.items()):
        arr = jnp.squeeze(arr)
        axs[i].set_title(label)
        for j in range(arr.shape[0]):
            axs[i].plot(arr[j].T, color=colors[j])
        axs[i].set_ylim(*utils.padded_bounds(arr))


def animate_arm2(
    xy: Float[Array, "time joints xy"],
    interval: int = 1,
):
    """Animated movement of a multi-segment arm.
    
    TODO:
    - n-link arm
    - don't hardcode the axes limits
    """
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

    return animation.FuncAnimation(
        fig, 
        animate, 
        frames=len(xy),
        interval=interval, 
        blit=True
    )
    


def animate_3D_rotate(
        fig, 
        ax, 
        azim_range: Tuple[int, int] = (0, 360), 
        elev: float = 10.,
        interval: int = 20,
) -> animation.FuncAnimation:
    """Rotate a 3D plot by `degrees` about the z axis."""
    def animate(i):
        ax.view_init(elev=elev, azim=i)
        return fig,

    frames = azim_range[1] - azim_range[0]

    return animation.FuncAnimation(
        fig, 
        animate, 
        frames=frames,
        interval=interval,
        blit=True,
    )


def plot_complex(x, fig=None, marker='o'):
    """Plot complex numbers as points in the complex plane."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(x.real, x.imag, marker)
    ax.axhline(c='grey')
    ax.axvline(c='grey')
    return fig, ax


def add_ax_labels(ax, labels):
    """Convenience function for when we have an iterable of axis labels, so we 
    don't have to call `set_xlabel`, `set_ylabel`, and maybe `set_zlabel`.
    
    Since `zip` is used, this will silently ignore any extra `labels`.
    """
    keys = ('xlabel', 'ylabel')
    if ax.name == '3d':
        keys = keys + ('zlabel',)
    ax.update(dict(zip(keys, labels)))
    return ax


def plot_hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.

    From https://matplotlib.org/3.1.0/gallery/specialty_plots/hinton_demo.html
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    return ax
    
    
def fig_to_img_arr(fig, norm=True):
    """
    Converts a matplotlib figure (given its handle) to an image represented
    by a numpy array, then closes the figure.

    Uses IO buffer for control over resolution etc.

    Note:
        Based on <https://stackoverflow.com/a/61443397>.
    """

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', facecolor='white', dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    img_arr = np.swapaxes(img_arr, 0, 2)  # in TF+TB version >= 1.8, RGB channel is dim 0
    img_arr = np.swapaxes(img_arr, 1, 2)  # rotate figure 90 deg

    plt.close(fig)

    return img_arr