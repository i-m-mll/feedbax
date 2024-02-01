"""Plotting utilities.

TODO:
- Some vmap-like option for plotting batches of trials on a single axis?
- Optional `fig` or `ax` argument that overrides figure generation

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import io
from itertools import zip_longest
import logging 
from typing import Any, Optional, Tuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Array, Int, PyTree
import matplotlib as mpl
from matplotlib import animation, gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feedbax.loss import LossDict
from feedbax.state import CartesianState2D
from feedbax.misc import corners_2d

if TYPE_CHECKING:
    from feedbax.trainer import TaskTrainerHistory

logger = logging.getLogger(__name__)


class SeabornFig2Grid: 
    """Inserts a seaborn plot as a subplot in a matplotlib figure.
    
    Certain seaborn plots create entire figures rather than plot on existing
    axes, which means they cannot be used directly to plot in a subplot grid.
    This class decomposes a seaborn plot's gridspec and inserts it in the 
    gridspec of a matplotlib figure.
    
    See `plot_endpoint_dists` for an example.
        
    From a stackoverflow answer by Luca Clissa:  https://stackoverflow.com/a/70666592
    """
    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """Move PairGrid or FacetGrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """Move JointGrid"""
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1, r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_2D_joint_positions(
        xy: Float[Array, "time links ndim=2"], 
        t0t1=(0, 1),  # (t0, t1)
        cmap: str = "viridis",
        length_unit=None, 
        ax=None, 
        add_root: bool = True,
        colorbar: bool = True,
        ms_trace: int = 6,
        lw_arm: int = 4,
        workspace: Optional[Float[Array, "bounds=2 xy=2"]] = None,
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
        xy = np.pad(xy, ((0, 0), (1, 0), (0, 0)))

    cmap_func = plt.get_cmap(cmap)
    cmap = cmap_func(np.linspace(0, 0.66, num=xy.shape[0], endpoint=True))
    cmap = mpl.colors.ListedColormap(cmap)
    
    # arms: beginning, midpoint, and end of trajectory
    for i in (len(xy), 2, 1):
        idx = len(xy) // i - 1
        c = cmap(idx / len(xy))
        # segment lines
        ax.plot(*xy[idx, :].T, c=c, lw=lw_arm, ms=0)
        # mobile joints
        ax.plot(*xy[idx, 1:].T, c=c, lw=0, marker='o', ms=5)
        # root joint
        ax.plot(*xy[idx, 0].T, c=c, lw=lw_arm, marker='s', ms=7)

    # full joint traces along trajectory
    for j in range(xy.shape[1]):
        ax.scatter(*xy[:, j].T, 
                   marker='.', s=ms_trace, linewidth=0, c=cmap.colors)

    if workspace is not None:
        corners = corners_2d(workspace)[:, jnp.array([0, 1, 3, 2, 0])]
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
    return fig, ax


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
    x: Float[Array, "batch time components"],
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


def plot_pos_vel_force_2D(
    states: PyTree[Float[Array, "batch time ..."] | Any],
    step: int = 1,  # plot every step-th trial
    leaf_func: Optional[Callable] = None,
    endpoints: Optional[Tuple[Float[Array, "batch xy"],
                              Float[Array, "batch xy"]]] = None, 
    straight_guides: bool = False,
    force_labels: Optional[Tuple[str, str, str]] = None,
    force_label_type: str = 'linear',
    cmap: str = 'viridis',
    color: Optional[str | Tuple[float, ...]] = None,
    colors: Optional[Sequence[str | Tuple[float, ...]]] = None,
    workspace: Optional[Float[Array, "bounds=2 xy=2"]] = None,
    fig=None, 
    ms: int = 3, 
    ms_source: int = 6, 
    ms_target: int = 7,
):
    """Plot trajectories of position, velocity, network output in 2D subplots.
    
    - [x, y, v_x, v_y] in last dim of `states`; [f_x, f_y] in last dim of `forces`.
    - First dim is batch, second dim is time step.
    """
    if leaf_func is None:
        positions, velocities, controls = (
            states.mechanics.effector.pos, 
            states.mechanics.effector.vel,
            states.network.output,
        )
    else:
        positions, velocities, controls = leaf_func(states)
        
    if endpoints is not None:
        endpoints = jnp.asarray(endpoints)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    if colors is None:
        cmap_func = plt.get_cmap(cmap)
        colors = [cmap_func(i) if color is None else color
                  for i in np.linspace(0, 1, positions.shape[0])]
   
    for i in range(0, positions.shape[0], step):
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
        
        # force (start at timestep 1; timestep 0 is always 0)
        axs[2].plot(controls[i, :, 0], controls[i, :, 1], '-o', color=colors[i], ms=ms)

    if force_labels is None:
        if force_label_type == 'linear':
            force_labels = ("Control force", r"$\mathrm{f}_x$", r"$\mathrm{f}_y$")
        elif force_label_type == 'torques':
            force_labels = ("Control torques", r"$\tau_1$", r"$\tau_2$")
        
    labels = [("Position", r"$x$", r"$y$"),
              ("Velocity", r"$\dot x$", r"$\dot y$"),
              force_labels]
    
    if workspace is not None:
        corners = corners_2d(workspace)[:, jnp.array([0, 1, 3, 2, 0])]
        axs[0].plot(*corners, 'w--', lw=0.8)

    for i, (title, xlabel, ylabel) in enumerate(labels):
        axs[i].set_title(title)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_aspect('equal')
        
    plt.tight_layout()

    return fig, axs


def plot_trajectories(
    states: PyTree[Float[Array, "batch time ..."] | Any],
    labels: Optional[Tuple[str, str, str]] = None,
    cmap: str = 'tab10',
    fig=None, 
    ms: int = 3, 
):
    """Plot trajectories of states.        
    """    
    state_arrays = jtu.tree_leaves(states, is_leaf=eqx.is_array)
    
    # TODO: clever row-col layout
    fig, axs = plt.subplots(1, len(state_arrays), figsize=(12, 6))

    cmap_func = plt.get_cmap(cmap)
    colors = [cmap_func(i) for i in np.linspace(0, 1, state_arrays[0].shape[0])]
   
    for j, array in enumerate(state_arrays):
        # Assumes constant batch size among state arrays.
        for i in range(state_arrays[0].shape[0]):
            axs[j].plot(
                array[i, 1:, 0], 
                array[i, 1:, 1], 
                '-o', 
                color=colors[i], 
                ms=ms
            )
        
        axs[j].set_aspect('equal')

    if labels is not None:
        # TODO: `labels` should be a PyTree with same structure as `states_tree`
        for i, (title, xlabel, ylabel) in enumerate(labels):
            axs[i].set_title(title)
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabel(ylabel)
            
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
    unit_includes = None,
    *, 
    key: jax.Array
):
    """Plot activity of a random sample of units over time.
    
    TODO:
    - optional vlines for epoch boundaries
    - Could generalize this to sampling any kind of time series. Depends
      on if there is a standard way we shape our state arrays.
    """
    xlabel = 'time step'
    
    unit_idxs = jr.choice(
        key, jnp.arange(activities.shape[-1]), (n_samples,), replace=False
    )
    if unit_includes is not None:
        unit_idxs = jnp.concatenate([unit_idxs, jnp.array(unit_includes)])
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
        
    return fig, axs
        

def plot_losses(
    train_history: "TaskTrainerHistory",
    xscale: str = 'log',
    yscale: str = 'log',
    cmap: str = 'Set1',
):
    """Line plot of loss terms and total.
    
    Each term in `losses` is an array where the first dimension is the
    training iteration, with an optional second batch dimension, e.g.
    for model replicates. Each term is plotted in a different color,
    with the same color used across the optional batch dimension.
    
    Uses 1-indexing for the training iteration, so that the first iteration
    is visible even when the x-axis is log-scaled.
    
    TODO:
    - Currently this assumes that any batch dimension (e.g. during ensembling)
      appears last, which is at odds with what happens elsewhere. So perhaps
      this function should assume the batch dimension would be first, check if
      it exists, and then transpose the data if necessary. Then we could return
      the losses in the shape `(replicates, iteration)` from `TaskTrainer`.
    """  
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    losses = train_history.loss
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, len(losses))]
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    xs = 1 + np.arange(len(losses.total))
        
    total = ax.plot(xs, losses.total, 'white', lw=3)
    
    all_handles = [total]
    
    for i, loss_term in enumerate(losses.values()):
        handles = ax.plot(xs, loss_term, lw=0.75, color=colors[i])
        all_handles.append(handles)
        
    ax.set_xlabel('Training iteration')
    ax.set_ylabel('Loss')
    
    ax.legend(
        # Only include the first plot for each loss term.
        [handles[0] for handles in all_handles],
        ['Total', *losses.keys()]
    )
    
    return fig, ax


def plot_mean_losses(
    losses: LossDict,
):
    """Similar to `plot_loss`, with mean-std over a batch dimension.
    
    TODO:
    - This is pretty slow. Maybe there's a way to construct only one 
    dataframe.
    """
    losses_terms_df = jax.tree_map(
        lambda losses: pd.DataFrame(
            losses.T, 
            index=range(losses.shape[1])
        ).melt(
            var_name='Time step', 
            value_name='Loss'
        ),
        dict(losses, total=losses.total),
    )
    
    fig, ax = plt.subplots()
    ax.set(xscale='log', yscale='log')
    
    for label, df in losses_terms_df.items():
        sns.lineplot(
            data=df, 
            x='Time step', 
            y='Loss', 
            errorbar='sd', 
            label=label, 
            ax=ax
        )
        
    return fig, ax


def plot_endpoint_pos_with_dists(
    trial_specs, 
    s=7,
    color=None,
    **kwargs
):
    """Plot the start and goal positions along with distributions.
    
    TODO:
    - Give the two joint plots the same axes limits.
    """
    if color is None:
        color = get_high_contrast_neutral_shade() 
        bbox_fc = dict(white='0.2', black='0.8')[color]
        bbox_ec = dict(white='0.8', black='0.2')[color]
    
    fig = plt.figure(figsize=(10, 5))

    endpoints = OrderedDict({
        'Start': trial_specs.init['mechanics.effector'], 
        'Goal': trial_specs.goal,
    })
    
    endpoint_pos_dfs = jax.tree_map(
        lambda arr: pd.DataFrame(arr.pos, columns=('x', 'y')),
        endpoints,
        is_leaf=lambda x: isinstance(x, CartesianState2D),
    )
    
    gs = gridspec.GridSpec(1, len(endpoint_pos_dfs))

    for i, (label, df) in enumerate(endpoint_pos_dfs.items()):
        joint_grid = sns.jointplot(
            data=df, 
            x='x', 
            y='y', 
            color=color,
            s=s,
            **kwargs,
        ) 
        joint_grid.ax_joint.annotate(
            label,
            xy=(-0.2, 1.1), 
            xycoords='axes fraction',
            ha='left', 
            va='center',
            color=color, 
            size=16,
            bbox=dict(boxstyle='round', fc=bbox_fc, ec=bbox_ec, lw=2)
        )  
        SeabornFig2Grid(joint_grid, fig, gs[i])   

    gs.tight_layout(fig)
    
    return fig, fig.axes


def plot_task_and_speed_profiles(
    speed: Float[Array, "batch time"], 
    task_variables: Mapping[str, Float[Array, "batch time"]] = dict(), 
    epoch_start_idxs: Optional[Int[Array, "batch epoch"]] = None,
    cmap: str = 'tab10',
    colors: Optional[Sequence[str | Tuple[float, ...]]] = None,
    **kwargs,
):
    """For visualizing learned movements versus task structure.
    
    For example: does the network start moving before the go cue is given?
    """
    task_rows = len(task_variables)
    
    if task_rows > 0:
        height_ratios = (1,) * task_rows + (task_rows,)
    else:
        height_ratios = (1,)

    if colors is None:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in np.linspace(0, 1, speed.shape[0])]

    fig, axs = plt.subplots(1 + task_rows, 1, height_ratios=height_ratios, 
                            sharex=True, constrained_layout=True)
                            
    if task_rows == 0:
        axs = [axs]

    axs[-1].set_title('speed profiles')
    for i in range(speed.shape[0]):
        axs[-1].plot(speed[i].T, color=colors[i], **kwargs)

    ymax = 1.5 * jnp.max(speed)
    if epoch_start_idxs is not None:
        axs[-1].vlines(epoch_start_idxs[:, 1], ymin=0, ymax=ymax, colors=colors, 
                       linestyles='dotted', linewidths=1, label='target ON')
        axs[-1].vlines(epoch_start_idxs[:, 3], ymin=0, ymax=ymax, colors=colors, 
                       linestyles='dashed', linewidths=1, label='fixation OFF')
        plt.legend()
    axs[-1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for i, (label, arr) in enumerate(task_variables.items()):
        arr = jnp.squeeze(arr)
        axs[i].set_title(label)
        for j in range(arr.shape[0]):
            axs[i].plot(arr[j].T, color=colors[j])
        axs[i].set_ylim(*padded_bounds(arr))
    
    axs[-1].set_xlabel('Time step')
    axs[-1].set_ylabel('Speed')
    
    return fig, axs


def animate_arm2(
    cartesian_state: Float[Array, "time joints xy"],
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

    arm_line, = ax.plot(*cartesian_state[0].T, 'k-')
    traj_line, = ax.plot(*cartesian_state[0, :, 2].T, 'g-')

    def animate(i):
        arm_line.set_data(*cartesian_state[i].T)
        traj_line.set_data(*cartesian_state[:i+1, :, 2].T)
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
    """Rotate a 3D plot through `axim_range` degrees about the z axis."""
    def animate(i):
        ax.view_init(elev=elev, azim=i)
        return fig,

    return animation.FuncAnimation(
        fig, 
        animate, 
        frames=azim_range[1] - azim_range[0],
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


def get_high_contrast_neutral_shade():
    """Return a high-contrast color depending on the current matplotlib style.
    
    Assumes that black will generally be the best choice, except when a dark 
    theme (i.e. black axes background color) is used.
    """
    axes_facecolor = plt.rcParams['axes.facecolor']
    if axes_facecolor == 'black':
        return 'white'
    else:
        return 'black'
    
    
def circular_hist(x, ax=None, bins=16, density=True, offset=0, gaps=True, plot_mean=False):
    """Produce a circular histogram of angles on ax.
    
    NOTE: Should probably replace this. See the original SO answer linked
    below. The area of the bars is not linear in the counts, but we tend to
    visually estimate proportions by areas.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
        
    From https://stackoverflow.com/a/55067613
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)
    
    if plot_mean:
        mean_angle = np.mean(x)
        ax.plot([mean_angle, mean_angle], [0, np.max(radius)], "r-", lw=2)
    
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return fig, ax #n, bins, patches


class preserve_axes_limits:
    """Context manager for preserving the axes limits of matplotlib axis/axes.
    
    For example, let's say we want to plot an `hlines` on an axis and use
    very large limits so that it definitely spans the visible region, but 
    we don't want the plot to adjust its limits to those of the `hlines`.    
    
    It is perhaps odd to use `jax.tree_map` for a stateful operation like 
    the one in `__exit__`, but it works.
    """
    def __init__(self, ax: mpl.axes.Axes | PyTree[mpl.axes.Axes]):
        if isinstance(ax, mpl.axes.Axes):
            axs = [ax]
        self._axs = axs
        self._lims = jax.tree_map(
            lambda ax: dict(
                x=ax.get_xlim(),
                y=ax.get_ylim(),
            ),
            axs,
            is_leaf=lambda x: isinstance(x, mpl.axes.Axes),
        )
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        jax.tree_map(
            lambda ax, lims: (
                ax.set_xlim(lims['x']), 
                ax.set_ylim(lims['y'])
            ),
            self._axs,
            self._lims,
        )
        
    
def hlines(ax, y, **kwargs):
    """Add a horizontal line across a plot, preserving the axes limits."""
    with preserve_axes_limits(ax):
        ax.hlines(y, -1e100, 1e100, **kwargs)


def vlines(ax, x, **kwargs):
    """Add a vertical line across a plot, preserving the axes limits."""
    with preserve_axes_limits(ax):
        ax.vlines(x, -1e100, 1e100, **kwargs)
        
        
def padded_bounds(x, p=0.2):
    """Return the lower and upper bounds of `x` with `p` percent padding."""
    bounds = jnp.array([jnp.min(x), jnp.max(x)])
    padding = (p * jnp.diff(bounds)).item()
    return bounds + jnp.array([-padding, padding])