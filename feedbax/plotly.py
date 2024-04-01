"""

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable, Sequence
import logging
from typing import TYPE_CHECKING, Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Shaped
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import polars as pl

from feedbax import tree_labels
from feedbax.bodies import SimpleFeedbackState
from feedbax.task import AbstractReachTrialSpec

if TYPE_CHECKING:
    from feedbax.train import TaskTrainerHistory


logger = logging.getLogger(__name__)


def color_add_alpha(rgb_str: str, alpha: float):
    return f"rgba{rgb_str[3:-1]}, {alpha})"


def columns_mean_std(dfs: PyTree[pl.DataFrame], index_col: str):
    return jax.tree_map(
        lambda df: df.select(**{
            index_col: pl.col(index_col),
            "mean": pl.concat_list(pl.col('*').exclude(index_col)).list.mean(),
            "std": pl.concat_list(pl.col('*').exclude(index_col)).list.std(),
        }),
        dfs,
    )

def errorbars(col_means_stds: PyTree[pl.DataFrame], n_std: int):
    return jax.tree_map(
        lambda df: df.select(
            lb=pl.col('mean') - n_std * pl.col('std'),
            ub=pl.col('mean') + n_std * pl.col('std'),
        ),
        col_means_stds,
    )

def loss_mean_history(
    train_history: "TaskTrainerHistory",
    colors: Optional[list[str]] = None,
    error_bars_alpha: float = 0.3,
    n_std_plot: int = 1,
) -> go.Figure:
    fig = go.Figure()

    losses = jax.tree_map(
        lambda x: np.array(x),
        train_history.loss,
    )

    timesteps = pl.DataFrame({"timestep": range(train_history.loss.total.shape[0])})

    dfs = jax.tree_map(
        lambda losses: timesteps.hstack(pl.DataFrame(losses)),
        dict(losses) | {"Total": losses.total},
    )

    loss_statistics = columns_mean_std(dfs, "timestep")
    error_bars_bounds = errorbars(loss_statistics, n_std_plot)

    colors_dict = {
        label: 'rgb(0,0,0)' if label == 'Total' else color
        for label, color in zip(dfs, colors)
    }

    for i, label in enumerate(dfs):
        # Mean
        fig.add_trace(go.Scatter(
            name=label,
            x=loss_statistics[label]["timestep"],
            y=loss_statistics[label]['mean'],
            line=dict(color=colors_dict[label])
        ))

        # Error bars
        fig.add_trace(go.Scatter(
            name="Upper bound",
            x=loss_statistics[label]["timestep"],
            y=error_bars_bounds[label]["ub"],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            name="Lower bound",
            x=loss_statistics[label]["timestep"],
            y=error_bars_bounds[label]["lb"],
            line=dict(color='rgba(255,255,255,0)'),
            fill="tonexty",
            fillcolor=color_add_alpha(colors_dict[label], error_bars_alpha),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(width=700, height=600)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    return fig


def activity_heatmap(
    activity: Float[Array, "time unit"],
    colorscale: str = 'viridis',
) -> go.Figure:
    """Plot activity of all units in a network layer over time, on a single trial.

    !!! Note
        This is a helper for Plotly's [`imshow`](https://plotly.com/python/imshow/),
        when the data is an array of neural network unit activities with shape
        `(time, unit)`.

    !!! Example
        When working with a `SimpleFeedback` model built with a `SimpleStagedNetwork`
        controller—for example, if we've constructed our `model` using
        [`point_mass_nn`][feedbax.xabdeef.models.point_mass_nn]—we can plot the activity
        of the hidden layer of the network:

        ```python
        from feedbax import tree_take

        states = task.eval(model, key=key_eval)  # States for all validation trials.
        states_trial0 = tree_take(states, 0)
        activity_heatmap(states_trial0.net.hidden)
        ```

    Arguments:
        activity: The array of activity over time for each unit in a network layer.
        colorscale: The name of the Plotly [color scale](https://plotly.com/python/builtin-colorscales/)
            to use.
    """
    fig = px.imshow(
        activity.T,
        aspect="auto",
        color_continuous_scale=colorscale,
        labels=dict(x="Time step", y="Unit"),
    )
    return fig


def profile(
    var: Float[Array, "*trial timestep"],
    var_label: str = "Value",
    colors: list[str] = px.colors.qualitative.Set1,
    **kwargs,
) -> go.Figure:
    return px.line(
        var.T,
        color_discrete_sequence=colors,
        labels=dict(x="Time step", y=var_label),
        **kwargs,
    )


def activity_sample_units(
    activities: Float[Array, "*trial time unit"],
    n_samples: int,
    unit_includes: Optional[Sequence[int]] = None,
    colors: Optional[list[str]] = None,
    height: Optional[int] = None,
    *,
    key: PRNGKeyArray,
    **kwargs
) -> go.Figure:
    """Plot activity over multiple trials for a random sample of network units.

    The result is a figure with `n_samples + len(unit_includes)` subplots, arranged
    in `cols` columns.

    When this function is called more than once in the course of an analysis, if the
    same `key` is passed and the network layer has the same number of units—that
    is, the last dimension of `activities` has the same size—then the same subset of
    units will be sampled.

    Arguments:
        activities: The array of trial-by-trial activity over time for each unit in a
            network layer.
        n_samples: The number of units to sample from the layer. Along with `unit_includes`,
            this determines the number of subplots in the figure.
        unit_includes: Indices of specific units to include in the plot, in addition to
            the `n_samples` randomly sampled units.
        cols: The number of columns in which to arrange the subplots.
        cmap_name: The name of the Matplotlib [colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
            to use. Each trial will be plotted in a different color.
        figsize: The size of the figure.
        key: A random key used to sample the units to plot.
    """

    # Make sure `activities` has shape (trials, time steps, units).
    # If multiple batch dimensions are present, flatten them.
    if len(activities.shape) == 2:
        activities = activities[None, ...]
    elif len(activities.shape) > 3:
        activities = activities.reshape(-1, *activities.shape[-2:])
    elif len(activities.shape) != 3:
        raise ValueError("Invalid shape for ")

    unit_idxs = jr.choice(
        key, np.arange(activities.shape[-1]), (n_samples,), replace=False
    )
    if unit_includes is not None:
        unit_idxs = np.concatenate([unit_idxs, np.array(unit_includes)])
    xs = np.array(activities[..., unit_idxs]).swapaxes(0, -1)

    # Join all the data into a dataframe.
    df = pl.concat([
        pl.DataFrame(dict(
            Unit=pl.repeat(f"{i}", x.shape[0], eager=True),
            Timestep=np.arange(x.shape[0]),
        )).hstack(
            pl.from_numpy(x, schema=[f"{j}" for j in range(x.shape[1])])
        ).melt(
            id_vars=["Timestep", "Unit"],
            variable_name="Trial",
            value_name="Activity",
        )
        for i, x in zip(unit_idxs, xs)
    ], how="vertical")

    if height is None:
        height = 150 * len(unit_idxs)

    fig = px.line(
        df,
        x='Timestep',
        y='Activity',
        color='Trial',
        facet_row='Unit',
        color_discrete_sequence=colors,
        height=height,
        **kwargs,
    )

    # Replace multiple y-axis labels with a single one.
    fig.for_each_yaxis(lambda y: y.update(title=""))
    fig.add_annotation(
        x=-0.07,
        y=0.5,
        text="Activity",
        textangle=-90,
        showarrow=False,
        font=dict(size=14),
        xref="paper",
        yref="paper",
    )

    return fig


def tree_of_2d_trial_timeseries_to_df(
    tree: PyTree[Shaped[Array, "trial timestep xy=2"], 'T'],
    labels: Optional[PyTree[str, 'T']] = None,
) -> pl.DataFrame:
    """Construct a single dataframe from a PyTree of spatial timeseries arrays,
    batched over trials."""

    array_spec = jtu.tree_map(eqx.is_array, tree)
    arrays_flat = map(np.array, jtu.tree_leaves(eqx.filter(tree, array_spec)))

    if labels is None:
        labels = tree_labels(tree, join_with=" ")

    labels_flat = jtu.tree_leaves(eqx.filter(labels, array_spec))

    def get_xy_df(array):
        # Concatenate all trials.
        return pl.concat([
            # For each trial, construct all timesteps of x, y data.
            pl.from_numpy(x, schema=["x", "y"]).hstack(pl.DataFrame({
                "Timestep": np.arange(x.shape[0]),
                "Trial": pl.repeat(str(i), x.shape[0], eager=True),
            }))
            for i, x in enumerate(array)
        ])

    xy_dfs = [get_xy_df(x) for x in arrays_flat]

    # Concatenate all variables.
    df = pl.concat([
        xy_df.hstack([pl.repeat(label, len(xy_df), eager=True).alias("var")])
        for xy_df, label in zip(xy_dfs, labels_flat)
    ])

    return df


def unshare_axes(fig: go.Figure):
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))


def effector_trajectories(
    states: SimpleFeedbackState | PyTree[Float[Array, "trial time ..."] | Any],
    where_data: Optional[Callable] = None,
    step: int = 1,  # plot every step-th trial
    trial_specs: Optional[AbstractReachTrialSpec] = None,
    endpoints: Optional[
        tuple[Float[Array, "trial xy=2"], Float[Array, "trial xy=2"]]
    ] = None,
    straight_guides: bool = False,
    workspace: Optional[Float[Array, "bounds=2 xy=2"]] = None,
    cmap_name: Optional[str] = None,
    colors: Optional[list[str]] = None,
    color: Optional[str | tuple[float, ...]] = None,
    ms: int = 5,
    ms_source: int = 6,
    ms_target: int = 7,
    control_labels: Optional[tuple[str, str, str]] = None,
    control_label_type: str = "linear",
):
    """Plot trajectories of position, velocity, network output.

    Arguments:
        states: A model state or PyTree of arrays from which the variables to be
            plotted can be extracted.
        where_data: If `states` is provided as an arbitrary PyTree of arrays,
            this function should be provided to extract the relevant arrays.
            It should take `states` and return a tuple of three arrays:
            position, velocity, and controller output/force.
        step: Plot every `step`-th trial. This is useful when `states` contains
            information about a very large set of trials, and we only want to
            plot a subset of them.
        trial_specs: The specifications for the trials being plotted. If supplied,
            this is used to plot markers at the initial and goal positions.
        endpoints: The initial and goal positions for the trials being plotted.
            Overrides `trial_specs`.
        straight_guides: If this is `True` and `endpoints` are provided, straight
            dashed lines will be drawn between the initial and goal positions.
        workspace: The workspace bounds. If provided, the bounds are drawn as a
            rectangle.
        cmap_name: The name of the Matplotlib [colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
            to use across trials.
        colors: A sequence of colors, one for each plotted trial. Overrides `cmap_name`.
        color: A single color to use for all trials. Overrides `cmap_name` but not `colors`.
        ms: Marker size for plots of states (trajectories).
        ms_source: Marker size for the initial position, if `trial_specs`/`endpoints`
            is provided.
        ms_target: Marker size for the goal position.
        control_label_type: If `'linear'`, labels the final (controller output/force)
            plot as showing Cartesian forces. If `'torques'`, labels the plot as showing
            the torques of a two-segment arm.
        control_labels: A tuple giving the labels for the title, x-axis, and y-axis
            of the final (controller output/force) plot. Overrides `control_label_type`.
    """
    if isinstance(states, SimpleFeedbackState):
        positions, velocities, controls = (
            states.mechanics.effector.pos,
            states.mechanics.effector.vel,
            states.efferent.output,
        )

    elif where_data is None:
        raise ValueError(
            "If `states` is not a `SimpleFeedbackState`, "
            "`where_data` must be provided."
        )
    else:
        positions, velocities, controls = where_data(states)

    df = tree_of_2d_trial_timeseries_to_df(
        (positions, velocities, controls),
        labels=("Position", "Velocity", "Force"),
    )

    n_vars = df['var'].n_unique()

    # if cmap_name is None:
    #     if positions.shape[0] < 10:
    #         cmap_name = "tab10"
    #     else:
    #         cmap_name = "viridis"

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Trial',
        facet_col='var',
        facet_col_spacing=0.05,
        color_discrete_sequence=colors,
    )

    fig.update_traces(marker_size=ms)

    # Constrain the axes of each subplot to be scaled equally.
    # That is, keep square aspect ratios.
    fig.update_layout({
        f"yaxis{i}": dict(scaleanchor=f"x{i}")
        for i in [''] + list(range(2, n_vars + 1))
    })

    # Omit the "var=" part of each subplot title
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1],
        font=dict(size=14),
    ))

    # Facet plot shares axes by default.
    unshare_axes(fig)

    return fig

