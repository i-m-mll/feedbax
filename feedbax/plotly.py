"""

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Callable, Sequence
import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

import equinox as eqx
from feedbax._tree import tree_infer_batch_size
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Shaped
import numpy as np
# pyright: reportMissingTypeStubs=false
from plotly.basedatatypes import BaseTraceType
from plotly.colors import convert_colors_to_same_type, sample_colorscale
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import polars as pl

from feedbax import tree_labels
from feedbax.bodies import SimpleFeedbackState
from feedbax.loss import LossDict
from feedbax.misc import where_func_to_labels

if TYPE_CHECKING:
    from feedbax.task import TaskTrialSpec
    from feedbax.train import TaskTrainerHistory


logger = logging.getLogger(__name__)


pio.templates.default = "plotly_white"
DEFAULT_COLORS = pio.templates[pio.templates.default].layout.colorway  # pyright: ignore


def color_add_alpha(rgb_str: str, alpha: float):
    return f"rgba{rgb_str[3:-1]}, {alpha})"


def columns_mean_std(dfs: PyTree[pl.DataFrame], index_col: Optional[str] = None):
    if index_col is not None:
        spec = {
            index_col: pl.col(index_col),
            "mean": pl.concat_list(pl.col('*').exclude(index_col)).list.mean(),
            "std": pl.concat_list(pl.col('*').exclude(index_col)).list.std(),
        }
    else:
        spec = {
            "mean": pl.concat_list(pl.col('*')).list.mean(),
            "std": pl.concat_list(pl.col('*')).list.std(),
        }

    return jax.tree_map(
        lambda df: df.select(**spec),
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


def loss_history(
    train_history: "TaskTrainerHistory",
    loss_context: Literal["training", "validation"] = "training",
    colors: Optional[list[str]] = None,
    error_bars_alpha: float = 0.3,
    n_std_plot: int = 1,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    **kwargs,
) -> go.Figure:
    fig = go.Figure(**kwargs)

    if loss_context == "training":
        loss_tree = train_history.loss
        scatter_mode = "markers+lines"
    elif loss_context == "validation":
        loss_tree = train_history.loss_validation
        # Validation losses are usually sparse in time, so don't connect with lines
        scatter_mode = "markers"
    else:
        raise ValueError(f"{loss_context} is not a valid loss context" )

    losses: LossDict | Array = jax.tree_map(
        lambda x: np.array(x),
        loss_tree,
    )

    losses_total = losses if isinstance(losses, Array) else losses.total

    timesteps = pl.DataFrame({"timestep": range(losses_total.shape[0])})

    dfs = jax.tree_map(
        lambda losses: pl.DataFrame(losses),
        OrderedDict({"Total": losses_total}) | dict(losses),
    )

    # TODO: Only apply this when yaxis is log scaled
    dfs = jax.tree_map(
        lambda df: df.select([
            np.log10(pl.all()),
        ]),
        dfs,
    )

    loss_statistics = columns_mean_std(dfs)
    error_bars_bounds = errorbars(loss_statistics, n_std_plot)

    # TODO: Only apply this when yaxis is log scaled
    loss_statistics, error_bars_bounds = jax.tree_map(
        lambda df: timesteps.hstack(df.select([
            np.power(10, pl.col("*")),  # type: ignore
        ])),
        (loss_statistics, error_bars_bounds),
    )

    if colors is None:
        colors = DEFAULT_COLORS

    colors_rgb: list[str]
    colors_rgb, _ = convert_colors_to_same_type(colors, colortype='rgb')  # type: ignore

    colors_dict = {
        label: 'rgb(0,0,0)' if label == 'Total' else color
        for label, color in zip(dfs, colors_rgb)
    }

    for i, label in enumerate(reversed(dfs)):
        # Mean
        trace = go.Scatter(
            name=label,
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=loss_statistics[label]['mean'],
            mode=scatter_mode,
            marker_size=3,
            line=dict(color=colors_dict[label]),
        )
        trace.update(scatter_kws)
        fig.add_trace(trace)

        # Error bars
        fig.add_trace(go.Scatter(
            name="Upper bound",
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=error_bars_bounds[label]["ub"],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            name="Lower bound",
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=error_bars_bounds[label]["lb"],
            line=dict(color='rgba(255,255,255,0)'),
            fill="tonexty",
            fillcolor=color_add_alpha(colors_dict[label], error_bars_alpha),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Training iteration",
        yaxis_title="Loss",
        # yaxis_tickformat="e",
        yaxis_exponentformat="E",
        margin=dict(l=80, r=10, t=30, b=60),
        legend_traceorder="reversed",
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def activity_heatmap(
    activity: Float[Array, "time unit"],
    colorscale: str = 'viridis',
    layout_kws: Optional[dict] = None,
    **kwargs,
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
        labels=dict(x="Time step", y="Unit", color="Activity"),
        **kwargs,
    )
    if layout_kws is not None:
        fig.update_layout(layout_kws)
    return fig


def profile(
    var: Float[Array, "*trial timestep"],
    var_label: str = "Value",
    colors: Optional[list[str]] = None,
    layout_kws: Optional[dict] = None,
    **kwargs,
) -> go.Figure:
    # TODO: vlines
    fig = px.line(
        var.T,
        color_discrete_sequence=colors,
        labels=dict(index="Time step", value=var_label, variable='Trial'),
        **kwargs,
    )
    if layout_kws is not None:
        fig.update_layout(layout_kws)
    return fig


def activity_sample_units(
    activities: Float[Array, "*trial time unit"],
    n_samples: int,
    unit_includes: Optional[Sequence[int]] = None,
    colors: Optional[list[str]] = None,
    row_height: int = 150,
    layout_kws: Optional[dict] = None,
    trial_label: str = "Trial",
    *,
    key: PRNGKeyArray,
    **kwargs,
) -> go.Figure:
    """Plot activity over multiple trials for a random sample of network units.

    The result is a figure with `n_samples + len(unit_includes)` subplots, arranged
    in `cols` columns.

    When this function is called more than once in the course of an analysis, if the
    same `key` is passed and the network layer has the same number of units—that
    is, the last dimension of `activities` has the same size—then the same subset of
    units will be sampled.

    Arguments:
        activities: An array of trial-by-trial activity over time for each unit in a
            network layer.
        n_samples: The number of units to sample from the layer. Along with `unit_includes`,
            this determines the number of subplots in the figure.
        unit_includes: Indices of specific units to include in the plot, in addition to
            the `n_samples` randomly sampled units.
        colors: A list of colors.
        row_height: How tall (in pixels) to make the figure, as a factor of units sampled.
        layout_kws: Additional kwargs with which to update the layout of the figure before
            returning.
        trial_label: The text label for the batch dimension. For example, if `activities`
            gives evaluations across model replicates, we may wish to pass
            `trial_label="Replicate"` to properly label the legend and tooltips.
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
    unit_idxs = np.sort(unit_idxs)
    unit_idx_strs = [str(i) for i in unit_idxs]

    xs = np.array(activities[..., unit_idxs])

    # Join all the data into a dataframe.
    df = pl.concat([
        # For each trial, construct all timesteps of x, y data.
        pl.from_numpy(x, schema=unit_idx_strs).hstack(pl.DataFrame({
            "Timestep": np.arange(x.shape[0]),
            # Note that "trial" here could be between- or within-condition.
            trial_label: pl.repeat(i, x.shape[0], eager=True),
        }))
        for i, x in enumerate(xs)
    ]).melt(
        id_vars=["Timestep", trial_label],
        value_name="Activity",
        variable_name="Unit",
    )

    fig = px.line(
        df,
        x='Timestep',
        y='Activity',
        color=trial_label,
        facet_row='Unit',
        color_discrete_sequence=colors,
        height=row_height * len(unit_idxs),
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

    # fig.update_yaxes(zerolinewidth=2, zerolinecolor='rgb(200,200,200)')
    fig.update_yaxes(zerolinewidth=0.5, zerolinecolor='black')

    # Improve formatting of subplot "Unit=" labels.
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.replace("=", " "),
        font=dict(size=14),
    ))

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def tree_of_2d_timeseries_to_df(
    tree: PyTree[Shaped[Array, "condition timestep xy=2"], 'T'],
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
                "Condition": pl.repeat(i, x.shape[0], eager=True),
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
    where_data: Optional[
        Callable[[PyTree[Array]], tuple[Shaped[Array, "*batch trial time xy=2"]]]
    ] = None,
    var_labels: Optional[tuple[str, ...]] = None,
    step: int = 1,  # plot every step-th trial
    trial_specs: Optional["TaskTrialSpec"] = None,
    endpoints: Optional[
        tuple[Float[Array, "trial xy=2"], Float[Array, "trial xy=2"]]
    ] = None,
    straight_guides: bool = False,
    workspace: Optional[Float[Array, "bounds=2 xy=2"]] = None,
    cmap_name: Optional[str] = None,
    colors: Optional[list[str]] = None,
    color: Optional[str | tuple[float, ...]] = None,
    mode: str = "markers+lines",
    ms: int = 5,
    ms_init: int = 12,
    ms_goal: int = 12,
    control_labels: Optional[tuple[str, str, str]] = None,
    control_label_type: str = "linear",
    layout_kws: Optional[dict] = None,
    **kwargs,
):
    """Plot trajectories of position, velocity, network output.

    Arguments:
        states: A model state or PyTree of arrays from which the variables to be
            plotted can be extracted.
        where_data: If `states` is provided as an arbitrary PyTree of arrays,
            this function should be provided to extract the relevant arrays.
            It should take `states` and return a tuple of arrays.
        var_labels: Labels for the variables selected by `where_data`.
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
    var_labels_ = var_labels

    if where_data is not None:
        vars_tuple = where_data(states)
        if var_labels is None:
            var_labels = where_func_to_labels(where_data)

        if isinstance(vars_tuple, (Array, np.ndarray)):
            vars_tuple = (vars_tuple,)
            var_labels_ = (var_labels,)

    elif isinstance(states, SimpleFeedbackState):
        vars_tuple = (
            states.mechanics.effector.pos,
            states.mechanics.effector.vel,
            states.efferent.output,
        )
        if var_labels is None:
            var_labels_ = ("Position", "Velocity", "Force")
    else:
        raise ValueError(
            "If `states` is not a `SimpleFeedbackState`, "
            "`where_data` must be provided."
        )

    if len(vars_tuple[0].shape) > 3:
        # Collapse to a single batch dimension
        vars_tuple = jax.tree_map(
            lambda arr: np.reshape(arr, (-1, *arr.shape[-3:])),
            vars_tuple,
        )
        dfs = [
            tree_of_2d_timeseries_to_df(v, labels=var_labels_)
            for v in zip(*jax.tree_map(tuple, vars_tuple))
        ]
        dfs = [
            df.hstack(pl.DataFrame({"Trial": pl.repeat(i, len(df), eager=True)}))
            for i, df in enumerate(dfs)
        ]
        df = pl.concat(dfs, how='vertical')

    else:
        df = tree_of_2d_timeseries_to_df(
            vars_tuple,
            labels=var_labels_,
        )

    n_vars = df['var'].n_unique()

    # if cmap_name is None:
    #     if positions.shape[0] < 10:
    #         cmap_name = "tab10"
    #     else:
    #         cmap_name = "viridis"

    # TODO: Use `go.Figure` for more control over trial vs. condition, in batched case
    # fig = go.Figure()
    # fig.add_traces()
    # TODO: Separate control/indexing of lines vs. markers; e.g. thin line-only traces,
    # plus markers only at the end of the reach

    
    if colors is None and (n_conditions := tree_infer_batch_size(states)) > 10:
        colors = [str(c) for c in sample_colorscale('phase', n_conditions)]

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color=df['Condition'].cast(pl.String),
        facet_col='var',
        facet_col_spacing=0.05,
        color_discrete_sequence=colors,
        labels=dict(color="Condition"),
        **kwargs,
    )

    fig.for_each_trace(lambda trace: trace.update(mode=mode))

    fig.update_traces(marker_size=ms)

    if endpoints is not None:
        endpoints_arr = np.array(endpoints)  #type: ignore
    else:
        if trial_specs is not None:
            target_specs = trial_specs.targets["mechanics.effector.pos"]
            if target_specs.value is not None:
                endpoints_arr = np.array(  # type: ignore
                    [
                        trial_specs.inits["mechanics.effector"].pos,
                        target_specs.value[:, -1],
                    ]
                )
            else:
                endpoints_arr = None
        else:
            endpoints_arr = None

    if endpoints_arr is not None:
        colors = [d.marker.color for d in fig.data[::n_vars]]

        n_trials = endpoints_arr.shape[1]

        # Add init and goal markers
        for j, (label, (ms, symbol)) in enumerate(
            {
                "Init": (ms_init, 'square'),
                "Goal": (ms_goal, 'circle'),
            }.items()
        ):
            fig.add_traces(
                [
                    go.Scatter(
                        name=f"{label} {i}",
                        meta=dict(label=label),
                        legendgroup=label, #str(i)
                        hoverinfo="name",
                        x=endpoints_arr[j, i, 0][None],
                        y=endpoints_arr[j, i, 1][None],
                        mode="markers",
                        marker=dict(
                            size=ms,
                            symbol=symbol,
                            color='rgba(255, 255, 255, 0)',
                            line=dict(
                                color=colors[i],
                                width=2,
                            ),
                        ),
                        xaxis="x1",
                        yaxis="y1",
                        # TODO: Show once in legend, for all markers of type j
                        showlegend=i < (n_trials - 1),
                    )
                    for i in range(n_trials)
                ]
            )

        # Add dashed straight lines from init to goal.
        if straight_guides:
            fig.add_traces(
                [
                    go.Scatter(
                        x=endpoints_arr[:, i, 0],
                        y=endpoints_arr[:, i, 1].T,
                        mode="lines",
                        line_dash='dash',
                        line_color=colors[i],
                        showlegend=False,
                        xaxis="x1",
                        yaxis="y1",
                    )
                    for i in range(endpoints_arr.shape[1])
                ]
            )

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

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def is_trace(element):
    return isinstance(element, BaseTraceType)


def plot_traj_3D(
    traj: Float[Array, "trials time 3"], 
    endpoint_symbol: Optional[str] = 'circle-open', 
    start_symbol: Optional[str] = None, 
    fig: Optional[go.Figure] = None, 
    colors: str | Sequence[str | None] | None = None, 
    mode: str = 'lines', 
    name: Optional[str] = "State trajectory",
    **kwargs,
):
    """Plot 3D trajectories."""
    if fig is None:
        fig = go.Figure(layout=dict(width=1000, height=1000))
        
    if colors is None or isinstance(colors, str):
        colors_func = lambda _: colors
    else:
        colors_func = lambda i: colors[i]
    
    if start_symbol is not None:
        fig.add_traces(
            [
                go.Scatter3d(
                    x=traj[:, 0, 0], 
                    y=traj[:, 0, 1], 
                    z=traj[:, 0, 2],
                    mode='markers',
                    marker_symbol=start_symbol,
                    marker_line_width=2,
                    marker_size=10,
                    marker_color="None",
                    marker_line_color=colors,
                    name=f'{name} start'
                )
            ]
        )
    fig.add_traces(
        [
            go.Scatter3d(
                x=traj[idx, :, 0], 
                y=traj[idx, :, 1], 
                z=traj[idx, :, 2],
                mode=mode,
                line_color=colors_func(idx),
                marker_color=colors_func(idx),
                # name='Reach trajectories',
                showlegend=(name is not None and idx == 0),
                name=name,
                **kwargs,
            )
            for idx in range(traj.shape[0])
        ]
    )
    if endpoint_symbol is not None:
        fig.add_traces(
            [
                go.Scatter3d(
                    x=traj[:, -1, 0], 
                    y=traj[:, -1, 1], 
                    z=traj[:, -1, 2],
                    mode='markers',
                    marker_symbol=endpoint_symbol,
                    marker_line_width=2,
                    marker_size=5,
                    marker_color=colors,
                    marker_line_color=colors,
                    name=f'{name} end'
                )
            ]
        )
    return fig


def plot_eigvals(
    eigvals: Float[Array, "batch eigvals"],   
    colors: str | Sequence[str] | None = None,  
    colorscale: str = 'phase',
    mode: str = 'markers',
    fig: Optional[go.Figure] = None,
    **kwargs,
):
    """Plot eigenvalues inside a unit circle with dashed axes."""
    if fig is None:
        fig = go.Figure(layout=dict(width=1000, height=1000))
        
    if colors is not None:
        if isinstance(colors, str):
            pass
        elif len(colors) != np.prod(eigvals.shape):
            colors = np.repeat(np.array(colors), eigvals.shape[1])

    # Add a unit circle
    fig.add_shape(
        type='circle',
        xref='x', yref='y',
        x0=-1, y0=-1, x1=1, y1=1,
    )
    
    # Add dashed axes lines
    fig.add_hline(0, line_dash='dot', line_color='grey')
    fig.add_vline(0, line_dash='dot', line_color='grey')

    # Plot eigenvalues
    fig.add_trace(
        go.Scatter(
            x=jnp.real(jnp.ravel(eigvals)),
            y=jnp.imag(jnp.ravel(eigvals)),
            mode=mode,
            marker_size=5,
            marker_color=colors,
            marker_colorscale=colorscale,
            **kwargs,
        )
    )
    
    return fig
