"""

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Literal,
    Optional,
    TypeVar,
)

import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import polars as pl
from jax_cookbook import identity
from jaxtyping import Array, Float, PRNGKeyArray
from plotly.subplots import make_subplots

# pyright: reportMissingTypeStubs=false
from plotly.colors import convert_colors_to_same_type

from feedbax.objectives.loss import TermTree
from feedbax.plot.colors import DEFAULT_COLORS, color_add_alpha
from feedbax.plot.misc import columns_mean_std, errorbars

logger = logging.getLogger(__name__)


T = TypeVar("T")


def _loss_scatter_mode(loss_context: Literal["training", "validation"]) -> str:
    if loss_context == "training":
        return "markers+lines"
    elif loss_context == "validation":
        # Validation losses are usually sparse in time, so don't connect with lines
        return "markers"
    else:
        raise ValueError(f"{loss_context} is not a valid loss context")


def _loss_trace_kws(scatter_kws: Optional[dict]) -> dict:
    trace_kws = dict(hovertemplate="%{y:.2e}")
    if scatter_kws is not None:
        trace_kws |= scatter_kws
    return trace_kws


def _loss_history_statistics(
    losses: TermTree | Array,
    n_std_plot: int,
) -> tuple[OrderedDict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    losses = jt.map(np.array, losses)

    if isinstance(losses, TermTree):
        losses_total = losses.aggregate(leaf_fn=identity)
        loss_entries = OrderedDict({"Total": losses_total}) | losses.flatten()
    else:
        losses_total = losses
        loss_entries = OrderedDict({"Total": losses})

    training_iterations = pl.DataFrame({"iteration": range(losses_total.shape[0])})

    dfs = jt.map(
        lambda losses: pl.DataFrame(np.array(losses)),
        loss_entries,
    )

    # TODO: Only apply this when yaxis is log scaled
    dfs = jt.map(
        lambda df: df.select(
            [
                np.log10(pl.all()),
            ]
        ),
        dfs,
    )

    loss_statistics = columns_mean_std(dfs)
    error_bars_bounds = errorbars(loss_statistics, n_std_plot)

    # TODO: Only apply this when yaxis is log scaled
    loss_statistics, error_bars_bounds = jt.map(
        lambda df: training_iterations.hstack(
            df.select(
                [
                    np.power(10, pl.col("*")),  # type: ignore
                ]
            )
        ),
        (loss_statistics, error_bars_bounds),
    )

    return dfs, loss_statistics, error_bars_bounds


def _plotly_colors(colors: Optional[list[str]]) -> list[str]:
    if colors is None:
        colors = DEFAULT_COLORS

    colors_rgb: list[str]
    colors_rgb, _ = convert_colors_to_same_type(colors, colortype="rgb")  # type: ignore
    return colors_rgb


def _add_loss_history_traces(
    fig: go.Figure,
    label: str,
    name: str,
    legendgroup: str,
    color: str,
    loss_statistics: Mapping[str, pl.DataFrame],
    error_bars_bounds: Mapping[str, pl.DataFrame],
    scatter_mode: str,
    scatter_kws: dict,
    error_bars_alpha: float,
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    showlegend: bool = True,
) -> None:
    trace = go.Scatter(
        name=name,
        legendgroup=legendgroup,
        x=loss_statistics[label]["iteration"],
        y=loss_statistics[label]["mean"],
        mode=scatter_mode,
        marker_size=3,
        line=dict(color=color),
        showlegend=showlegend,
    )
    trace.update(scatter_kws)
    fig.add_trace(trace, row=row, col=col)

    fig.add_trace(
        go.Scatter(
            name=f"{name} upper bound",
            legendgroup=legendgroup,
            x=loss_statistics[label]["iteration"],
            y=error_bars_bounds[label]["ub"],
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            name=f"{name} lower bound",
            legendgroup=legendgroup,
            x=loss_statistics[label]["iteration"],
            y=error_bars_bounds[label]["lb"],
            line=dict(color="rgba(255,255,255,0)"),
            fill="tonexty",
            fillcolor=color_add_alpha(color, error_bars_alpha),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def loss_history(
    losses: TermTree | Array,
    loss_context: Literal["training", "validation"] = "training",
    colors: Optional[list[str]] = None,
    error_bars_alpha: float = 0.3,
    n_std_plot: int = 1,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    hover_compare: bool = True,
    **kwargs,
) -> go.Figure:
    scatter_kws = _loss_trace_kws(scatter_kws)

    fig = go.Figure(
        layout_modebar_add=["v1hovermode"],
        **kwargs,
    )

    scatter_mode = _loss_scatter_mode(loss_context)
    dfs, loss_statistics, error_bars_bounds = _loss_history_statistics(losses, n_std_plot)

    colors_rgb = _plotly_colors(colors)

    colors_dict = {
        label: "rgb(0,0,0)" if label == "Total" else color for label, color in zip(dfs, colors_rgb)
    }

    for i, label in enumerate(reversed(dfs)):
        _add_loss_history_traces(
            fig,
            label,
            label,
            str(i),
            colors_dict[label],
            loss_statistics,
            error_bars_bounds,
            scatter_mode,
            scatter_kws,
            error_bars_alpha,
        )

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

    if hover_compare:
        fig.update_layout(hovermode="x")

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def _history_losses(
    history: Any,
    loss_context: Literal["training", "validation"],
) -> TermTree | Array:
    if loss_context == "training":
        return getattr(history, "loss", history)
    elif loss_context == "validation":
        return getattr(history, "loss_validation", history)
    else:
        raise ValueError(f"{loss_context} is not a valid loss context")


def _shared_loss_terms(
    loss_entries: Mapping[str, Mapping[str, pl.DataFrame]],
    terms: Literal["shared", "union"] | Sequence[str],
) -> list[str]:
    if terms == "shared":
        first_terms = list(next(iter(loss_entries.values())).keys())
        common_terms = set(first_terms)
        for entries in loss_entries.values():
            common_terms &= set(entries)
        return [term for term in first_terms if term in common_terms]

    if terms == "union":
        ordered_terms: list[str] = []
        for entries in loss_entries.values():
            for term in entries:
                if term not in ordered_terms:
                    ordered_terms.append(term)
        return ordered_terms

    requested_terms = list(terms)
    missing_terms = {
        run_label: [term for term in requested_terms if term not in entries]
        for run_label, entries in loss_entries.items()
    }
    missing_terms = {run_label: missing for run_label, missing in missing_terms.items() if missing}
    if missing_terms:
        raise ValueError(f"Requested loss terms are missing from histories: {missing_terms}")

    return requested_terms


def loss_history_compare(
    histories: Mapping[str, Any],
    loss_context: Literal["training", "validation"] = "training",
    terms: Literal["shared", "union"] | Sequence[str] = "shared",
    colors: Optional[list[str]] = None,
    error_bars_alpha: float = 0.2,
    n_std_plot: int = 1,
    n_cols: int = 1,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    hover_compare: bool = True,
    **kwargs,
) -> go.Figure:
    """Compare labelled training histories on shared loss-term axes.

    Args:
        histories: Mapping from run label to history-like objects.
            Values may also be raw loss trees/arrays, matching `loss_history`.
        loss_context: Whether to plot `history.loss` or `history.loss_validation`.
        terms: Which term axes to include. `"shared"` plots the terms present in
            every history, `"union"` plots all terms, and a sequence requests an
            explicit ordered subset.
        colors: Plotly colors used for run labels.
        error_bars_alpha: Alpha value for replicate error bands.
        n_std_plot: Number of standard deviations included in error bands.
        n_cols: Number of subplot columns used for term axes.
        layout_kws: Optional layout overrides.
        scatter_kws: Optional mean-trace overrides.
        hover_compare: Whether to use Plotly x-hover comparison.
        **kwargs: Additional keyword arguments passed to `make_subplots`.

    Returns:
        A Plotly figure with one subplot per loss term and one mean/error-band
        trace bundle per run.
    """
    if len(histories) == 0:
        raise ValueError("At least one history is required")

    if n_cols < 1:
        raise ValueError("n_cols must be at least 1")

    histories = OrderedDict(histories)
    scatter_kws = _loss_trace_kws(scatter_kws)
    scatter_mode = _loss_scatter_mode(loss_context)

    stats_by_run = OrderedDict(
        (
            run_label,
            _loss_history_statistics(_history_losses(history, loss_context), n_std_plot),
        )
        for run_label, history in histories.items()
    )

    dfs_by_run = OrderedDict(
        (run_label, stats[0]) for run_label, stats in stats_by_run.items()
    )
    term_labels = _shared_loss_terms(dfs_by_run, terms)

    if not term_labels:
        raise ValueError("No loss terms are shared across histories")

    n_cols = min(n_cols, len(term_labels))
    n_rows = int(np.ceil(len(term_labels) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=term_labels,
        shared_xaxes=True,
        shared_yaxes=True,
        **kwargs,
    )

    colors_rgb = _plotly_colors(colors)
    colors_dict = {
        run_label: colors_rgb[i % len(colors_rgb)] for i, run_label in enumerate(histories)
    }

    for term_i, term_label in enumerate(term_labels):
        row = (term_i // n_cols) + 1
        col = (term_i % n_cols) + 1
        for run_label, (_, loss_statistics, error_bars_bounds) in stats_by_run.items():
            if term_label not in loss_statistics:
                continue

            _add_loss_history_traces(
                fig,
                term_label,
                run_label,
                run_label,
                colors_dict[run_label],
                loss_statistics,
                error_bars_bounds,
                scatter_mode,
                scatter_kws,
                error_bars_alpha,
                row=row,
                col=col,
                showlegend=term_i == 0,
            )

    fig.update_layout(
        width=800,
        height=max(300, 240 * n_rows),
        # yaxis_tickformat="e",
        yaxis_exponentformat="E",
        margin=dict(l=80, r=10, t=30, b=60),
    )

    if hover_compare:
        fig.update_layout(hovermode="x")

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    fig.update_xaxes(title_text="Training iteration")
    fig.update_yaxes(title_text="Loss")

    return fig


def activity_heatmap(
    activity: Float[Array, "time unit"],
    colorscale: str = "viridis",
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
        import jax_cookbook.tree as jtree

        states = task.eval(model, key=key_eval)  # States for all validation trials.
        states_trial0 = jtree.take(states, 0)
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


def activity_sample_units(
    activities: Float[Array, "*trial time unit"],
    n_samples: int = 4,
    unit_includes: Optional[Sequence[int]] = None,
    colors: Optional[list[str]] = None,
    row_height: int = 150,
    layout_kws: Optional[dict] = None,
    trial_label: str = "Trial",  # TODO: Rename
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

    unit_idxs = jr.choice(key, np.arange(activities.shape[-1]), (n_samples,), replace=False)
    if unit_includes is not None:
        unit_idxs = np.concatenate([unit_idxs, np.array(unit_includes)])
    unit_idxs = np.sort(unit_idxs)
    unit_idx_strs = [str(i) for i in unit_idxs]

    xs = np.array(activities[..., unit_idxs])

    # Join all the data into a dataframe.
    df = pl.concat(
        [
            # For each trial, construct all timesteps of x, y data.
            pl.from_numpy(x, schema=unit_idx_strs).hstack(
                pl.DataFrame(
                    {
                        "Timestep": np.arange(x.shape[0]),
                        # Note that "trial" here could be between- or within-condition.
                        trial_label: pl.repeat(i, x.shape[0], eager=True),
                    }
                )
            )
            for i, x in enumerate(xs)
        ]
    ).melt(
        id_vars=["Timestep", trial_label],
        value_name="Activity",
        variable_name="Unit",
    )

    fig = px.line(
        df,
        x="Timestep",
        y="Activity",
        color=trial_label,
        facet_row="Unit",
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
    fig.update_yaxes(zerolinewidth=0.5, zerolinecolor="black")

    # Improve formatting of subplot "Unit=" labels.
    fig.for_each_annotation(
        lambda a: a.update(
            text=a.text.replace("=", " "),
            font=dict(size=14),
        )
    )

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig
