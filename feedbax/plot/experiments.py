from typing import Optional

import jax.numpy as jnp
import plotly.graph_objects as go
from jaxtyping import Array, Float

from feedbax.plot import apply_default_template

# Plotly copies the process-default template into each figure as it is constructed, so
# the Feedbax default has to be in place before any function here runs. Importing a
# figure-drawing module is itself use of the plotting surface, unlike importing the
# package, which entry points that never draw anything do transitively.
apply_default_template()


def add_endpoint_traces(
    fig: go.Figure,
    pos_endpoints: Float[Array, "ends=2 *trials xy=2"],
    visible: tuple[bool, bool] = (True, True),
    colorscale: Optional[str] = None,
    colorscale_axis: int = 0,  # of `trials` axes
    init_marker_kws: Optional[dict] = None,
    goal_marker_kws: Optional[dict] = None,
    straight_guides: bool = False,
    straight_guide_kws: Optional[dict] = None,
    behind_traces: bool = True,
    **kwargs,
):
    """
    Add endpoint markers and optional straight guide lines to a Plotly figure.

    Args:
        fig: Plotly figure to add traces to
        pos_endpoints: Array with start and goal positions. Shape: [2, *trials, 2] where
                      the first dimension is [start, goal], middle dimensions are trials,
                      and last dimension is [x, y] coordinates
        visible: tuple of booleans indicating if start and goal markers should be visible
        colorscale: Optional color scale to apply to the markers
        colorscale_axis: Which trial axis to use for color mapping (if colorscale provided)
        init_marker_kws: Additional keyword arguments for the start markers
        goal_marker_kws: Additional keyword arguments for the goal markers
        straight_guides: Whether to add straight lines connecting start and goal points
        straight_guide_kws: Additional keyword arguments for the guide lines
        **kwargs: Additional keyword arguments for all marker traces

    Returns:
        Updated Plotly figure with added traces
    """
    marker_kws = {
        "Start": dict(
            size=10,
            symbol="square-open",
            color="rgb(25, 25, 25)",
            line=dict(width=2, color="rgb(25, 25, 25)"),
        ),
        "Goal": dict(
            size=10,
            symbol="circle-open",
            color="rgb(25, 25, 25)",
            line=dict(width=2, color="rgb(25, 25, 25)"),
        ),
    }
    default_guide_kws = dict(
        mode="lines",
        line=dict(
            color="rgba(100, 100, 100, 0.5)",
            dash="dot",
            width=0.5,
        ),
        hoverinfo="skip",
    )

    if len(pos_endpoints.shape) == 2:
        pos_endpoints = jnp.expand_dims(pos_endpoints, axis=1)

    if colorscale is not None:
        # Calculate colors based on trial dimensions only
        trial_shape = pos_endpoints.shape[1:-1]
        if not trial_shape:  # Handle case with only one trial dimension
            trial_shape = (pos_endpoints.shape[1],)

        color_linspace = jnp.linspace(
            0, 1, pos_endpoints.shape[colorscale_axis + 1], endpoint=False
        )

        # Axes within trial_shape that are *not* the colorscale_axis
        expand_dims_axes = tuple(i for i in range(len(trial_shape)) if i != colorscale_axis)

        # Expand and broadcast linspace to match the full trial shape
        broadcasted_colors = jnp.broadcast_to(
            jnp.expand_dims(color_linspace, expand_dims_axes),
            trial_shape,
        )

        # Flatten colors to match the flattened trial dimension used later
        flat_colors = jnp.reshape(broadcasted_colors, (-1,))

        for i, key in enumerate(marker_kws):
            marker_kws[key].update(
                # Apply the flat color array to markers
                line_color=flat_colors,
                color=flat_colors,
                cmin=0,
                cmax=1,
                # colorscale=colorscale, # This should be set on the trace, not marker
            )

    if init_marker_kws is not None:
        marker_kws["Start"].update(init_marker_kws)
    if goal_marker_kws is not None:
        marker_kws["Goal"].update(goal_marker_kws)

    if len(pos_endpoints.shape) > 3:
        pos_endpoints = jnp.reshape(pos_endpoints, (2, -1, 2))

    traces = []

    # Add straight guide lines between start and goal points if requested
    if straight_guides:
        if straight_guide_kws is not None:
            default_guide_kws.update(straight_guide_kws)

        for i in range(pos_endpoints.shape[1]):
            # For each trial, create a line from start to goal
            traces.append(
                go.Scatter(
                    x=[pos_endpoints[0, i, 0], pos_endpoints[1, i, 0]],
                    y=[pos_endpoints[0, i, 1], pos_endpoints[1, i, 1]],
                    name="Straight path",
                    showlegend=False,
                    legend="legend2",
                    **default_guide_kws,
                )
            )

    for j, (label, kws) in enumerate(marker_kws.items()):
        traces.append(
            go.Scatter(
                name=f"{label}",
                meta=dict(label=label),
                legend="legend2",
                hovertemplate=f"{label}<extra></extra>",
                x=pos_endpoints[j, ..., 0],
                y=pos_endpoints[j, ..., 1],
                visible=visible[j],
                mode="markers",
                marker=kws,
                marker_colorscale=colorscale,
                showlegend=True,
                **kwargs,
            )
        )

    if behind_traces:
        existing_traces = list(fig.data)
        fig.data = ()
        fig.add_traces(traces + existing_traces)
    else:
        fig.add_traces(traces)

    fig.update_layout(
        legend2=dict(
            title_text="Guides",
            xanchor="left",
            yanchor="top",
            xref="paper",
            yref="paper",
            y=0.99,
            x=1.22,
        ),
    )

    return fig


# TODO
# TODO: annotate types
# TODO
# def plot_fp_loss(all_fps, fp_loss_fn, n_bins=50):
#     fp_tols = list(all_fps.keys())

#     f1, ax = plt.subplots(figsize=(12, 6))
#     for tol in fp_tols:
#         ax.semilogy(all_fps[tol]['losses']);
#         ax.set_xlabel('Fixed point #')
#         ax.set_ylabel('Fixed point loss');
#     f1.legend(fp_tols)
#     ax.set_title('Fixed point loss by fixed point (sorted) and stop tolerance')

#     f2, axs = plt.subplots(1, len(fp_tols), figsize=(12,4))

#     for i, tol in enumerate(fp_tols):
#         axs[i].hist(np.log10(fp_loss_fn(all_fps[tol]['fps'])), n_bins)
#         axs[i].set_xlabel('log10(FP loss)')
#         axs[i].set_title('Tolerance: ' + str(tol))

#     return f1, f2


#! On preliminary tests this isn't actually very useful for anchored axes;
#! in principle what we need to do for anchored axes is convert x and y
#! extrema into a common frame (based on the scale ratio) and then scale
#! that frame (i.e. only x xor y axis range). Does this achieve that?
