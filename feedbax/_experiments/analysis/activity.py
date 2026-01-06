from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Optional

import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from equinox import field
from feedbax.bodies import SimpleFeedbackState
from jax_cookbook import is_type
from jaxtyping import PRNGKeyArray, PyTree

from feedbax._experiments.analysis.analysis import AbstractAnalysis, NoPorts
from feedbax._experiments.analysis.pca import PCA
from feedbax._experiments.constants import REPLICATE_CRITERION
from feedbax._experiments.plot_utils import calculate_array_minmax, get_label_str
from feedbax._experiments.types import AnalysisInputData, LDict, TreeNamespace


def activity_sample_units(
    activities: LDict,
    n_units_sample: int = 4,
    unit_includes: Optional[Sequence[int]] = None,
    colors: Optional[dict] = None,
    row_height: int = 150,
    layout_kws: Optional[dict] = None,
    legend_title: Optional[str] = None,
    *,
    key: PRNGKeyArray,
    **kwargs,
) -> go.Figure:
    """Plot activity over multiple conditions for a random sample of network units.

    Combines all values from activities dictionary into a single figure, with each value
    given a different color in the legend.

    Arguments:
        activities: A dictionary or LDict of activity arrays.
            Each activity array should have shape (trials, time steps, units).
        n_units_sample: The number of units to sample from the layer.
        unit_includes: Indices of specific units to include in the plot.
        colors: A dictionary mapping keys from activities to colors.
        row_height: How tall (in pixels) to make the figure, as a factor of units sampled.
        layout_kws: Additional kwargs for the figure layout.
        legend_title: Title for the legend. If None, attempts to derive from activities.label.
        key: A random key used to sample the units to plot.
    """
    if colors is None:
        colors = {}

    # Get legend title from activities.label if available and not provided
    if legend_title is None:
        legend_title = get_label_str(activities.label)

    # Get a sample of units
    n_units_total = jt.leaves(activities)[0].shape[-1]
    unit_idxs = jr.choice(key, np.arange(n_units_total), (n_units_sample,), replace=False)
    if unit_includes is not None:
        unit_idxs = np.concatenate([unit_idxs, np.array(unit_includes)])
    unit_idxs = np.sort(unit_idxs)
    unit_idx_strs = [str(i) for i in unit_idxs]

    # Create figure with a subplot for each unit)
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=len(unit_idxs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    fig.update_layout(height=row_height * len(unit_idxs))

    condition_values = list(activities.keys())

    # Process each condition
    for condition_idx, condition in enumerate(condition_values):
        activity_data = activities[condition]

        # Normalize the activities to shape (trials, time, units)
        if len(activity_data.shape) == 2:  # (time, units)
            activities_3d = activity_data[None, :, :]  # Add trial dimension
        elif len(activity_data.shape) > 3:  # Extra dimensions
            # Reshape to (trials, time, units)
            activities_3d = activity_data.reshape(
                -1, activity_data.shape[-2], activity_data.shape[-1]
            )
        else:  # Already 3D
            activities_3d = activity_data

        # Extract the selected units
        activities_units = activities_3d[..., unit_idxs]

        # Flatten batch dimensions if needed - we'll use only first element of second dim
        if activities_units.shape[1] == 1:
            # If single timestep, just squeeze
            batch_activities = activities_units[:, 0, :]
        else:
            # Keep as is
            batch_activities = activities_units

        # Get the color for this condition
        color = colors.get(condition, None)

        # For each eval (trial)
        for trial_idx, batch_activity in enumerate(batch_activities):
            # Ensure batch_activity has shape (time, n_selected_units)
            if batch_activity.ndim == 1:
                # If it's 1D (just time), reshape to (time, 1)
                batch_activity = batch_activity.reshape(-1, 1)

            # Create a separate trace for each unit
            for unit_idx, unit_str in enumerate(unit_idx_strs):
                unit_activity = batch_activity[:, unit_idx]
                timesteps = np.arange(len(unit_activity))

                trace_name = str(condition)

                # Add a trace for this trial
                fig.add_trace(
                    go.Scatter(
                        x=timesteps,
                        y=unit_activity,
                        mode="lines",
                        name=trace_name,
                        # legendgroup=trace_name,  # Group in legend by condition
                        showlegend=unit_idx == 0
                        and trial_idx == 0,  # Show legend only once per condition
                        line_color=color,
                        **kwargs,
                    ),
                    row=unit_idx + 1,  # 1-indexed subplot row
                    col=1,
                )

    # Calculate global min and max for y-axis scaling
    y_min, y_max = calculate_array_minmax(activities, indices=unit_idxs)

    # Update layout
    fig.update_layout(
        legend_title=legend_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Set all y-axes to have the same range for comparison
    fig.update_yaxes(
        title="",
        zerolinewidth=0.5,
        zerolinecolor="black",
        range=[y_min, y_max],  # Set consistent y-range
    )

    # Add Unit labels on the right side
    for i, unit_str in enumerate(unit_idx_strs):
        fig.add_annotation(
            text=f"Unit {unit_str}",
            x=1.02,  # Position to the right of the plot
            y=0,  # Will be adjusted for each subplot
            xref="paper",
            yref=f"y{i + 1 if i > 0 else ''}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            textangle=90,  # Vertical text
            font=dict(size=14),
        )

    # Add the Activity label with more space
    fig.add_annotation(
        x=-0.12,  # Move further left to avoid tick labels
        y=0.5,
        text="Activity",
        textangle=-90,
        showarrow=False,
        font=dict(size=14),
        xref="paper",
        yref="paper",
    )

    # Update layout if provided
    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


class NetworkActivity_SampleUnits(AbstractAnalysis[NoPorts]):
    variant: Optional[str] = "small"
    fig_params: Mapping = MappingProxyType(
        dict(
            n_units_sample=4,
            key=jr.PRNGKey(0),
            layout_kws=dict(
                width=700,
                height=500,
            ),
            scatter_kws=dict(
                line_width=1,
            ),
            # legend_title="Reach direction",
        )
    )
    # colorscale_key: str = "reach_condition"

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        colors,
        **kwargs,
    ):
        activities = jt.map(
            lambda states: states.net.hidden,
            data.states[self.variant],
            is_leaf=is_type(SimpleFeedbackState),
        )

        figs = LDict.of(activities.label)(
            {
                outer_value: activity_sample_units(
                    activities=inner_dict,
                    n_units_sample=self.fig_params.get("n_units_sample"),
                    colors=colors[inner_dict.label].dark,
                    key=self.fig_params.get("key"),
                    **self.fig_params.get("scatter_kws", {}),
                )
                for outer_value, inner_dict in activities.items()
            }
        )

        figs = jt.map(
            lambda fig: fig.update_layout(**self.fig_params.get("layout_kws", {})),
            figs,
            is_leaf=is_type(go.Figure),
        )

        return figs

    def _params_to_save(
        self, hps: PyTree[TreeNamespace], *, replicate_info, train_pert_std, **kwargs
    ):
        return dict(
            i_replicate=replicate_info[train_pert_std]["best_replicates"][REPLICATE_CRITERION],
        )
