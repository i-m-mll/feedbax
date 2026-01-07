from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Literal, Optional

import feedbax.plot as fbp
import jax.tree as jt
import jax_cookbook.tree as jtree
import plotly.graph_objects as go
from equinox import Module, field
from feedbax.plot import AxesLabels
from feedbax.task import AbstractTask
from feedbax.types import SeqOfT
from jax_cookbook import is_module, is_type
from jaxtyping import Array, PyTree

from feedbax.analysis.aligned import DEFAULT_VARSET, get_varset_labels
from feedbax.analysis.analysis import AbstractAnalysis, NoPorts
from feedbax.analysis.state_utils import get_pos_endpoints
from feedbax._experiments.colors import COLORSCALES
from feedbax.config import PLOTLY_CONFIG
from feedbax._experiments.constants import REPLICATE_CRITERION
from feedbax._experiments.misc import deep_merge
from feedbax._experiments.plot import add_endpoint_traces
from feedbax._experiments.plot_utils import get_label_str
from feedbax.types import AnalysisInputData, TreeNamespace, VarSpec

MEAN_LIGHTEN_FACTOR = PLOTLY_CONFIG.mean_lighten_factor


def plot_2d_effector_trajectories[T](
    plot_data: SeqOfT[Array, T],
    var_labels: SeqOfT[str, T],
    colorscale,
    # Corresponding to axis 0 of `states`:
    legend_title="Reach direction",
    **kwargs,
):
    """Helper to define the usual formatting for effector trajectory plots."""
    return fbp.trajectories_2D(
        plot_data,
        var_labels=var_labels,
        axes_labels=AxesLabels("x", "y"),
        #! TODO: Replace with `colorscales` (common analysis dependency)
        colorscale=colorscale,
        legend_title=legend_title,
        # scatter_kws=dict(line_width=0.5),
        layout_kws=dict(
            width=100 + len(var_labels) * 300,
            height=400,
            legend_tracegroupgap=1,
        ),
        **kwargs,
    )


class EffectorTrajectories(AbstractAnalysis[NoPorts]):
    variant: Optional[str] = "small"
    varset: PyTree[VarSpec] = field(default_factory=lambda: DEFAULT_VARSET)
    fig_params: Mapping = MappingProxyType(
        dict(
            # legend_title="Reach direction",
            mean_exclude_axes=(),
            curves_mode="lines",
            legend_labels=None,
            darken_mean=MEAN_LIGHTEN_FACTOR,
            scatter_kws=dict(line_width=0.75, opacity=0.4),
            mean_scatter_kws=dict(line_width=2.5),
        )
    )
    colorscale_key: Optional[str] = None
    colorscale_axis: Optional[int] = None
    pos_endpoints: bool = True
    straight_guides: bool = True
    label_fmt: Literal["short", "medium", "full"] = "medium"

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        colorscales,
        **kwargs,
    ):
        #! TODO: Add a general way to include callables in `fig_params`;
        #! however this probably requires passing `fig_params` to `AbstractAnalysis.make_figs`...
        if self.fig_params.get("legend_title") is None and self.colorscale_key is not None:
            fig_params = MappingProxyType(
                deep_merge(self.fig_params, {"legend_title": get_label_str(self.colorscale_key)})
            )
        else:
            fig_params = self.fig_params

        var_labels = getattr(get_varset_labels(self.varset), self.label_fmt)

        def _make_fig(states):
            plot_data = jt.map(
                lambda spec: spec.where(states), self.varset, is_leaf=is_type(VarSpec)
            )
            return plot_2d_effector_trajectories(
                plot_data.values(),
                var_labels=var_labels.values(),
                colorscale=colorscales[self.colorscale_key],
                colorscale_axis=self.colorscale_axis,
                **fig_params,
            )

        figs = jt.map(_make_fig, data.states[self.variant], is_leaf=is_module)

        if self.pos_endpoints:
            #! Assume all tasks are straight reaches with the same length.
            #! TODO: Remove this assumption. Depending on `_pre_ops`/`_fig_ops`, the
            #! PyTree structure of `data.tasks[self.variant]` may differ from that of `figs`
            #! and thus we have to be careful about how to perform the mapping.
            #! (In the simplest case, without ops, the task PyTree is a prefix of `figs`)
            task_0 = jt.leaves(data.tasks[self.variant], is_leaf=is_type(AbstractTask))[0]
            pos_endpoints = get_pos_endpoints(task_0.validation_trials)

            if self.colorscale_key == "reach_condition":
                colorscale = COLORSCALES["reach_condition"]
            else:
                colorscale = None

            init_marker_kws = dict(color="rgb(25, 25, 25)")

            figs = jt.map(
                lambda fig: add_endpoint_traces(
                    fig,
                    pos_endpoints,
                    xaxis="x1",
                    yaxis="y1",
                    colorscale=colorscale,
                    init_marker_kws=init_marker_kws,
                    straight_guides=self.straight_guides,
                ),
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
