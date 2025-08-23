from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Optional

import equinox as eqx
import feedbax.plotly as fbp
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from equinox import Module
from jax_cookbook import is_type
from jaxtyping import Array, PyTree

from feedbax_experiments.analysis.aligned import AlignedVars, get_varset_labels
from feedbax_experiments.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf
from feedbax_experiments.plot_utils import get_label_str
from feedbax_experiments.tree_utils import move_ldict_level_above, tree_level_labels
from feedbax_experiments.types import AnalysisInputData, LDict, TreeNamespace


class ProfilesPorts(AbstractAnalysisPorts):
    """Input ports for Profiles analysis."""

    vars: InputOf[Array] = AlignedVars()


class Profiles(AbstractAnalysis[ProfilesPorts]):
    """Generates figures for

    Assumes that all the aligned vars have the same number of coordinates (i.e.
    length of final array axis), and that these coordinates can be labeled similarly
    by `coord_labels`. For example, this is the case when we align position, velocity,
    acceleration, and force in 2D.
    """

    Ports = ProfilesPorts
    inputs: ProfilesPorts = eqx.field(
        default_factory=ProfilesPorts, converter=ProfilesPorts.converter
    )

    # variant: Optional[str] = "full"
    fig_params: Mapping = MappingProxyType(
        dict(
            mode="std",  # or 'curves'
            n_std_plot=1,
            layout_kws=dict(
                width=600,
                height=400,
                legend_tracegroupgap=1,
            ),
        )
    )
    var_level_label: str = "var"
    vrect_kws_func: Optional[Callable[[TreeNamespace], dict]] = None
    # var_labels: Optional[dict[str, str]] = None  # e.g. for mapping "pos" to "position"
    coord_labels: Optional[tuple[str, str]] = (
        "parallel",
        "lateral",
    )  # None for vars with single, unlabelled coordinates (e.g. deviations)
    varset: Optional[PyTree[TreeNamespace]] = None  # e.g. for aligned vars with metadata

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        vars,
        colors,
        hps_common,
        **kwargs,
    ):
        if self.varset is not None:
            labels = get_varset_labels(self.varset).medium
        else:
            labels = None

        def _get_fig(fig_data, i, coord_label, var_key, colors):
            # if self.var_labels is not None:
            #     var_label = self.var_labels[var_key]
            # else:
            #     var_label = var_key
            if labels is not None:
                var_label = labels[var_key]
            else:
                var_label = var_key

            if coord_label:
                label = f"{coord_label} {var_label}"
            else:
                label = var_label

            if isinstance(fig_data, LDict):
                colors = list(colors[fig_data.label].dark.values())
                legend_title = get_label_str(fig_data.label)
            else:
                colors = None
                legend_title = None

            return fbp.profiles(
                jtree.take(fig_data, i, -1),
                varname=label.capitalize(),
                legend_title=legend_title,
                hline=dict(y=0, line_color="grey"),
                colors=colors,
                # stride_curves=500,
                # curves_kws=dict(opacity=0.7),
                **self.fig_params,
            )

        def _get_figs_by_coord(var_key, var_data):
            if self.coord_labels is None:
                return _get_fig(var_data, 0, "", var_key, colors)
            else:
                return LDict.of("coord")(
                    {
                        coord_label: _get_fig(var_data, coord_idx, coord_label, var_key, colors)
                        for coord_idx, coord_label in enumerate(self.coord_labels)
                    }
                )

        figs = jt.map(
            lambda results_by_var: LDict.of(self.var_level_label)(
                {
                    var_key: _get_figs_by_coord(var_key, var_data)
                    for var_key, var_data in results_by_var.items()
                }
            ),
            vars,
            is_leaf=LDict.is_of(self.var_level_label),
        )

        if self.vrect_kws_func is not None:
            vrect_kws = self.vrect_kws_func(hps_common)
            jt.map(
                lambda fig: fig.add_vrect(**vrect_kws),
                figs,
                is_leaf=is_type(go.Figure),
            )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, vars, **kwargs):
        return dict(n=int(np.prod(jt.leaves(vars)[0].shape[:-2])))
