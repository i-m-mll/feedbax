from collections.abc import Mapping
from types import MappingProxyType

import equinox as eqx
import jax.tree as jt
import numpy as np
from jaxtyping import PyTree

from feedbax_experiments.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    InputOf,
)
from feedbax_experiments.misc import deep_merge
from feedbax_experiments.plot import get_violins
from feedbax_experiments.plot_utils import get_label_str
from feedbax_experiments.tree_utils import (
    ldict_label_only_func,
    tree_level_types,
)
from feedbax_experiments.types import AnalysisInputData, LDict, LDictConstructor, TreeNamespace

#! TODO
# measure_ranges = {
#     key: (
#             jnp.nanmin(measure_data_stacked),
#             jnp.nanmax(measure_data_stacked),
#     )
#     for key, measure_data_stacked in {
#         key: jnp.stack(jt.leaves(measure_data))
#         for key, measure_data in all_measure_values.items()
#     }.items()
# }


class ViolinsPorts(AbstractAnalysisPorts):
    """Input ports for `Violins` analysis.

    - `data`: required PyTree to plot; expects the innermost two LDict levels
      at figure time to represent [legend groups, x-axis bins]. Use fig-ops
      like `.map_figs_at_level` to slice deeper trees.
    - `data_split`: optional matching PyTree for split violins.
    """

    input: InputOf[PyTree]
    input_split: InputOf[PyTree] | None = None


class Violins(AbstractAnalysis[ViolinsPorts]):
    Ports = ViolinsPorts
    inputs: ViolinsPorts = eqx.field(default_factory=ViolinsPorts, converter=ViolinsPorts.converter)

    fig_params: Mapping = MappingProxyType(
        dict(
            violinmode="overlay",
            zero_hline=False,
            arr_axis_labels=None,
            yaxis_title=None,  # Often provided per-slice via fig-ops
        )
    )

    def make_figs(
        self, data: AnalysisInputData, *, result, colors, input: PyTree, input_split=None, **kwargs
    ):
        # Determine the two innermost LDict levels for grouping and x-axis
        level_types = tree_level_types(input)
        level_labels = [ldict_label_only_func(node_type) for node_type in level_types]

        if len(level_labels) < 2 or any(
            not isinstance(t, LDictConstructor) for t in level_types[-2:]
        ):
            raise ValueError(
                "Violins expects at least two inner LDict levels to determine group/x axes."
            )

        group_label, x_label = level_labels[-2:]

        plot_kwargs = dict(
            split_mode="whole" if input_split is None else "split",
            legend_title=get_label_str(group_label),
            xaxis_title=get_label_str(x_label),
            colors=colors[group_label].dark,
        )

        def _make_fig(node, node_split=None):
            return get_violins(
                node,
                data_split=node_split,
                **deep_merge(plot_kwargs, self.fig_params),
            )

        # Map over any outer levels; at the group level we build one figure
        if input_split is None:
            figs = jt.map(_make_fig, input, is_leaf=LDict.is_of(group_label))
        else:
            figs = jt.map(_make_fig, input, input_split, is_leaf=LDict.is_of(group_label))

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, input, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(input)[0].shape)),
        )
