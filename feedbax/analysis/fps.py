from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Any, Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import plotly.graph_objects as go
from equinox import field
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from optax import GradientTransformation

from feedbax.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    InputOf,
)
from feedbax.analysis.fp_finder import (
    FixedPointFinder,
    fp_adam_optimizer,
)
from feedbax.analysis.pca import PCAResults, StatesPCA
from feedbax.plot.experiments import plot_fp_pcs
from feedbax.tree_utils import ldict_level_to_bottom
from feedbax.types import AnalysisInputData, LDict

T = TypeVar("T")

__all__ = [
    "FixedPoints",
    "FixedPointsPorts",
    "PlotInPCSpace",
    "PlotInPCSpacePorts",
]


class FixedPointsPorts(AbstractAnalysisPorts):
    """Input ports for FixedPoints analysis."""

    fns: InputOf[Callable]  # Functions to find fixed points for, e.g. RNN cells
    candidates: InputOf[Array]  # Candidate states to initialize the fixed point search
    fn_args: Optional[tuple[InputOf[Any], ...]] = (
        None  # Optional positional arguments to pass to functions
    )


class FixedPoints(AbstractAnalysis[FixedPointsPorts]):
    """Canonical Feedbax analysis for steady-state fixed points.

    This analysis is intentionally model-agnostic: callers provide one or more
    transition functions and matching candidate state arrays. Workflow-specific
    helpers for SimpleReaches or RLRMP analyses belong downstream or in analysis
    recipes that construct these inputs explicitly.

    Inputs:
        fns: Functions to find fixed points for, e.g. RNN cells.
            The functions should have signature (`func(*args, state)`) where `state` is the
            optimized tensor that is initialized with `candidates`.
        candidates: Candidate states to initialize the fixed point search.
        fn_args Optional positional arguments to pass to the functions as `*args`.
    """

    Ports = FixedPointsPorts
    inputs: FixedPointsPorts = eqx.field(
        default_factory=FixedPointsPorts, converter=FixedPointsPorts.converter
    )

    variant: Optional[str] = "full"
    cache_result: bool = True

    # ss_fn: Callable = get_ss_rnn_fn
    fp_tol: Scalar = eqx.field(default=1e-7, converter=jnp.array)
    unique_tol: float = 0.025
    outlier_tol: float = 1.0
    stride_candidates: int = 1
    fp_optimizer: Callable[[], GradientTransformation] = fp_adam_optimizer
    key: PRNGKeyArray = field(default_factory=lambda: jr.PRNGKey(0))

    @property
    def fpfinder(self):
        """Low-level fixed-point finder used by this analysis."""
        return FixedPointFinder(self.fp_optimizer())

    @property
    def fpf_fn(self):
        """Partial function for finding and filtering fixed points."""
        return partial(
            self.fpfinder.find_and_filter,
            loss_tol=jnp.array(self.fp_tol),
            outlier_tol=self.outlier_tol,
            unique_tol=self.unique_tol,
            key=self.key,
        )

    def compute(
        self,
        data: AnalysisInputData,
        *,
        fns: PyTree[Callable, "T"],
        candidates: PyTree[Array, "T"],
        fn_args: Optional[tuple[PyTree[Any, "T"], ...]] = None,
        **kwargs,
    ):
        if fn_args is None:
            fn_args = tuple()

        def get_fps(func, cands, *args):
            return self.fpf_fn(
                lambda h: func(*args, h),
                cands[:: self.stride_candidates],
            )

        return jt.map(
            get_fps,
            fns,
            candidates,
            *fn_args,
            is_leaf=callable,
        )


# ----------------------------------------------------------------
# Plotting classes
# ----------------------------------------------------------------


class PlotInPCSpacePorts(AbstractAnalysisPorts):
    """Input ports for PlotInPCSpace analysis."""

    pca_results: InputOf[StatesPCA]
    plot_data: InputOf[Array]


class PlotInPCSpace(AbstractAnalysis[PlotInPCSpacePorts]):
    """Plot data in PCA space."""

    Ports = PlotInPCSpacePorts
    inputs: PlotInPCSpacePorts = eqx.field(
        default_factory=PlotInPCSpacePorts, converter=PlotInPCSpacePorts.converter
    )
    variant: Optional[str] = "full"
    fig_params: Mapping = MappingProxyType(dict())

    # n_pcs: Literal[2, 3] = 3
    spread_label: str = "sisu"

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        pca_results: PyTree[PCAResults],
        plot_data: PyTree[Array],
        colors,
        **kwargs,
    ):
        plot_data = ldict_level_to_bottom(self.spread_label, plot_data)

        def generate_fig(plot_data, pca):
            fig = go.Figure(
                layout=dict(
                    width=800,
                    height=800,
                    scene=dict(
                        xaxis_title="PC 1",
                        yaxis_title="PC 2",
                        zaxis_title="PC 3",
                    ),
                    legend=dict(
                        title=self.spread_label,
                        itemsizing="constant",
                        y=0.85,
                    ),
                )
            )

            plot_data_pc = pca.batch_transform(plot_data)
            for value in plot_data:
                fig = plot_fp_pcs(
                    plot_data_pc[value],
                    fig=fig,
                    label=value,
                    colors=colors[self.spread_label].dark[value],
                )
            return fig

        figs = jt.map(generate_fig, plot_data, pca_results, is_leaf=LDict.is_of(self.spread_label))

        return figs
