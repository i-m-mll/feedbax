from collections.abc import Callable, Hashable, Mapping
from functools import partial
from struct import pack
from types import MappingProxyType
from typing import Any, ClassVar, Optional, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import plotly.graph_objects as go
from equinox import field
from feedbax._vendor.jax_cookbook import LDict
from jaxtyping import Array, PRNGKeyArray, PyTree, PyTreeDef, Scalar
from optax import GradientTransformation

from feedbax.analysis.types import AnalysisInputData
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
from feedbax.config.tree import ldict_level_to_bottom
from feedbax.contracts.computation_revision import ANALYSIS_ENGINE_REVISION

T = TypeVar("T")

__all__ = [
    "FixedPoints",
    "FixedPointsPorts",
    "PlotInPCSpace",
    "PlotInPCSpacePorts",
]


class _StaticLeaf:
    """Key Python leaves without conflating distinct types or opaque objects."""

    def __init__(self, value: Any):
        self.value = value
        value_type = type(value)
        if value_type in (bool, int, str, bytes):
            self._key = (value_type, value)
        elif value_type in (float, complex):
            self._key = (value_type, pack("!dd", value.real, value.imag))
        elif callable(value) or isinstance(value, Hashable):
            self._key = id(value)
        else:
            # Opaque, unhashable arguments may be mutable. Trace them per binding.
            self._key = object()

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticLeaf) and self._key == other._key


class _BoundCall(eqx.Module):
    """Expose array leaves to JAX and restore the original PyTree on call."""

    leaves: tuple[Any, ...]
    treedef: PyTreeDef = eqx.field(static=True)

    def __init__(self, func: Callable, *args: Any, static_func: bool = False):
        leaves, self.treedef = jt.flatten(
            (func, args), is_leaf=(lambda leaf: leaf is func) if static_func else None
        )
        self.leaves = tuple(leaf if eqx.is_array(leaf) else _StaticLeaf(leaf) for leaf in leaves)

    def __call__(self, *runtime_args, **kwargs):
        leaves = (leaf.value if isinstance(leaf, _StaticLeaf) else leaf for leaf in self.leaves)
        func, args = jt.unflatten(self.treedef, leaves)
        return func(*args, *runtime_args, **kwargs)


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
        fn_args: Optional positional arguments to pass to the functions as `*args`.
            Array leaves are runtime values. Treat callables and other static
            Python configuration as immutable for compiled solves.
    """

    Ports = FixedPointsPorts
    computation_revision: ClassVar[str] = ANALYSIS_ENGINE_REVISION
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
    def fpfinder(self) -> FixedPointFinder:
        """Construct a low-level finder with this analysis's optimizer."""
        return FixedPointFinder(self.fp_optimizer())

    @property
    def fpf_fn(self):
        """Partial function for finding and filtering fixed points."""
        factory = self.fp_optimizer
        # Concrete factory parameters can participate in Python control flow.
        factory_is_concrete = not any(
            isinstance(leaf, jax.core.Tracer) for leaf in jt.leaves(factory)
        )
        return partial(
            FixedPointFinder._find_and_filter_with_factory,
            _BoundCall(factory, static_func=factory_is_concrete),
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
        def get_fps(func, cands, *args):
            return self.fpf_fn(
                _BoundCall(func, *args),
                cands[:: self.stride_candidates],
            )

        return jt.map(
            get_fps,
            fns,
            candidates,
            *(fn_args or ()),
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
    spread_label: str = eqx.field(kw_only=True)
    variant: Optional[str] = "full"
    fig_params: Mapping = MappingProxyType(dict())

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
