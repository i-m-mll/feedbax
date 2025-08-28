import inspect
import logging
from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from functools import partial
from inspect import Parameter
from operator import is_
from re import A
from types import MappingProxyType
from typing import Any, Concatenate, Generic, Optional, ParamSpec, Self, TypeVar

import equinox as eqx
import feedbax.plotly as fbp
import jax.tree as jt
import plotly.graph_objects as go
from jax_cookbook import is_type
from jax_cookbook.misc import deep_merge
from jax_cookbook.progress import map_rich
from jaxtyping import Array, PyTree

from feedbax_experiments.analysis.analysis import (
    AbstractAnalysis,
    PortsType,
    SinglePort,
)
from feedbax_experiments.config import PLOTLY_CONFIG
from feedbax_experiments.hyperparams import flat_key_to_where_func
from feedbax_experiments.plot import set_axes_bounds_equal
from feedbax_experiments.plot_utils import get_label_str
from feedbax_experiments.tree_utils import ldict_level_to_bottom
from feedbax_experiments.types import AnalysisInputData, LDict

logger = logging.getLogger(__name__)


def _validate_defaults_against_callable(
    cls: type, fn: Callable[..., Any], defaults: Mapping[str, Any]
) -> None:
    """Raise at import time if subclass `fig_params` has keys not accepted by `fig_fn`."""
    sig = inspect.signature(fn)
    params = sig.parameters

    # If the function accepts **kwargs, allow any keys
    has_var_kw = any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return

    # Otherwise only these names are legal as keyword arguments:
    legal = {
        name
        for name, p in params.items()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }

    # Enforce identifier-ness (so they can be shown as parameters)
    bad_ident = [k for k in defaults if not str(k).isidentifier()]
    if bad_ident:
        raise TypeError(f"{cls.__name__}.fig_params contains non-identifier keys: {bad_ident!r}")

    extraneous = [k for k in defaults if k not in legal]
    if extraneous:
        raise TypeError(
            f"{cls.__name__}.fig_params contains keys not accepted by {getattr(fn, '__name__', fn)}: "
            f"{extraneous!r}. Legal kwargs: {sorted(legal)}"
        )


FigFnParams = ParamSpec("FigFnParams")


class AbstractPlotter(AbstractAnalysis[PortsType], Generic[PortsType, FigFnParams]):
    """Base class for analyses that produce plotly figures.

    Note:
        By making this class and its subclasses generics of `FigFnParams`, type inference is
        supplied for the signature of `with_fig_params`. When subclassing, it is necessary to
        properly bind `FigFnParams` to the signature of the `fig_fn` field. See `ScatterN2D` for
        an example.
    """

    fig_fn: Callable[Concatenate[..., FigFnParams], go.Figure] = eqx.field(kw_only=True)

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # Try to obtain a callable to validate against. Subclasses usually set a default.
        fn = getattr(cls, "fig_fn", None)
        if isinstance(fn, staticmethod):  # just in case someone uses staticmethod
            fn = fn.__func__
        if not callable(fn):
            # Can't validate or synthesize a signature without a callable.
            # We'll do a runtime check in __post_init__ instead (below).
            return

        _validate_defaults_against_callable(cls, fn, cls.fig_params)

    def __post_init__(self):
        """If a subclass didn't provide a class-level `fig_fn`, validate at instance time."""
        fn = self.fig_fn
        if callable(fn):
            _validate_defaults_against_callable(type(self), fn, type(self).fig_params)

    def with_fig_params(self, *args: FigFnParams.args, **kwargs: FigFnParams.kwargs) -> Self:
        """Returns a copy of this analysis with updated figure parameters."""
        if any(args):
            logger.warning(
                "No positional args should be given to `with_fig_params`; if your type checker / "
                "IDE suggested you should provide a positional arg, the cause is likely that the "
                f"signature of `fig_fn` was not properly bound in {type(self).__name__}."
            )
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            MappingProxyType(deep_merge(self.fig_params, kwargs)),
        )


class ScatterPlots(AbstractPlotter[SinglePort, FigFnParams]):
    """General 2D plotter (lines/markers) over a PyTree of arrays.

    Example:
        Use the output of `AlignedVars` as input. Set `subplot_level` to "vars" to get one
        subplot per aligned variable. Set `colorscale_axis=0` and `colorscale_key="pert__amp"`,
        and modify the instance with `after_stacking("pert__amp")` to get one series/curve per
        "pert__amp" value on each subplot.

    Fields:
        subplot_level: If set, maps keys at this level to subplots.
        colorscale_axis: Which axis of the array leaves to plot as separate series.
        colorscale_key: If set, determines the colorscale to use across the series.
    """

    Ports = SinglePort[Array]
    inputs: SinglePort[Array] = eqx.field(  # pyright: ignore[reportGeneralTypeIssues]
        kw_only=True, converter=SinglePort[Array].converter
    )
    fig_fn: Callable[Concatenate[PyTree, FigFnParams], go.Figure] = fbp.trajectories  # pyright: ignore[reportAssignmentType]

    subplot_level: str | None = None
    colorscale_key: Optional[str] = None

    # Node-level default fig parameters; everything is optional.
    fig_params: Mapping[str, Any] = MappingProxyType(
        dict(
            colorscale_axis=0,
            # colorscale=PLOTLY_CONFIG.default_colorscale,
            lighten_mean=PLOTLY_CONFIG.mean_lighten_factor,
            n_curves_max=20,
            layout_kws=dict(
                width=1200,
                height=300,
                margin_t=50,
                margin_b=20,
                legend_tracegroupgap=1,
                legend_itemsizing="constant",
            ),
            scatter_kws=dict(
                line_width=0.5,
                opacity=0.5,
                mode="lines",
            ),
            mean_scatter_kws=dict(
                line_width=2.5,
                opacity=1,
            ),
        )
    )

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        input,
        colorscales,
        hps_common,
        **kwargs,
    ) -> PyTree[go.Figure]:
        fig_params = self.fig_params

        if self.colorscale_key is not None:
            updates = {}
            if fig_params.get("colorscale") is None:
                updates["colorscale"] = colorscales[self.colorscale_key]
            if fig_params.get("legend_title") is None:
                updates["legend_title"] = get_label_str(self.colorscale_key)
            if fig_params.get("legend_labels") is None:
                updates["legend_labels"] = flat_key_to_where_func(self.colorscale_key)(hps_common)

            if updates:
                fig_params = deep_merge(fig_params, updates)

        #! This is because `fig_fn = fbp.trajectories` which does not like mappings...
        #! However, we should not hardcode this here! Either update `fbp.trajectories` to
        #! handle mappings, or fix `fig_fn` to `fbp.trajectories` / functions with a particular
        #! signature
        fig_params: Mapping[str, Any] = {
            k: v.values() if isinstance(v, LDict) else v for k, v in fig_params.items()
        }

        # Use dummy *args to convince the static checker that `FigFnParams` is satisfied
        # (the ParamSpec could specify more args, but we only use it for kwargs).
        def _make_fig(node_data: PyTree[Array], *_) -> go.Figure:
            return self.fig_fn(jt.leaves(node_data), *_, **fig_params)

        # one_series can be a single array OR an LDict of leaves (arrays -> subplots)
        if self.subplot_level is not None:
            input = ldict_level_to_bottom(self.subplot_level, input)
            is_leaf = LDict.is_of(self.subplot_level)
        else:
            is_leaf = None

        return map_rich(
            _make_fig, input, is_leaf=is_leaf, description="Making figures (ScatterPlots)"
        )
