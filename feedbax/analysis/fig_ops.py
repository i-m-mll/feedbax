"""Figure and final-operation execution helpers for analyses."""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, Optional, Union

import equinox as eqx
import jax.tree as jt
import plotly.graph_objects as go
from equinox import field
from jax_cookbook import LDict, is_type
from jaxtyping import Array, PyTree

from feedbax.analysis.types import AnalysisInputData
from feedbax.config.utils import deep_merge


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigIterCtx:
    """Context object passed to fig_params_fn during figure iteration."""

    level: Optional[str]
    key: Any
    idx: int
    depth: int
    path: tuple[tuple[Optional[str], Any, int], ...]


class FigureSaveTask(NamedTuple):
    """Container for figure save operation parameters."""

    fig: go.Figure
    figure_hash: str
    eval_dir: Path
    save_formats: list[str]
    params: dict[str, Any]


class _PrepOp(NamedTuple):
    name: str
    dep_name: Optional[Union[str, Sequence[str]]]
    fn: Callable[..., Any]
    wrapped_fn: Callable[..., Any]
    params: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class _FigOp(NamedTuple):
    """Figure operation that iteratively generates and aggregates figures."""

    name: str
    dep_name: Optional[Union[str, Sequence[str]]]
    is_leaf: Callable[[Any], bool]
    slice_fn: Callable[[Any, Any], Any]
    items_fn: Callable[[Any], list]
    agg_fn: Callable[[list, list], Any]
    fig_params_fn: Optional[Callable] = None
    params: Optional[dict] = None
    metadata: Optional[dict[str, Any]] = None
    pre_slice_hook: Optional[Callable] = None
    post_agg_hook: Optional[Callable] = None


class _FinalOp(NamedTuple):
    name: str
    fn: Callable[[PyTree[go.Figure]], PyTree[go.Figure]]
    wrapped_fn: Callable[[PyTree[go.Figure]], PyTree[go.Figure]]
    params: Optional[dict[str, Any]] = None
    is_leaf: Optional[Callable[[Any], bool]] = None
    metadata: Optional[dict[str, Any]] = None


class _AnalysisVmapSpec(eqx.Module):
    """Immutable container holding accumulated vmap information for an analysis."""

    in_axes_spec: Mapping[str, tuple[int | None, ...]] = field(default_factory=dict)
    vmapped_dep_names: tuple[str, ...] = ()
    in_axes_sequence: tuple[tuple[int | None, ...], ...] = ()


_FinalOpKeyType = Literal["results", "figs"]


def _combine_figures(
    figs_list: list[PyTree[go.Figure]],
    items_iterated: Iterable,
) -> PyTree[go.Figure]:
    """Merge traces from multiple figures into a single one."""

    def combine_figs(*figs):
        if not figs:
            return None

        layout = figs[0].layout

        if layout.legend.traceorder == "reversed":
            layout.legend.traceorder = "grouped+reversed"

        if layout.legend.grouptitlefont.style is None:
            layout.legend.grouptitlefont.style = "italic"

        traces = [trace for fig in figs for trace in fig.data]

        fig = go.Figure(data=traces, layout=layout)
        return fig

    return jt.map(
        combine_figs,
        *figs_list,
        is_leaf=is_type(go.Figure),
    )


def _axis_items_fn(axis: int, leaf: Array) -> Iterable:
    if not isinstance(leaf, Array) or axis >= leaf.ndim:
        raise ValueError(f"Combine target for axis {axis} is not Array or axis out of bounds.")
    return range(leaf.shape[axis])


def _axis_slice_fn(axis: int, node: Array, idx: int) -> Array:
    return node[(slice(None),) * axis + (idx,)]


def _level_slice_fn(node: LDict, item: Any) -> Any:
    return node[item]


def _level_items_fn(level: str, leaf: LDict) -> Iterable:
    if not LDict.is_of(level)(leaf):
        raise TypeError(f"Map target for level '{level}' is not an LDict with that label.")
    return leaf.keys()


def _reconstruct_ldict_aggregator(
    level: str, figs_list: list[PyTree], items_iterated: Iterable
) -> LDict:
    return LDict.of(level)(dict(zip(items_iterated, figs_list)))


def _call_fig_params_fn(fn, fp, ctx, **kwargs):
    """Call fig_params_fn with ctx-style signature."""
    return fn(fp, ctx, **kwargs)


def _apply_fig_ops(
    analysis: Any, data: AnalysisInputData, depth: int, path: tuple, **kwargs
):
    """Recursively apply figure operations outer to inner."""
    if depth >= len(analysis._fig_ops):
        return analysis.make_figs(data, **kwargs)

    op = analysis._fig_ops[depth]

    target_dep_names = analysis._get_target_dependency_names(op.dep_name, kwargs, "Fig op")
    deps = {k: kwargs[k] for k in target_dep_names if k in kwargs}
    if not deps:
        logger.warning("No varying dependencies for fig-op %s; falling back to make_figs.", op.name)
        return analysis.make_figs(data, **kwargs)

    first_dep = next(iter(deps.values()))
    leaves = jt.leaves(first_dep, is_leaf=op.is_leaf)
    if not leaves:
        logger.error("Fig-op %s found no matching leaves; falling back.", op.name)
        return analysis.make_figs(data, **kwargs)
    sample_leaf = leaves[0]

    ref_keys = list(op.items_fn(sample_leaf))
    for name, dep in deps.items():
        keys = list(op.items_fn(jt.leaves(dep, is_leaf=op.is_leaf)[0]))
        if keys != ref_keys:
            logger.warning(
                "Fig-op %s: keys for %r differ from reference; proceeding with reference ordering.",
                op.name,
                name,
            )

    children = []

    for i, key in enumerate(ref_keys):
        sliced_kwargs = dict(kwargs)
        for k, v in deps.items():
            sliced_kwargs[k] = jt.map(
                lambda x: op.slice_fn(x, key) if op.is_leaf(x) else x,
                v,
                is_leaf=op.is_leaf,
            )

        data_i = data
        if "data.states" in sliced_kwargs:
            data_i = eqx.tree_at(lambda d: d.states, data, sliced_kwargs["data.states"])
            del sliced_kwargs["data.states"]

        ctx = FigIterCtx(
            level=op.params.get("level"),
            key=key,
            idx=i,
            depth=depth,
            path=path,
        )

        if op.pre_slice_hook is not None:
            data_i, sliced_kwargs = op.pre_slice_hook(data_i, sliced_kwargs, ctx)

        analysis_i = analysis
        if op.fig_params_fn is not None:
            new_fp = op.fig_params_fn(analysis.fig_params, ctx, **sliced_kwargs)
            analysis_i = eqx.tree_at(
                lambda a: a.fig_params,
                analysis,
                MappingProxyType(deep_merge(analysis.fig_params, new_fp)),
            )

        child = _apply_fig_ops(
            analysis_i, data_i, depth + 1, path + ((ctx.level, key, i),), **sliced_kwargs
        )
        children.append(child)

    figs = op.agg_fn(children, ref_keys)

    if op.post_agg_hook is not None:
        figs = op.post_agg_hook(figs, ref_keys, path)

    return figs


def _apply_final_ops(
    analysis: Any, kind: Literal["figs", "results"], tree: PyTree, **kwargs
):
    for final_op in analysis._final_ops_by_type.get(kind, ()):
        try:
            tree = jt.map(
                lambda leaf: final_op.wrapped_fn(leaf, **kwargs),
                tree,
                is_leaf=final_op.is_leaf,
            )
        except Exception as e:
            logger.error(
                f"Error during execution of final {kind} op '{final_op.name}'",
                exc_info=True,
            )
            raise e

    return tree
