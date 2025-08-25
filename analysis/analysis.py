import dataclasses
import functools
import inspect
import logging
import pprint
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial, wraps
from itertools import chain
from pathlib import Path
from types import EllipsisType, MappingProxyType, SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Self,
    TypeAlias,
    TypeVar,
    Union,
)

import dill as pickle
import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
import jax_cookbook.tree as jtree
import plotly.graph_objects as go
from equinox import Module, field
from feedbax.task import AbstractTask
from jax_cookbook import is_module, is_none, is_type, vmap_multi
from jax_cookbook._vmap import AxisSpec, expand_axes_spec
from jaxtyping import Array, ArrayLike, PyTree
from ruamel.yaml import YAML
from sqlalchemy.orm import Session

from feedbax_experiments.config import PATHS, STRINGS
from feedbax_experiments.database import EvaluationRecord, add_evaluation_figure, savefig
from feedbax_experiments.misc import (
    camel_to_snake,
    deep_merge,
    field_names,
    get_dataclass_fields,
    get_md5_hexdigest,
    get_name_of_callable,
    get_origin_type,
    is_json_serializable,
)
from feedbax_experiments.plot_utils import figs_flatten_with_paths
from feedbax_experiments.tree_utils import (
    DoNotHashTree,
    _hash_pytree,
    first_shape,
    hash_callable_leaves,
    ldict_label_only_func,
    ldict_level_to_bottom,
    ldict_level_to_top,
    move_ldict_level_above,
    rearrange_ldict_levels,
    subdict,
    tree_level_labels,
)
from feedbax_experiments.types import AnalysisInputData, LDict, TreeNamespace

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar  # noqa: F401


logger = logging.getLogger(__name__)


yaml = YAML(typ="safe")
yaml.default_flow_style = None


PARAM_SEQ_LEN_TRUNCATE = 9
RESULTS_CACHE_SUBDIR = "results"


@dataclass(frozen=True)
class AnalysisRef[T]:
    """A thin, typed pointer to something that will yield a PyTree[T]."""

    target: Union[str, "AbstractAnalysis"]  # name or instance


# The user asked for AbstractAnalysis without a type parameter in InputOf.
# We leave it raw here – static tools won't check graph consistency, only help
# with autocomplete / annotations.
type InputOf[T] = Union[
    AnalysisRef[PyTree[T]],
    str,
    AbstractAnalysis,  # intentionally un-parameterised
    _DataField,
    Transformed,
    ExpandTo,
    LiteralInput,
]


class AbstractAnalysisPorts(Module, Mapping):
    """Base class for typed analysis input ports."""

    @classmethod
    def converter(cls, inputs: Self | Mapping[str, Any]) -> Self:
        """Convert inputs to the appropriate format."""
        if isinstance(inputs, Mapping):
            return cls(**inputs)
        else:
            return inputs

    def __getitem__(self, key: str):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return (f.name for f in dataclasses.fields(self))

    def __len__(self):
        return len(dataclasses.fields(self))


class NoPorts(AbstractAnalysisPorts):
    """An empty Ports dataclass for analyses with no additional inputs."""

    pass


T = TypeVar("T")


class SinglePort(AbstractAnalysisPorts, Generic[T]):
    """A Ports dataclass with a single port named 'input'."""

    input: InputOf[T] = eqx.field(kw_only=True)


PortsType = TypeVar("PortsType", bound=AbstractAnalysisPorts)


# Define a string representer for objects PyYAML doesn't know how to handle
def represent_undefined(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.representer.add_representer(object, represent_undefined)


@dataclass(frozen=True, slots=True)
class _DataField:
    """Description of what to extract from `AnalysisInputData`.

    Parameters
    ----------
    attr
        Name of the attribute to forward (must be one of the dataclass fields
        of `AnalysisInputData`).
    where, is_leaf
        If *where* is provided the forwarded value becomes

        ``jax.tree.map(where, getattr(data, attr), is_leaf=is_leaf)``.
    """

    attr: str
    where: Optional[Callable] = None
    is_leaf: Optional[Callable[[Any], bool]] = is_module

    # Allow the author to write `Data.states(where=..., is_leaf=...)`
    def __call__(
        self,
        *,
        where: Optional[Callable] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> Self:
        """Return a new `_DataField` overriding *where* and/or *is_leaf*.

        Any argument left as ``None`` inherits the value from the receiver
        instance so that `Data.states(where=...)` keeps the default
        ``is_leaf=is_module``.
        """
        return type(self)(
            self.attr,
            where if where is not None else self.where,
            is_leaf if is_leaf is not None else self.is_leaf,
        )

    def __repr__(self):  # noqa: D401
        return f"Data.{self.attr}"


T = TypeVar("T")


@jtu.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class ExpandTo:
    """Specifies that a PyTree input should be prefix-expanded to match another input's structure.

    This is useful when one input lacks the inner structure necessary for tree operations
    with another input. For example, if `funcs` has structure ['sisu', 'train__pert__std']
    but `func_args` is a tuple lacking the 'train__pert__std' level, you can use:

        func_args=ExpandTo("funcs", tuple((Data.hps(...), Data.tasks(...))))

    The `source` will be prefix-expanded to match the structure of `target`.

    Parameters
    ----------
    target
        Reference to the input whose structure should be matched. Can be either:
        - A string referring to another input in the same analysis
        - A _DataField (e.g., Data.models)
    source
        The PyTree to be prefix-expanded
    where
        Optional function to select a subtree of the target for expansion.
        For example: where=lambda func_args: func_args[0] to expand to just the
        first element of a tuple target instead of the entire tuple structure.
    is_leaf, is_leaf_prefix
        Optional leaf predicates passed to the prefix expansion operation
    """

    target: Union[str, _DataField]
    source: PyTree
    where: Optional[Callable] = None
    is_leaf: Optional[Callable] = None
    is_leaf_prefix: Optional[PyTree[Callable]] = None

    @classmethod
    def map(
        cls,
        target: Union[str, _DataField],
        source: T,
        is_leaf_prefix: Optional[PyTree[Callable]] = None,
        **kwargs,
    ) -> T:
        """Convenience, for mapping over the first level of a tree."""
        #! TODO: Support mapping to arbitrary levels / leaves, not just first
        #! Though this might get kind of confusing with `is_leaf`, `is_leaf_prefix` already
        #! allowed as params
        source_with_is_leaf = jt.map(
            lambda is_leaf_node_prefix, node: (node, is_leaf_node_prefix),
            is_leaf_prefix,
            source,
            is_leaf=is_none,
        )
        nodes_with_is_leaf, treedef = eqx.tree_flatten_one_level(source_with_is_leaf)
        return jt.unflatten(
            treedef,
            [
                cls(target, node, is_leaf_prefix=is_leaf_node_prefix, **kwargs)
                for node, is_leaf_node_prefix in nodes_with_is_leaf
            ],
        )

    def tree_flatten(self):
        return [self.source], SimpleNamespace(
            target=self.target,
            where=self.where,
            is_leaf=self.is_leaf,
            is_leaf_prefix=self.is_leaf_prefix,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            target=aux_data.target,
            source=children[0],
            where=aux_data.where,
            is_leaf=aux_data.is_leaf,
            is_leaf_prefix=aux_data.is_leaf_prefix,
        )


@jtu.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Transformed:
    """Applies a transformation to resolved dependencies before passing to compute.

    This allows in-place transformation of dependency results. For example:

        some_input=Transformed("states_pca", lambda result: result.batch_transform)

    Or with PyTree sources:

        some_input=Transformed((Data.states, Data.tasks), lambda tree: custom_transform(tree))

    The `source` will be resolved as dependencies first, then transformed.

    Parameters
    ----------
    source
        The PyTree of dependencies to resolve and transform
    transform
        Function to apply to the resolved source dependencies
    """

    source: PyTree
    transform: Callable

    def tree_flatten(self):
        return [self.source], (self.transform,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (transform,) = aux_data
        return cls(source=children[0], transform=transform)

    @classmethod
    def map(cls, source: PyTree, transform: Callable, is_leaf: Optional[Callable] = None) -> PyTree:
        def _map_transform(tree):
            return jt.map(transform, tree, is_leaf=is_leaf)

        return cls(source=source, transform=_map_transform)


@dataclass(frozen=True, slots=True)
class LiteralInput:
    """Wraps a constant value to be passed directly as an analysis input.

    This allows passing literal values without going through dependency resolution.
    For example:

        custom_inputs=dict(some_param=LiteralInput(42), other_param=LiteralInput(jnp.array([1, 2, 3])))

    The wrapped value will be passed directly to the analysis's compute method.

    Parameters
    ----------
    value
        The constant value to pass to the analysis
    """

    value: Any


class _DataProxy:
    """Expose only valid `AnalysisInputData` attributes.

    Any attempt to access a non-existent field fails *eagerly* at import time.
    """

    _allowed = tuple(AnalysisInputData.__annotations__.keys())

    def __getattr__(self, item: str) -> _DataField:  # noqa: D401
        if item not in self._allowed:
            raise AttributeError(
                f"'Data' has no attribute '{item}'. "
                f"Valid attributes are: {', '.join(self._allowed)}"
            )
        return _DataField(item)

    def __repr__(self):  # noqa: D401
        return "Data"


"""Sentinel for forwarding attributes from `AnalysisInputData` to analysis input ports."""
Data = _DataProxy()


@dataclass(frozen=True)
class FigIterCtx:
    """Context object passed to fig_params_fn during figure iteration.

    Provides information about the current iteration state for context-aware
    figure parameter updates.
    """

    level: Optional[str]  # the LDict level label for this mapping step (or None for axis maps)
    key: Any  # the selected key at this level (or index for axis maps)
    idx: int  # 0-based index of `key` within items at this level
    depth: int  # 0 = outermost mapped level
    path: tuple[tuple[Optional[str], Any, int], ...]  # cumulative selections from outermost→current


class _PrepOp(NamedTuple):
    name: str
    dep_name: Optional[Union[str, Sequence[str]]]  # Dependencies to transform
    transform_func: Callable[..., Any]
    params: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class _FigOp(NamedTuple):
    name: str
    dep_name: Optional[Union[str, Sequence[str]]]
    is_leaf: Callable[[Any], bool]
    slice_fn: Callable[[Any, Any], Any]  # (leaf, item) -> sliced_leaf
    items_fn: Callable[[Any], list]  # (leaf) -> items at this level
    agg_fn: Callable[[list, list], Any]  # (list(child_figs), items) -> aggregated_figs_tree
    fig_params_fn: Optional[Callable] = None
    params: Optional[dict] = None
    metadata: Optional[dict[str, Any]] = None

    # Optional hooks (default None)
    pre_slice_hook: Optional[Callable] = None
    post_agg_hook: Optional[Callable] = None


class _FinalOp(NamedTuple):
    name: str
    transform_func: Callable[[PyTree[go.Figure]], PyTree[go.Figure]]
    params: Optional[dict[str, Any]] = None
    is_leaf: Optional[Callable[[Any], bool]] = None
    metadata: Optional[dict[str, Any]] = None


def _process_param(param: Any) -> Any:
    """
    Process parameter values for serialization in the database.
    """
    # Process value based on its type
    if isinstance(param, Callable):
        return get_name_of_callable(param)
    elif isinstance(param, Mapping):
        # Preserve structure but ensure keys are strings
        return {str(mk): _process_param(mv) for mk, mv in param.items()}
    elif isinstance(param, (list, tuple)) or eqx.is_array(param):
        # Convert to list
        param_list = list(str(p) for p in param)
        if len(param_list) > PARAM_SEQ_LEN_TRUNCATE:
            return f"[{', '.join(param_list[:PARAM_SEQ_LEN_TRUNCATE])}, ..., {param_list[-1]}]"
        else:
            return f"[{', '.join(param_list)}]"
    else:
        # Simple types
        return param


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


def _format_level_str(label: str):
    return label.replace(STRINGS.hps_level_label_sep, "-").replace("_", "")


def _merge_metadata(*metadata_dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple metadata dictionaries, handling list-valued keys specially.

    For keys "descriptions" and "labels", assumes values are lists and concatenates them.
    For other keys, uses normal dict update behavior (later arguments take precedence).
    None arguments are skipped gracefully.

    Returns:
        A new metadata dictionary with merged contents.
    """
    result = dict(descriptions=[], labels=[])

    for metadata in metadata_dicts:
        if metadata is None:
            continue

        for key, value in metadata.items():
            if key in ("descriptions", "labels"):
                # Concatenate list values
                if isinstance(value, list):
                    result[key].extend(value)
                else:
                    result[key].append(value)
            else:
                # Normal dict update behavior
                result[key] = value

    return result


def _format_dict_of_params(d: dict, join_str: str = ", "):
    # For constructing parts of `AbstractAnalysis.__str__`
    return join_str.join([f"{k}={_process_param(v)}" for k, v in d.items()])


_NAME_NORMALIZATION = {"states": "data.states"}


def _normalize_name(name: str) -> str:
    return _NAME_NORMALIZATION.get(name, name)


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
    # items_iterated here will be the keys from the LDict level
    # Rebuild the LDict using the original level label
    return LDict.of(level)(dict(zip(items_iterated, figs_list)))


def get_validation_trial_specs(task: AbstractTask):
    # TODO: Support any number of extra axes (i.e. for analyses that vmap over multiple axes in their task/model objects)
    # if len(task.workspace.shape) == 3:
    #     #! I don't understand why/if this is necessary
    #     return eqx.filter_vmap(lambda task: task.validation_trials)(task)
    # else:
    return task.validation_trials


def _extract_vmapped_kwargs_to_args(func, vmapped_dep_names: Sequence[str]):
    """Convert specified kwargs to positional args for vmapping."""

    def modified_func(data, *vmapped_deps, **remaining_kwargs):
        # Reconstruct the full kwargs dict
        full_kwargs = remaining_kwargs | dict(zip(vmapped_dep_names, vmapped_deps))
        return func(data, **full_kwargs)

    return modified_func


def _build_in_axes_sequence(
    dependency_axes: Mapping[str, tuple[int | None, ...]],
    vmapped_dep_names: tuple[str, ...],
) -> tuple[tuple[int | None, ...], ...]:
    """Build the in_axes_sequence for vmapping over inputs to `AbstractAnalysis.compute`."""
    if not dependency_axes:
        return ()

    max_levels = max(len(axes) for axes in dependency_axes.values())

    in_axes_sequence = []
    for level in range(max_levels):
        # Data axis
        data_axis = None
        if "data.states" in dependency_axes:
            state_axes = dependency_axes["data.states"]
            if level < len(state_axes):
                data_axis = AnalysisInputData(None, None, state_axes[level], None, None)
            else:
                data_axis = AnalysisInputData(None, None, None, None, None)
        else:
            data_axis = AnalysisInputData(None, None, None, None, None)

        # Vmapped dependency axes
        vmapped_dep_axes = []
        for dep_name in vmapped_dep_names:
            if dep_name in dependency_axes:
                dep_axes = dependency_axes[dep_name]
                if level < len(dep_axes):
                    vmapped_dep_axes.append(dep_axes[level])
                else:
                    vmapped_dep_axes.append(None)
            else:
                vmapped_dep_axes.append(None)

        in_axes_sequence.append((data_axis, *vmapped_dep_axes))

    return tuple(in_axes_sequence)


class _AnalysisVmapSpec(eqx.Module):
    """Immutable container holding all accumulated vmap information
    for an AbstractAnalysis instance."""

    # Logical → per-level axis schedule (one tuple entry per vmap level)
    in_axes_spec: Mapping[str, tuple[int | None, ...]] = field(default_factory=dict)
    # Names of dependencies (kwargs) that must be popped before vmapping
    vmapped_dep_names: tuple[str, ...] = ()
    # Fully-resolved positional in_axes_sequence for vmap_multi
    in_axes_sequence: tuple[tuple[int | None, ...], ...] = ()


_FinalOpKeyType = Literal["results", "figs"]


# By using `strict=False`, we can define non-abstract fields, i.e. without needing to
# implement them trivially in subclasses. This violates the abstract-final design
# pattern. This is intentional. If it leads to problems, I will learn from that.
def _apply_fig_ops(analysis, data, kwargs, depth: int, path: tuple):
    """Recursively apply figure operations outer→inner.

    Args:
        analysis: The AbstractAnalysis instance
        data: AnalysisInputData
        kwargs: Dependency kwargs
        depth: Current recursion depth (0 = outermost)
        path: Cumulative path selections from outermost→current

    Returns:
        PyTree of figures
    """
    if depth >= len(analysis._fig_ops):
        # Base case: no more fig-ops
        return analysis.make_figs(data, **kwargs)

    op = analysis._fig_ops[depth]

    # Choose dependencies that will vary at this level
    target_dep_names = analysis._get_target_dependency_names(op.dep_name, kwargs, "Fig op")
    deps = {k: kwargs[k] for k in target_dep_names if k in kwargs}
    if not deps:
        logger.warning("No varying dependencies for fig-op %s; falling back to make_figs.", op.name)
        return analysis.make_figs(data, **kwargs)

    # Find a representative leaf for items
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
        # Slice kwargs for this item
        sliced_kwargs = dict(kwargs)
        for k, v in deps.items():
            sliced_kwargs[k] = jt.map(
                lambda x: op.slice_fn(x, key) if op.is_leaf(x) else x,
                v,
                is_leaf=op.is_leaf,
            )

        # (Optional) move states in/out as in current pipeline
        data_i = data
        if "data.states" in sliced_kwargs:
            data_i = eqx.tree_at(lambda d: d.states, data, sliced_kwargs["data.states"])
            del sliced_kwargs["data.states"]

        # Context
        ctx = FigIterCtx(
            level=op.params.get("level"),
            key=key,
            idx=i,
            depth=depth,
            path=path,
        )

        # Optional hook before recursion
        if op.pre_slice_hook is not None:
            data_i, sliced_kwargs = op.pre_slice_hook(data_i, sliced_kwargs, ctx)

        # Per-level fig params (ctx-aware)
        analysis_i = analysis
        if op.fig_params_fn is not None:
            new_fp = _call_fig_params_fn(op.fig_params_fn, analysis.fig_params, ctx)
            analysis_i = eqx.tree_at(
                lambda a: a.fig_params,
                analysis,
                MappingProxyType(deep_merge(analysis.fig_params, new_fp)),
            )

        # Recurse
        child = _apply_fig_ops(
            analysis_i, data_i, sliced_kwargs, depth + 1, path + ((ctx.level, key, i),)
        )
        children.append(child)

    # Aggregate children at this depth
    figs = op.agg_fn(children, ref_keys)

    # Optional post-aggregation hook
    if op.post_agg_hook is not None:
        figs = op.post_agg_hook(figs, ref_keys, path)

    return figs


def _call_fig_params_fn(fn, fp, ctx):
    """Call fig_params_fn with ctx-style signature.

    Args:
        fn: The fig_params_fn to call
        fp: Current figure parameters
        ctx: Context object

    Returns:
        Updated figure parameters
    """
    return fn(fp, ctx)


def _get_vmap_spec_debug_str(
    node: eqx.Module,
    vmap_spec: _AnalysisVmapSpec,
    prepped_data: AnalysisInputData,
    vmapped_deps: list[PyTree[Any]],
) -> str:
    def _pformat(obj: Any) -> str:
        """Pretty-print an object, ensuring it fits within a reasonable width."""
        return pprint.pformat(obj, width=60, compact=True).replace("\n", "\n\t\t")

    example_leaf_shapes_str = "\n\t\t".join(
        [
            "example leaf shapes:",
            f"data.states: {first_shape(prepped_data.states)}",
            *[
                f"{k}: {first_shape(dep)}"
                for k, dep in zip(vmap_spec.vmapped_dep_names, vmapped_deps)
            ],
        ]
    )

    return "\n\t".join(
        [
            f"Vectorizing {node.__class__.__name__}.compute according to:",
            f"vmapped_dep_names:\n\t\t{_pformat(vmap_spec.vmapped_dep_names)}",
            f"in_axes_spec:\n\t\t{_pformat(vmap_spec.in_axes_spec)}",
            f"in_axes_sequence:\n\t\t{_pformat(vmap_spec.in_axes_sequence)}",
            example_leaf_shapes_str,
        ]
    )


class AbstractAnalysis(Module, Generic[PortsType], strict=False):
    """Component in an analysis pipeline.

    In `run_analysis`, multiple sets of evaluations may be performed
    prior to analysis. In particular, we may evaluate a full/large set
    of task conditions for statistical purposes, and evaluate a smaller
    version for certain visualizations. Thus `AbstractAnalysis`
    subclasses expect arguments `models`, `tasks`, `states`, and `hps` all
    of which are PyTrees. The top-level structure of these PyTrees is always
    a #TODO

    Now, while it may be the case that an analysis would depend on both the
    larger and smaller variants (in our example), we still must specify only a
    single `variant`, since this determines the hyperparameters that are passed
    to `analysis.save`. Thus it is assumed that all figures that result from a
    call to some `AbstractAnalysis.make_figs` will be best associated with only
    one (and always the same one) of the eval variants.

    TODO: If we return the hps on a fig-by-fig basis from within `make_figs`, then
    we could avoid this limitation.

    Class attributes:
        Ports: The structure and defaults for the input ports of the analysis subclass.

    Fields:
        inputs: The port->input mapping for the analysis instance.
        variant: Label of the evaluation variant this analysis uses (primarily).
    """

    _exclude_fields = (
        "inputs",
        "fig_params",
        "cache_result",
        "_prep_ops",
        "_fig_ops",
        "_final_ops_by_type",
        "_extra_inputs",
        "_estimate_mem_preflight",
    )

    Ports: ClassVar[type[PortsType]] = NoPorts  # type: ignore
    inputs: PortsType = field(default_factory=NoPorts, converter=NoPorts.converter)

    variant: Optional[str] = (
        None  #! TODO: Eliminate this. Should be in `tasks` PyTree, and dealt with explicitly with ops
    )
    fig_params: Mapping[str, Any] = MappingProxyType(dict())
    cache_result: bool = False

    #! TODO: Make these `init=False` so they don't appear in the constructor signature IDE
    #! suggestions. (However, this leads to issues with `eqx.tree_at` for some reason.)
    #! One solution might be (for one) to make a `_PrepOps` wrapper class for the tuple.
    _prep_ops: tuple[_PrepOp, ...] = field(default=())
    _vmap_spec: Optional[_AnalysisVmapSpec] = field(default=None)
    _fig_ops: tuple[_FigOp, ...] = field(default=())
    _final_ops_by_type: Mapping[_FinalOpKeyType, tuple[_FinalOp, ...]] = field(
        default_factory=lambda: MappingProxyType({"results": (), "figs": ()})
    )
    _extra_inputs: dict[str, Any] = field(default_factory=dict)
    _estimate_mem_preflight: Literal["off", "stages", "final"] = field(default="off")

    def __post_init__(self):
        """Validate inputs instance and check for unresolved required inputs."""
        # Validate that inputs is an instance of the expected Ports class
        ports_origin_type = get_origin_type(self.Ports)
        if not isinstance(self.inputs, ports_origin_type):
            raise TypeError(f"Expected inputs of type {self.Ports}, got {type(self.inputs)}")

    def _compute_with_ops(
        self,
        data: AnalysisInputData,
        **kwargs,
    ) -> dict[str, PyTree[Any]]:
        """Perform computations with prep-ops, vmap, and result final-ops applied."""
        # Transform inputs prior to performing the analysis
        # e.g. see `after_stacking` for an example of defining a pre-op
        prepped_kwargs = self._run_prep_ops(data, kwargs)
        # Transformed `data.states` end up in own temporary key, so move back into `data`
        prepped_data = eqx.tree_at(lambda d: d.states, data, prepped_kwargs["data.states"])
        del prepped_kwargs["data.states"]

        def _run_compute():
            if self._vmap_spec is not None:
                compute_func = _extract_vmapped_kwargs_to_args(
                    self.compute, self._vmap_spec.vmapped_dep_names
                )
                vmapped_deps = [
                    prepped_kwargs.pop(name) for name in self._vmap_spec.vmapped_dep_names
                ]
                logger.debug(
                    _get_vmap_spec_debug_str(self, self._vmap_spec, prepped_data, vmapped_deps)
                )
                compute_func = partial(compute_func, **prepped_kwargs)
                return vmap_multi(compute_func, in_axes_sequence=self._vmap_spec.in_axes_sequence)(
                    prepped_data,
                    *vmapped_deps,
                )
            else:
                compute_fn = partial(self.compute, **prepped_kwargs)
                return compute_fn(prepped_data)

        def _try_load_result_from_cache() -> tuple[Optional[Path], Optional[Any]]:
            """Attempt to load the result from cache."""
            cache_root = PATHS.cache / RESULTS_CACHE_SUBDIR
            cache_root.mkdir(parents=True, exist_ok=True)

            try:
                inputs_hash = _hash_pytree((prepped_data, prepped_kwargs))
            except Exception as e:
                logger.error(
                    f"Failed to hash inputs for caching of {self.name}: {e}", exc_info=True
                )
                # Fallback: disable caching for this invocation
                return None, None

            cache_fname = f"{self.name}_{self.md5_str}_{inputs_hash}.pkl"
            cache_path = cache_root / cache_fname

            if cache_path.exists():
                try:
                    logger.info(f"Loading cached result for {self.name}")
                    with open(cache_path, "rb") as f:
                        return None, pickle.load(f)
                except Exception as e:
                    logger.warning(
                        f"Could not load cached result for {self.name} (will recompute): {e}"
                    )

            return cache_path, None

        result = None
        cache_path, cached = None, None

        if self.cache_result:
            cache_path, result = _try_load_result_from_cache()

        if result is None:
            result = _run_compute()

        if self.cache_result and cache_path is not None:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                logger.info(f"Saved cache for {self.name} to {cache_path}")
            except Exception as e:
                logger.warning(f"Could not save cache for {self.name}: {e}")

        for final_op in self._final_ops_by_type.get("results", ()):
            try:
                result = _call_user_func(
                    final_op.transform_func,
                    result,
                    data=data,
                    **prepped_kwargs,
                )
            except Exception as e:
                logger.error(f"Error during execution of final op '{final_op.name}'", exc_info=True)
                raise e

        return result

    def _make_figs_with_ops(
        self,
        data: AnalysisInputData,
        result: dict[str, PyTree[Any]],
        **kwargs,
    ) -> PyTree[go.Figure]:
        """Generate figures with fig-ops and figure final-ops applied."""

        # Transform dependencies prior to making figures

        prepped_kwargs = self._run_prep_ops(data, kwargs)

        prepped_data = eqx.tree_at(lambda d: d.states, data, prepped_kwargs["data.states"])
        del prepped_kwargs["data.states"]

        figs: PyTree[go.Figure] = None
        if self._fig_ops:
            # Prepare by amalgamating all kwargs including results
            prepped_kwargs_with_results = prepped_kwargs.copy()
            prepped_kwargs_with_results["data.states"] = prepped_data.states
            prepped_kwargs_with_results["result"] = result

            figs = _apply_fig_ops(self, prepped_data, prepped_kwargs_with_results, 0, ())
        else:
            figs = self.make_figs(prepped_data, result=result, **prepped_kwargs)

        for final_fig_op in self._final_ops_by_type.get("figs", ()):
            try:
                figs = _call_user_func(
                    final_fig_op.transform_func,
                    figs,
                    data=prepped_data,
                    result=result,
                    **prepped_kwargs,
                )
            except Exception as e:
                logger.error(
                    f"Error during execution of final op '{final_fig_op.name}'", exc_info=True
                )
                raise e

        return figs

    def __call__(
        self,
        data: AnalysisInputData,
        **kwargs,
    ) -> tuple[PyTree[Any], PyTree[go.Figure]]:
        """Perform analysis: compute results and generate figures."""
        result = self._compute_with_ops(data, **kwargs)
        figs = self._make_figs_with_ops(data, result, **kwargs)
        return result, figs

    @classmethod
    def _input_leaf_types(cls) -> tuple[type, ...]:
        """Get the set of valid input leaf types for this analysis."""
        return (str, _DataField, AbstractAnalysis, LiteralInput)

    def is_analysis_input_leaf(self, leaf: Any) -> bool:
        """Determine if a leaf is a valid analysis input type."""
        return isinstance(leaf, self._input_leaf_types())

    @property
    def _flattened_inputs(self) -> dict[str, list]:
        """Get flattened dependency sources for each input name.

        Each PyTree input is flattened to a list of leaves. ExpandTo and Transformed
        are registered as PyTrees so only their .source children are included in leaves.
        """
        flattened = {}
        for name, source in chain(self.inputs.items(), self._extra_inputs.items()):
            # Flatten the PyTree - ExpandTo/Transformed registration ensures only actual dependencies are leaves
            leaves = jt.leaves(source, is_leaf=self.is_analysis_input_leaf)

            # Validate all leaves are valid dependency types
            for leaf in leaves:
                if not (
                    isinstance(leaf, (type, str))
                    or isinstance(leaf, (_DataField, AbstractAnalysis, LiteralInput))
                ):
                    valid_types = ", ".join([t.__name__ for t in self._input_leaf_types()])
                    raise ValueError(
                        f"Invalid dependency leaf in '{name}': {type(leaf)}. "
                        f"All leaves must be one of: {valid_types}"
                    )
            flattened[name] = leaves
        return flattened

    @property
    def _input_treedefs(self) -> dict[str, Any]:
        """Get tree definitions for reconstructing PyTree inputs.

        Returns TreeDef for dependency trees. ExpandTo/Transformed PyTree registration
        automatically handles source substitution during flattening/unflattening.
        """
        return {
            name: jt.structure(source, is_leaf=self.is_analysis_input_leaf)
            for name, source in chain(self.inputs.items(), self._extra_inputs.items())
        }

    def _get_target_dependency_names(
        self,
        dep_name_spec: Optional[Union[str, Sequence[str]]],
        available_kwargs: Dict[str, Any],
        op_context: str,  # e.g., "Prep-op", "Fig op"
    ) -> list[str]:
        """
        Determines the list of valid dependency names based on the specification
        and available kwargs.
        """
        target_names: list[str] = []

        if dep_name_spec is None:
            # Use all dependencies relevant to this analysis instance found in kwargs
            target_names = [k for k in self.inputs if k in available_kwargs]
            # Process the states and `compute` results by default
            target_names.append("data.states")
            # ? Don't include 'result' by default?
            if available_kwargs.get("result") is not None:
                target_names.append("result")
            if not target_names and self.inputs:  # Log only if dependencies were expected
                logger.warning(
                    f"{op_context} needs dependencies (dep_name_spec=None), but none found in kwargs."
                )
        else:
            # dep_name_spec = jt.map(_normalize_name, dep_name_spec)
            if isinstance(dep_name_spec, str):
                # Single dependency name
                if dep_name_spec in available_kwargs:
                    target_names = [dep_name_spec]
                else:
                    logger.warning(
                        f"{op_context} dependency '{dep_name_spec}' not found in available kwargs."
                    )
            elif isinstance(dep_name_spec, Sequence):
                # Sequence of dependency names
                target_names = [name for name in dep_name_spec if name in available_kwargs]
                if len(target_names) < len(dep_name_spec):
                    missing = set(dep_name_spec) - set(target_names)
                    logger.warning(
                        f"{op_context} dependencies missing from kwargs: {missing}. Proceeding with available: {target_names}"
                    )
                if not target_names:
                    logger.warning(
                        f"{op_context} specified dependencies {dep_name_spec}, but none were found in kwargs."
                    )
            else:
                logger.error(f"Invalid type for {op_context} dep_name_spec: {type(dep_name_spec)}")

        return target_names

    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Return kwargs to be used when instantiating dependencies.

        Subclasses can override this method to provide parameters for their dependencies.
        Returns a dictionary mapping dependency name to a dictionary of kwargs.
        """
        return {}

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ) -> PyTree:
        """Perform computations for the analysis.

        The return value is passed as `result` to `make_figs`, and is also made available to other
        subclasses of `AbstractAnalysis` as defined in their respective`dependencies` attribute.

        Note that the outer (task variant) `dict` level should be retained in the returned PyTree, since generally
        a subclass that implements `compute` is implicitly available as a dependency for other subclasses
        which may depend on data for any variant.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute(). "
            "Either implement this method or the analysis will be skipped during computation."
        )

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        result: Optional[Any],
        **kwargs,
    ) -> PyTree[go.Figure]:
        """Generate figures for this analysis.

        Figures are returned, but are not made available to other subclasses of `AbstractAnalysis`
        which depend on the subclass implementing this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement make_figs(). "
            "Either implement this method or the analysis will be skipped during figure generation."
        )

    def save_figs(
        self,
        db_session: Session,
        eval_info: EvaluationRecord,
        result,
        figs: PyTree[go.Figure],
        hps: PyTree[TreeNamespace],  # dict level: variant
        model_info=None,
        dump_path: Optional[Path] = None,
        dump_formats: Sequence[str] = ("html",),
        label: Optional[str] = None,
        **dependencies,
    ) -> None:
        """
        Save to disk and record in the database each figure in a PyTree of figures, for this analysis.
        """
        # `sep="_"`` switches the label dunders for single underscores, so
        # in `_params_to_save` we can use an argument e.g. `train_pert_std` rather than `train__pert__std`
        param_keys = tree_level_labels(
            figs, label_func=ldict_label_only_func, is_leaf=is_type(go.Figure), sep="_"
        )

        if dump_path is not None:
            dump_path = Path(dump_path)
            dump_path.mkdir(exist_ok=True, parents=True)

        figs_with_paths_flat = figs_flatten_with_paths(figs)

        # Construct this for reference to hps that should only vary with the task variant.
        hps_0 = jt.leaves(hps.get(self.variant, hps), is_leaf=is_type(TreeNamespace))[0]

        ops_params_dict = self._extract_ops_info()

        for i, (path, fig) in enumerate(figs_with_paths_flat):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))

            # Include fields from this instance, but only if they are JSON serializable
            field_params = {k: v for k, v in self._field_params.items() if is_json_serializable(v)}

            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree
                **field_params,  # From the fields of the analysis subclass instance
                **self._params_to_save(  # Implemented by the subclass
                    hps,
                    result=result,
                    **path_params,
                    **dependencies,  # Specified by the subclass `dependency_kwargs`, via `run_analysis`
                ),
                eval_n=hps_0.eval_n,  # ? Some things should always be included
            )

            if ops_params_dict:
                params["ops"] = ops_params_dict

            add_evaluation_figure(
                db_session,
                eval_info,
                fig,
                camel_to_snake(self.name),
                model_records=model_info,
                **params,
            )

            # Additionally dump to specified path if provided
            if dump_path is not None:
                # Create a unique filename using label (if provided), class name and hash
                if label is not None:
                    filename = f"{label}_{self.name}_{i}"
                else:
                    filename = f"{self.name}_{self.md5_str}_{i}"

                savefig(fig, filename, dump_path, dump_formats, metadata=params)

                # Save parameters as YAML
                params_path = dump_path / f"{filename}.yaml"
                try:
                    with open(params_path, "w") as f:
                        yaml.dump(params, f)
                except Exception as e:
                    logger.error(
                        f"Error saving fig dump parameters to {params_path}: {e}", exc_info=True
                    )

    @property
    def _all_ops(self) -> tuple:
        all_final_ops = sum(self._final_ops_by_type.values(), ())
        return self._prep_ops + self._fig_ops + all_final_ops

    def _extract_ops_info(self):
        """
        Extract information about all operations (prep ops and fig op).

        Returns:
            - ops_params_dict: Dictionary with all operations info
        """
        ops_params_dict = {
            op.name: {k: _process_param(v) for k, v in op.params.items()} for op in self._all_ops
        }

        # ops_filename_str = "__".join(op.label for op in self._all_ops)

        return ops_params_dict

    def __str__(self) -> str:
        field_params = dict(variant=self.variant) | dict(self._non_default_field_params)
        op_params_strs = [f"{op.name}({_format_dict_of_params(op.params)})" for op in self._all_ops]
        return ".".join([f"{self.name}({_format_dict_of_params(field_params)})", *op_params_strs])

    def estimate_memory(self, mode: Literal["off", "stages", "final"] = "final") -> Self:
        """Returns a copy of this analysis with memory estimation enabled."""
        if self.compute is AbstractAnalysis.compute:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement compute(). "
                "Memory estimation requires a compute method."
            )

        return eqx.tree_at(lambda x: x._estimate_mem_preflight, self, mode)

    def with_fig_params(self, **kwargs) -> Self:
        """Returns a copy of this analysis with updated figure parameters."""
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            MappingProxyType(deep_merge(self.fig_params, kwargs)),
        )

    def after_indexing(
        self,
        axis: int,
        idxs: ArrayLike | Callable[[Array], ArrayLike],
        axis_label: Optional[str] = None,
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that slices its inputs along an axis before proceeding.

        Args:
            axis: The axis to index along.
            idxs: The indices to take along the axis. Can be an array-like object or a callable
                that takes the shape of the array to be indexed and returns array-like indices.
            axis_label: Optional label for the axis, used to construct the operation label.
            dependency_name: Optional name of one or more dependencies to transform. If None,
                transforms all inputs.
            metadata: Optional metadata containing descriptions and labels.
        """

        if axis_label is None:
            label = f"axis{axis}-idx{idxs}"
        else:
            label = f"{axis_label}-idx{idxs}"

        dep_str = dependency_name or "dependencies"
        description = f"Index {dep_str} along axis {axis} with indices {idxs}"

        def index_func(dep_data, **kwargs):
            return jtree.take(dep_data, idxs, axis)

        # Build metadata with description and label
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name="after_indexing",
            dep_name=dependency_name,
            transform_func=index_func,
            params=dict(axis=axis, idxs=idxs),
            metadata=final_metadata,
        )

    def after_map(
        self,
        func: Callable[[Any], Any],
        is_leaf: Optional[Callable[[Any], bool]] = None,
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that maps a function over the input PyTrees.
        """
        dep_str = dependency_name or "dependencies"
        description = f"Map function {get_name_of_callable(func)} over {dep_str}"
        label = f"map-{get_name_of_callable(func)}"

        # Build metadata with description and label
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name="map",
            dep_name=dependency_name,
            transform_func=lambda dep_data, **kwargs: jt.map(func, dep_data, is_leaf=is_leaf),
            params=dict(func=func),
            metadata=final_metadata,
        )

    def after_transform_states(
        self,
        func: Callable[..., Any],
        level: Optional[str | Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Returns a copy of this analysis that transforms the evaluated states before proceeding."""
        # Generate description for this specific method
        if level is None:
            description = f"Transform states with {get_name_of_callable(func)}"
        else:
            level_str = level if isinstance(level, str) else ",".join(level)
            description = (
                f"Transform states at LDict level '{level_str}' with {get_name_of_callable(func)}"
            )

        # Build metadata with our description
        final_metadata = _merge_metadata(
            metadata,
            dict(
                descriptions=[description],
                labels=[f"transform-states_{get_name_of_callable(func)}"],
            ),
        )

        return self.after_transform(
            func=func,
            level=level,
            dependency_names="data.states",
            metadata=final_metadata,
        )

    def after_transform_inputs(
        self,
        func: Callable[..., Any],
        level: Optional[str | Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Returns a copy of this analysis that transforms the instance's port inputs before proceeding."""
        # Generate description for this specific method
        if level is None:
            description = f"Transform inputs with {get_name_of_callable(func)}"
        else:
            level_str = level if isinstance(level, str) else ",".join(level)
            description = (
                f"Transform inputs at LDict level '{level_str}' with {get_name_of_callable(func)}"
            )

        # Build metadata with our description
        final_metadata = _merge_metadata(
            metadata,
            dict(
                descriptions=[description],
                labels=[f"transform-inputs_{get_name_of_callable(func)}"],
            ),
        )

        return self.after_transform(
            func=func,
            level=level,
            dependency_names=field_names(self.Ports),
            metadata=final_metadata,
        )

    def after_transform(
        self,
        func: Callable[..., Any],
        level: Optional[str | Sequence[str]] = None,
        dependency_names: Optional[str | Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that transforms its inputs before proceeding.

        Applies a function to one or more `LDict` levels, or to the entire input PyTree.
        This is less general than `after_map`.

        Args:
            func: The function to transform the inputs with.
            level: The `LDict` level to apply the transformation to. If None, the transformation is applied to the entire input PyTree.
            dependency_names: The name(s) of the dependencies to transform.
            metadata: Optional metadata containing descriptions and labels.
        """

        dep_str = f"{dependency_names}" if dependency_names else "dependencies"

        if level is None:
            description = f"Transform {dep_str} with {get_name_of_callable(func)}"
            label = f"pre-transform_{get_name_of_callable(func)}"
            transform_func = func

        else:
            if isinstance(level, str):
                levels = [level]
            else:
                levels = level

            description = (
                f"Transform {dep_str} at LDict levels {levels} with {get_name_of_callable(func)}"
            )
            label = f"pre-transform-{', '.join(levels)}_{get_name_of_callable(func)}"

            def _transform_levels(dep_data, levels=levels, **kwargs):
                tree = dep_data
                for level in levels:
                    tree = jt.map(
                        lambda node: _call_user_func(func, node, **kwargs),
                        tree,
                        is_leaf=LDict.is_of(level),
                    )
                return tree

            transform_func = _transform_levels

        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name="after_transform",
            dep_name=dependency_names,
            transform_func=transform_func,
            params=dict(level=level, transform_func=transform_func),
            metadata=final_metadata,
        )

    def after_unstacking(
        self,
        axis: int,
        level_label: str,
        keys: Optional[Sequence[Hashable]] = None,
        dependency_name: Optional[str] = None,
        above_level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that unpacks an array axis into an `LDict` level.

        Args:
            axis: The array axis to unpack
            level_label: The label for the new LDict level
            keys: The keys to use for the LDict entries. If given, must match the length of the axis.
                By default, uses integer keys starting from zero.
            dependency_name: Optional name of specific dependency to transform
            above_level: Optional level to move the new level above
            metadata: Optional metadata containing descriptions and labels.
        """

        dep_str = dependency_name or "dependencies"
        description = f"Unstack {dep_str} axis {axis} into LDict level '{level_label}'"
        label = f"unstack-axis{axis}-to-{_format_level_str(level_label)}"

        def unpack_axis(data_, **kwargs):
            def transform_array(arr):
                nonlocal keys
                if keys is None:
                    keys = range(arr.shape[axis])
                else:
                    # Check if keys length matches the axis length
                    if arr.shape[axis] != len(keys):
                        raise ValueError(
                            f"Length of keys ({len(keys)}) must match the length of axis {axis} ({arr.shape[axis]})"
                        )

                # Move the specified axis to position 0
                arr_moved = jnp.moveaxis(arr, axis, 0)

                # Create an LDict with the specified label
                return LDict.of(level_label)(
                    {key: slice_data for key, slice_data in zip(keys, arr_moved)}
                )

            unstacked = jt.map(
                transform_array,
                data_,
                is_leaf=eqx.is_array,
            )

            if above_level is not None:
                unstacked = jt.map(
                    lambda subtree, above_level=above_level: move_ldict_level_above(
                        level_label,
                        above_level,
                        subtree,
                    ),
                    unstacked,
                    is_leaf=is_type(LDict),
                )

            return unstacked

        # Build metadata with description and label
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name="after_unstacking",
            dep_name=dependency_name,
            transform_func=unpack_axis,
            params=dict(axis=axis, level_label=level_label, above_level=above_level),
            metadata=final_metadata,
        )

    def after_stacking(
        self,
        level: str,
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that stacks its inputs along an `LDict` PyTree level before proceeding.

        This is useful when we have a PyTree of results with an `LDict` level representing
        the values across some variable we want to visually compare, and our analysis
        uses a plotting function that compares across the first axis of input arrays.
        By stacking first, we collapse the `LDict` level into the first axis so that the
        plotting function will compare (e.g. colour differently) across the variable.

        Args:
            level: The label of the `LDict` level in the PyTree to stack by.
            dependency_name: Optional name of the specific dependency to stack.
                If None, will stack all dependencies listed in self.inputs.

        Returns:
            A copy of this analysis with stacking operation and updated parameters
        """

        # Define the stacking function
        def stack_dependency(dep_data, **kwargs):
            # jtree.stack stacks subtrees, so we don't need to move `level` to bottom
            return jt.map(
                lambda d: jtree.stack(list(d.values())),
                dep_data,
                is_leaf=LDict.is_of(level),
            )

        #! TODO: Refactor/remove; this is not universal anymore.
        #! I'm not sure how this would be generalized;
        #! perhaps subclasses could specify that certain axes (assumed axis=0 in this case)
        #! are colorscale axes or similar; then when we perform operations such as this,
        #! we could use `level: str` to modify them respectively.
        # modified_analysis = eqx.tree_at(
        #     lambda obj: (obj.colorscale_key, obj.colorscale_axis, obj.fig_params),
        #     self,
        #     (
        #         level,
        #         0,
        #         self.fig_params | dict(legend_title=get_label_str(level)),
        #     ),
        #     is_leaf=is_none,
        # )

        description = f"Stack dependency {dependency_name} LDict level '{level}' into axis 0"
        label = f"stack_{_format_level_str(level)}"
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name="after_stacking",
            dep_name=dependency_name,
            transform_func=stack_dependency,
            params=dict(level=level),
            metadata=final_metadata,
        )

    def after_level_to_top(
        self,
        label: str,
        is_leaf: Callable[[Any], bool] | None = None,  # LDict.is_of('var'),
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that will transpose `LDict` levels of its inputs.

        This is useful when our analysis uses a plotting function that compares across
        the outer PyTree level, but for whatever reason this level is not already
        the outer level of our results PyTree.
        """

        def transpose_dependency(dep_data, **kwargs):
            return LDict.of("task_variant")(
                {
                    variant_label: ldict_level_to_top(
                        label, dep_data[variant_label], is_leaf=is_leaf
                    )
                    for variant_label in dep_data
                }
            )

        description = (
            f"Move LDict level '{label}' to the top of the PyTree structure of "
            f"dependency {dependency_name}"
        )
        label_str = f"{_format_level_str(label)}_to-top"
        final_metadata = _merge_metadata(
            metadata, dict(descriptions=[description], labels=[label_str])
        )

        return self._add_prep_op(
            name="after_level_to_top",
            dep_name=dependency_name,
            transform_func=transpose_dependency,
            params=dict(label=label),
            metadata=final_metadata,
        )

    def after_rearrange_levels(
        self,
        spec: Sequence[str | EllipsisType],
        is_leaf: Callable[[Any], bool] | None = None,  # LDict.is_of('var'),
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that will transpose `LDict` levels of its inputs.

        This is useful when our analysis uses a plotting function that compares across
        the outer PyTree level, but for whatever reason this level is not already
        the outer level of our results PyTree.
        """

        def transpose_dependency(dep_data, **kwargs):
            return rearrange_ldict_levels(dep_data, spec, is_leaf=is_leaf)

        spec_str = "-".join(
            str(...) if s is Ellipsis else _format_level_str(s)
            for s in spec  # type: ignore[call-arg]
        )

        description = (
            f"Rearrange LDict levels of dependency {dependency_name} "
            f"according to specification: {spec_str}"
        )
        label = f"levels-to-{spec_str}"
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name="after_rearrange_levels",
            dep_name=dependency_name,
            transform_func=transpose_dependency,
            params=dict(spec=spec),
            metadata=final_metadata,
        )

    def after_level_to_bottom(
        self,
        label: str,
        is_leaf: Callable[[Any], bool] | None = None,  # LDict.is_of('var'),
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that will transpose `LDict` levels of its inputs.

        This is useful when our analysis uses a plotting function that compares across
        the outer PyTree level, but for whatever reason this level is not already
        the outer level of our results PyTree.
        """

        def transpose_dependency(dep_data, **kwargs):
            return LDict.of("task_variant")(
                {
                    variant_label: ldict_level_to_bottom(
                        label, dep_data[variant_label], is_leaf=is_leaf
                    )
                    for variant_label in dep_data
                }
            )

        description = (
            f"Move LDict level '{label}' to the bottom of the PyTree structure "
            f"of dependency {dependency_name}"
        )
        label_str = f"{_format_level_str(label)}_to-bottom"
        final_metadata = _merge_metadata(
            metadata, dict(descriptions=[description], labels=[label_str])
        )

        return self._add_prep_op(
            name="after_level_to_bottom",
            dep_name=dependency_name,
            transform_func=transpose_dependency,
            params=dict(label=label),
            metadata=final_metadata,
        )

    def after_subdict_at_level(
        self,
        level: str,
        keys: Optional[Sequence[Hashable]] = None,
        idxs: Optional[Sequence[int]] = None,
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that keeps certain keys of an `LDict` level before proceeding.

        Either `keys` or `idxs` must be provided, but not both: `keys` specifies the exact keys to keep,
        whereas `idxs` specifies the indices of the keys to keep in terms of their ordering in the `LDict`.
        """

        if keys is not None and idxs is not None:
            raise ValueError("Cannot provide both `keys` and `idxs`.")

        if keys is None and idxs is None:
            raise ValueError("Must provide either `keys` or `idxs`.")

        label = f"subdict-at-{_format_level_str(level)}_"
        if keys is not None:

            def _select_func_keys(d: dict, keys=keys) -> dict:
                return subdict(d, keys)

            select_func = _select_func_keys
            selection_str = f"keys {','.join(str(k) for k in keys)}"
            label += ",".join(str(k) for k in keys)

        elif idxs is not None:
            if not isinstance(idxs, Sequence) or not all(isinstance(i, int) for i in idxs):
                raise ValueError("`idxs` must be a sequence of integers.")

            def _select_func_idxs(d: dict, idxs=idxs) -> dict:
                return subdict(d, [list(d.keys())[i] for i in idxs])

            select_func = _select_func_idxs
            selection_str = f"items with indices {','.join(str(i) for i in idxs)}"
            label += f"idxs-{','.join(str(i) for i in idxs)}"

        else:
            raise ValueError("Either `keys` or `idxs` must be provided.")

        # Generate description for this specific method
        dep_str = dependency_name or "dependencies"
        description = f"Select {selection_str} at LDict level '{level}' of dependency {dep_str}"

        # Build metadata with our description
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self.after_transform(
            func=select_func,
            level=level,
            dependency_names=dependency_name,
            metadata=final_metadata,
        )

    def after_getitem_at_level(
        self,
        level: str,
        key: Hashable,
        dependency_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that selects a single key from an `LDict` level,
        completely removing that level.

        Args:
            level: The `LDict` level to select from
            key: The specific key to select from the `LDict`
            dependency_name: Optional name of specific dependency to transform
            metadata: Optional metadata containing descriptions and labels.
        """

        def getitem(d, key=key):
            return d[key]

        dep_str = dependency_name or "dependencies"
        description = f"Select key '{key}' from LDict level '{level}' of {dep_str}"
        label = f"getitem-at-{_format_level_str(level)}_{key}"

        # Build metadata with our description
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self.after_transform(
            func=getitem,
            level=level,
            dependency_names=dependency_name,
            metadata=final_metadata,
        )

    def vmap(self, in_axes: Mapping[str, AxisSpec]) -> Self:
        """
        Return a new instance whose `compute` is wrapped in one or more
        nested jax.vmap layers (via vmap_multi).

        `in_axes` maps dependency names (keys in `inputs` or
        "data.states") to *singular* axis specs: int/None, PyTrees thereof,
        or MultiVmapAxes for nested vmaps.  These are expanded by
        expand_axes_spec into per-level dicts.
        """
        # 1) normalize names (aliases → canonical)
        norm: dict[str, AxisSpec] = {_normalize_name(k): v for k, v in in_axes.items()}

        # 2) expand nested specs → list of dicts, one per new vmap level
        per_level = expand_axes_spec(norm)
        n_new = len(per_level)

        # 3) build name → tuple of axes across these new levels
        in_axes_spec: dict[str, tuple[int | None, ...]] = {
            name: tuple(level.get(name, None) for level in per_level) for name in norm
        }

        # 4) kwargs to pop (everything except "data.states")
        new_dep_names = tuple(n for n in in_axes_spec if n != "data.states")

        # 5) build positional in_axes_sequence for these levels
        new_sequence = _build_in_axes_sequence(in_axes_spec, new_dep_names)

        # 6) compose with any prior `.vmap` calls on this instance
        if self._vmap_spec is None:
            combined_spec = in_axes_spec
            combined_sequence = new_sequence
            combined_dep_names = new_dep_names
        else:
            prev = self._vmap_spec
            n_prev = len(prev.in_axes_sequence)

            # a) append positional specs
            combined_sequence = prev.in_axes_sequence + new_sequence

            # b) merge dict specs, padding with None
            combined_spec: dict[str, tuple[int | None, ...]] = {}
            for nm in set(prev.in_axes_spec) | set(in_axes_spec):
                left = prev.in_axes_spec.get(nm, (None,) * n_prev)
                right = in_axes_spec.get(nm, (None,) * n_new)
                combined_spec[nm] = left + right

            combined_dep_names = tuple(nm for nm in combined_spec if nm != "data.states")

        # 7) build the new frozen spec
        new_spec = _AnalysisVmapSpec(
            in_axes_spec=combined_spec,
            vmapped_dep_names=combined_dep_names,
            in_axes_sequence=combined_sequence,
        )

        # 8) return an eqx-tree-updated copy of self
        return eqx.tree_at(
            lambda a: a._vmap_spec,
            self,
            new_spec,
            is_leaf=is_none,
        )

    def then_transform_result(
        self,
        func: Callable[..., Any],
        level: Optional[str] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Returns a copy of this analysis that transforms its PyTree of results.

        The transformation occurs prior to the generation of figures, thus affects them.
        """

        return self._then_transform(
            op_type="results",
            func=func,
            level=level,
            is_leaf=is_leaf,
            metadata=metadata,
        )

    #! TODO: Generalize `map_figs_at_level` to map at any PyTree node
    #! For example, maybe our input is a tuple of LDict and we want to map `make_figs`
    #! separately, for each element of the tuple.
    # def map_figs(
    #     self,
    #     dependency_name: Optional[str] = None,
    # ):
    #     ...

    def map_figs_at_level(
        self,
        level,
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that maps over the input PyTrees, down to a certain `LDict` level.

        This is useful when e.g. the analysis calls a plotting function that expects a two-level PyTree,
        but we've evaluated a deeper PyTree of states, where the two levels are inner.

        Args:
            level: str | Sequence[str] | Sequence[tuple[str, Callable]]
                - str: append a single fig-op that maps by this LDict level.
                - Sequence[str]: expand into a chain, appending one fig-op per level in order,
                  reusing `fig_params_fn` for each op.
                - Sequence[tuple[str, Callable]]: per-level override for `fig_params_fn`.
            dependency_name: Optional name of specific dependency to map
            fig_params_fn: Optional function for fig parameter updates
            metadata: Optional metadata containing descriptions and labels.
        """
        if isinstance(level, (list, tuple)):
            ana = self
            for spec in level:
                if isinstance(spec, (list, tuple)) and len(spec) == 2:
                    lv, fn = spec
                    ana = ana.map_figs_at_level(lv, dependency_name, fn, metadata)
                else:
                    ana = ana.map_figs_at_level(spec, dependency_name, fig_params_fn, metadata)
            return ana

        dep_str = dependency_name or "dependencies"
        description = f"Map figures at LDict level '{level}' of dependency {dep_str}"
        label = f"map_figs_at-{_format_level_str(level)}"

        # Build metadata with description and label
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._append_fig_op(
            name="map_figs_at_level",
            dep_name=dependency_name,
            is_leaf=LDict.is_of(level),
            slice_fn=_level_slice_fn,
            items_fn=partial(_level_items_fn, level),
            # Use the new aggregator specific to mapping
            agg_fn=partial(_reconstruct_ldict_aggregator, level),
            fig_params_fn=fig_params_fn,
            params=dict(level=level),
            metadata=final_metadata,
        )

    def map_figs_by_axis(
        self,
        level: str,
        *,
        axis_func: Callable[[Any], str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Map figures by creating new axis labels using axis_func."""
        description = "DEBUG THIS"
        final_metadata = _merge_metadata(metadata, dict(descriptions=[description]))

        return self._append_fig_op(
            name="map_figs_by_axis",
            dep_name=None,
            axis_func=axis_func,
            level=level,
            metadata=final_metadata,
        )

    def combine_figs_by_axis(
        self,
        axis: int,
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable[[Mapping, FigIterCtx], Mapping]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that will merge individual figures generated by slicing along
        the specified axis of the arrays in the dependency PyTree(s).

        This is useful when we want to include an additional dimension of comparison in a figure.
        For example, our plotting function may already compare across the first axis of the input
        arrays, or the outer level of the input PyTree; but perhaps we also want a secondary
        comparison across a different axis of the input arrays.

        Args:
            axis: The axis to slice and combine along
            dependency_name: Optional name of specific dependency to slice.
                If None, will slice all dependencies listed in self.inputs.
        """
        description = (
            f"Combine figures by slicing along axis {axis} of dependency {dependency_name}"
        )
        final_metadata = _merge_metadata(
            metadata, dict(descriptions=[description], label=f"combine_by-axis{axis}")
        )

        return self._append_fig_op(
            name="combine_figs_by_axis",
            dep_name=dependency_name,
            is_leaf=eqx.is_array,
            slice_fn=partial(_axis_slice_fn, axis),
            items_fn=partial(_axis_items_fn, axis),
            fig_params_fn=fig_params_fn,
            # Use the default aggregator that matches the new signature
            agg_fn=_combine_figures,
            params=dict(axis=axis),
            metadata=final_metadata,
        )

    def combine_figs_by_level(
        self,
        level: str,
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable[[Mapping, FigIterCtx], Mapping]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that will merge individual figures generated by iterating over
        the keys of an LDict level in the dependency PyTree(s).

        This is useful when we want to include an additional dimension of comparison in a figure.
        For example, our plotting function may already compare across the first axis of the input
        arrays, or the outer level of the input PyTree; but perhaps we also want a secondary
        comparison across a different level of the input PyTree.

        Args:
            level: The LDict level to iterate over and combine across
            dependency_name: Optional name of specific dependency to iterate over.
                If None, will iterate over all dependencies listed in self.inputs.
        """
        description = f"Combine figures over LDict level '{level}' of dependency {dependency_name}"
        final_metadata = _merge_metadata(
            metadata,
            dict(descriptions=[description], label=f"combine_by-{_format_level_str(level)}"),
        )

        return self._append_fig_op(
            name="combine_figs_by_level",
            dep_name=dependency_name,
            is_leaf=LDict.is_of(level),
            slice_fn=_level_slice_fn,
            items_fn=partial(_level_items_fn, level),
            fig_params_fn=fig_params_fn,
            agg_fn=_combine_figures,
            params=dict(level=level),
            metadata=final_metadata,
        )

    def then_transform_figs(
        self,
        func: Callable[..., Any],
        level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that transforms its output PyTree of figures
        """
        return self._then_transform(
            op_type="figs",
            func=func,
            level=level,
            is_leaf=is_type(go.Figure),
            metadata=metadata,
        )

    def _then_transform(
        self,
        op_type: _FinalOpKeyType,
        func: Callable[..., Any],
        level: Optional[str] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        label = f"post-transform-{op_type}_{get_name_of_callable(func)}"
        func_name = get_name_of_callable(func)
        if op_type == "results":
            description = f"Transform analysis results using {func_name}"
        else:  # op_type == "figs"
            description = f"Transform output figures using {func_name}"

        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], label=label))

        if level is not None:
            description += f" at LDict level '{level}'"

            # Apply the transformation leafwise across the `level` LDict level;
            # e.g. suppose there are two keys in the `level` LDict, then the transformation
            # will be applied to 2-tuples of figures; following the transformation, reconsistute
            # the `level` LDict with the transformed figures.
            @wraps(func)
            def _transform_func(tree):
                _Tuple = jtree.make_named_tuple_subclass("ColumnTuple")

                def _transform_level(ldict_node):
                    if not LDict.is_of(level)(ldict_node):
                        return ldict_node

                    zipped = jtree.zip_(
                        *ldict_node.values(),
                        is_leaf=is_leaf,
                        zip_cls=_Tuple,
                    )
                    transformed = jt.map(func, zipped, is_leaf=is_type(_Tuple))
                    unzipped = jtree.unzip(transformed, tuple_cls=_Tuple)
                    return LDict.of(level)(dict(zip(ldict_node.keys(), unzipped)))

                return jt.map(_transform_level, tree, is_leaf=LDict.is_of(level))

            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                transform_func=_transform_func,
                params=dict(level=level, transform_func=func),
                is_leaf=is_leaf,
                metadata=final_metadata,
            )
        else:
            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                transform_func=func,
                params=dict(transform_func=func),
                is_leaf=is_leaf,
                metadata=final_metadata,
            )

    def _add_prep_op(
        self,
        name: str,
        dep_name: Optional[str | Sequence[str]],
        transform_func: Callable,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        # If the transform consumes extra dependencies, ensure the analysis
        # instance knows about them so the graph builder evaluates them.
        analysis_with_deps = self
        spec_map = getattr(transform_func, "_extra_inputs", None)  # {port -> dep_spec}
        if spec_map:
            # Each dep_spec can be a PyTree; store it verbatim under the port.
            analysis_with_deps = eqx.tree_at(
                lambda a: a._extra_inputs,
                self,
                self._extra_inputs | spec_map,
            )

        # Log the most specific description if available
        if metadata and "descriptions" in metadata and metadata["descriptions"]:
            logger.debug(f"Add prep-op to {self.__class__.__name__}: {metadata['descriptions'][0]}")

        return eqx.tree_at(
            lambda a: a._prep_ops,
            analysis_with_deps,
            analysis_with_deps._prep_ops
            + (
                _PrepOp(
                    name=name,
                    dep_name=dep_name,
                    transform_func=transform_func,
                    params=params or {},
                    metadata=metadata or {},
                ),
            ),
        )

    def _append_fig_op(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Self:
        """Append a new figure operation to the chain."""
        # Log the most specific description if available
        if metadata and "descriptions" in metadata and metadata["descriptions"]:
            logger.debug(f"Add fig-op to {self.__class__.__name__}: {metadata['descriptions'][0]}")

        new_op = _FigOp(metadata=metadata or {}, **kwargs)
        return eqx.tree_at(lambda a: a._fig_ops, self, self._fig_ops + (new_op,))

    def _add_final_op(
        self,
        op_type: _FinalOpKeyType,
        name: str,
        transform_func: Callable,
        params: Optional[Dict[str, Any]] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        # Log the most specific description if available
        if metadata and "descriptions" in metadata and metadata["descriptions"]:
            logger.debug(
                f"Add final-op to {self.__class__.__name__}: {metadata['descriptions'][0]}"
            )

        current_ops = self._final_ops_by_type.get(op_type, ())
        new_op = _FinalOp(
            name=name,
            transform_func=transform_func,
            params=params or {},
            is_leaf=is_leaf,
            metadata=metadata or {},
        )
        updated_ops_for_type = current_ops + (new_op,)

        # Create a new dictionary for _final_ops_by_type to ensure immutability if needed by equinox
        new_final_ops_by_type = dict(self._final_ops_by_type)
        new_final_ops_by_type[op_type] = updated_ops_for_type

        return eqx.tree_at(
            lambda a: a._final_ops_by_type,
            self,
            MappingProxyType(new_final_ops_by_type),
        )

    @cached_property
    def _field_params(self):
        # TODO: Inherit from dependencies?
        return get_dataclass_fields(
            self,
            exclude=AbstractAnalysis._exclude_fields,
            include_internal=False,
        )

    #! TODO: Remove this (probably)
    @cached_property
    def _non_default_field_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of fields that have non-default values.
        Works without knowing field names in advance.
        """
        result = {}

        # Get all dataclass fields for this instance
        for field_ in dataclasses.fields(self):
            # Exclude `variant` since we explicitly include it first, in dump file names
            if field_.name in AbstractAnalysis._exclude_fields or field_.name == "variant":
                continue

            # Skip fields that are marked as subclass-internal
            if field_.metadata.get("internal", False):
                continue

            current_value = getattr(self, field_.name)

            # Check if this field has a default value defined
            has_default = field_.default is not dataclasses.MISSING
            has_default_factory = field_.default_factory is not dataclasses.MISSING

            if has_default and current_value != field_.default:
                # Field has a different value than its default
                result[field_.name] = current_value
            elif has_default_factory:
                # For default_factory fields, we can't easily tell if the value
                # was explicitly provided, so we include the current value
                # This is an approximation - we'll include fields with default_factory
                result[field_.name] = current_value
            elif not has_default and not has_default_factory:
                # Field has no default, so it must have been provided
                result[field_.name] = current_value

        return result

    @cached_property
    def md5_str(self):
        """An md5 hash string that identifies this analysis.

        The hash is computed from the analysis parameter values and not the instance itself.
        Any callable leaves are first replaced with their whitespace-stripped source strings,
        which should generally capture when the implementation is identical, and should not
        result in any
        """
        # ? TODO: De-duplicate computations *prior* to ops (i.e. some analyses get repeated with
        # ? different ops)
        ops_params = self._extract_ops_info()

        params = dict(cls_name=self.__class__.__name__) | ops_params | self._field_params

        #! TODO: add leaves from `self.inputs` and make sure any referenced `AbstractAnalysis`
        #! instances are resolved to their own `md5_str`.

        params = hash_callable_leaves(
            params,
            is_leaf=is_type(DoNotHashTree),
            ignore=(LDict, is_module, DoNotHashTree),
        )
        return get_md5_hexdigest(params)

    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        """Additional parameters to save.

        Note that `**kwargs` here may not only contain the dependencies, but that `save`
        passes the key-value pairs of parameters inferred from the `figs` PyTree.
        """
        return dict()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _run_prep_ops(self, data: AnalysisInputData, kwargs: Dict[str, Any]):
        """Apply all prep-ops in sequence, consuming extra CallWithDeps deps."""

        prepped_kwargs = kwargs.copy()  # Start with original kwargs for modification
        prepped_kwargs["data.states"] = data.states

        for prep_op in self._prep_ops:
            dep_names_to_process = self._get_target_dependency_names(
                prep_op.dep_name, prepped_kwargs, "Prep-op"
            )

            for name in dep_names_to_process:
                try:
                    prepped_kwargs[name] = _call_user_func(
                        prep_op.transform_func,
                        prepped_kwargs[name],
                        data=data,
                        **kwargs,
                    )
                except Exception as e:
                    logger.error(f"Error applying prep_op transform to '{name}'", exc_info=True)
                    raise e

            # Pop any extra dependencies that are not part of the analysis interface
            extra_ports = getattr(prep_op.transform_func, "_ports", ())
            for port in extra_ports:
                if port not in self.inputs:
                    prepped_kwargs.pop(port, None)

        return prepped_kwargs


# Keep InputType for validation purposes
InputType: TypeAlias = PyTree[
    type[AbstractAnalysis]
    | AbstractAnalysis
    | _DataField
    | str
    | ExpandTo
    | Transformed
    | LiteralInput
]


class _DummyAnalysis(AbstractAnalysis[NoPorts]):
    """An empty analysis, for debugging."""

    def compute(self, data: AnalysisInputData, **kwargs) -> PyTree[Any]:
        return None

    def make_figs(self, data: AnalysisInputData, **kwargs) -> PyTree[go.Figure]:
        return None


def _call_user_func(func, *args, **kwargs):
    """Invoke *func* with *dep_data* and the subset of *extra_kwargs* it accepts.

    - If `func` exposes `_ports`, it's a wrapper:
        * Verify all required port kwargs exist, then pass through *args/**kwargs.
        * The wrapper handles placement/filtering.
    - Otherwise (plain callable):
        * Drop unknown kwargs if the function doesn't accept **kwargs.
    """
    ports = getattr(func, "_ports", None)

    if ports:
        missing = [p for p in ports if p not in kwargs]
        if missing:
            raise KeyError(
                f"Missing dependency ports {missing}; available keys: {sorted(kwargs.keys())}"
            )
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_varkw:
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return func(*args, **kwargs)


class CallWithDeps:
    """
    Plug (PyTrees of) dependency specs into arbitrary argument positions.

    Positional spec slots:
      - None -> consume the NEXT *caller*-provided positional arg.
      - Any other object -> treat as a dependency *spec*; it will be computed
        under a private port and injected here when the wrapper is called.

    Keyword spec slots:
      - name=spec -> inject the computed dependency as keyword `name`
        unless the caller explicitly provides `name` (caller wins).
    """

    _counter = 0

    def __init__(self, *pos_specs: Any, **kw_specs: Any):
        self.pos_specs = pos_specs
        self.kw_specs = kw_specs

    @staticmethod
    def _alloc_port() -> str:
        CallWithDeps._counter += 1
        return f"__cwd_{CallWithDeps._counter:x}"

    def __call__(self, func: Callable):
        # Plan structures
        spec_map: Dict[str, Any] = {}  # {private_port -> spec (leaf or PyTree)}
        pos_tokens: list[Optional[str]] = []  # [None | private_port]
        kw_ports: Dict[str, str] = {}  # {param_name -> private_port}

        def port_for(spec: Any) -> str:
            p = self._alloc_port()
            spec_map[p] = spec
            return p

        for item in self.pos_specs:
            pos_tokens.append(None if item is None else port_for(item))
        for name, item in self.kw_specs.items():
            kw_ports[name] = port_for(item)

        # Introspect the *wrapped* function once, outside the wrapper body
        sig = inspect.signature(func)
        param_names = set(sig.parameters.keys())
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        @functools.wraps(func)
        def wrapper(
            *caller_args: Any,
            _spec_map=spec_map,  # capture to avoid late binding
            _pos_tokens=tuple(pos_tokens),
            _kw_ports=kw_ports,
            _param_names=param_names,
            _has_varkw=has_varkw,
            **caller_kwargs: Any,
        ) -> Any:
            # 1) Extract dependency payloads from private ports
            deps: Dict[str, Any] = {}
            for p in _spec_map.keys():
                if p in caller_kwargs:
                    deps[p] = caller_kwargs.pop(p)

            missing = [p for p in _spec_map.keys() if p not in deps]
            if missing:
                raise KeyError(
                    f"Missing dependency ports {missing}; have keys={sorted(deps.keys())}"
                )

            # 2) Build positional args: None consumes a caller positional
            args: list[Any] = []
            it = iter(caller_args)
            for tok in _pos_tokens:
                if tok is None:
                    try:
                        args.append(next(it))
                    except StopIteration:
                        raise TypeError(
                            "Not enough positional arguments from caller "
                            "to satisfy `None` placeholders."
                        ) from None
                else:
                    args.append(deps[tok])

            # Any remaining caller positionals go after the tokens
            args.extend(list(it))

            # 3) Build keyword args: caller wins; fill missing with deps
            mapped_kwargs: Dict[str, Any] = dict(caller_kwargs)
            for name, port in _kw_ports.items():
                if name not in mapped_kwargs:
                    mapped_kwargs[name] = deps[port]

            # 4) Filter unknown kwargs if the underlying function has no **kwargs
            if not _has_varkw:
                mapped_kwargs = {k: v for k, v in mapped_kwargs.items() if k in _param_names}

            return func(*args, **mapped_kwargs)

        # Expose to the engine:
        wrapper._extra_inputs = spec_map  # {port -> spec}  # type: ignore[attr-defined]
        wrapper._ports = tuple(spec_map.keys())  # type: ignore[attr-defined]
        return wrapper
