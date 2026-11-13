import dataclasses
import hashlib
import logging
import pprint
from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cached_property, partial, wraps
from itertools import chain
from pathlib import Path, PurePath
from types import EllipsisType, FunctionType, MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Optional,
    Self,
    TypeAlias,
    Union,
)

import equinox as eqx
import jax.tree as jt
import feedbax._vendor.jax_cookbook.tree as jtree
import plotly.graph_objects as go
from equinox import Module, field
from feedbax._vendor.jax_cookbook import (
    LDict,
    LDictConstructor,
    is_none,
    is_type,
    vmap_multi,
)
from feedbax._vendor.jax_cookbook._func import wrap_to_accept_var_kwargs
from feedbax._vendor.jax_cookbook._vmap import AxisSpec, expand_axes_spec
from jaxtyping import PyTree

from feedbax.config import PATHS, STRINGS
from feedbax.config.namespace import TreeNamespace
from feedbax.config.utils import deep_merge
from feedbax.analysis.support import (
    camel_to_snake,
    get_dataclass_fields,
    get_name_of_callable,
    get_origin_type,
    is_json_serializable,
)
from feedbax.config.tree import (
    _align_trees_to_structure,
    ldict_label_only_fn,
    tree_level_labels,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.analysis.fig_ops import (
    FigIterCtx,
    FigureSaveTask,
    _AnalysisVmapSpec,
    _FinalOp,
    _FinalOpKeyType,
    _FigOp,
    _PrepOp,
    _apply_fig_ops,
    _apply_final_ops,
    _call_fig_params_fn,
)
from feedbax.analysis.inputs import (
    AbstractAnalysisPorts,
    AnalysisRef,
    CallWithDeps,
    Data,
    ExpandTo,
    InputOf,
    LiteralInput,
    NoPorts,
    PortsType,
    SinglePort,
    Transformed,
    _DataField,
)
from feedbax.analysis.result_cache import (
    ANALYSIS_RESULT_CACHE_SCHEMA_VERSION,
    RESULTS_CACHE_SUBDIR,
    AnalysisResultCacheCorruption,
    _UnsupportedCacheInput,
    _CACHE_LOAD_ERRORS,
    _CACHE_SAVE_ERRORS,
    _fingerprint_cache_inputs,
    _fingerprint_semantic_values,
    _load_analysis_result_cache,
    _write_analysis_result_cache,
)
from feedbax.contracts.computation_revision import validate_computation_revision

if TYPE_CHECKING:
    from feedbax.tasks import AbstractTask
    from typing import ClassVar as AbstractClassVar

    from feedbax.analysis.context import AnalysisRunContext
else:
    from equinox import AbstractClassVar  # noqa: F401


logger = logging.getLogger(__name__)

__all__ = [
    "AbstractAnalysis",
    "AbstractAnalysisPorts",
    "AnalysisRef",
    "AnalysisResultCacheCorruption",
    "ANALYSIS_RESULT_CACHE_SCHEMA_VERSION",
    "CallWithDeps",
    "Data",
    "ExpandTo",
    "FigIterCtx",
    "FigureSaveTask",
    "IdentityNode",
    "InputOf",
    "InputType",
    "LiteralInput",
    "NoPorts",
    "PortsType",
    "RESULTS_CACHE_SUBDIR",
    "SinglePort",
    "Transformed",
    "_AnalysisVmapSpec",
    "_CACHE_LOAD_ERRORS",
    "_CACHE_SAVE_ERRORS",
    "_DataField",
    "_FigOp",
    "_FinalOp",
    "_FinalOpKeyType",
    "_PrepOp",
    "_apply_fig_ops",
    "_apply_final_ops",
    "_call_fig_params_fn",
    "_load_analysis_result_cache",
    "_write_analysis_result_cache",
    "get_validation_trial_specs",
]


PARAM_SEQ_LEN_TRUNCATE = 9
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


def get_validation_trial_specs(task: "AbstractTask"):
    # TODO: Support any number of extra axes (i.e. for analyses that vmap over multiple axes in their task/model objects)
    # if len(task.workspace.shape) == 3:
    #     #! I don't understand why/if this is necessary
    #     return eqx.filter_vmap(lambda task: task.validation_trials)(task)
    # else:
    return task.validation_trials


def _extract_vmapped_kwargs_to_args(fn, vmapped_dep_names: Sequence[str]):
    """Convert specified kwargs to positional args for vmapping."""

    def modified_fn(data, *vmapped_deps, **remaining_kwargs):
        # Reconstruct the full kwargs dict
        full_kwargs = remaining_kwargs | dict(zip(vmapped_dep_names, vmapped_deps))
        return fn(data, **full_kwargs)

    return modified_fn


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
            f"data.states: {jtree.first_leaf_shape(prepped_data.states)}",
            *[
                f"{k}: {jtree.first_leaf_shape(dep)}"
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


_ThenTransformTuple = jtree.make_named_tuple_subclass("_ThenTransformTuple")


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
        computation_revision: Revision of the raw computation, including its implementation
            dependencies. Declare it as a ClassVar on each cacheable concrete class and change it when
            the computation changes. Subclasses must declare their own revision.

    Fields:
        inputs: The port->input mapping for the analysis instance.
        variant: Label of the evaluation variant this analysis uses (primarily).
        cache_result: Reuse raw computation results for a declared computation revision
            and supported prepared values. Cached computations must treat their inputs
            as read-only values. Opaque inputs such as Python functions run without
            result-cache reads or writes.

    Treat node parameters as read-only once used in an analysis graph.
    Registering a caller-supplied operation callback separately creates a distinct live node.
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
    computation_revision: ClassVar[str | None] = None
    inputs: PortsType = field(default_factory=NoPorts, converter=NoPorts.converter)

    variant: Optional[str] = (
        None  #! TODO: Eliminate this. Should be in `tasks` PyTree, and dealt with explicitly with ops
    )
    #! TODO: Rename to `default_fig_params`
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
    # Fields for controlling execution granularity via tree mapping
    _map_compute_spec: Optional[tuple[Callable[[Any], bool], Optional[str | Sequence[str]]]] = (
        field(default=None)
    )
    _map_make_figs_spec: Optional[tuple[Callable[[Any], bool], Optional[str | Sequence[str]]]] = (
        field(default=None)
    )

    def __check_init__(self) -> None:
        if any(member.name == "computation_revision" for member in dataclasses.fields(self)):
            raise TypeError("computation_revision must be declared as a ClassVar")

    def __post_init__(self):
        """Validate inputs instance and check for unresolved required inputs."""
        # Validate that inputs is an instance of the expected Ports class
        ports_origin_type = get_origin_type(self.Ports)
        if not isinstance(self.inputs, ports_origin_type):
            raise TypeError(f"Expected inputs of type {self.Ports}, got {type(self.inputs)}")

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

    def emit_artifacts(
        self,
        context: "AnalysisRunContext",
        data: AnalysisInputData,
        *,
        result: PyTree[Any],
        **kwargs,
    ) -> PyTree[Any] | None:
        """Optionally record non-figure artifacts and return the downstream result.

        Subclasses may use ``context.record_artifact*`` helpers here. Returning a
        value replaces the result exposed to downstream analysis nodes; returning
        ``None`` keeps the original result.
        """
        return None

    def _emit_artifacts_with_context(
        self,
        context: "AnalysisRunContext",
        data: AnalysisInputData,
        result: PyTree[Any],
        **kwargs,
    ) -> PyTree[Any]:
        """Run the artifact hook and record any refs embedded in the result payload."""
        if self.__class__.emit_artifacts is AbstractAnalysis.emit_artifacts:
            context.record_artifact_refs_from_value(result)
            return result

        emitted = self.emit_artifacts(context, data, result=result, **kwargs)
        output = result if emitted is None else emitted
        context.record_artifact_refs_from_value(output)
        return output

    def _compute_with_ops(
        self,
        data: AnalysisInputData,
        *,
        analysis_context: Optional["AnalysisRunContext"] = None,
        **kwargs,
    ) -> dict[str, PyTree[Any]]:
        """Perform computations with prep-ops, vmap, and result final-ops applied."""
        # Transform inputs prior to performing the analysis
        prepped_kwargs = self._run_prep_ops(data, kwargs)
        prepped_data = eqx.tree_at(lambda d: d.states, data, prepped_kwargs["data.states"])
        del prepped_kwargs["data.states"]

        def _run_compute():
            # Define the core compute function that may or may not be vmapped
            def _compute_core(data_for_compute, kwargs_for_compute):
                if self._vmap_spec is not None:
                    # Extract vmapped dependencies
                    vmapped_names = self._vmap_spec.vmapped_dep_names
                    compute_fn = _extract_vmapped_kwargs_to_args(self.compute, vmapped_names)
                    try:
                        vmapped_deps = [kwargs_for_compute[name] for name in vmapped_names]
                    except KeyError as e:
                        raise KeyError(
                            f"Unknown dependency '{e.args[0]}' for analysis {self.name} "
                            "supplied in `in_axes` to `vmap`."
                        ) from e
                    logger.debug(
                        _get_vmap_spec_debug_str(
                            self, self._vmap_spec, data_for_compute, vmapped_deps
                        )
                    )
                    compute_fn = partial(
                        compute_fn,
                        **{k: v for k, v in kwargs_for_compute.items() if k not in vmapped_names},
                    )
                    return vmap_multi(
                        compute_fn, in_axes_sequence=self._vmap_spec.in_axes_sequence
                    )(
                        data_for_compute,
                        *vmapped_deps,
                    )
                else:
                    compute_fn = partial(self.compute, **kwargs_for_compute)
                    return compute_fn(data_for_compute)

            # Check if we should map compute execution
            if self._map_compute_spec is not None:
                is_leaf, dep_names = self._map_compute_spec

                # Determine which dependencies to map over
                target_dep_names = self._get_target_dependency_names(
                    dep_names, prepped_kwargs, "Compute mapping"
                )

                # Separate mapped and unmapped kwargs
                mapped_kwargs = {
                    k: prepped_kwargs[k] for k in target_dep_names if k in prepped_kwargs
                }
                unmapped_kwargs = {
                    k: v for k, v in prepped_kwargs.items() if k not in target_dep_names
                }

                # Map compute over the specified tree nodes
                def compute_at_leaf(*mapped_deps):
                    # Reconstruct kwargs for this leaf
                    leaf_kwargs = unmapped_kwargs.copy()
                    leaf_kwargs.update(dict(zip(target_dep_names, mapped_deps)))

                    # Handle data.states if it was mapped
                    if "data.states" in target_dep_names and target_dep_names.index(
                        "data.states"
                    ) < len(mapped_deps):
                        states_idx = target_dep_names.index("data.states")
                        leaf_data = eqx.tree_at(
                            lambda d: d.states, prepped_data, mapped_deps[states_idx]
                        )
                        # Remove data.states from leaf_kwargs since it's in the data object
                        remaining_deps = list(mapped_deps)
                        remaining_names = list(target_dep_names)
                        remaining_deps.pop(states_idx)
                        remaining_names.pop(states_idx)
                        leaf_kwargs.update(dict(zip(remaining_names, remaining_deps)))
                    else:
                        leaf_data = prepped_data

                    # Apply compute (with potential vmap inside)
                    result = _compute_core(leaf_data, leaf_kwargs)
                    return result

                # Map over the tree structure
                return jt.map(compute_at_leaf, *mapped_kwargs.values(), is_leaf=is_leaf)
            else:
                # No tree mapping, just apply compute (with potential vmap)
                result = _compute_core(prepped_data, prepped_kwargs)
                return result

        def _compute_cached(computation_revision: str):
            # Declared default functions are part of the stable analysis implementation.
            config = {
                name: value
                for name, value in self._field_params.items()
                if name != "_map_make_figs_spec"
                and not (
                    isinstance(value, FunctionType)
                    and value is self.__dataclass_fields__[name].default
                )
            }
            try:
                inputs_hash = _fingerprint_cache_inputs(
                    type(self), computation_revision, config, prepped_data, prepped_kwargs
                )
            except _UnsupportedCacheInput as exc:
                logger.debug("Skipping result cache for %s: %s", self.name, exc)
                return _run_compute()

            cache_root = (
                analysis_context.results_cache_dir
                if analysis_context is not None
                else PATHS.cache / RESULTS_CACHE_SUBDIR
            )
            cache_path = cache_root / f"{ANALYSIS_RESULT_CACHE_SCHEMA_VERSION}_{inputs_hash}.pkl"
            try:
                result = _load_analysis_result_cache(cache_path, inputs_hash=inputs_hash)
            except FileNotFoundError:
                pass
            except AnalysisResultCacheCorruption as exc:
                logger.warning(
                    "Ignoring invalid cached result for %s at %s (will recompute): %s",
                    self.name,
                    cache_path,
                    exc,
                )
            else:
                logger.info("Loaded cached result for %s from %s", self.name, cache_path)
                return result

            result = _run_compute()
            try:
                _write_analysis_result_cache(cache_path, inputs_hash=inputs_hash, result=result)
                logger.info("Saved cache for %s to %s", self.name, cache_path)
            except _CACHE_SAVE_ERRORS as exc:
                logger.warning("Could not save cache for %s: %s", self.name, exc)
            return result

        use_result_cache = self.cache_result and (
            analysis_context is None or analysis_context.use_result_cache
        )
        computation_revision = (
            validate_computation_revision(vars(type(self)).get("computation_revision"))
            if use_result_cache
            else None
        )
        result = _compute_cached(computation_revision) if computation_revision else _run_compute()

        result = _apply_final_ops(self, "results", result, data=prepped_data, **prepped_kwargs)
        if analysis_context is not None:
            result = self._emit_artifacts_with_context(
                analysis_context,
                prepped_data,
                result,
                **prepped_kwargs,
            )

        return result

    def _make_figs_with_ops(
        self,
        data: AnalysisInputData,
        result: dict[str, PyTree[Any]],
        **kwargs,
    ) -> PyTree[go.Figure]:
        """Generate figures with fig-ops and figure final-ops applied."""
        from feedbax.plot import apply_default_template

        apply_default_template()

        # Transform dependencies prior to making figures
        prepped_kwargs = self._run_prep_ops(data, kwargs)
        prepped_data = eqx.tree_at(lambda d: d.states, data, prepped_kwargs["data.states"])
        del prepped_kwargs["data.states"]

        figs: PyTree[go.Figure] = None

        # Define the core make_figs function that can use fig ops or direct call
        def _make_figs_core(data_for_figs, result_for_figs, kwargs_for_figs):
            if self._fig_ops:
                # Use fig ops (complex behavior with context tracking)
                kwargs_with_results = kwargs_for_figs.copy()
                kwargs_with_results["data.states"] = data_for_figs.states
                kwargs_with_results["result"] = result_for_figs
                return _apply_fig_ops(self, data_for_figs, 0, (), **kwargs_with_results)
            else:
                # Direct call to make_figs
                return self.make_figs(data_for_figs, result=result_for_figs, **kwargs_for_figs)

        # Check if we should use simple tree mapping
        if self._map_make_figs_spec is not None:
            is_leaf, dep_names = self._map_make_figs_spec

            # Determine which dependencies to map over
            target_dep_names = self._get_target_dependency_names(
                dep_names, prepped_kwargs, "Make-figs mapping"
            )

            # Include result in potential mapping targets
            all_kwargs = prepped_kwargs.copy()
            all_kwargs["result"] = result

            # Separate mapped and unmapped kwargs
            mapped_kwargs = {k: all_kwargs[k] for k in target_dep_names if k in all_kwargs}
            unmapped_kwargs = {k: v for k, v in all_kwargs.items() if k not in target_dep_names}

            # Map make_figs over the specified tree nodes
            def make_figs_at_leaf(*mapped_deps):
                # Reconstruct kwargs for this leaf
                leaf_kwargs = {k: v for k, v in unmapped_kwargs.items() if k != "result"}
                mapped_dict = dict(zip(target_dep_names, mapped_deps))

                # Handle result specially
                if "result" in mapped_dict:
                    leaf_result = mapped_dict.pop("result")
                else:
                    leaf_result = unmapped_kwargs.get("result", result)

                # Handle data.states specially
                if "data.states" in mapped_dict:
                    leaf_data = eqx.tree_at(
                        lambda d: d.states, prepped_data, mapped_dict.pop("data.states")
                    )
                else:
                    leaf_data = prepped_data

                # Add remaining mapped deps to kwargs
                leaf_kwargs.update(mapped_dict)

                # Apply make_figs (with potential fig ops inside)
                return _make_figs_core(leaf_data, leaf_result, leaf_kwargs)

            # Map over the tree structure
            figs = jt.map(make_figs_at_leaf, *mapped_kwargs.values(), is_leaf=is_leaf)
        else:
            # No tree mapping, just apply make_figs (with potential fig ops)
            figs = _make_figs_core(prepped_data, result, prepped_kwargs)

        # Apply final ops
        figs = _apply_final_ops(
            self, "figs", figs, data=prepped_data, result=result, **prepped_kwargs
        )

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
        return (str, AnalysisRef, _DataField, AbstractAnalysis, LiteralInput)

    def is_analysis_input_leaf(self, leaf: Any) -> bool:
        """Determine if a leaf is a valid analysis input type."""
        return isinstance(leaf, self._input_leaf_types())

    @cached_property
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
                    or isinstance(leaf, (AnalysisRef, _DataField, AbstractAnalysis, LiteralInput))
                ):
                    valid_types = ", ".join([t.__name__ for t in self._input_leaf_types()])
                    raise ValueError(
                        f"Invalid dependency leaf in '{name}': {type(leaf)}. "
                        f"All leaves must be one of: {valid_types}"
                    )
            flattened[name] = leaves
        return flattened

    @cached_property
    def _input_treedefs(self) -> dict[str, Any]:
        """Get tree definitions for reconstructing PyTree inputs.

        Returns TreeDef for dependency trees. ExpandTo/Transformed PyTree registration
        automatically handles source substitution during flattening/unflattening.
        """
        return {
            name: jt.structure(source, is_leaf=self.is_analysis_input_leaf)
            for name, source in chain(self.inputs.items(), self._extra_inputs.items())
        }

    def _hashable_input_value(self, value: Any) -> Any:
        """Project dependency wiring into a stable hash payload."""
        if isinstance(value, AbstractAnalysis):
            return {"analysis": value.md5_str}
        if isinstance(value, AnalysisRef):
            return {
                "analysis_ref": self._hashable_input_value(value.target),
            }
        if isinstance(value, _DataField):
            return {
                "data_field": value.attr,
                "where": value.where,
                "is_leaf": value.is_leaf,
            }
        if isinstance(value, LiteralInput):
            return {"literal": value.value}
        if isinstance(value, ExpandTo):
            return {
                "expand_to": {
                    "target": self._hashable_input_value(value.target),
                    "source": self._hashable_input_value(value.source),
                    "where": value.where,
                    "is_leaf": value.is_leaf,
                    "is_leaf_prefix": value.is_leaf_prefix,
                }
            }
        if isinstance(value, Transformed):
            return {
                "transformed": {
                    "source": self._hashable_input_value(value.source),
                    "transform": value.transform,
                }
            }
        if isinstance(value, Mapping):
            children, treedef = eqx.tree_flatten_one_level(value)
            node = treedef.node_data()
            if node is None:
                raise TypeError(f"Unregistered analysis input mapping: {type(value).__qualname__}")
            cls, metadata = node
            # Match the tree used at execution: dict keys are sorted by JAX,
            # while ordered or custom mappings carry their own order and metadata.
            return (
                b"feedbax.analysis.input-mapping.v2",
                cls.__module__,
                cls.__qualname__,
                metadata,
                tuple(self._hashable_input_value(child) for child in children),
            )
        if isinstance(value, tuple):
            return tuple(self._hashable_input_value(item) for item in value)
        if isinstance(value, list):
            return [self._hashable_input_value(item) for item in value]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field_.name: self._hashable_input_value(getattr(value, field_.name))
                for field_ in dataclasses.fields(value)
            }
        return value

    def _hashable_inputs(self) -> dict[str, Any]:
        return {
            name: self._hashable_input_value(source)
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

    def save_outputs(
        self,
        context: "AnalysisRunContext",
        result,
        figs: PyTree[go.Figure],
        hps: PyTree[TreeNamespace],
        dump_path: Optional[Path] = None,
        dump_formats: Sequence[str] = ("html",),
        label: Optional[str] = None,
        **dependencies,
    ) -> None:
        """Save analysis figures through a manifest-canonical run context."""
        if figs is None:
            return
        for i, _path, fig, params in self._figure_output_records(
            result,
            figs,
            hps,
            dependencies,
        ):
            context.record_figure(
                fig=fig,
                analysis_name=camel_to_snake(self.name),
                analysis_label=label,
                ordinal=i,
                params=params,
                dump_path=dump_path,
                dump_formats=dump_formats,
            )

    def _figure_output_records(
        self,
        result,
        figs: PyTree[go.Figure],
        hps: PyTree[TreeNamespace],
        dependencies: Mapping[str, Any],
    ) -> list[tuple[int, Any, go.Figure, dict[str, Any]]]:
        """Return flattened figure records with legacy-compatible metadata."""
        from feedbax.plot.utils import figs_flatten_with_paths

        # `sep="_"`` switches the label dunders for single underscores, so
        # in `_params_to_save` we can use an argument e.g. `train_pert_std` rather than
        # `train__pert__std`.
        param_keys = tree_level_labels(
            figs, label_fn=ldict_label_only_fn, is_leaf=is_type(go.Figure), sep="_"
        )
        figs_with_paths_flat = figs_flatten_with_paths(figs)

        # Construct this for reference to hps that should only vary with the task variant.
        hps_0 = jt.leaves(hps.get(self.variant, hps), is_leaf=is_type(TreeNamespace))[0]
        ops_params_dict = self._extract_ops_info()
        records = []

        for i, (path, fig) in enumerate(figs_with_paths_flat):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))

            # Include fields from this instance, but only if they are JSON serializable.
            field_params = {k: v for k, v in self._field_params.items() if is_json_serializable(v)}

            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree.
                **field_params,  # From the fields of the analysis subclass instance.
                **self._params_to_save(
                    hps,
                    result=result,
                    **path_params,
                    **dependencies,
                ),
                eval_n=hps_0.task.eval_n,
            )

            if ops_params_dict:
                params["ops"] = ops_params_dict
            records.append((i, path, fig, params))

        return records

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
        """Returns a copy of this analysis with updated figure parameters.

        Deep-merges with existing parameters.
        """
        #! TODO:
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            MappingProxyType(deep_merge(self.fig_params, kwargs)),
        )

    def _after_transform(
        self,
        fn: Callable[..., Any],
        level: str | Sequence[str] | None,
        dependency_names: str | Sequence[str] | None,
        metadata: dict[str, Any] | None,
        *,
        name: str,
        params: dict[str, Any],
        uses_callback: bool = False,
    ) -> Self:
        dep_str = f"{dependency_names}" if dependency_names else "dependencies"

        if level is None:
            description = f"Transform {dep_str} with {get_name_of_callable(fn)}"
            label = f"pre-transform_{get_name_of_callable(fn)}"
            transform_fn = fn

        else:
            if isinstance(level, str):
                levels = [level]
            else:
                levels = level

            description = (
                f"Transform {dep_str} at LDict levels {levels} with {get_name_of_callable(fn)}"
            )
            label = f"pre-transform-{', '.join(levels)}_{get_name_of_callable(fn)}"

            func_wrapped = wrap_to_accept_var_kwargs(fn)

            def _transform_levels(dep_data, levels=levels, **kwargs):
                tree = dep_data
                for level in levels:
                    tree = jt.map(
                        lambda node: func_wrapped(node, **kwargs),
                        tree,
                        is_leaf=LDict.is_of(level),
                    )
                return tree

            transform_fn = _transform_levels

        final_metadata = _merge_metadata(metadata, dict(descriptions=[description], labels=[label]))

        return self._add_prep_op(
            name=name,
            dep_name=dependency_names,
            fn=transform_fn,
            params=dict(level=level, **params),
            metadata=final_metadata,
            uses_callback=uses_callback,
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

        return self._after_transform(
            getitem,
            level,
            dependency_name,
            final_metadata,
            name="after_getitem_at_level",
            params=dict(key=key),
        )

    #! TODO: Implement this. Note that `axis_fn` isn't implemented at this point.
    #! Also there is no notion of a vectorized *figure*, so this might as well just
    #! be a convenience for unstacking the axis and then tree mapping.
    # def map_figs_by_axis(
    #     self,
    #     level: str,
    #     *,
    #     axis_fn: Callable[[Any], str],
    #     metadata: Optional[Dict[str, Any]] = None,
    # ) -> Self:
    #     """Map figures by creating new axis labels using axis_fn."""
    #     description = "DEBUG THIS"
    #     final_metadata = _merge_metadata(metadata, dict(descriptions=[description]))

    #     return self._append_fig_op(
    #         name="map_figs_by_axis",
    #         dep_name=None,
    #         axis_fn=axis_fn,
    #         level=level,
    #         metadata=final_metadata,
    #     )

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
        fn: Callable[..., Any],
        is_leaf: Optional[Callable[[Any], bool]] = lambda _: True,
        # level_to_tuple_arg: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Returns a copy of this analysis that transforms its PyTree of results.

        The transformation occurs prior to the generation of figures, thus affects them.

        By default, the transformation is applied to the entire results PyTree. To apply the
        transformation to standard leaves (arraylikes etc.) pass `is_leaf=None`.
        """
        return self._then_transform(
            op_type="results",
            fn=fn,
            is_leaf=is_leaf,
            metadata=metadata,
        )

    def _then_transform(
        self,
        op_type: _FinalOpKeyType,
        fn: Callable[..., Any],
        level_to_tuple_arg: Optional[str] = None,
        levels: Optional[Sequence[str | LDictConstructor | type]] = None,
        invert_levels: bool = False,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Self:
        label = f"post-transform-{op_type}_{get_name_of_callable(fn)}"
        func_name = get_name_of_callable(fn)
        if op_type == "results":
            description = f"Transform analysis results using {func_name}"
        else:  # op_type == "figs"
            description = f"Transform output figures using {func_name}"

        func_with_var_kwargs = wrap_to_accept_var_kwargs(fn)

        # Only enter here if `op_type == "figs"``
        if levels is not None:
            if levels:
                description += f" over LDict levels {levels}"

                if invert_levels:
                    description += " (inverted selection)"
            elif invert_levels:
                # If levels=() and invert_levels=True, that means all levels
                description += " over all LDict levels"

            # Normalize to the same form as returned by `jtree.tree_level_types`
            levels = [LDict.of(lvl) if isinstance(lvl, str) else lvl for lvl in levels]

            def _transform_levels(figs, **kwargs):
                # Move levels to map over to the bottom and map over them
                level_types = jtree.tree_level_types(figs, is_leaf=is_type(go.Figure))
                if invert_levels:
                    levels_ = [lvl for lvl in level_types if lvl not in levels]
                else:
                    levels_ = levels
                figs = jtree.rearrange_uniform_tree(
                    figs, spec=[Ellipsis] + list(levels_), is_leaf=is_type(go.Figure)
                )

                #! TODO: Refactor? I think this is redundant with something in `jtree`
                leaf_level = levels_[0]
                if isinstance(leaf_level, LDictConstructor):
                    is_leaf_ = leaf_level.predicate
                else:
                    is_leaf_ = is_type(leaf_level)

                figs = jt.map(
                    lambda x: func_with_var_kwargs(x, **kwargs),
                    figs,
                    is_leaf=is_leaf_,
                )
                # ? figs = jtree.rearrange_uniform_tree(
                # ?     figs, spec=level_types, is_leaf=is_type(go.Figure)
                # ? )
                return figs

            final_metadata = _merge_metadata(
                metadata, dict(descriptions=[description], label=label)
            )
            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                fn=_transform_levels,
                params=dict(level=level_to_tuple_arg),
                is_leaf=lambda _: True,
                metadata=final_metadata,
                uses_callback=True,
            )

        if level_to_tuple_arg is not None:
            description += f" at LDict level '{level_to_tuple_arg}'"

            # Apply the transformation leafwise across the `level` LDict level;
            # e.g. suppose there are two keys in the `level` LDict, then the transformation
            # will be applied to 2-tuples of leaves; following the transformation, reconsistute
            # the `level` LDict with the transformed leaves.
            @wraps(fn)
            def _transform_level(ldict_node, **kwargs):
                if not LDict.is_of(level_to_tuple_arg)(ldict_node):
                    return ldict_node

                zipped = jtree.zip_(
                    *ldict_node.values(),
                    is_leaf=is_leaf,
                    zip_cls=_ThenTransformTuple,
                )
                transformed = jt.map(
                    lambda x: func_with_var_kwargs(x, **kwargs),
                    zipped,
                    is_leaf=is_type(_ThenTransformTuple),
                )
                unzipped = jtree.unzip(transformed, tuple_cls=_ThenTransformTuple)
                return LDict.of(level_to_tuple_arg)(dict(zip(ldict_node.keys(), unzipped)))

            final_metadata = _merge_metadata(
                metadata, dict(descriptions=[description], label=label)
            )

            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                fn=_transform_level,
                params=dict(level=level_to_tuple_arg),
                is_leaf=LDict.is_of(level_to_tuple_arg),
                metadata=final_metadata,
                uses_callback=True,
            )
        else:
            final_metadata = _merge_metadata(
                metadata, dict(descriptions=[description], label=label)
            )
            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                fn=fn,
                is_leaf=is_leaf,
                metadata=final_metadata,
                uses_callback=True,
            )

    def _add_prep_op(
        self,
        name: str,
        dep_name: Optional[str | Sequence[str]],
        fn: Callable,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        uses_callback: bool,
    ) -> Self:
        # If the transform consumes extra dependencies, ensure the analysis
        # instance knows about them so the graph builder evaluates them.
        analysis_with_deps = self
        spec_map = getattr(fn, "_extra_inputs", None)  # {port -> dep_spec}
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

        # Ensure `CallWithDeps` extra deps are not excluded by the wrapper (allowed_extra)
        wrapped_fn = wrap_to_accept_var_kwargs(fn, allowed_extra=getattr(fn, "_ports", ()))

        return eqx.tree_at(
            lambda a: a._prep_ops,
            analysis_with_deps,
            analysis_with_deps._prep_ops
            + (
                _PrepOp(
                    name=name,
                    dep_name=dep_name,
                    wrapped_fn=wrapped_fn,
                    fn=fn,
                    params=params or {},
                    metadata=metadata or {},
                    callback_token=object() if uses_callback else None,
                ),
            ),
        )

    def _add_final_op(
        self,
        op_type: _FinalOpKeyType,
        name: str,
        fn: Callable,
        params: Optional[Dict[str, Any]] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        uses_callback: bool,
    ) -> Self:
        # Log the most specific description if available
        if metadata and "descriptions" in metadata and metadata["descriptions"]:
            logger.debug(
                f"Add final-op to {self.__class__.__name__}: {metadata['descriptions'][0]}"
            )

        current_ops = self._final_ops_by_type.get(op_type, ())
        new_op = _FinalOp(
            name=name,
            wrapped_fn=wrap_to_accept_var_kwargs(fn),
            fn=fn,
            params=params or {},
            is_leaf=is_leaf,
            metadata=metadata or {},
            callback_token=object() if uses_callback else None,
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
    def _logical_identity(self) -> tuple[str, tuple[Any, ...]]:
        opaque_leaves: list[Any] = []

        def project_leaf(value: Any) -> Any:
            if isinstance(value, PurePath):
                return value.parts
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return tuple(
                    (field.name, getattr(value, field.name))
                    for field in dataclasses.fields(value)
                )
            # Keep opaque PyTree metadata alive along with its identity.
            opaque_leaves.append(value)
            return id(value)

        # A caller callback is opaque. Only copies of the same registration
        # share its live identity, even if two registrations use the same callable.
        prep_params = tuple(
            (op.name, op.dep_name, op.params, op.callback_token) for op in self._prep_ops
        )
        fig_params = tuple(
            (op.name, op.dep_name, op.params, op.callback_token) for op in self._fig_ops
        )
        final_params = {
            kind: tuple((op.name, op.params, op.callback_token) for op in ops)
            for kind, ops in self._final_ops_by_type.items()
        }
        cls = type(self)
        fingerprint = _fingerprint_semantic_values(
            self._hashable_inputs(),
            self._field_params,
            prep_params,
            fig_params,
            final_params,
            header=("feedbax.analysis.logical-node", cls.__module__, cls.__qualname__),
            project_leaf=project_leaf,
        )
        node_id = hashlib.md5(bytes.fromhex(fingerprint), usedforsecurity=False).hexdigest()
        return node_id, tuple(opaque_leaves)

    @property
    def md5_str(self) -> str:
        """Identify a logical node within a live analysis graph."""
        return self._logical_identity[0]

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
                    prepped_kwargs[name] = prep_op.wrapped_fn(
                        prepped_kwargs[name],
                        data=data,
                        **prepped_kwargs,
                    )
                except Exception as e:
                    logger.error(f"Error applying prep_op transform to '{name}'", exc_info=True)
                    raise e

            # Pop any extra dependencies that are not part of the analysis interface
            extra_ports = getattr(prep_op.wrapped_fn, "_ports", ())
            for port in extra_ports:
                if port not in self.inputs:
                    prepped_kwargs.pop(port, None)

        return prepped_kwargs

    #! TEMPORARY
    #! TODO: Simplify this. It is overcomplicated and performs needlessly redundant work.
    #! We should only need a single prep-op to align dependencies -- as long as we modify
    #! `_run_prep_ops` so that *all* `dependency_names` dependencies get passed and modified
    #! by each prep op.
    #! Likewise, the spec alignment might be refactorable with the logic of
    #! `jtree.rearrange_uniform_tree`, especially re: specifying specs that can include non-LDict
    #! level types.
    def _add_explicit_alignment_prep_op(
        self,
        dep_names: list[str],
        output_spec: list[str],
        map_type: str,
    ) -> Self:
        """Add prep op to align dependencies to explicit output structure."""

        def make_align_fn(this_dep_name, all_dep_names):
            """Create alignment function for a specific dependency."""

            def align_to_spec(dep_tree, output_spec=output_spec, **kwargs):
                """Align this dependency to the explicit output structure."""
                # Collect all dependencies for reference
                all_deps = {}
                for name in all_dep_names:
                    if name == this_dep_name:
                        all_deps[name] = dep_tree
                    elif name in kwargs:
                        all_deps[name] = kwargs[name]

                # Align just this dependency using others as reference
                aligned = _align_trees_to_structure(
                    {this_dep_name: dep_tree},
                    output_spec,
                    reference_trees={k: v for k, v in all_deps.items() if k != this_dep_name},
                )
                return aligned[this_dep_name]

            return align_to_spec

        modified = self
        for dep_name in dep_names:
            align_fn = make_align_fn(dep_name, dep_names)
            description = f"Align {dep_name} to structure {output_spec} for {map_type} mapping"
            modified = modified._add_prep_op(
                name="align_to_explicit_spec",
                dep_name=dep_name,
                fn=align_fn,
                params=dict(output_spec=output_spec, map_type=map_type),
                metadata=dict(descriptions=[description]),
                uses_callback=False,
            )

        return modified

    def _add_ellipsis_alignment_prep_op(
        self,
        dep_names: list[str],
        output_spec: list[str | EllipsisType],
        is_leaf: Callable,
        map_type: str,
    ) -> Self:
        """Add prep op for alignment with ellipsis in spec."""

        # Find ellipsis position
        ellipsis_idx = output_spec.index(Ellipsis)
        before_ellipsis = output_spec[:ellipsis_idx]
        after_ellipsis = output_spec[ellipsis_idx + 1 :]

        def make_align_fn(this_dep_name, all_dep_names):
            """Create alignment function for a specific dependency."""

            def align_with_ellipsis(
                dep_tree, before=before_ellipsis, after=after_ellipsis, is_leaf=is_leaf, **kwargs
            ):
                """Expand ellipsis based on tree structure and align."""
                # Get current levels up to is_leaf
                current_levels = tree_level_labels(dep_tree, is_leaf=is_leaf)

                # Determine what goes in the ellipsis position
                middle_levels = [
                    level for level in current_levels if level not in before and level not in after
                ]

                # Construct full target structure
                target_structure = list(before) + middle_levels + list(after)

                # Collect all dependencies for reference
                all_deps = {}
                for name in all_dep_names:
                    if name == this_dep_name:
                        all_deps[name] = dep_tree
                    elif name in kwargs:
                        all_deps[name] = kwargs[name]

                # Align using helper
                aligned = _align_trees_to_structure(
                    {this_dep_name: dep_tree},
                    target_structure,
                    is_leaf=is_leaf,
                    reference_trees={k: v for k, v in all_deps.items() if k != this_dep_name},
                )
                return aligned[this_dep_name]

            return align_with_ellipsis

        modified = self
        for dep_name in dep_names:
            align_fn = make_align_fn(dep_name, dep_names)
            spec_str = f"{list(before_ellipsis)}...{list(after_ellipsis)}"
            description = f"Align {dep_name} to structure {spec_str} for {map_type} mapping"
            modified = modified._add_prep_op(
                name="align_with_ellipsis",
                dep_name=dep_name,
                fn=align_fn,
                params=dict(output_spec=output_spec, map_type=map_type),
                metadata=dict(descriptions=[description]),
                uses_callback=True,
            )

        return modified

    def _add_inferred_alignment_prep_op(
        self,
        dep_names: list[str],
        is_leaf: Callable,
        map_type: str,
    ) -> Self:
        """Add prep op that infers structure from all dependencies."""

        def make_align_fn(this_dep_name, all_dep_names):
            """Create alignment function for a specific dependency."""

            def infer_and_align(dep_tree, is_leaf=is_leaf, **kwargs):
                """Infer structure from all dependencies and align."""
                # Collect all dependencies
                all_deps = {}
                for name in all_dep_names:
                    if name == this_dep_name:
                        all_deps[name] = dep_tree
                    elif name in kwargs:
                        all_deps[name] = kwargs[name]

                # Infer target structure by taking union of all levels
                all_levels = []
                level_order = {}  # Track first occurrence position

                for dep in all_deps.values():
                    dep_levels = tree_level_labels(dep, is_leaf=is_leaf)
                    for i, level in enumerate(dep_levels):
                        if level not in level_order:
                            level_order[level] = len(all_levels)
                            all_levels.append(level)

                # Sort by first occurrence to maintain reasonable order
                target_structure = sorted(level_order.keys(), key=lambda x: level_order[x])

                # Align this tree to the inferred structure
                aligned = _align_trees_to_structure(
                    {this_dep_name: dep_tree},
                    target_structure,
                    is_leaf=is_leaf,
                    reference_trees={k: v for k, v in all_deps.items() if k != this_dep_name},
                )
                return aligned[this_dep_name]

            return infer_and_align

        modified = self
        for dep_name in dep_names:
            align_fn = make_align_fn(dep_name, dep_names)
            description = f"Infer structure and align {dep_name} for {map_type} mapping"
            modified = modified._add_prep_op(
                name="infer_and_align",
                dep_name=dep_name,
                fn=align_fn,
                params=dict(map_type=map_type),
                metadata=dict(descriptions=[description]),
                uses_callback=True,
            )

        return modified


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


def implements_analysis_method(
    analysis: AbstractAnalysis, method: Literal["compute", "make_figs"]
) -> bool:
    """Check whether the resolved class method replaces the default stub."""
    return getattr(type(analysis), method) is not getattr(AbstractAnalysis, method)


class IdentityNode(AbstractAnalysis[SinglePort[Any]]):
    """An analysis that simply returns its input as output.

    Useful for debugging or as a placeholder.
    """

    Ports = SinglePort[Any]
    inputs: SinglePort[Any] = eqx.field(
        default_factory=SinglePort[Any], converter=SinglePort[Any].converter
    )

    def compute(self, data: AnalysisInputData, *, input, **kwargs) -> Any:
        return input
