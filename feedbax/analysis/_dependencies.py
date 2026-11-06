"""Compile and execute the dependencies reachable from analysis outputs."""

import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any, Callable, Optional

import equinox as eqx
import jax.tree as jt
import feedbax._vendor.jax_cookbook.tree as jtree
from jax.tree_util import PyTreeDef, treedef_is_leaf
from feedbax._vendor.jax_cookbook import is_type
from feedbax._vendor.jax_cookbook.progress import progress_piter
from feedbax._vendor.jax_cookbook.tree import collect_aux_data
from jaxtyping import PyTree

from feedbax.analysis.analysis import (
    AbstractAnalysis,
    AnalysisRef,
    ExpandTo,
    LiteralInput,
    NoPorts,
    Transformed,
    _DataField,
    _FinalOp,
)
from feedbax.analysis.types import AnalysisInputData

logger = logging.getLogger(__name__)


def _format_output_keys(keys: set[str]) -> str:
    return ", ".join(repr(key) for key in sorted(keys))


class _DataForwarder(AbstractAnalysis[NoPorts]):
    """Forwards a single attribute of `AnalysisInputData`.

    This node exists only to integrate `Data.<attr>` references into the
    dependency graph.  It performs no computation other than returning the
    chosen attribute and produces no figures.

    NOTE: See also `_DataField` which is used to specify the attribute to forward.
    """

    # Name of the attribute (e.g. "states", "models", …) to be forwarded
    attr: str = ""
    where: Optional[Callable] = None
    is_leaf: Optional[Callable[[Any], bool]] = None

    # Pure forwarding
    def compute(self, data: AnalysisInputData, **kwargs):
        value = getattr(data, self.attr)
        if self.where is not None:
            value = jt.map(self.where, value, is_leaf=self.is_leaf)
        return value

    def __post_init__(self):
        if not self.attr:
            raise ValueError("_DataForwarder.attr must be provided")


@dataclass(frozen=True, slots=True)
class _PortBinding:
    name: str
    treedef: PyTreeDef
    leaves: tuple[str | LiteralInput, ...]


@dataclass(frozen=True, slots=True)
class _AnalysisStep:
    analysis: AbstractAnalysis
    params: dict[str, Any]
    ports: tuple[_PortBinding, ...]


def _resolve_source(
    source: AbstractAnalysis | AnalysisRef | _DataField | str,
    lookup: dict[str, AbstractAnalysis],
) -> AbstractAnalysis:
    if isinstance(source, AnalysisRef):
        source = source.target
    if isinstance(source, str):
        if source not in lookup:
            raise ValueError(
                f"String dependency '{source}' could not be resolved. "
                "Provide dependency_lookup with all available keys."
            )
        return lookup[source]
    if isinstance(source, _DataField):
        return _DataForwarder(attr=source.attr, where=source.where, is_leaf=source.is_leaf)
    return source


def _validate_signature(func, required_names: list[str], func_label: str, analysis_name: str):
    """Check that *func* lists all *required_names* unless it has **kwargs."""
    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    missing = [name for name in required_names if name not in sig.parameters]
    if missing:
        raise ValueError(
            f"Analysis '{analysis_name}' expects parameters {missing} in `{func_label}` but they are not present. "
            "Add them to the signature or accept **kwargs."
        )


def _port_order(analysis: AbstractAnalysis) -> list[str]:
    """Schedule each ExpandTo target before the port that uses it."""
    targets = {
        name: tuple(
            op.target for op in collect_aux_data(treedef, ExpandTo) if isinstance(op.target, str)
        )
        for name, treedef in analysis._input_treedefs.items()
    }
    try:
        return list(TopologicalSorter(targets).static_order())
    except CycleError as exc:
        raise ValueError(f"Circular dependency detected at node {exc.args[1][0]}") from exc


def _compile_plan(
    roots: Iterable[AbstractAnalysis],
    lookup: dict[str, AbstractAnalysis],
) -> dict[str, _AnalysisStep]:
    """Resolve each reachable node once and return steps in dependency order."""
    seen: dict[str, AbstractAnalysis] = {}
    plan: dict[str, _AnalysisStep] = {}

    def visit(analysis: AbstractAnalysis, params: dict[str, Any]) -> str:
        node_id = analysis.md5_str
        if node_id in seen:
            previous = seen[node_id]
            if type(previous) is not type(analysis) and vars(type(previous)).get(
                "computation_revision"
            ) != vars(type(analysis)).get("computation_revision"):
                raise ValueError(
                    f"Analysis {analysis.name!r} has conflicting computation revisions "
                    f"for logical node {node_id}"
                )
            if node_id not in plan:
                raise ValueError(f"Circular dependency detected at node {node_id}")
            return node_id

        seen[node_id] = analysis
        names = list(analysis.inputs.keys())
        _validate_signature(analysis.compute, names, "compute", analysis.name)
        _validate_signature(analysis.make_figs, names + ["result"], "make_figs", analysis.name)

        ports = []
        kwargs_by_port = analysis.dependency_kwargs()
        for name in _port_order(analysis):
            leaves: list[str | LiteralInput] = []
            for source in analysis._flattened_inputs[name]:
                if isinstance(source, LiteralInput):
                    leaves.append(source)
                else:
                    dependency = _resolve_source(source, lookup)
                    kwargs = {} if isinstance(source, _DataField) else kwargs_by_port.get(name, {})
                    leaves.append(visit(dependency, kwargs))
            ports.append(_PortBinding(name, analysis._input_treedefs[name], tuple(leaves)))
        plan[node_id] = _AnalysisStep(analysis, params, tuple(ports))
        return node_id

    for root in roots:
        visit(root, {})
    return plan


#! Could add the transformation / unpacking logic to the `unflatten` methods of `ExpandTo` and
#! `Transformed`, which would apply the transformations inside-out at unflatten time;
#! this would greatly simplify this function, but we'd need a custom recursive function to
#! conservatively operate on PyTrees containing `ExpandTo` and `Transformed` nodes.
#! https://chatgpt.com/share/68878682-f454-8006-8ff3-433acdcf3f95
def _apply_transformations(tree: Any, dep_kwargs: dict[str, Any]) -> Any:
    """Apply ExpandTo and Transformed transformations recursively from innermost to outermost."""

    # Handle transformation objects first
    if isinstance(tree, ExpandTo):
        # First, recursively transform the source
        transformed_source = _apply_transformations(tree.source, dep_kwargs)

        # Then apply ExpandTo transformation
        if isinstance(tree.target, str):
            target_structure = dep_kwargs[tree.target]
        elif isinstance(tree.target, _DataField):
            # For _DataField targets, we need to resolve them like regular dependencies
            # This is a simplified approach - in practice you might need more sophisticated target
            # resolution
            #! TODO
            raise NotImplementedError(
                "_DataField targets in ExpandTo not yet supported in new PyTree system"
            )
        else:
            raise TypeError(f"Invalid ExpandTo target type: {type(tree.target)}")

        # Apply where function if provided to select subtree of target
        if tree.where is not None:
            target_structure = tree.where(target_structure)

        return jtree.prefix_expand(
            transformed_source,
            target_structure,
            is_leaf=tree.is_leaf,
            is_leaf_prefix=tree.is_leaf_prefix,
        )

    elif isinstance(tree, Transformed):
        # First, recursively transform the source
        transformed_source = _apply_transformations(tree.source, dep_kwargs)

        # Then apply the transformation
        return tree.transform(transformed_source)

    elif isinstance(tree, LiteralInput):
        # LiteralInput will not contain any further transformations
        return tree.value

    elif treedef_is_leaf(jt.structure(tree)):
        # If it's a leaf node, return it as is
        return tree

    else:
        leaves, tree_def = jt.flatten(tree, is_leaf=is_type(ExpandTo, Transformed))

        # For other PyTree structures, recursively apply transformations to children
        transformed_leaves = [_apply_transformations(leaf, dep_kwargs) for leaf in leaves]

        return jt.unflatten(tree_def, transformed_leaves)


def _reconstruct_dependencies(
    ports: tuple[_PortBinding, ...], computed_results: dict[str, Any]
) -> dict[str, Any]:
    dep_kwargs = {}
    for port in ports:
        leaves = [
            leaf.value if isinstance(leaf, LiteralInput) else computed_results[leaf]
            for leaf in port.leaves
        ]
        tree = jt.unflatten(port.treedef, leaves)
        dep_kwargs[port.name] = _apply_transformations(tree, dep_kwargs)
    return dep_kwargs


def _stage_collector(ops, tree, **kwargs):
    outs = [tree]
    for op in ops:
        tree = jt.map(
            lambda leaf: op.wrapped_fn(leaf, **kwargs),
            tree,
            is_leaf=op.is_leaf,
        )
        outs.append(tree)
    return tuple(outs)


def _replace_results_final_ops(inst, new_ops_tuple):
    """Return a clone with result final-ops replaced."""
    d = dict(inst._final_ops_by_type)
    d["results"] = new_ops_tuple
    return eqx.tree_at(lambda a: a._final_ops_by_type, inst, MappingProxyType(d))


def _run_preflight_memory_estimation(
    dep_instance: AbstractAnalysis,
    data: AnalysisInputData,
    dep_kwargs: dict[str, Any],
    log_name: str,
) -> None:
    """Run preflight memory estimation for an analysis instance.

    Args:
        dep_instance: The analysis instance to estimate memory for
        data: Input data for analysis
        dep_kwargs: Keyword arguments for the analysis compute method
        log_name: Name for logging purposes
    """
    preflight_mode = getattr(dep_instance, "_estimate_mem_preflight")

    if preflight_mode != "off":
        try:
            # Avoid caching during the eval_shape pass
            preflight_inst = eqx.tree_at(lambda obj: obj.cache_result, dep_instance, False)

            if preflight_mode == "final":
                final_shapes = eqx.filter_eval_shape(
                    lambda obj, d, **kw: obj._compute_with_ops(d, **kw),
                    preflight_inst,
                    data,
                    **dep_kwargs,
                )
                final_gb = jtree.struct_bytes(final_shapes) / 1e9
                logger.info(f"Estimated final memory usage for {log_name}: {final_gb:.2f} GB")

            elif preflight_mode == "stages":
                ops = preflight_inst._final_ops_by_type.get("results", ())

                collected_fn = partial(_stage_collector, ops)
                tap = _FinalOp(
                    name="__shape_mem_tap__",
                    fn=lambda *args, **kwargs: None,
                    wrapped_fn=collected_fn,
                    params={},
                    is_leaf=None,
                    metadata=dict(label="shape-mem-tap"),
                )
                tapped = _replace_results_final_ops(preflight_inst, (tap,))

                stages_shapes = eqx.filter_eval_shape(
                    lambda inst, d, **kw: inst._compute_with_ops(d, **kw),
                    tapped,
                    data,
                    **dep_kwargs,
                )
                # stages_shapes is a tuple: (pre_final, after_op1, after_op2, ...)
                stages_gb = [jtree.struct_bytes(s) / 1e9 for s in stages_shapes]
                names = ["pre-final"] + [f"{op.name}/{op.label}" for op in ops]
                peak = max(stages_gb)

                parts = ", ".join(f"{n} {gb:.2f}GB" for n, gb in zip(names, stages_gb))
                logger.info(
                    f"Estimated memory usage for {log_name} across chain of final ops: "
                    f"{parts} (peak≈{peak:.2f} GB)"
                )

        except Exception as e:
            logger.warning(f"Failed to estimate memory usage for {log_name}: {e}")


def compute_dependency_results(
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    custom_dependencies: dict[str, AbstractAnalysis] | None = None,
    requested_outputs: set[str] | None = None,
    analysis_context: Any | None = None,
    **kwargs,
) -> list[dict[str, PyTree[Any]]]:
    """Compute all dependencies in correct order.

    Args:
        analyses: Analysis outputs keyed by name
        data: Input data for analysis
        custom_dependencies: Additional analysis instances available by name
        requested_outputs: If provided, only compute the analyses whose keys
            appear in this set (plus their transitive dependencies).  When
            ``None``, every output and its dependencies are computed.
        **kwargs: Additional baseline dependencies
    """
    if custom_dependencies is None:
        custom_dependencies = {}

    selected = analyses
    if requested_outputs is not None:
        available_outputs = set(analyses)
        matched = available_outputs & requested_outputs
        if not matched:
            raise ValueError(
                "None of the requested analysis outputs matched available analysis keys. "
                f"requested_outputs=[{_format_output_keys(requested_outputs)}]; "
                f"available_analysis_keys=[{_format_output_keys(available_outputs)}]"
            )

        selected = {name: analysis for name, analysis in analyses.items() if name in matched}

    dependency_lookup = custom_dependencies | analyses
    plan = _compile_plan(selected.values(), dependency_lookup)
    labels_by_instance = {id(analysis): key for key, analysis in dependency_lookup.items()}
    output_ids = [analysis.md5_str for analysis in selected.values()]
    leaf_node_ids = set(output_ids)

    leaf_dependencies = {}  # Cache reconstructed dependencies for leaf analyses

    baseline_kwargs = kwargs.copy()
    computed_results = {}

    # Execute all dependencies in topological order
    with progress_piter(plan, description="Computing analysis nodes", eta_halflife=10) as (
        nodes,
        update_piter,
    ):
        for node_id in nodes:
            step = plan[node_id]
            dep_instance = step.analysis
            node_label = labels_by_instance.get(
                id(dep_instance), f"{type(dep_instance).__name__} ({node_id[:7]})"
            )
            update_piter.subdescription(node_label)

            # Reconstruct dependencies and add instance parameters
            dep_kwargs = baseline_kwargs.copy()
            dep_kwargs.update(step.params)

            reconstructed_deps = _reconstruct_dependencies(step.ports, computed_results)
            dep_kwargs.update(reconstructed_deps)

            if node_id in leaf_node_ids:
                # Cache dependency reconstruction for leaf analyses to avoid redundant work
                leaf_dependencies[node_id] = reconstructed_deps

            if dep_instance.__class__.compute is AbstractAnalysis.compute:
                # Skip this node -- no implementation of compute()
                logger.debug(f"Skipping analysis node: {node_label} (no compute implementation)")
                result = None
            else:
                if not node_label.startswith("_DataForwarder"):
                    logger.debug(f"Computing analysis node: {node_label}")

                # Run preflight memory estimation
                _run_preflight_memory_estimation(dep_instance, data, dep_kwargs, node_label)

                # Execute analysis and store result
                result = dep_instance._compute_with_ops(
                    data,
                    analysis_context=analysis_context,
                    **dep_kwargs,
                )
            computed_results[node_id] = result

    # Assemble final results for each requested analysis
    all_dependency_results = []
    for node_id in output_ids:
        dependency_results = baseline_kwargs.copy()
        dependency_results["result"] = computed_results[node_id]
        dependency_results.update(leaf_dependencies[node_id])
        all_dependency_results.append(dependency_results)

    return all_dependency_results
