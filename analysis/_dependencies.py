"""
Interpret analysis graphs, to avoid re-computing shared dependencies.

This is useful because it allows us to specify the dependencies as class attributes of the
subclasses of `AbstractAnalysis`, and they will automatically be computed only once for all
of the requested analyses (in `run_analysis.py`).

An alternative is to allow for statefulness of the analysis classes. Then analyses callback to
their dependencies, and also memoize their results so that repeat work is avoided. However,
I've decided to use `eqx.Module` and stick with a stateless solution. So we need to explicitly
parse the graph.
"""

import hashlib
import inspect
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Dict, Optional, Set

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
from jax.tree_util import treedef_is_leaf
from jax_cookbook import is_type
from jax_cookbook.progress import piter, progress_piter
from jax_cookbook.tree import collect_aux_data
from jaxtyping import PyTree

from feedbax_experiments.analysis.analysis import (
    AbstractAnalysis,
    ExpandTo,
    LiteralInput,
    NoPorts,
    Transformed,
    _DataField,
    _FinalOp,
    _format_dict_of_params,
)
from feedbax_experiments.misc import get_md5_hexdigest
from feedbax_experiments.tree_utils import prefix_expand
from feedbax_experiments.types import AnalysisInputData

logger = logging.getLogger(__name__)


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


def param_hash(params: Dict[str, Any]) -> str:
    """Create a hash of parameter values to uniquely identify dependency configurations."""
    # Convert params to a stable string representation and hash it
    params_formatted = _format_dict_of_params(params)
    param_str = json.dumps(params_formatted, sort_keys=True)
    return get_md5_hexdigest(param_str)


def resolve_dependency_node(analysis, dep_name, dep_source, dependency_lookup=None):
    """Resolve a dependency source to an analysis instance and create a graph node ID.

    Args:
        analysis: The analysis instance requesting the dependency
        dep_name: The name of the dependency port
        dep_source: Either a class type, string reference, analysis instance, or LiteralInput
        dependency_lookup: Optional dict for resolving string references
    Returns:
        tuple: (node_id, params, analysis_instance) or None for LiteralInput
    """
    # Handle LiteralInput - skip dependency resolution
    if isinstance(dep_source, LiteralInput):
        return None

    # Handle None inputs by skipping them
    if dep_source is None:
        return None

    # Handle forwarding of attributes from AnalysisInputData via the `Data` sentinel
    if isinstance(dep_source, _DataField):
        # Treat each attribute as a unique forwarding analysis node, including any transform
        analysis_instance = _DataForwarder(
            attr=dep_source.attr,
            where=dep_source.where,
            is_leaf=dep_source.is_leaf,
        )
        node_id = analysis_instance.md5_str
        return node_id, {}, analysis_instance

    class_params = analysis.dependency_kwargs().get(dep_name, {})
    # Recursively resolve string dependencies
    if isinstance(dep_source, str):
        if dependency_lookup is not None and dep_source in dependency_lookup:
            dep_instance = dependency_lookup[dep_source]
            node_id = dep_source  # Use the string key as node_id for deduplication
            return node_id, class_params, dep_instance
        else:
            raise ValueError(
                f"String dependency '{dep_source}' could not be resolved. Provide dependency_lookup with all available keys."
            )

    else:
        # Already an analysis instance - use its hash directly
        if dependency_lookup is not None:
            for k, v in dependency_lookup.items():
                if v is dep_source:
                    node_id = k
                    return node_id, class_params, dep_source
        node_id = dep_source.md5_str
        return node_id, class_params, dep_source


def _process_dependency_sources(analysis, dep_name, dep_sources, dependency_lookup, callback):
    """Helper to iterate over dependency sources and apply callback to non-LiteralInput deps."""
    for dep_source in dep_sources:
        dep_result = resolve_dependency_node(analysis, dep_name, dep_source, dependency_lookup)
        if dep_result is not None:  # Skip LiteralInput
            callback(dep_result)


def build_dependency_graph(
    analyses: Sequence[AbstractAnalysis],
    dependency_lookup=None,
) -> tuple[dict[str, set], dict]:
    """Build a directed acyclic graph of analysis dependencies."""
    graph = defaultdict(set)  # Maps node_id -> set of dependency node_ids
    nodes = {}  # Maps md5 hash (str) -> (analysis_instance, params)

    def _validate_signature(func, required_names: list[str], func_label: str, analysis_name: str):
        """Check that *func* lists all *required_names* unless it has **kwargs."""
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return  # **kwargs catches all
        missing = [n for n in required_names if n not in sig.parameters]
        if missing:
            raise ValueError(
                f"Analysis '{analysis_name}' expects parameters {missing} in `{func_label}` but they are not present. "
                "Add them to the signature or accept **kwargs."
            )

    def _add_deps(analysis: AbstractAnalysis):
        """Recursively add analysis and its dependencies to the graph."""
        md5_id = analysis.md5_str
        if md5_id not in nodes:
            nodes[md5_id] = (analysis, {})

            # Validate that compute/make_figs accept required parameters
            deps_list = list(analysis.inputs.keys())
            _validate_signature(analysis.compute, deps_list, "compute", analysis.name)
            _validate_signature(
                analysis.make_figs, deps_list + ["result"], "make_figs", analysis.name
            )

            # Recursively process all flattened dependencies
            for dep_name, dep_sources in analysis._flattened_inputs.items():

                def add_dependency(dep_result):
                    dep_node_id, params, dep_instance = dep_result
                    if dep_instance.md5_str not in nodes:
                        nodes[dep_instance.md5_str] = (dep_instance, params)
                        _add_deps(dep_instance)

                _process_dependency_sources(
                    analysis, dep_name, dep_sources, dependency_lookup, add_dependency
                )

    # Add all analyses from dependency lookup and main analysis list
    for analysis in (dependency_lookup or {}).values():
        _add_deps(analysis)
    for analysis in analyses:
        _add_deps(analysis)

    # Build graph edges from analyses to their direct dependencies
    for analysis in analyses:
        for dep_name, dep_sources in analysis._flattened_inputs.items():

            def add_edge(dep_result):
                _, _, dep_instance = dep_result
                analysis_node_id = analysis.md5_str
                dep_node_id = dep_instance.md5_str
                graph[analysis_node_id].add(dep_node_id)
                # Ensure every edge target is also present in `nodes`
                if dep_node_id not in nodes:
                    nodes[dep_node_id] = (dep_instance, {})

            _process_dependency_sources(
                analysis, dep_name, dep_sources, dependency_lookup, add_edge
            )

    # Build graph edges for transitive dependencies (dependencies of dependencies)
    for node_id, (dep_instance, _) in list(nodes.items()):
        for subdep_name, subdep_sources in dep_instance._flattened_inputs.items():

            def add_transitive_edge(dep_result):
                _, _, sub_dep_instance = dep_result
                sub_dep_id = sub_dep_instance.md5_str
                graph[node_id].add(sub_dep_id)
                if sub_dep_id not in nodes:
                    nodes[sub_dep_id] = (sub_dep_instance, {})

            _process_dependency_sources(
                dep_instance, subdep_name, subdep_sources, dependency_lookup, add_transitive_edge
            )

    # Ensure all nodes have graph entries (even if no dependencies)
    for node_id in nodes:
        if node_id not in graph:
            graph[node_id] = set()
    return dict(graph), nodes


def topological_sort(graph: dict[str, Set[str]]) -> list[str]:
    """Return dependencies in order they should be computed."""
    visited = set()
    temp_marks = set()
    order = []

    def visit(node: str):
        if node in temp_marks:
            raise ValueError(f"Circular dependency detected at node {node}")
        if node in visited:
            return

        temp_marks.add(node)

        # Visit all dependencies first
        for dep in graph.get(node, set()):
            visit(dep)

        temp_marks.remove(node)
        visited.add(node)
        order.append(node)

    # Visit all nodes
    for node in graph:
        if node not in visited:
            visit(node)

    return order


#! Could add the transformation / unpacking logic to the `unflatten` methods of `ExpandTo` and
#! `Transformed`, which would apply the transformations inside-out at unflatten time;
#! this would greatly simplify this function, but we'd need a custom recursive function to
#! conservatively operate on PyTrees containing `ExpandTo` and `Transformed` nodes.
#! https://chatgpt.com/share/68878682-f454-8006-8ff3-433acdcf3f95
def _apply_transformations(
    tree: Any, dep_kwargs: dict[str, Any], dependency_lookup: dict[str, AbstractAnalysis]
) -> Any:
    """Apply ExpandTo and Transformed transformations recursively from innermost to outermost."""

    # Handle transformation objects first
    if isinstance(tree, ExpandTo):
        # First, recursively transform the source
        transformed_source = _apply_transformations(tree.source, dep_kwargs, dependency_lookup)

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

        return prefix_expand(
            transformed_source,
            target_structure,
            is_leaf=tree.is_leaf,
            is_leaf_prefix=tree.is_leaf_prefix,
        )

    elif isinstance(tree, Transformed):
        # First, recursively transform the source
        transformed_source = _apply_transformations(tree.source, dep_kwargs, dependency_lookup)

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
        transformed_leaves = [
            _apply_transformations(leaf, dep_kwargs, dependency_lookup) for leaf in leaves
        ]

        return jt.unflatten(tree_def, transformed_leaves)


def _reconstruct_dependencies(
    analysis: AbstractAnalysis,
    computed_results: dict[str, Any],
    dependency_lookup: dict[str, AbstractAnalysis],
) -> dict[str, Any]:
    """Reconstruct PyTree dependencies for an analysis from computed leaf results."""
    dep_kwargs = {}

    # Extract ExpandTo field dependencies so we process our inputs in the correct order
    inputs_graph = {
        port_name: {
            aux_data.target
            for aux_data in collect_aux_data(treedef, ExpandTo)
            if isinstance(aux_data.target, str)
        }
        for port_name, treedef in analysis._input_treedefs.items()
    }

    # Process fields in dependency order to ensure ExpandTo targets are available
    for dep_name in topological_sort(inputs_graph):
        dep_sources = analysis._flattened_inputs[dep_name]
        leaf_results = []

        # Gather results for all leaves in this dependency
        for dep_source in dep_sources:
            if isinstance(dep_source, LiteralInput):
                # Add LiteralInput values directly to leaf_results
                leaf_results.append(dep_source.value)
            else:
                dep_result = resolve_dependency_node(
                    analysis, dep_name, dep_source, dependency_lookup=dependency_lookup
                )
                if dep_result is not None:
                    _, _, dep_instance = dep_result
                    dep_hash = dep_instance.md5_str
                    if dep_hash in computed_results:
                        leaf_results.append(computed_results[dep_hash])
                    else:
                        raise RuntimeError(
                            f"Missing dependency result for '{dep_name}' in analysis '{analysis.__class__.__name__}'. "
                            f"Dependency hash '{dep_hash}' not found in computed results."
                        )

        # Ensure all dependencies were resolved before reconstruction
        if len(leaf_results) != len(dep_sources):
            missing = len(dep_sources) - len(leaf_results)
            raise RuntimeError(
                f"Failed to resolve {missing}/{len(dep_sources)} dependencies for "
                f"'{dep_name}' in analysis '{analysis.__class__.__name__}'. "
                f"This indicates a bug in dependency resolution or circular dependencies."
            )

        # Reconstruct PyTree from leaf results - this gives us the structure with ExpandTo/Transformed objects
        tree_def = analysis._input_treedefs[dep_name]
        reconstructed_tree = jt.unflatten(tree_def, leaf_results)

        # Apply transformations recursively from innermost to outermost
        dep_kwargs[dep_name] = _apply_transformations(
            reconstructed_tree, dep_kwargs, dependency_lookup
        )

    return dep_kwargs


def _stage_collector(ops, x):
    outs = [x]
    for op in ops:
        x = op.transform_func(x)
        outs.append(x)
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

                tap = _FinalOp(
                    name="__shape_mem_tap__",
                    label="shape-mem-tap",
                    transform_func=partial(_stage_collector, ops),
                    params={},
                    is_leaf=None,
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
    custom_dependencies: Optional[Dict[str, AbstractAnalysis]] = None,
    **kwargs,
) -> list[dict[str, PyTree[Any]]]:
    """Compute all dependencies in correct order.

    Args:
        analyses: Analysis instances to process (sequence or dict)
        data: Input data for analysis
        custom_dependencies: Optional dict of custom dependency instances (from DEPENDENCIES)
        **kwargs: Additional baseline dependencies
    """
    if custom_dependencies is None:
        custom_dependencies = {}
    analyses_list = list(analyses.values())
    dependency_lookup = custom_dependencies | analyses

    # Build computation graph and determine execution order
    graph, dep_instances = build_dependency_graph(
        analyses_list, dependency_lookup=dependency_lookup
    )
    comp_order = topological_sort(graph)

    # Create reverse lookup for better logging
    node_labels = {}
    for key, instance in dependency_lookup.items():
        node_labels[instance.md5_str] = key

    for node_id, (instance, _) in dep_instances.items():
        if node_id not in node_labels:
            node_labels[node_id] = f"{instance.__class__.__name__} ({node_id[:7]})"

    # Track which nodes are leaf analyses to avoid redundant dependency reconstruction
    leaf_node_ids = {analysis.md5_str for analysis in analyses_list}
    leaf_dependencies = {}  # Cache reconstructed dependencies for leaf analyses

    baseline_kwargs = kwargs.copy()
    computed_results = {}

    # Execute all dependencies in topological order
    with progress_piter(comp_order, description="Computing analysis nodes", eta_halflife=10) as (
        nodes,
        update_piter,
    ):
        for node_id in nodes:
            node_label = node_labels.get(node_id, "[Unknown]")
            update_piter.subdescription(node_label)
            dep_instance, params = dep_instances[node_id]

            # Reconstruct dependencies and add instance parameters
            dep_kwargs = baseline_kwargs.copy()
            dep_kwargs.update(params)

            reconstructed_deps = _reconstruct_dependencies(
                dep_instance, computed_results, dependency_lookup
            )
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
                result = dep_instance._compute_with_ops(data, **dep_kwargs)
            computed_results[node_id] = result

    # Assemble final results for each requested analysis
    all_dependency_results = []
    for analysis in analyses_list:
        dependency_results = baseline_kwargs.copy()

        # Add the analysis result itself
        analysis_hash = analysis.md5_str
        if analysis_hash in computed_results:
            dependency_results["result"] = computed_results[analysis_hash]

        # Reuse cached dependency reconstruction for leaf analyses
        dependency_results.update(leaf_dependencies[analysis_hash])
        all_dependency_results.append(dependency_results)

    return all_dependency_results
