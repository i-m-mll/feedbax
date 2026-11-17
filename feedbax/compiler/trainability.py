"""Resolve which graph parameters the optimizer may update."""

from collections.abc import Callable, Iterable
from typing import Any, Literal

import equinox as eqx
import jax.tree as jt
from equinox.nn import StateIndex

from feedbax.component_registry.declarations import DeclaredComponent
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.runtime.graph import Graph

type TrainingMode = Literal["frozen", "parameters", "interior"]


def component_training_mode(
    node: ComponentSpec, declaration: DeclaredComponent | None, interior: object,
) -> TrainingMode:
    """Resolve the authored override before defaulting to the owner's declaration."""
    explicit = node.params.get("trainable") if declaration is not None else None
    if explicit is False:
        return "frozen"
    if isinstance(interior, GraphSpec):
        return "interior"
    if explicit is True or (declaration is not None and declaration.trainable_by_default):
        return "parameters"
    return "frozen"


def _graph_trainability[T](
    graph: Graph, root: T, components: Callable[[Graph, T], Iterable[tuple[str, TrainingMode, T]]],
) -> tuple[tuple[str, ...], Any]:
    trainable_paths: list[str] = []

    def graph_filter(runtime: Graph, source: T, path: tuple[str, ...]) -> Any:
        node_filters = dict.fromkeys(runtime.nodes, False)
        for node_id, mode, child_source in components(runtime, source):
            if mode == "frozen":
                continue
            component = runtime.nodes[node_id]
            node_path = (*path, node_id)
            if mode == "interior":
                if not isinstance(component, Graph):
                    raise TypeError(f"Expected a runtime graph at {'.'.join(node_path)!r}")
                node_filters[node_id] = graph_filter(component, child_source, node_path)
                continue
            leaves, structure = jt.flatten(component, is_leaf=lambda value: isinstance(value, StateIndex))
            mask = [eqx.is_inexact_array(leaf) for leaf in leaves]
            if any(mask):
                node_filters[node_id] = structure.unflatten(mask)
                trainable_paths.append(".".join(node_path))

        # Equinox accepts mask prefixes, so unselected node subtrees share one False.
        return jt.map(
            lambda value: node_filters if value is runtime.nodes else False,
            runtime,
            is_leaf=lambda value: value is runtime.nodes,
        )

    filter_spec = graph_filter(graph, root, ())
    return tuple(trainable_paths), filter_spec


def model_trainability(graph: Graph) -> tuple[tuple[str, ...], Any]:
    """Select trainable parameters of an already realized, registered graph."""
    def components(runtime: Graph, _source: None):
        for name, binding in runtime.component_bindings.items():
            yield name, binding.training_mode, None

    return _graph_trainability(graph, None, components)


def compile_trainability(graph_spec: GraphSpec, graph: Graph, registry: Any) -> tuple[tuple[str, ...], Any]:
    """Select trainable parameters against an explicit specification for a programmatic graph."""
    def components(runtime: Graph, spec: GraphSpec):
        subgraphs = spec.subgraphs or {}
        for name, node in spec.nodes.items():
            interior = subgraphs.get(name)
            binding = runtime.component_bindings.get(name)
            mode = (binding.training_mode if binding is not None else
                    component_training_mode(node, registry.get(node.type), interior))
            yield name, mode, interior

    return _graph_trainability(graph, graph_spec, components)
