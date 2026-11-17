"""Scoped model occurrences shared by checkpoint and observation consumers."""

from collections.abc import Iterator

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec


def graph_leaf_specs(
    graph: GraphSpec, scope: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], ComponentSpec]]:
    for path, node, _ in graph_leaf_occurrences(graph, scope):
        yield path, node


def graph_leaf_occurrences(
    graph: GraphSpec, scope: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], ComponentSpec, AcausalGraphSpec | None]]:
    for name, node in graph.nodes.items():
        path = (*scope, name)
        interior = (graph.subgraphs or {}).get(name)
        if isinstance(interior, GraphSpec):
            yield from graph_leaf_occurrences(interior, path)
        else:
            yield path, node, interior


def graph_output_sources(graph: GraphSpec) -> Iterator[tuple[tuple[str, ...], str]]:
    """Follow the declared output bindings to their producing leaves."""
    for port in graph.output_bindings:
        scope: tuple[str, ...] = ()
        interior = graph
        while True:
            node, port = interior.output_bindings[port]
            scope = (*scope, node)
            child = (interior.subgraphs or {}).get(node)
            if not isinstance(child, GraphSpec):
                yield scope, port
                break
            interior = child
