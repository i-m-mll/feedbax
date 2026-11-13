"""Scoped model occurrences shared by checkpoint and observation consumers."""

from collections.abc import Iterator

from feedbax.contracts.graph import ComponentSpec, GraphSpec


def graph_leaf_specs(
    graph: GraphSpec, scope: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], ComponentSpec]]:
    for name, node in graph.nodes.items():
        path = (*scope, name)
        interior = (graph.subgraphs or {}).get(name)
        if isinstance(interior, GraphSpec):
            yield from graph_leaf_specs(interior, path)
        else:
            yield path, node
