"""Test-only convenience for assertions against process-local graph objects."""

from typing import Any, Mapping

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.contracts.graph import GraphSpec
from feedbax.runtime.graph import Graph


def spec_to_graph(
    spec: GraphSpec,
    component_registry: Any,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
) -> Graph:
    return compile_graph(
        GraphDocument(graph=spec),
        component_registry,
        input_prototypes=input_prototypes,
    ).graph
