from __future__ import annotations
from typing import Any

from feedbax.web.models.graph import GraphSpec


def graph_to_spec(graph: Any) -> GraphSpec:
    """Serialize a Graph-like object to GraphSpec.

    Note: feedbax.graph is not yet available in this codebase, so this function
    is a placeholder. Implement when Graph is introduced.
    """
    raise NotImplementedError('graph_to_spec requires feedbax.graph.Graph')


def spec_to_graph(spec: GraphSpec, component_registry: dict) -> Any:
    """Instantiate a Graph-like object from GraphSpec.

    Note: feedbax.graph is not yet available in this codebase, so this function
    is a placeholder. Implement when Graph is introduced.
    """
    raise NotImplementedError('spec_to_graph requires feedbax.graph.Graph')
