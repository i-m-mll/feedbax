"""Studio authoring normalization for persisted graph specs."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from feedbax.compiler.feedback_migration import lower_legacy_feedback
from feedbax.compiler.resolution import resolve_component_node
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.acausal_interface import normalize_acausal_interfaces_for_graph
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphProject,
    GraphSpec,
)


def normalize_graph_for_studio_authoring(
    graph: GraphSpec,
    *,
    component_registry: Any | None = None,
    component_versions: Literal["authoring", "historical"] | None = None,
) -> GraphSpec:
    """Apply the supplied registry's component resolution and dynamic-port policy."""

    graph = lower_legacy_feedback(normalize_acausal_interfaces_for_graph(graph), recurse=False)
    nodes: dict[str, ComponentSpec] = {}
    for node_id, node_spec in graph.nodes.items():
        next_type = node_spec.type
        if component_registry is not None and component_versions is not None:
            node_spec = resolve_component_node(
                node_spec,
                component_registry,
                historical_unversioned=component_versions == "historical",
            )
            next_type = node_spec.type
        params = dict(node_spec.params)
        input_ports = list(node_spec.input_ports)
        output_ports = list(node_spec.output_ports)
        if component_registry is not None:
            derive_layout = getattr(component_registry, "dynamic_port_layout", None)
            if callable(derive_layout):
                layout = derive_layout(next_type, params)
                if layout is not None:
                    if not input_ports:
                        input_ports = list(layout.input_ports)
                    if not output_ports:
                        output_ports = list(layout.output_ports)
        nodes[node_id] = node_spec.model_copy(
            update={
                "type": next_type,
                "params": params,
                "input_ports": input_ports,
                "output_ports": output_ports,
            }
        )

    subgraphs = None
    if graph.subgraphs:
        subgraphs = {}
        for node_id, subgraph in graph.subgraphs.items():
            if isinstance(subgraph, GraphSpec):
                subgraphs[node_id] = normalize_graph_for_studio_authoring(
                    subgraph,
                    component_registry=component_registry,
                    component_versions=component_versions,
                )
            elif component_registry is not None and component_versions is not None:
                subgraphs[node_id] = stamp_component_versions(
                    subgraph,
                    component_registry,
                    historical_unversioned=component_versions == "historical",
                )
            else:
                subgraphs[node_id] = subgraph
    return graph.model_copy(
        update={
            "nodes": nodes,
            "subgraphs": subgraphs,
        }
    )


_Graph = TypeVar("_Graph", GraphSpec, AcausalGraphSpec)


def stamp_component_versions(
    graph: _Graph,
    component_registry: Any,
    *,
    historical_unversioned: bool = False,
) -> _Graph:
    """Resolve persisted node identities without completing their parameters."""
    nodes = {
        name: resolve_component_node(
            node, component_registry, historical_unversioned=historical_unversioned
        )
        for name, node in graph.nodes.items()
    }
    subgraphs = (
        {
            name: stamp_component_versions(
                subgraph, component_registry, historical_unversioned=historical_unversioned
            )
            for name, subgraph in graph.subgraphs.items()
        }
        if graph.subgraphs is not None
        else None
    )
    return graph.model_copy(update={"nodes": nodes, "subgraphs": subgraphs})


def normalize_project_for_studio_authoring(
    project: GraphProject,
    *,
    component_registry: Any | None = None,
    component_versions: Literal["authoring", "historical"] | None = None,
) -> GraphProject:
    graph = normalize_graph_for_studio_authoring(
        project.graph,
        component_registry=component_registry,
        component_versions=component_versions,
    )
    if graph is project.graph:
        return project
    return project.model_copy(update={"graph": graph})
