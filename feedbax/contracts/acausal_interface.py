"""Derived interface rules for acausal graph interiors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec, GraphSubgraphSpec


ACTUATION_INPUT_TYPE = "ActuationInput"
SENSOR_OUTPUT_TYPE = "SensorOutput"
BOUNDARY_PORT_TYPE = "BoundaryPort"
ACAUSAL_SYSTEM_TYPE = "AcausalSystem"


class AcausalInterfaceError(ValueError):
    """Raised when an acausal graph cannot expose a valid derived interface."""


@dataclass(frozen=True)
class DerivedAcausalInterface:
    """Causal and conserving ports derived from an acausal graph interior."""

    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    boundary_ports: tuple[str, ...]
    boundary_port_types: PortTypeSpec


def signal_port_type() -> PortType:
    """Return the static type for a causal signal boundary port."""

    return PortType(dtype="scalar", kind="signal")


def conserving_port_type(
    *,
    physical_domain: str | None = None,
    across_vars: Iterable[str] | None = None,
    through_var: str | None = None,
) -> PortType:
    """Return a conserving port type declaration."""

    return PortType(
        dtype="scalar",
        kind="conserving",
        physical_domain=physical_domain,
        across_vars=list(across_vars) if across_vars is not None else None,
        through_var=through_var,
    )


def _param_string(params: dict[str, Any], key: str, default: str) -> str:
    value = params.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise AcausalInterfaceError(f"{key} must be a non-empty string")
    return value


def _param_order(params: dict[str, Any]) -> int:
    value = params.get("order", 0)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AcausalInterfaceError("order must be an integer") from exc


def _adapter_ports(
    graph: AcausalGraphSpec,
    *,
    component_type: Literal["ActuationInput", "SensorOutput", "BoundaryPort"],
    default_port_name: str,
) -> tuple[str, ...]:
    entries: list[tuple[int, str, str]] = []
    for node_id, node in graph.nodes.items():
        if node.type != component_type:
            continue
        params = dict(node.params)
        port_name = _param_string(params, "port_name", default_port_name)
        entries.append((_param_order(params), port_name, node_id))
    entries.sort(key=lambda item: (item[0], item[1]))

    seen: dict[str, str] = {}
    ordered: list[str] = []
    for _order, port_name, node_id in entries:
        previous = seen.get(port_name)
        if previous is not None:
            raise AcausalInterfaceError(
                f"Duplicate {component_type} port_name {port_name!r} on nodes "
                f"{previous!r} and {node_id!r}"
            )
        seen[port_name] = node_id
        ordered.append(port_name)
    return tuple(ordered)


def derive_acausal_interface(graph: AcausalGraphSpec) -> DerivedAcausalInterface:
    """Derive parent-facing ports from explicit acausal boundary components."""

    inputs = _adapter_ports(
        graph,
        component_type=ACTUATION_INPUT_TYPE,
        default_port_name="u",
    )
    outputs = _adapter_ports(
        graph,
        component_type=SENSOR_OUTPUT_TYPE,
        default_port_name="y",
    )
    boundary_ports = _adapter_ports(
        graph,
        component_type=BOUNDARY_PORT_TYPE,
        default_port_name="port",
    )
    boundary_port_types = PortTypeSpec(
        inputs={name: conserving_port_type() for name in boundary_ports},
        outputs={},
    )
    return DerivedAcausalInterface(
        input_ports=inputs,
        output_ports=outputs,
        boundary_ports=boundary_ports,
        boundary_port_types=boundary_port_types,
    )


def normalize_acausal_graph_interfaces(graph: AcausalGraphSpec) -> AcausalGraphSpec:
    """Normalize nested acausal composite nodes from their BoundaryPort markers."""

    if not graph.subgraphs:
        return graph

    normalized_subgraphs = {
        node_id: normalize_acausal_graph_interfaces(subgraph)
        for node_id, subgraph in graph.subgraphs.items()
    }
    nodes: dict[str, ComponentSpec] = dict(graph.nodes)
    for node_id, subgraph in normalized_subgraphs.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        boundary = derive_acausal_interface(subgraph)
        nodes[node_id] = node.model_copy(
            update={
                "input_ports": list(boundary.boundary_ports),
                "output_ports": [],
            }
        )
    return graph.model_copy(update={"nodes": nodes, "subgraphs": normalized_subgraphs})


def normalize_acausal_interfaces_for_graph(graph: GraphSpec) -> GraphSpec:
    """Recompute all AcausalSystem outer ports from their acausal interiors."""

    subgraphs = dict(graph.subgraphs or {})
    if not subgraphs:
        return graph

    nodes: dict[str, ComponentSpec] = dict(graph.nodes)
    normalized_subgraphs: dict[str, GraphSubgraphSpec] = {}
    changed = False

    for node_id, subgraph in subgraphs.items():
        if isinstance(subgraph, AcausalGraphSpec):
            normalized = normalize_acausal_graph_interfaces(subgraph)
            normalized_subgraphs[node_id] = normalized
            node = nodes.get(node_id)
            if node is not None and node.type == ACAUSAL_SYSTEM_TYPE:
                interface = derive_acausal_interface(normalized)
                nodes[node_id] = node.model_copy(
                    update={
                        "input_ports": list(interface.input_ports),
                        "output_ports": list(interface.output_ports),
                    }
                )
                changed = True
            if normalized is not subgraph:
                changed = True
        elif isinstance(subgraph, GraphSpec):
            normalized = normalize_acausal_interfaces_for_graph(subgraph)
            normalized_subgraphs[node_id] = normalized
            if normalized is not subgraph:
                changed = True
        else:
            normalized_subgraphs[node_id] = subgraph

    return (
        graph.model_copy(update={"nodes": nodes, "subgraphs": normalized_subgraphs or None})
        if changed
        else graph
    )
