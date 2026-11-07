"""Derived interface rules for acausal graph interiors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from feedbax._acausal_payload import (
    ACTIVATION_PAYLOAD as ACTIVATION_PAYLOAD,
    FORCE_PAYLOAD as FORCE_PAYLOAD,
    MUSCLE_EXCITATION_PAYLOAD as MUSCLE_EXCITATION_PAYLOAD,
    PRESCRIBED_MOTION_PAYLOAD as PRESCRIBED_MOTION_PAYLOAD,
    SourcePayload as SourcePayload,
    TORQUE_PAYLOAD as TORQUE_PAYLOAD,
)
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec, GraphSubgraphSpec


ACTUATION_INPUT_TYPE = "ActuationInput"
SENSOR_OUTPUT_TYPE = "SensorOutput"
BOUNDARY_PORT_TYPE = "BoundaryPort"
ACAUSAL_SYSTEM_TYPE = "AcausalSystem"


class AcausalInterfaceError(ValueError):
    """Raised when an acausal graph cannot expose a valid derived interface."""


ACTUATION_PAYLOADS = {
    "force": FORCE_PAYLOAD,
    "torque": TORQUE_PAYLOAD,
    "prescribed_motion": PRESCRIBED_MOTION_PAYLOAD,
    "muscle_excitation_vector": MUSCLE_EXCITATION_PAYLOAD,
}
SIGNAL_INPUT_PAYLOADS = {"RigidTendonHillMuscle": ACTIVATION_PAYLOAD, "feedbax.RigidTendonHillMuscle": ACTIVATION_PAYLOAD}


def is_acausal_type(type_id: str, builtin: str) -> bool:
    """Match a fixed built-in compiler role in authored or canonical input."""
    return type_id in (builtin, f"feedbax.{builtin}")


def actuation_source_payload(node: ComponentSpec) -> SourcePayload:
    """Resolve the payload of an authored ActuationInput."""
    kind = _param_string(node.params, "source_kind", "force")
    if kind not in ACTUATION_PAYLOADS:
        raise AcausalInterfaceError(f"Unknown ActuationInput source_kind {kind!r}")
    return ACTUATION_PAYLOADS[kind]


@dataclass(frozen=True)
class DerivedAcausalInterface:
    """Causal and conserving ports derived from an acausal graph interior."""

    input_payloads: dict[str, SourcePayload]
    output_ports: tuple[str, ...]
    boundary_ports: tuple[str, ...]
    boundary_port_types: PortTypeSpec

    @property
    def input_ports(self) -> tuple[str, ...]:
        return tuple(self.input_payloads)


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
) -> dict[str, str]:
    entries: list[tuple[int, str, str]] = []
    for node_id, node in graph.nodes.items():
        if not is_acausal_type(node.type, component_type):
            continue
        params = dict(node.params)
        port_name = _param_string(params, "port_name", default_port_name)
        entries.append((_param_order(params), port_name, node_id))
    entries.sort(key=lambda item: (item[0], item[1]))

    seen: dict[str, str] = {}
    for _order, port_name, node_id in entries:
        previous = seen.get(port_name)
        if previous is not None:
            raise AcausalInterfaceError(
                f"Duplicate {component_type} port_name {port_name!r} on nodes "
                f"{previous!r} and {node_id!r}"
            )
        seen[port_name] = node_id
    return seen


def derive_acausal_interface(graph: AcausalGraphSpec) -> DerivedAcausalInterface:
    """Derive parent-facing ports from explicit acausal boundary components."""

    adapter_inputs = _adapter_ports(
        graph,
        component_type=ACTUATION_INPUT_TYPE,
        default_port_name="u",
    )
    input_payloads = {
        port: actuation_source_payload(graph.nodes[node_id])
        for port, node_id in adapter_inputs.items()
    }
    for node_id, node in graph.nodes.items():
        if node.type in SIGNAL_INPUT_PAYLOADS:
            if node_id in input_payloads:
                raise AcausalInterfaceError(f"Duplicate acausal input port {node_id!r}")
            input_payloads[node_id] = SIGNAL_INPUT_PAYLOADS[node.type]
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
        input_payloads=input_payloads,
        output_ports=tuple(outputs),
        boundary_ports=tuple(boundary_ports),
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
            if node is not None and is_acausal_type(node.type, ACAUSAL_SYSTEM_TYPE):
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
