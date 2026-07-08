"""Compile durable acausal graph specs into runtime ``AcausalSystem`` objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Type

import diffrax as dfx
import optimistix as optx

from feedbax.acausal.base import AcausalConnection, AcausalElement
from feedbax.acausal.rotational import (
    AngleSensor,
    AngularVelocitySensor,
    GearRatio,
    Inertia,
    RotationalDamper,
    RotationalGround,
    TorqueSensor,
    TorqueSource,
    TorsionalSpring,
)
from feedbax.acausal.system import AcausalSystem
from feedbax.acausal.translational import (
    ForceSensor,
    ForceSource,
    Ground,
    LinearDamper,
    LinearSpring,
    Mass,
    PositionSensor,
    PrescribedMotion,
    VelocitySensor,
)
from feedbax.contracts.acausal import (
    ACAUSAL_GRAPH_SCHEMA_ID,
    AcausalGraphSpec,
    RootFinderSpec,
    SolverConfigSpec,
)
from feedbax.contracts.acausal_interface import (
    ACTUATION_INPUT_TYPE,
    BOUNDARY_PORT_TYPE,
    SENSOR_OUTPUT_TYPE,
    ACAUSAL_SYSTEM_TYPE,
    derive_acausal_interface,
)
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ComponentSpec


_ELEMENT_BUILDERS: dict[str, Type[AcausalElement]] = {
    cls.__name__: cls
    for cls in (
        Mass,
        LinearSpring,
        LinearDamper,
        Ground,
        ForceSource,
        PrescribedMotion,
        PositionSensor,
        VelocitySensor,
        ForceSensor,
        Inertia,
        TorsionalSpring,
        RotationalDamper,
        RotationalGround,
        TorqueSource,
        GearRatio,
        AngleSensor,
        AngularVelocitySensor,
        TorqueSensor,
    )
}

_SOURCE_KIND_BUILDERS: dict[str, Type[AcausalElement]] = {
    "force": ForceSource,
    "torque": TorqueSource,
    "prescribed_motion": PrescribedMotion,
}

_SENSOR_QUANTITY_BUILDERS: dict[str, Type[AcausalElement]] = {
    "position": PositionSensor,
    "velocity": VelocitySensor,
    "force": ForceSensor,
    "angle": AngleSensor,
    "angular_velocity": AngularVelocitySensor,
    "torque": TorqueSensor,
}

_SOLVER_TYPES: dict[str, Type[dfx.AbstractSolver]] = {
    "euler": dfx.Euler,
    "implicit_euler": dfx.ImplicitEuler,
    "kvaerno5": dfx.Kvaerno5,
    "tsit5": dfx.Tsit5,
}


Endpoint = tuple[str, str]


@dataclass
class _FlattenedAcausalGraph:
    elements: dict[str, AcausalElement] = field(default_factory=dict)
    connections: list[AcausalConnection] = field(default_factory=list)
    boundary_ports: dict[str, list[Endpoint]] = field(default_factory=dict)
    input_bindings: dict[str, str] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)


def compile_acausal_graph(
    interior_spec: AcausalGraphSpec,
    node_name: str,
    component_registry: Any,
) -> AcausalSystem:
    """Compile an ``AcausalGraphSpec`` into an executable ``AcausalSystem``."""

    if not isinstance(interior_spec, AcausalGraphSpec):
        raise ValueError(
            f"Acausal compiler expected {AcausalGraphSpec.__name__} for node "
            f"{node_name!r}, got {type(interior_spec).__name__}"
        )
    if interior_spec.schema_id != ACAUSAL_GRAPH_SCHEMA_ID:
        raise ValueError(
            f"Acausal compiler expected schema_id {ACAUSAL_GRAPH_SCHEMA_ID!r} "
            f"for node {node_name!r}, got {interior_spec.schema_id!r}"
        )

    interface = derive_acausal_interface(interior_spec)
    flattened = _flatten_acausal_graph(
        interior_spec,
        component_registry=component_registry,
        path=node_name,
    )
    solver_type, root_finder = _solver_runtime(interior_spec.solver)
    system = AcausalSystem(
        elements=flattened.elements,
        connections=flattened.connections,
        dt=interior_spec.solver.dt,
        solver_type=solver_type,
        root_finder=root_finder,
        input_bindings=flattened.input_bindings,
        output_bindings=flattened.output_bindings,
    )

    if system.input_ports != interface.input_ports:
        raise ValueError(
            f"Acausal compiler internal error for node {node_name!r}: compiled "
            f"input ports {system.input_ports!r} do not match derived interface "
            f"{interface.input_ports!r}"
        )
    if system.output_ports != interface.output_ports:
        raise ValueError(
            f"Acausal compiler internal error for node {node_name!r}: compiled "
            f"output ports {system.output_ports!r} do not match derived interface "
            f"{interface.output_ports!r}"
        )
    return system


def _flatten_acausal_graph(
    graph: AcausalGraphSpec,
    *,
    component_registry: Any,
    path: str,
    namespace: str = "",
) -> _FlattenedAcausalGraph:
    flattened = _FlattenedAcausalGraph()
    boundary_nodes: dict[str, str] = {}
    composite_boundaries: dict[str, dict[str, list[Endpoint]]] = {}

    for node_id, node_spec in graph.nodes.items():
        node_type = node_spec.type
        node_path = f"{path}.{node_id}" if path else node_id
        if node_type == BOUNDARY_PORT_TYPE:
            boundary_nodes[node_id] = _string_param(
                node_spec,
                "port_name",
                default="port",
                node_path=node_path,
            )
            continue

        if node_type == ACAUSAL_SYSTEM_TYPE:
            subgraph = (graph.subgraphs or {}).get(node_id)
            if subgraph is None:
                raise ValueError(
                    f"Acausal composite node {node_path!r} requires an acausal "
                    "interior subgraph."
                )
            child = _flatten_acausal_graph(
                subgraph,
                component_registry=component_registry,
                path=node_path,
                namespace=f"{namespace}{node_id}.",
            )
            _merge_unique(flattened.elements, child.elements, "acausal element")
            flattened.connections.extend(child.connections)
            flattened.input_bindings.update(child.input_bindings)
            flattened.output_bindings.update(child.output_bindings)
            composite_boundaries[node_id] = child.boundary_ports
            continue

        element = _instantiate_node(
            node_id=f"{namespace}{node_id}",
            node_spec=node_spec,
            component_registry=component_registry,
            node_path=node_path,
            flattened=flattened,
        )
        if element is not None:
            flattened.elements[element.name] = element

    for connection in graph.connections:
        a_node, a_port = connection.a
        b_node, b_port = connection.b
        a_boundary = boundary_nodes.get(a_node)
        b_boundary = boundary_nodes.get(b_node)

        if a_boundary is not None and b_boundary is not None:
            raise ValueError(
                f"Acausal connection in {path!r} connects two BoundaryPort nodes: "
                f"{a_node!r} and {b_node!r}"
            )

        if a_boundary is not None:
            flattened.boundary_ports.setdefault(a_boundary, []).extend(
                _resolve_endpoint(
                    b_node,
                    b_port,
                    namespace=namespace,
                    composite_boundaries=composite_boundaries,
                    graph=graph,
                    path=path,
                )
            )
            continue
        if b_boundary is not None:
            flattened.boundary_ports.setdefault(b_boundary, []).extend(
                _resolve_endpoint(
                    a_node,
                    a_port,
                    namespace=namespace,
                    composite_boundaries=composite_boundaries,
                    graph=graph,
                    path=path,
                )
            )
            continue

        left = _resolve_endpoint(
            a_node,
            a_port,
            namespace=namespace,
            composite_boundaries=composite_boundaries,
            graph=graph,
            path=path,
        )
        right = _resolve_endpoint(
            b_node,
            b_port,
            namespace=namespace,
            composite_boundaries=composite_boundaries,
            graph=graph,
            path=path,
        )
        for left_endpoint in left:
            for right_endpoint in right:
                flattened.connections.append(AcausalConnection(left_endpoint, right_endpoint))

    for port_name, endpoints in list(flattened.boundary_ports.items()):
        flattened.boundary_ports[port_name] = _dedupe_endpoints(endpoints)
    return flattened


def _instantiate_node(
    *,
    node_id: str,
    node_spec: ComponentSpec,
    component_registry: Any,
    node_path: str,
    flattened: _FlattenedAcausalGraph,
) -> AcausalElement | None:
    node_type = node_spec.type
    params = dict(node_spec.params)
    _validate_acausal_component_meta(node_type, component_registry, node_path)

    if node_type == ACTUATION_INPUT_TYPE:
        source_kind = str(params.get("source_kind", "force"))
        builder = _SOURCE_KIND_BUILDERS.get(source_kind)
        if builder is None:
            raise ValueError(
                f"Unknown ActuationInput source_kind {source_kind!r} at {node_path!r}"
            )
        port_name = _string_param(node_spec, "port_name", default="u", node_path=node_path)
        flattened.input_bindings[port_name] = node_id
        return builder(node_id)

    if node_type == SENSOR_OUTPUT_TYPE:
        quantity = str(params.get("quantity", "position"))
        builder = _SENSOR_QUANTITY_BUILDERS.get(quantity)
        if builder is None:
            raise ValueError(f"Unknown SensorOutput quantity {quantity!r} at {node_path!r}")
        port_name = _string_param(node_spec, "port_name", default="y", node_path=node_path)
        flattened.output_bindings[port_name] = node_id
        return builder(node_id)

    builder = _ELEMENT_BUILDERS.get(node_type)
    if builder is None:
        raise ValueError(
            f"Unsupported acausal component type {node_type!r} at {node_path!r}"
        )
    return builder(node_id, **params)


def _validate_acausal_component_meta(
    node_type: str,
    component_registry: Any,
    node_path: str,
) -> None:
    get_meta = getattr(component_registry, "get", None)
    meta = get_meta(node_type) if callable(get_meta) else None
    if meta is None:
        raise ValueError(
            f"Unsupported acausal component type {node_type!r} at {node_path!r}: "
            "not present in the component registry"
        )
    if getattr(meta, "domain", None) != ACAUSAL_DOMAIN_ID:
        raise ValueError(
            f"Component type {node_type!r} at {node_path!r} belongs to domain "
            f"{getattr(meta, 'domain', None)!r}, not {ACAUSAL_DOMAIN_ID!r}"
        )


def _resolve_endpoint(
    node_id: str,
    port_name: str,
    *,
    namespace: str,
    composite_boundaries: Mapping[str, Mapping[str, list[Endpoint]]],
    graph: AcausalGraphSpec,
    path: str,
) -> list[Endpoint]:
    if node_id in composite_boundaries:
        endpoints = composite_boundaries[node_id].get(port_name)
        if not endpoints:
            raise ValueError(
                f"Acausal composite endpoint {path}.{node_id}.{port_name} has no "
                "matching BoundaryPort in its interior."
            )
        return list(endpoints)
    if node_id not in graph.nodes:
        raise ValueError(f"Acausal connection in {path!r} references unknown node {node_id!r}")
    return [(f"{namespace}{node_id}", port_name)]


def _solver_runtime(
    solver: SolverConfigSpec,
) -> tuple[Type[dfx.AbstractSolver], optx.AbstractRootFinder | None]:
    solver_type = _SOLVER_TYPES[solver.solver_type]
    if not issubclass(solver_type, dfx.AbstractImplicitSolver):
        return solver_type, None
    root_spec = solver.root_finder or RootFinderSpec()
    if root_spec.method != "newton":
        raise ValueError(f"Unsupported root finder method {root_spec.method!r}")
    return (
        solver_type,
        optx.Newton(
            rtol=root_spec.rtol,
            atol=root_spec.atol,
        ),
    )


def _string_param(
    node_spec: ComponentSpec,
    key: str,
    *,
    default: str,
    node_path: str,
) -> str:
    value = dict(node_spec.params).get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{node_path!r} parameter {key!r} must be a non-empty string")
    return value


def _merge_unique(target: dict[str, Any], source: Mapping[str, Any], noun: str) -> None:
    overlap = set(target) & set(source)
    if overlap:
        raise ValueError(f"Duplicate flattened {noun} names: {sorted(overlap)!r}")
    target.update(source)


def _dedupe_endpoints(endpoints: list[Endpoint]) -> list[Endpoint]:
    seen: set[Endpoint] = set()
    deduped: list[Endpoint] = []
    for endpoint in endpoints:
        if endpoint in seen:
            continue
        seen.add(endpoint)
        deduped.append(endpoint)
    return deduped
