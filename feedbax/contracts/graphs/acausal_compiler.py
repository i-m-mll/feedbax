"""Compile durable acausal graph specs into runtime ``AcausalSystem`` objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Type

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import optimistix as optx

from feedbax.acausal.base import AcausalConnection, AcausalElement
from feedbax.acausal.analysis import StructuralAnalysis, analyze_system
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
from feedbax.acausal.mechanics import RigidTendonHillMuscle
from feedbax.acausal.multibody import (
    Anchor,
    MusclePath,
    PlanarLink,
    PointMarker,
    RevoluteJoint,
    WorldFrame,
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
    acausal_interior_content_hash,
)
from feedbax.contracts.acausal_interface import (
    ACTUATION_INPUT_TYPE,
    BOUNDARY_PORT_TYPE,
    SENSOR_OUTPUT_TYPE,
    ACAUSAL_SYSTEM_TYPE,
    derive_acausal_interface,
    signal_port_type,
)
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID, MECHANICS_DOMAIN_ID
from feedbax.contracts.domain import (
    DomainCompileReport,
    DomainDiagnostic,
    report_status_for_diagnostics,
)
from feedbax.contracts.graph import ComponentSpec
from feedbax.contracts.graphs.builders import build_component
from feedbax.mechanics.skeleton.arm import TwoLinkArm


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
        RigidTendonHillMuscle,
        PlanarLink,
        RevoluteJoint,
        WorldFrame,
        Anchor,
        MusclePath,
        PointMarker,
    )
}
_ELEMENT_BUILDERS.update(
    {
        "MechanicsMass": Mass,
        "MechanicsLinearSpring": LinearSpring,
        "MechanicsLinearDamper": LinearDamper,
        "MechanicsGround": Ground,
        "MechanicsForceSource": ForceSource,
        "MechanicsPositionSensor": PositionSensor,
        "MechanicsVelocitySensor": VelocitySensor,
        "MechanicsForceSensor": ForceSensor,
    }
)

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
    if interior_spec.physical_domain == "planar_multibody":
        return _compile_planar_multibody_graph(
            interior_spec,
            node_name=node_name,
            component_registry=component_registry,
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
        input_bindings=flattened.input_bindings or None,
        output_bindings=flattened.output_bindings or None,
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


def compile_acausal_authoring_report(
    interior_spec: AcausalGraphSpec,
    *,
    node_path: list[str],
    component_registry: Any,
) -> DomainCompileReport:
    """Return a structural authoring compile report without constructing JAX closures."""

    content_hash = acausal_interior_content_hash(interior_spec)
    diagnostics = _diagnose_acausal_spec(
        interior_spec,
        component_registry=component_registry,
        path=".".join(node_path) if node_path else "<root>",
    )
    derived_interface = None
    analysis: StructuralAnalysis | None = None
    flattened = _FlattenedAcausalGraph()

    if interior_spec.physical_domain == "planar_multibody":
        try:
            interface = derive_acausal_interface(interior_spec)
            derived_interface = {
                "inputs": {
                    name: signal_port_type().model_dump(mode="json")
                    for name in interface.input_ports
                },
                "outputs": {
                    name: signal_port_type().model_dump(mode="json")
                    for name in interface.output_ports
                },
            }
        except Exception as exc:  # noqa: BLE001 - authoring endpoint reports, not raises.
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.internal",
                    message=f"Internal acausal interface failure for {node_path!r}: {exc}",
                    node_ids=list(node_path),
                )
            )
    elif not any(diagnostic.severity == "error" for diagnostic in diagnostics):
        try:
            interface = derive_acausal_interface(interior_spec)
            derived_interface = {
                "inputs": {
                    name: signal_port_type().model_dump(mode="json")
                    for name in interface.input_ports
                },
                "outputs": {
                    name: signal_port_type().model_dump(mode="json")
                    for name in interface.output_ports
                },
            }
            flattened = _flatten_acausal_graph(
                interior_spec,
                component_registry=component_registry,
                path=".".join(node_path) if node_path else "<root>",
            )
            analysis = analyze_system(flattened.elements, flattened.connections)
        except Exception as exc:  # noqa: BLE001 - authoring endpoint reports, not raises.
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.internal",
                    message=f"Internal acausal compile failure for {node_path!r}: {exc}",
                    node_ids=list(node_path),
                )
            )

    summary = (
        _compile_planar_multibody_summary(interior_spec)
        if interior_spec.physical_domain == "planar_multibody"
        else _compile_summary(interior_spec, flattened, analysis)
    )
    return DomainCompileReport(
        status=report_status_for_diagnostics(diagnostics),
        interior_content_hash=content_hash,
        diagnostics=diagnostics,
        derived_interface=derived_interface,
        summary=summary,
    )


def _diagnose_acausal_spec(
    graph: AcausalGraphSpec,
    *,
    component_registry: Any,
    path: str,
) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    if not graph.nodes:
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="acausal.empty_interior",
                message=f"Acausal interior {path!r} has no elements.",
                counts={"n_elements": 0},
            )
        )
        return diagnostics

    diagnostics.extend(_diagnose_component_types(graph, component_registry, path))
    diagnostics.extend(_diagnose_boundary_ports(graph, path))
    diagnostics.extend(_diagnose_adapter_units(graph, path))

    for node_id, subgraph in sorted((graph.subgraphs or {}).items()):
        diagnostics.extend(
            _diagnose_acausal_spec(
                subgraph,
                component_registry=component_registry,
                path=f"{path}.{node_id}",
            )
        )
    if graph.physical_domain == "planar_multibody":
        diagnostics.extend(_diagnose_planar_multibody_graph(graph, path))
        return diagnostics
    if not any(diagnostic.severity == "error" for diagnostic in diagnostics):
        try:
            flattened = _flatten_acausal_graph(
                graph,
                component_registry=component_registry,
                path=path,
            )
            diagnostics.extend(_diagnose_flattened_domains(graph, flattened, path))
            if not any(diagnostic.severity == "error" for diagnostic in diagnostics):
                diagnostics.extend(
                    _diagnose_structural_analysis(
                        analyze_system(
                            flattened.elements,
                            flattened.connections,
                        )
                    )
                )
        except Exception as exc:  # noqa: BLE001 - converted to structured report.
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.internal",
                    message=f"Internal acausal structural analysis failure for {path!r}: {exc}",
                )
            )
    return diagnostics


def _diagnose_component_types(
    graph: AcausalGraphSpec,
    component_registry: Any,
    path: str,
) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    get_meta = getattr(component_registry, "get", None)
    for node_id, node_spec in graph.nodes.items():
        if node_spec.type in {ACTUATION_INPUT_TYPE, SENSOR_OUTPUT_TYPE, BOUNDARY_PORT_TYPE}:
            continue
        if node_spec.type == ACAUSAL_SYSTEM_TYPE:
            continue
        meta = get_meta(node_spec.type) if callable(get_meta) else None
        if meta is not None and getattr(meta, "domain", None) not in {
            ACAUSAL_DOMAIN_ID,
            MECHANICS_DOMAIN_ID,
        }:
            meta_domain = getattr(meta, "domain", None)
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.domain_mismatch",
                    message=(
                        f"Acausal node {path}.{node_id!s} has component domain "
                        f"{meta_domain!r}, expected {ACAUSAL_DOMAIN_ID!r} or "
                        f"{MECHANICS_DOMAIN_ID!r}."
                    ),
                    node_ids=[node_id],
                )
            )
            continue
        builder_known = node_spec.type in _ELEMENT_BUILDERS
        if not builder_known or meta is None:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.unknown_element_type",
                    message=(
                        f"Acausal node {path}.{node_id!s} uses unknown element type "
                        f"{node_spec.type!r}."
                    ),
                    node_ids=[node_id],
                    counts={"n_elements": len(graph.nodes)},
                )
            )
            continue
    return diagnostics


def _diagnose_flattened_domains(
    graph: AcausalGraphSpec,
    flattened: _FlattenedAcausalGraph,
    path: str,
) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    for element_name, element in flattened.elements.items():
        for port_name, port in element.ports.items():
            if port.domain.value != graph.physical_domain:
                diagnostics.append(
                    DomainDiagnostic(
                        severity="error",
                        code="acausal.domain_mismatch",
                        message=(
                            f"Acausal element {path}.{element_name} port {port_name!r} "
                            f"has physical domain {port.domain.value!r}, expected "
                            f"{graph.physical_domain!r}."
                        ),
                        node_ids=[element_name],
                        ports=[(element_name, port_name)],
                    )
                )
    return diagnostics


def _diagnose_boundary_ports(graph: AcausalGraphSpec, path: str) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    seen: dict[str, str] = {}
    connected_boundary_nodes = {
        node for connection in graph.connections for node in (connection.a[0], connection.b[0])
    }
    for node_id, node_spec in graph.nodes.items():
        if node_spec.type != BOUNDARY_PORT_TYPE:
            continue
        port_name = _string_param(node_spec, "port_name", default="port", node_path=node_id)
        if port_name in seen:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.duplicate_boundary_port_name",
                    message=(
                        f"Boundary port name {port_name!r} is used by both "
                        f"{seen[port_name]!r} and {node_id!r} in {path!r}."
                    ),
                    node_ids=[seen[port_name], node_id],
                    ports=[(seen[port_name], port_name), (node_id, port_name)],
                )
            )
        seen[port_name] = node_id
        if node_id not in connected_boundary_nodes:
            diagnostics.append(
                DomainDiagnostic(
                    severity="warning",
                    code="acausal.dangling_conserving_port",
                    message=(
                        f"Boundary port {port_name!r} on node {node_id!r} in {path!r} "
                        "is not connected to any conserving endpoint."
                    ),
                    node_ids=[node_id],
                    ports=[(node_id, port_name)],
                )
            )
    return diagnostics


def _diagnose_adapter_units(graph: AcausalGraphSpec, path: str) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    allowed_sources = {
        "translational": {"force", "prescribed_motion"},
        "rotational": {"torque", "prescribed_motion"},
        "planar_multibody": {"muscle_excitation_vector"},
    }
    allowed_sensors = {
        "translational": {"position", "velocity", "force"},
        "rotational": {"angle", "angular_velocity", "torque"},
        "planar_multibody": {"effector", "state", "dof", "muscle_length", "muscle_velocity"},
    }
    for node_id, node_spec in graph.nodes.items():
        params = dict(node_spec.params)
        if node_spec.type == ACTUATION_INPUT_TYPE:
            source_kind = str(params.get("source_kind", "force"))
            if source_kind not in allowed_sources[graph.physical_domain]:
                diagnostics.append(
                    DomainDiagnostic(
                        severity="warning",
                        code="acausal.adapter_unit_mismatch",
                        message=(
                            f"ActuationInput {path}.{node_id} source_kind {source_kind!r} "
                            f"does not match physical domain {graph.physical_domain!r}."
                        ),
                        node_ids=[node_id],
                    )
                )
        elif node_spec.type == SENSOR_OUTPUT_TYPE:
            quantity = str(params.get("quantity", "position"))
            if quantity not in allowed_sensors[graph.physical_domain]:
                diagnostics.append(
                    DomainDiagnostic(
                        severity="warning",
                        code="acausal.adapter_unit_mismatch",
                        message=(
                            f"SensorOutput {path}.{node_id} quantity {quantity!r} "
                            f"does not match physical domain {graph.physical_domain!r}."
                        ),
                        node_ids=[node_id],
                    )
                )
    return diagnostics


def _compile_planar_multibody_graph(
    graph: AcausalGraphSpec,
    *,
    node_name: str,
    component_registry: Any,
) -> Any:
    diagnostics = [
        diagnostic
        for diagnostic in _diagnose_acausal_spec(
            graph,
            component_registry=component_registry,
            path=node_name,
        )
        if diagnostic.severity == "error"
    ]
    if diagnostics:
        messages = "; ".join(diagnostic.message for diagnostic in diagnostics)
        raise ValueError(f"Planar multibody graph {node_name!r} is not compilable: {messages}")

    values = _analytical_planar_values(graph, node_name)
    analytical_meta = component_registry.get("AnalyticalMusculoskeletalPlant")
    if analytical_meta is None:
        raise ValueError("AnalyticalMusculoskeletalPlant is missing from the component registry.")
    builder_params = dict(analytical_meta.default_params)
    builder_params["dt"] = graph.solver.dt
    mechanics = build_component(
        node_name,
        "AnalyticalMusculoskeletalPlant",
        builder_params,
        component_registry=component_registry,
    )
    skeleton = TwoLinkArm(
        l=values.segment_lengths,
        m=values.segment_masses,
        I=values.segment_inertias,
        s=values.segment_centers,
        B=jnp.diag(jnp.asarray(values.joint_damping)),
    )
    plant = eqx.tree_at(
        lambda value: (
            value.skeleton,
            value.segment_lengths,
            value.moment_arms,
            value.muscle_gear,
            value.mt_reference_length,
        ),
        mechanics.plant,
        (
            skeleton,
            jnp.asarray(values.segment_lengths),
            jnp.asarray(values.moment_arms),
            jnp.asarray(values.maximum_isometric_forces),
            jnp.asarray(values.reference_lengths),
        ),
    )
    return eqx.tree_at(lambda value: value.plant, mechanics, plant)


@dataclass(frozen=True)
class _AnalyticalPlanarValues:
    segment_lengths: tuple[float, float]
    segment_masses: tuple[float, float]
    segment_centers: tuple[float, float]
    segment_inertias: tuple[float, float]
    joint_damping: tuple[float, float]
    moment_arms: tuple[tuple[float, float], ...]
    reference_lengths: tuple[float, ...]
    maximum_isometric_forces: tuple[float, ...]


def _analytical_planar_values(
    graph: AcausalGraphSpec,
    node_name: str,
) -> _AnalyticalPlanarValues:
    if graph.solver.solver_type != "euler" or graph.solver.root_finder is not None:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} requests solver "
            f"{graph.solver.solver_type!r} or a root finder, but the canonical "
            "AnalyticalMusculoskeletalPlant builder supports Euler without a root finder."
        )
    if graph.subgraphs:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} cannot compile nested subgraphs "
            "into the analytical two-link plant."
        )

    by_type: dict[str, list[tuple[str, ComponentSpec]]] = {}
    for node_id, node in graph.nodes.items():
        by_type.setdefault(node.type, []).append((node_id, node))
    supported_types = {
        "ActuationInput",
        "Anchor",
        "MusclePath",
        "PlanarLink",
        "PointMarker",
        "RevoluteJoint",
        "SensorOutput",
        "WorldFrame",
    }
    unsupported_types = sorted(set(by_type) - supported_types)
    if unsupported_types:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} contains unsupported node types "
            f"{unsupported_types!r}."
        )

    worlds = by_type.get("WorldFrame", [])
    anchors = by_type.get("Anchor", [])
    links = dict(by_type.get("PlanarLink", []))
    joints = dict(by_type.get("RevoluteJoint", []))
    if (len(worlds), len(anchors), len(links), len(joints)) != (1, 1, 2, 2):
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} must contain exactly one WorldFrame, "
            "one Anchor, two PlanarLinks, and two RevoluteJoints; got "
            f"{len(worlds)}, {len(anchors)}, {len(links)}, and {len(joints)}."
        )
    world_id, world = worlds[0]
    _require_supported_params(world_id, world, set())

    root_joints = [
        (joint_id, joint)
        for joint_id, joint in joints.items()
        if _param(joint, "parent_frame", "") == f"{world_id}.frame"
    ]
    if len(root_joints) != 1:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} must have one root joint attached "
            f"to {world_id + '.frame'!r}."
        )
    root_joint_id, root_joint = root_joints[0]
    root_link_id = _proximal_link_id(
        root_joint_id,
        _param(root_joint, "child_frame", ""),
        links,
    )

    child_joints = [
        (joint_id, joint)
        for joint_id, joint in joints.items()
        if joint_id != root_joint_id
        and _param(joint, "parent_frame", "") == f"{root_link_id}.distal"
    ]
    if len(child_joints) != 1:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} is not the supported rooted serial "
            f"two-link chain after {root_link_id + '.distal'!r}."
        )
    child_joint_id, child_joint = child_joints[0]
    child_link_id = _proximal_link_id(
        child_joint_id,
        _param(child_joint, "child_frame", ""),
        links,
    )
    if child_link_id == root_link_id:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} attaches both joints to "
            f"PlanarLink {root_link_id!r}."
        )

    anchor_id, anchor = anchors[0]
    _require_supported_params(anchor_id, anchor, {"frame", "world_frame"})
    if (
        _param(anchor, "world_frame", "world.frame") != f"{world_id}.frame"
        or _param(anchor, "frame", "") != f"{root_link_id}.proximal"
    ):
        raise NotImplementedError(
            f"Anchor {anchor_id!r} must attach {world_id + '.frame'!r} to "
            f"{root_link_id + '.proximal'!r}."
        )
    _require_serial_connections(
        graph,
        node_name=node_name,
        world_id=world_id,
        anchor_id=anchor_id,
        root_joint_id=root_joint_id,
        root_link_id=root_link_id,
        child_joint_id=child_joint_id,
        child_link_id=child_link_id,
    )

    ordered_links = ((root_link_id, links[root_link_id]), (child_link_id, links[child_link_id]))
    lengths: list[float] = []
    masses: list[float] = []
    centers: list[float] = []
    inertias: list[float] = []
    for link_id, link in ordered_links:
        _require_supported_params(link_id, link, {"length", "mass", "com", "inertia"})
        length = _positive_param(link_id, link, "length", 0.3)
        mass = _positive_param(link_id, link, "mass", 1.0)
        com = _number_param(link_id, link, "com", 0.5)
        expected_inertia = (1.0 / 3.0) * mass * length**2
        inertia = _positive_param(link_id, link, "inertia", expected_inertia)
        if not math.isclose(com, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise NotImplementedError(
                f"PlanarLink {link_id!r} com={com} is unsupported; the analytical "
                "plant requires the center of mass at half the link length."
            )
        if not math.isclose(inertia, expected_inertia, rel_tol=1e-9, abs_tol=1e-12):
            raise NotImplementedError(
                f"PlanarLink {link_id!r} inertia={inertia} is unsupported; the analytical "
                f"plant requires uniform-rod inertia {expected_inertia}."
            )
        lengths.append(length)
        masses.append(mass)
        centers.append(length * com)
        inertias.append(inertia)

    damping: list[float] = []
    for joint_id, joint in ((root_joint_id, root_joint), (child_joint_id, child_joint)):
        _require_supported_params(
            joint_id,
            joint,
            {"parent_frame", "child_frame", "damping", "rest_angle", "limit_min", "limit_max"},
        )
        rest_angle = _number_param(joint_id, joint, "rest_angle", 0.0)
        if rest_angle != 0.0:
            raise NotImplementedError(
                f"RevoluteJoint {joint_id!r} rest_angle={rest_angle} is unsupported; "
                "the analytical plant uses zero-angle joint coordinates."
            )
        for limit_name in ("limit_min", "limit_max"):
            if _param(joint, limit_name, None) is not None:
                raise NotImplementedError(
                    f"RevoluteJoint {joint_id!r} parameter {limit_name!r} is unsupported "
                    "by the analytical plant."
                )
        joint_damping = _number_param(joint_id, joint, "damping", 0.0)
        if joint_damping < 0.0:
            raise ValueError(
                f"RevoluteJoint {joint_id!r} damping must be non-negative, got {joint_damping}."
            )
        damping.append(joint_damping)

    muscles = sorted(by_type.get("MusclePath", []))
    if len(muscles) != 6:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} has {len(muscles)} MusclePaths; "
            "the canonical analytical builder requires exactly six."
        )
    expected_frames = [
        f"{world_id}.frame",
        f"{root_link_id}.distal",
        f"{child_link_id}.distal",
    ]
    moment_arms: list[tuple[float, float]] = []
    reference_lengths: list[float] = []
    maximum_forces: list[float] = []
    for muscle_id, muscle in muscles:
        _require_supported_params(
            muscle_id,
            muscle,
            {"path_points", "moment_arm", "reference_length", "max_isometric_force"},
        )
        _require_supported_muscle_path(muscle_id, muscle, expected_frames)
        raw_moment_arm = _param(muscle, "moment_arm", [])
        if not isinstance(raw_moment_arm, list) or len(raw_moment_arm) != 2:
            raise ValueError(
                f"MusclePath {muscle_id!r} moment_arm must contain exactly two numbers."
            )
        moment_arms.append(
            (
                _finite_number(muscle_id, "moment_arm[0]", raw_moment_arm[0]),
                _finite_number(muscle_id, "moment_arm[1]", raw_moment_arm[1]),
            )
        )
        reference_lengths.append(_positive_param(muscle_id, muscle, "reference_length", 0.2))
        maximum_forces.append(_positive_param(muscle_id, muscle, "max_isometric_force", 1.0))

    _require_analytical_boundary(by_type, node_name, child_link_id)
    return _AnalyticalPlanarValues(
        segment_lengths=(lengths[0], lengths[1]),
        segment_masses=(masses[0], masses[1]),
        segment_centers=(centers[0], centers[1]),
        segment_inertias=(inertias[0], inertias[1]),
        joint_damping=(damping[0], damping[1]),
        moment_arms=tuple(moment_arms),
        reference_lengths=tuple(reference_lengths),
        maximum_isometric_forces=tuple(maximum_forces),
    )


def _proximal_link_id(
    joint_id: str,
    frame: Any,
    links: Mapping[str, ComponentSpec],
) -> str:
    if not isinstance(frame, str) or not frame.endswith(".proximal"):
        raise NotImplementedError(
            f"RevoluteJoint {joint_id!r} child_frame must be a PlanarLink proximal frame; "
            f"got {frame!r}."
        )
    link_id = frame.removesuffix(".proximal")
    if link_id not in links:
        raise NotImplementedError(
            f"RevoluteJoint {joint_id!r} child_frame references missing PlanarLink {link_id!r}."
        )
    return link_id


def _require_serial_connections(
    graph: AcausalGraphSpec,
    *,
    node_name: str,
    world_id: str,
    anchor_id: str,
    root_joint_id: str,
    root_link_id: str,
    child_joint_id: str,
    child_link_id: str,
) -> None:
    expected = {
        frozenset(((world_id, "frame"), (anchor_id, "world"))),
        frozenset(((anchor_id, "frame"), (root_link_id, "proximal"))),
        frozenset(((world_id, "frame"), (root_joint_id, "parent"))),
        frozenset(((root_joint_id, "child"), (root_link_id, "proximal"))),
        frozenset(((root_link_id, "distal"), (child_joint_id, "parent"))),
        frozenset(((child_joint_id, "child"), (child_link_id, "proximal"))),
    }
    actual = {frozenset((connection.a, connection.b)) for connection in graph.connections}
    if actual != expected:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} connection topology is not the "
            "supported rooted serial two-link chain."
        )


def _require_supported_muscle_path(
    muscle_id: str,
    muscle: ComponentSpec,
    expected_frames: list[str],
) -> None:
    path_points = _param(muscle, "path_points", [])
    if not isinstance(path_points, list):
        raise ValueError(f"MusclePath {muscle_id!r} path_points must be a list.")
    frames: list[str] = []
    for point in path_points:
        if not isinstance(point, dict) or set(point) - {"frame", "offset"}:
            raise NotImplementedError(
                f"MusclePath {muscle_id!r} contains a path point the analytical "
                f"plant cannot represent: {point!r}."
            )
        frames.append(str(point.get("frame", "")))
        if point.get("offset", [0.0, 0.0]) != [0.0, 0.0]:
            raise NotImplementedError(
                f"MusclePath {muscle_id!r} offset {point.get('offset')!r} is unsupported "
                "by the constant-moment-arm analytical plant."
            )
    if frames != expected_frames:
        raise NotImplementedError(
            f"MusclePath {muscle_id!r} frame path {frames!r} is unsupported; "
            f"expected {expected_frames!r}."
        )


def _require_analytical_boundary(
    by_type: Mapping[str, list[tuple[str, ComponentSpec]]],
    node_name: str,
    terminal_link_id: str,
) -> None:
    inputs = by_type.get("ActuationInput", [])
    sensors = by_type.get("SensorOutput", [])
    markers = by_type.get("PointMarker", [])
    if (len(inputs), len(sensors), len(markers)) != (1, 2, 1):
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} requires one ActuationInput, "
            "two SensorOutputs, and one PointMarker for the analytical plant."
        )
    input_id, input_node = inputs[0]
    _require_supported_params(input_id, input_node, {"port_name", "order", "source_kind", "units"})
    if (
        _param(input_node, "port_name", "u") != "excitation"
        or _param(input_node, "source_kind", "force") != "muscle_excitation_vector"
        or _param(input_node, "order", 0) != 0
        or _param(input_node, "units", None) is not None
    ):
        raise NotImplementedError(
            f"ActuationInput {input_id!r} must be the unadorned 'excitation' muscle vector."
        )
    observed: dict[str, str] = {}
    for sensor_id, sensor in sensors:
        _require_supported_params(sensor_id, sensor, {"port_name", "order", "quantity", "units"})
        if _param(sensor, "order", 0) != 0 or _param(sensor, "units", None) is not None:
            raise NotImplementedError(
                f"SensorOutput {sensor_id!r} order or units are unsupported by the "
                "analytical plant boundary."
            )
        observed[str(_param(sensor, "quantity", "position"))] = str(
            _param(sensor, "port_name", "y")
        )
    if observed != {"effector": "effector", "state": "state"}:
        raise NotImplementedError(
            f"Planar multibody graph {node_name!r} must expose 'effector' and 'state'; "
            f"got {observed!r}."
        )
    marker_id, marker = markers[0]
    _require_supported_params(marker_id, marker, {"frame", "offset", "marker_name"})
    if (
        _param(marker, "frame", "") != f"{terminal_link_id}.distal"
        or _param(marker, "offset", [0.0, 0.0]) != [0.0, 0.0]
        or _param(marker, "marker_name", None) not in {None, "effector"}
    ):
        raise NotImplementedError(
            f"PointMarker {marker_id!r} must be the zero-offset effector on "
            f"{terminal_link_id + '.distal'!r}."
        )


def _param(node: ComponentSpec, key: str, default: Any) -> Any:
    return node.params[key] if key in node.params else default


def _require_supported_params(node_id: str, node: ComponentSpec, allowed: set[str]) -> None:
    unsupported = sorted(set(node.params) - allowed)
    if unsupported:
        raise NotImplementedError(
            f"{node.type} {node_id!r} has unsupported authored parameters {unsupported!r}."
        )


def _finite_number(node_id: str, key: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Node {node_id!r} parameter {key!r} must be a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Node {node_id!r} parameter {key!r} must be finite.")
    return result


def _number_param(node_id: str, node: ComponentSpec, key: str, default: float) -> float:
    return _finite_number(node_id, key, _param(node, key, default))


def _positive_param(node_id: str, node: ComponentSpec, key: str, default: float) -> float:
    value = _number_param(node_id, node, key, default)
    if value <= 0.0:
        raise ValueError(f"Node {node_id!r} parameter {key!r} must be positive, got {value}.")
    return value


def _frame_ref(node_id: str, port_name: str) -> str:
    return f"{node_id}.{port_name}"


def _valid_planar_frames(graph: AcausalGraphSpec) -> set[str]:
    frames: set[str] = set()
    for node_id, node_spec in graph.nodes.items():
        if node_spec.type == "PlanarLink":
            frames.add(_frame_ref(node_id, "proximal"))
            frames.add(_frame_ref(node_id, "distal"))
        elif node_spec.type == "WorldFrame":
            frames.add(_frame_ref(node_id, "frame"))
    return frames


def _node_param_string(node_spec: ComponentSpec, key: str) -> str:
    value = node_spec.params.get(key, "")
    return value if isinstance(value, str) else ""


def _diagnose_planar_multibody_graph(
    graph: AcausalGraphSpec,
    path: str,
) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    link_ids = [node_id for node_id, node in graph.nodes.items() if node.type == "PlanarLink"]
    joint_ids = [node_id for node_id, node in graph.nodes.items() if node.type == "RevoluteJoint"]
    world_ids = [node_id for node_id, node in graph.nodes.items() if node.type == "WorldFrame"]
    anchor_ids = [node_id for node_id, node in graph.nodes.items() if node.type == "Anchor"]
    valid_frames = _valid_planar_frames(graph)

    if not world_ids or not anchor_ids:
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="mechanics.unanchored_chain",
                message=(
                    f"Planar multibody interior {path!r} must include a WorldFrame "
                    "and Anchor for the root link."
                ),
                node_ids=[*world_ids, *anchor_ids],
                counts={"n_world_frames": len(world_ids), "n_anchors": len(anchor_ids)},
            )
        )

    for anchor_id in anchor_ids:
        frame = _node_param_string(graph.nodes[anchor_id], "frame")
        if frame not in valid_frames:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="mechanics.unanchored_chain",
                    message=(f"Anchor {path}.{anchor_id} references missing root frame {frame!r}."),
                    node_ids=[anchor_id],
                    variables=[frame],
                )
            )

    for joint_id in joint_ids:
        joint = graph.nodes[joint_id]
        parent_frame = _node_param_string(joint, "parent_frame")
        child_frame = _node_param_string(joint, "child_frame")
        missing = [frame for frame in (parent_frame, child_frame) if frame not in valid_frames]
        if missing:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="mechanics.joint_frame_mismatch",
                    message=(
                        f"RevoluteJoint {path}.{joint_id} references unknown frame(s) {missing!r}."
                    ),
                    node_ids=[joint_id],
                    variables=missing,
                )
            )
        if parent_frame.endswith(".proximal") and child_frame.endswith(".distal"):
            diagnostics.append(
                DomainDiagnostic(
                    severity="warning",
                    code="mechanics.joint_frame_mismatch",
                    message=(
                        f"RevoluteJoint {path}.{joint_id} connects proximal-to-distal; "
                        "planar arm trees normally connect parent distal/world to child proximal."
                    ),
                    node_ids=[joint_id],
                    variables=[parent_frame, child_frame],
                )
            )

    if len(joint_ids) > len(link_ids):
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="mechanics.open_kinematic_loop",
                message=(
                    f"Planar multibody interior {path!r} has {len(joint_ids)} joints "
                    f"for {len(link_ids)} links; closed loops are not supported by "
                    "the minimal-coordinate compiler."
                ),
                node_ids=joint_ids,
                counts={"n_joints": len(joint_ids), "n_links": len(link_ids)},
            )
        )

    for node_id, node_spec in graph.nodes.items():
        if node_spec.type != "MusclePath":
            continue
        path_points = node_spec.params.get("path_points", [])
        if not isinstance(path_points, list) or len(path_points) < 2:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="mechanics.muscle_path_missing_frame",
                    message=f"MusclePath {path}.{node_id} must contain at least two path points.",
                    node_ids=[node_id],
                )
            )
            continue
        missing_frames = [
            str(point.get("frame", ""))
            for point in path_points
            if not isinstance(point, dict) or str(point.get("frame", "")) not in valid_frames
        ]
        if missing_frames:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="mechanics.muscle_path_missing_frame",
                    message=(
                        f"MusclePath {path}.{node_id} references missing frame(s) "
                        f"{missing_frames!r}."
                    ),
                    node_ids=[node_id],
                    variables=missing_frames,
                )
            )

    diagnostics.append(
        DomainDiagnostic(
            severity="info",
            code="mechanics.dof_summary",
            message=(
                f"Planar multibody interior {path!r} has {len(joint_ids)} generalized "
                f"coordinate(s) from revolute joints {joint_ids!r}."
            ),
            node_ids=joint_ids,
            counts={
                "n_dof": len(joint_ids),
                "n_links": len(link_ids),
                "n_muscles": sum(1 for node in graph.nodes.values() if node.type == "MusclePath"),
            },
        )
    )
    return diagnostics


def _diagnose_structural_analysis(analysis: StructuralAnalysis) -> list[DomainDiagnostic]:
    diagnostics: list[DomainDiagnostic] = []
    for network in analysis.networks:
        counts = network.counts
        if counts["equations"] != counts["unknowns"]:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.unbalanced",
                    message=(
                        "Acausal network is structurally unbalanced: "
                        f"elements={list(network.elements)!r}, "
                        f"equations={counts['equations']}, unknowns={counts['unknowns']}, "
                        f"variables={list(network.differential_variables)!r}."
                    ),
                    node_ids=list(network.elements),
                    variables=list(network.differential_variables),
                    counts=counts,
                )
            )
        if not network.has_ground:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.ungrounded_network",
                    message=(
                        "Acausal network has no Ground or RotationalGround: "
                        f"elements={list(network.elements)!r}."
                    ),
                    node_ids=list(network.elements),
                    counts={"n_elements": len(network.elements)},
                )
            )
        if len(network.across_source_elements) > 1:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.parallel_across_sources",
                    message=(
                        "Acausal network has multiple prescribed across-variable sources: "
                        f"elements={list(network.across_source_elements)!r}."
                    ),
                    node_ids=list(network.across_source_elements),
                    counts={"n_sources": len(network.across_source_elements)},
                )
            )
        if len(network.source_elements) > 1:
            diagnostics.append(
                DomainDiagnostic(
                    severity="error",
                    code="acausal.series_through_sources",
                    message=(
                        "Acausal network has multiple through-variable sources in one "
                        f"connected path: elements={list(network.source_elements)!r}."
                    ),
                    node_ids=list(network.source_elements),
                    counts={"n_sources": len(network.source_elements)},
                )
            )
    if analysis.through_equation_cycle is not None:
        diagnostics.append(
            DomainDiagnostic(
                severity="error",
                code="acausal.series_through_sources",
                message=(
                    "Acausal through-variable equations contain a cycle: "
                    f"variables={list(analysis.through_equation_cycle)!r}."
                ),
                variables=list(analysis.through_equation_cycle),
                counts={"n_variables": len(analysis.through_equation_cycle)},
            )
        )
    return diagnostics


def _compile_summary(
    graph: AcausalGraphSpec,
    flattened: _FlattenedAcausalGraph,
    analysis: StructuralAnalysis | None,
) -> dict[str, int]:
    if analysis is None:
        return {
            "n_equations": 0,
            "n_unknowns": 0,
            "n_states": 0,
            "n_networks": 0,
            "n_elements": len(graph.nodes),
        }
    return {
        "n_equations": sum(len(network.equations) for network in analysis.networks),
        "n_unknowns": len(analysis.layout._differential),
        "n_states": analysis.layout.total_size,
        "n_networks": len(analysis.networks),
        "n_elements": len(flattened.elements),
    }


def _compile_planar_multibody_summary(graph: AcausalGraphSpec) -> dict[str, int]:
    n_joints = sum(1 for node in graph.nodes.values() if node.type == "RevoluteJoint")
    return {
        "n_equations": 0,
        "n_unknowns": 0,
        "n_states": 2 * n_joints,
        "n_networks": 1 if graph.nodes else 0,
        "n_elements": len(graph.nodes),
        "n_dof": n_joints,
        "n_muscles": sum(1 for node in graph.nodes.values() if node.type == "MusclePath"),
    }


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
                    f"Acausal composite node {node_path!r} requires an acausal interior subgraph."
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
            raise ValueError(f"Unknown ActuationInput source_kind {source_kind!r} at {node_path!r}")
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
        raise ValueError(f"Unsupported acausal component type {node_type!r} at {node_path!r}")
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
    if getattr(meta, "domain", None) not in {ACAUSAL_DOMAIN_ID, MECHANICS_DOMAIN_ID}:
        raise ValueError(
            f"Component type {node_type!r} at {node_path!r} belongs to domain "
            f"{getattr(meta, 'domain', None)!r}, not {ACAUSAL_DOMAIN_ID!r} "
            f"or {MECHANICS_DOMAIN_ID!r}"
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
