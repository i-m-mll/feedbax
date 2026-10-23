"""Acausal element and boundary-adapter component metadata."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Protocol

from feedbax.acausal.base import AcausalElement, AcausalPort
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
from feedbax.contracts.acausal_interface import conserving_port_type, signal_port_type
from feedbax.contracts.component import PortTypeSpec
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID, MECHANICS_DOMAIN_ID
from feedbax.contracts.graph import ParamSchema
from feedbax.contracts.representation import RepresentationSpec

from .declarations import DeclaredComponent, declare_component


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


_POSITIVE_PARAMS = {
    "damping",
    "inertia",
    "mass",
    "stiffness",
}


_TRANSLATIONAL_ELEMENTS: tuple[type[AcausalElement], ...] = (
    Mass,
    LinearSpring,
    LinearDamper,
    Ground,
    ForceSource,
    PrescribedMotion,
    PositionSensor,
    VelocitySensor,
    ForceSensor,
)


_ROTATIONAL_ELEMENTS: tuple[type[AcausalElement], ...] = (
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


_PLANAR_MULTIBODY_ELEMENTS: tuple[type[AcausalElement], ...] = (
    PlanarLink,
    RevoluteJoint,
    WorldFrame,
    Anchor,
    MusclePath,
    PointMarker,
)


def _schema_type(annotation: Any, default: Any) -> str:
    if isinstance(default, (list, tuple)):
        return "array"
    if isinstance(default, dict):
        return "object"
    if annotation in {int, "int"} or isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if annotation in {bool, "bool"} or isinstance(default, bool):
        return "bool"
    if annotation in {str, "str"} or isinstance(default, str):
        return "str"
    return "float"


def _param_schema_from_constructor(element_type: type[AcausalElement]) -> list[ParamSchema]:
    schema: list[ParamSchema] = []
    signature = inspect.signature(element_type.__init__)
    for name, parameter in signature.parameters.items():
        if name in {"self", "name"}:
            continue
        default = parameter.default
        has_default = default is not inspect.Parameter.empty
        schema.append(
            ParamSchema(
                name=name,
                type=_schema_type(parameter.annotation, default if has_default else None),
                default=default if has_default else None,
                min=0.0 if name in _POSITIVE_PARAMS else None,
                required=not has_default,
            )
        )
    return schema


def _port_type_spec(ports: Iterable[AcausalPort]) -> PortTypeSpec:
    return PortTypeSpec(
        inputs={
            port.name: conserving_port_type(
                physical_domain=port.domain.value,
                across_vars=port.across_vars,
                through_var=port.through_var,
            )
            for port in ports
        },
        outputs={},
    )


def _element_meta(
    element_type: type[AcausalElement],
    *,
    category: str,
    icon: str,
    name: str | None = None,
    domain: str = ACAUSAL_DOMAIN_ID,
    representation: RepresentationSpec | None = None,
    param_schema: list[ParamSchema] | None = None,
) -> DeclaredComponent:
    probe = element_type("__probe__")
    port_names = sorted(probe.ports)
    return declare_component(
        name=name or element_type.__name__,
        category=category,
        description=(inspect.getdoc(element_type) or f"{element_type.__name__} acausal element."),
        param_schema=(
            param_schema
            if param_schema is not None
            else _param_schema_from_constructor(element_type)
        ),
        input_ports=port_names,
        output_ports=[],
        icon=icon,
        port_types=_port_type_spec(probe.ports.values()),
        domain=domain,
        builder=None,
        representation=representation,
    )


def _literal_binding(value: Any, *, dim: int | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "literal", "value": value}
    if dim is not None:
        binding["dim"] = dim
    return binding


def _param_binding(
    path: str, *, expected_type: str | None = None, dim: int | None = None
) -> dict[str, Any]:
    binding: dict[str, Any] = {"kind": "param_path", "path": path}
    if expected_type is not None:
        binding["expected_type"] = expected_type
    if dim is not None:
        binding["dim"] = dim
    return binding


def _representation(data: dict[str, Any]) -> RepresentationSpec:
    return RepresentationSpec.model_validate(data)


def _planar_link_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "proximal",
                    "semantic_role": "joint",
                    "binding": _literal_binding([0.0, 0.0], dim=2),
                    "units": "m",
                    "dim": 2,
                },
                {
                    "id": "distal",
                    "semantic_role": "endpoint",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"computed_from": "length_and_orientation"},
                },
            ],
            "elements": [
                {
                    "id": "link",
                    "archetype": "planar_chain",
                    "anchors": ["proximal", "distal"],
                    "bindings": {
                        "link_lengths": _param_binding("length", expected_type="float", dim=1)
                    },
                    "metadata": {"chain_kind": "planar_link_segment"},
                }
            ],
        }
    )


def _muscle_path_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "path",
                    "semantic_role": "path",
                    "units": "m",
                    "dim": 2,
                    "metadata": {"collection": "path_points"},
                }
            ],
            "elements": [
                {
                    "id": "muscle-path",
                    "archetype": "muscle_path",
                    "anchors": ["path"],
                    "bindings": {
                        "path_points": _param_binding("path_points", expected_type="array")
                    },
                    "metadata": {"activation_input": "activation"},
                }
            ],
        }
    )


def _point_marker_representation() -> RepresentationSpec:
    return _representation(
        {
            "frame": "world.xy",
            "units": "m",
            "dim": 2,
            "anchors": [
                {
                    "id": "point",
                    "semantic_role": "endpoint",
                    "binding": _param_binding("offset", expected_type="array", dim=2),
                    "units": "m",
                    "dim": 2,
                }
            ],
            "elements": [
                {
                    "id": "marker",
                    "archetype": "marker",
                    "anchors": ["point"],
                    "metadata": {"frame_param": "frame"},
                }
            ],
        }
    )


def _multibody_param_schema(element_type: type[AcausalElement]) -> list[ParamSchema] | None:
    if element_type is MusclePath:
        return [
            ParamSchema(name="path_points", type="array", default=[], required=True),
            ParamSchema(name="moment_arm", type="array", default=[], required=False),
            ParamSchema(name="reference_length", type="float", default=0.2, min=0.0),
            ParamSchema(name="max_isometric_force", type="float", default=1.0, min=0.0),
        ]
    if element_type is PointMarker:
        return [
            ParamSchema(name="frame", type="str", default="", required=True),
            ParamSchema(name="offset", type="array", default=[0.0, 0.0], required=False),
            ParamSchema(name="marker_name", type="str", default=None, required=False),
        ]
    return None


def _multibody_representation(element_type: type[AcausalElement]) -> RepresentationSpec | None:
    if element_type is PlanarLink:
        return _planar_link_representation()
    if element_type is MusclePath:
        return _muscle_path_representation()
    if element_type is PointMarker:
        return _point_marker_representation()
    return None


def _adapter_metas() -> tuple[DeclaredComponent, ...]:
    return (
        declare_component(
            name="ActuationInput",
            category="Boundary",
            description="Causal signal input to an acausal conserving source.",
            param_schema=[
                ParamSchema(name="port_name", type="str", default="u", required=True),
                ParamSchema(name="order", type="int", default=0, required=True),
                ParamSchema(
                    name="source_kind",
                    type="enum",
                    options=["force", "torque", "prescribed_motion"],
                    default="force",
                    required=True,
                ),
                ParamSchema(name="units", type="str", default=None, required=False),
            ],
            input_ports=["u", "flange"],
            output_ports=[],
            icon="LogIn",
            port_types=PortTypeSpec(
                inputs={
                    "u": signal_port_type(),
                    "flange": conserving_port_type(),
                },
                outputs={},
            ),
            domain=ACAUSAL_DOMAIN_ID,
            builder=None,
        ),
        declare_component(
            name="SensorOutput",
            category="Boundary",
            description="Acausal measurement exposed as a causal signal output.",
            param_schema=[
                ParamSchema(name="port_name", type="str", default="y", required=True),
                ParamSchema(name="order", type="int", default=0, required=True),
                ParamSchema(
                    name="quantity",
                    type="enum",
                    options=[
                        "position",
                        "velocity",
                        "force",
                        "torque",
                        "angle",
                        "angular_velocity",
                    ],
                    default="position",
                    required=True,
                ),
                ParamSchema(name="units", type="str", default=None, required=False),
            ],
            input_ports=["flange"],
            output_ports=["y"],
            icon="LogOut",
            port_types=PortTypeSpec(
                inputs={"flange": conserving_port_type()},
                outputs={"y": signal_port_type()},
            ),
            domain=ACAUSAL_DOMAIN_ID,
            builder=None,
        ),
        declare_component(
            name="BoundaryPort",
            category="Boundary",
            description="Named conserving port exposed by a nested acausal composite.",
            param_schema=[
                ParamSchema(name="port_name", type="str", default="port", required=True),
                ParamSchema(name="order", type="int", default=0, required=True),
            ],
            input_ports=["flange"],
            output_ports=[],
            icon="Unplug",
            port_types=PortTypeSpec(
                inputs={"flange": conserving_port_type()},
                outputs={},
            ),
            domain=ACAUSAL_DOMAIN_ID,
            builder=None,
        ),
    )


def register_acausal_components(registry: _Registry) -> None:
    """Register acausal element and boundary-adapter palette metadata."""

    for element_type in _TRANSLATIONAL_ELEMENTS:
        registry.register(
            _element_meta(
                element_type,
                category="Acausal / Translational",
                icon="MoveHorizontal",
            )
        )
    for element_type in _ROTATIONAL_ELEMENTS:
        registry.register(
            _element_meta(
                element_type,
                category="Acausal / Rotational",
                icon="RotateCw",
            )
        )
    for element_type in _PLANAR_MULTIBODY_ELEMENTS:
        registry.register(
            _element_meta(
                element_type,
                category="Acausal / Planar Multibody",
                icon="Orbit",
                domain=MECHANICS_DOMAIN_ID,
                representation=_multibody_representation(element_type),
                param_schema=_multibody_param_schema(element_type),
            )
        )
    for meta in _adapter_metas():
        registry.register(meta)

    for alias, element_type, category, icon in (
        ("MechanicsMass", Mass, "Bodies", "Circle"),
        ("MechanicsGround", Ground, "Boundary", "Anchor"),
        ("MechanicsLinearSpring", LinearSpring, "Forces", "MoveHorizontal"),
        ("MechanicsLinearDamper", LinearDamper, "Forces", "Gauge"),
        ("MechanicsForceSource", ForceSource, "Boundary", "LogIn"),
        ("MechanicsPositionSensor", PositionSensor, "Sensors", "Ruler"),
        ("MechanicsVelocitySensor", VelocitySensor, "Sensors", "Activity"),
        ("MechanicsForceSensor", ForceSensor, "Sensors", "Gauge"),
    ):
        registry.register(
            _element_meta(
                element_type,
                name=alias,
                category=category,
                icon=icon,
                domain=MECHANICS_DOMAIN_ID,
            )
        )
    registry.register(
        _element_meta(
            RigidTendonHillMuscle,
            category="Muscles",
            icon="Dumbbell",
            domain=MECHANICS_DOMAIN_ID,
        )
    )
