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
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ParamSchema

from .meta import ComponentMeta


class _Registry(Protocol):
    def register(self, meta: ComponentMeta) -> None: ...


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


def _schema_type(annotation: Any, default: Any) -> str:
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
) -> ComponentMeta:
    probe = element_type("__probe__")
    port_names = sorted(probe.ports)
    return ComponentMeta(
        name=element_type.__name__,
        category=category,
        description=(inspect.getdoc(element_type) or f"{element_type.__name__} acausal element."),
        param_schema=_param_schema_from_constructor(element_type),
        input_ports=port_names,
        output_ports=[],
        icon=icon,
        port_types=_port_type_spec(probe.ports.values()),
        domain=ACAUSAL_DOMAIN_ID,
        builder=None,
    )


def _adapter_metas() -> tuple[ComponentMeta, ...]:
    return (
        ComponentMeta(
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
        ComponentMeta(
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
        ComponentMeta(
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
    for meta in _adapter_metas():
        registry.register(meta)
