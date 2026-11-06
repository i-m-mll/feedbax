"""Acausal graph boundary descriptors and the built-in element module allowlist."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from feedbax.acausal import _component, mechanics, multibody, rotational, translational
from feedbax.contracts.acausal_interface import (
    ACTUATION_PAYLOADS,
    FORCE_PAYLOAD,
    conserving_port_type,
    signal_port_type,
)
from feedbax.contracts.component import PortTypeSpec
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ParamSchema

from .declarations import DeclaredComponent, declare_component
from .discovery import ComponentPack, compile_owner, discover_modules


ACAUSAL_ELEMENT_MODULES = (mechanics, multibody, rotational, translational)


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...


def acausal_boundary_definitions() -> tuple[DeclaredComponent, ...]:
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
                    options=list(ACTUATION_PAYLOADS),
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
                    "u": FORCE_PAYLOAD.port_type(),
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


def acausal_component_pack() -> ComponentPack:
    """Compose the built-in acausal element owners and boundary descriptors."""

    elements = discover_modules(*ACAUSAL_ELEMENT_MODULES)
    return ComponentPack(
        components=(*elements.components, *acausal_boundary_definitions()),
        sources=(Path(_component.__file__), Path(__file__)),
    )


def register_acausal_components(registry: _Registry) -> None:
    """Project acausal owners into the legacy built-in definition collector."""

    for owner in acausal_component_pack().components:
        if isinstance(owner, type):
            definitions, _, _ = compile_owner(owner)
            for definition in definitions:
                registry.register(definition)
        elif isinstance(owner, DeclaredComponent):
            registry.register(owner)
