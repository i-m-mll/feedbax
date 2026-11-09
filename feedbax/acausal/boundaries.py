"""Typed descriptors for acausal graph interfaces."""

from dataclasses import dataclass
from typing import Literal

from feedbax.contracts.portable import (
    ComponentOwner,
    ComponentOptions,
    Ports,
    validate_configuration,
)
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID
from feedbax.contracts.acausal_interface import (
    conserving_port_type,
    signal_port_type,
    FORCE_PAYLOAD,
)


@dataclass
class ActuationInput(ComponentOwner):
    """Causal signal input to an acausal conserving source."""

    __post_init__ = validate_configuration
    component_options = ComponentOptions(domain=ACAUSAL_DOMAIN_ID)
    ports = Ports({"u": FORCE_PAYLOAD.port_type(), "flange": conserving_port_type()}, {})
    port_name: str = "u"
    order: int = 0
    source_kind: Literal[
        "force", "torque", "prescribed_motion", "muscle_excitation_vector", "actuator_control",
    ] = "force"
    size: int | None = None
    units: str | None = None


@dataclass
class SensorOutput(ComponentOwner):
    """Acausal measurement exposed as a causal signal output."""

    __post_init__ = validate_configuration
    component_options = ComponentOptions(domain=ACAUSAL_DOMAIN_ID)
    ports = Ports({"flange": conserving_port_type()}, {"y": signal_port_type()})
    port_name: str = "y"
    order: int = 0
    quantity: Literal[
        "position", "velocity", "force", "torque", "angle", "angular_velocity", "effector", "state",
    ] = "position"
    units: str | None = None


@dataclass
class BoundaryPort(ComponentOwner):
    """Named conserving port exposed by a nested acausal composite."""

    __post_init__ = validate_configuration
    component_options = ComponentOptions(domain=ACAUSAL_DOMAIN_ID)
    ports = Ports({"flange": conserving_port_type()}, {})
    port_name: str = "port"
    order: int = 0
