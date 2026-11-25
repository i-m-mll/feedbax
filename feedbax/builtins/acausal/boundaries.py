"""Typed descriptors for acausal graph interfaces."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from feedbax.contracts.portable import (
    ComponentOwner,
    ComponentOptions,
    OutputSignals,
    Ports,
    SignalProperties,
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


_SENSOR_SIGNALS = {
    "position": SignalProperties("m", "point"),
    "velocity": SignalProperties("m/s", "free"),
    "force": SignalProperties("N", "free"),
    "torque": SignalProperties("N*m"),
    "angle": SignalProperties("rad"),
    "angular_velocity": SignalProperties("rad/s"),
}


def _sensor_signals(params: Mapping[str, Any]) -> OutputSignals:
    quantity = params["quantity"]
    unit = params.get("units")
    if quantity == "effector":
        return {
            ("y", "pos"): _SENSOR_SIGNALS["position"],
            ("y", "vel"): _SENSOR_SIGNALS["velocity"],
            ("y", "force"): _SENSOR_SIGNALS["force"],
        }
    declared = _SENSOR_SIGNALS.get(quantity, SignalProperties())
    return {("y",): replace(declared, unit=unit)} if unit is not None else {("y",): declared}


@dataclass
class SensorOutput(ComponentOwner):
    """Acausal measurement exposed as a causal signal output."""

    __post_init__ = validate_configuration
    component_options = ComponentOptions(domain=ACAUSAL_DOMAIN_ID)
    ports = Ports(
        {"flange": conserving_port_type()}, {"y": signal_port_type()},
        output_leaves=_sensor_signals,
    )
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
