"""Portable direct-force mechanics owners."""

from __future__ import annotations
from typing import Annotated, Any
from pydantic import Field
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import Ports, ConfigBinding
from feedbax.runtime.graph import Component
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.arm import TwoLinkArm as ArmSkeleton
from feedbax.mechanics.skeleton.pointmass import PointMass as PointMassSkeleton

_MECHANICS_PORTS = Ports(
    inputs={"force": PortType(dtype="vector")},
    outputs={"effector": PortType(dtype="state"), "state": PortType(dtype="state")},
)


class TwoLinkArm(Mechanics, Component):
    """Two-link arm plant with direct force input."""

    ports = _MECHANICS_PORTS
    config_bindings = (ConfigBinding(path="plant.skeleton.l", name="link_lengths"),)

    def __init__(
        self,
        dt: Annotated[float, Field(ge=0.001)] = 0.01,
        link_lengths: Annotated[Any, Field(json_schema_extra={"feedbax": {"type": "array"}})] = (
            0.30,
            0.33,
        ),
    ):
        super().__init__(plant=DirectForceInput(ArmSkeleton(l=link_lengths)), dt=dt)


class PointMass(Mechanics, Component):
    """Point-mass plant with direct force input."""

    ports = _MECHANICS_PORTS
    config_bindings = (
        ConfigBinding(path="plant.skeleton.mass", name="mass"),
        ConfigBinding(path="plant.skeleton.damping", name="damping"),
    )

    def __init__(
        self,
        dt: Annotated[float, Field(ge=0.001)] = 0.01,
        mass: Annotated[float, Field(ge=0)] = 1.0,
        damping: Annotated[float, Field(ge=0)] = 0.0,
    ):
        super().__init__(
            plant=DirectForceInput(PointMassSkeleton(mass=mass, damping=damping)), dt=dt
        )
