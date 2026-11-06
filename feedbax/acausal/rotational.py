"""Rotational-domain acausal elements.

Mirrors the translational elements with rotational semantics:
- Across variables: ``(angle, angular_vel)``
- Through variable: ``torque``

Elements
--------
- **Inertia** -- rotational inertia, ``tau = J * alpha``.
- **TorsionalSpring** -- ``tau = k * (angle_a - angle_b)``.
- **RotationalDamper** -- ``tau = b * (omega_a - omega_b)``.
- **TorqueSource** -- external torque from the causal world.
- **RotationalGround** -- fixed rotational reference.
- **GearRatio** -- kinematic gear constraint.
- **AngleSensor / AngularVelocitySensor / TorqueSensor** -- read-only sensors.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.acausal._component import NonNegativeFloat, element_options
from feedbax.acausal._scalar_mechanics import ROTATIONAL
from feedbax.acausal.base import (
    AcausalElement,
    AcausalPort,
    Domain,
)
from feedbax._acausal_payload import TORQUE_PAYLOAD
from feedbax.component_registry.discovery import component
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID


@component(
    "Inertia",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class Inertia(AcausalElement):
    """Rotational inertia.

    Port:
        flange -- rotational, across=(angle, angular_vel), through=torque.

    Equations:
        d(angle)/dt = angular_vel
        d(angular_vel)/dt = net_torque / J
    """

    def __init__(self, name: str, inertia: NonNegativeFloat = 1.0):
        super().__init__(name=name)
        ROTATIONAL.inertia(self, inertia)


@component(
    "TorsionalSpring",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class TorsionalSpring(AcausalElement):
    """Torsional spring: ``tau = k * (angle_a - angle_b)``.

    Ports:
        flange_a, flange_b -- rotational.
    """

    def __init__(self, name: str, stiffness: NonNegativeFloat = 1.0):
        super().__init__(name=name)
        ROTATIONAL.coupling(self, "stiffness", stiffness, ROTATIONAL.coordinate)


@component(
    "RotationalDamper",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class RotationalDamper(AcausalElement):
    """Rotational damper: ``tau = b * (omega_a - omega_b)``."""

    def __init__(self, name: str, damping: NonNegativeFloat = 1.0):
        super().__init__(name=name)
        ROTATIONAL.coupling(self, "damping", damping, ROTATIONAL.velocity)


@component(
    "TorqueSource",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class TorqueSource(AcausalElement):
    """External torque from the causal world."""

    source_payload = TORQUE_PAYLOAD

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.one_port(self, "force_source")


@component(
    "RotationalGround",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class RotationalGround(AcausalElement):
    """Fixed rotational reference (angle = 0, angular_vel = 0)."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.one_port(self, "ground")


@component(
    "GearRatio",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class GearRatio(AcausalElement):
    """Ideal gear constraint.

    Kinematic relations::

        angle_b   = ratio * angle_a
        omega_b   = ratio * omega_a
        reaction_a + ratio * reaction_b = 0   (zero reaction power)

    Assembly pulls shaft efforts and inertias back to one coordinate;
    physical shaft observables retain their ratio scaling.

    Ports:
        flange_a (input shaft), flange_b (output shaft) -- rotational.
    """

    def __init__(self, name: str, ratio: float = 1.0):
        super().__init__(name=name)
        for port_name in ("flange_a", "flange_b"):
            self.ports[port_name] = AcausalPort(
                name=port_name,
                domain=Domain.ROTATIONAL,
                across_vars=("angle", "angular_vel"),
                through_var="torque",
            )
        self.params[f"{name}.ratio"] = ratio
        self.element_type = "gear_ratio"


@component(
    "AngleSensor",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class AngleSensor(AcausalElement):
    """Reads angle at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.sensor(self, ROTATIONAL.coordinate)


@component(
    "AngularVelocitySensor",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class AngularVelocitySensor(AcausalElement):
    """Reads angular velocity at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.sensor(self, ROTATIONAL.velocity)


@component(
    "TorqueSensor",
    configure=element_options,
    category="Acausal / Rotational",
    icon="RotateCw",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class TorqueSensor(AcausalElement):
    """Reads net torque at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.sensor(self, ROTATIONAL.effort)
