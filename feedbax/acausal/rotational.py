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

from feedbax.acausal._scalar_mechanics import ROTATIONAL
from feedbax.acausal.base import (
    AcausalElement,
    AcausalPort,
    Domain,
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

    def __init__(self, name: str, inertia: float = 1.0):
        super().__init__(name=name)
        ROTATIONAL.inertia(self, inertia)


@dataclass
class TorsionalSpring(AcausalElement):
    """Torsional spring: ``tau = k * (angle_a - angle_b)``.

    Ports:
        flange_a, flange_b -- rotational.
    """

    def __init__(self, name: str, stiffness: float = 1.0):
        super().__init__(name=name)
        ROTATIONAL.coupling(self, "stiffness", stiffness, ROTATIONAL.coordinate)


@dataclass
class RotationalDamper(AcausalElement):
    """Rotational damper: ``tau = b * (omega_a - omega_b)``."""

    def __init__(self, name: str, damping: float = 1.0):
        super().__init__(name=name)
        ROTATIONAL.coupling(self, "damping", damping, ROTATIONAL.velocity)


@dataclass
class TorqueSource(AcausalElement):
    """External torque from the causal world."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.one_port(self, "force_source")


@dataclass
class RotationalGround(AcausalElement):
    """Fixed rotational reference (angle = 0, angular_vel = 0)."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.one_port(self, "ground")


@dataclass
class GearRatio(AcausalElement):
    """Ideal gear constraint.

    Kinematic relations::

        angle_b   = ratio * angle_a
        omega_b   = ratio * omega_a
        torque_a  = ratio * torque_b   (power conservation)

    During assembly, port_b across variables are *eliminated* and expressed
    in terms of port_a's variables scaled by the ratio.

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


@dataclass
class AngleSensor(AcausalElement):
    """Reads angle at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.sensor(self, ROTATIONAL.coordinate)


@dataclass
class AngularVelocitySensor(AcausalElement):
    """Reads angular velocity at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.sensor(self, ROTATIONAL.velocity)


@dataclass
class TorqueSensor(AcausalElement):
    """Reads net torque at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        ROTATIONAL.sensor(self, ROTATIONAL.effort)
