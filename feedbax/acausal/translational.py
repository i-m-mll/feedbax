"""Translational-domain acausal elements.

Each element is a *construction-time descriptor* (plain Python ``@dataclass``).
It declares ports, equations, and parameters that the assembly algorithm
compiles into a single JAX-traceable vector field.

Elements
--------
- **Mass** -- point mass, ``F = m a``.
- **LinearSpring** -- Hooke's law, ``F = k (x_a - x_b)``.
- **LinearDamper** -- viscous damping, ``F = b (v_a - v_b)``.
- **Ground** -- fixed reference (across vars = 0).
- **ForceSource** -- external force from the causal world.
- **PrescribedMotion** -- position driven from causal input.
- **PositionSensor / VelocitySensor / ForceSensor** -- read-only sensors.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.acausal._component import NonNegativeFloat, element_options
from feedbax.acausal._scalar_mechanics import TRANSLATIONAL
from feedbax.acausal.base import AcausalElement
from feedbax._acausal_payload import FORCE_PAYLOAD, PRESCRIBED_MOTION_PAYLOAD
from feedbax.component_registry.discovery import component
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID, MECHANICS_DOMAIN_ID


@component(
    "Mass",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsMass": {"category": "Bodies", "icon": "Circle", "domain": MECHANICS_DOMAIN_ID}
    },
)
@dataclass
class Mass(AcausalElement):
    """Point mass in the translational domain.

    Ports:
        flange -- translational, across=(pos, vel), through=force.

    Equations:
        d(pos)/dt = vel
        d(vel)/dt = net_force / mass

    The ``net_force`` is the sum of all through variables connected to the
    flange (computed by the assembly algorithm's force-balance step).
    """

    def __init__(self, name: str, mass: NonNegativeFloat = 1.0):
        super().__init__(name=name)
        TRANSLATIONAL.inertia(self, mass)


@component(
    "LinearSpring",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsLinearSpring": {
            "category": "Forces",
            "icon": "MoveHorizontal",
            "domain": MECHANICS_DOMAIN_ID,
        }
    },
)
@dataclass
class LinearSpring(AcausalElement):
    """Linear spring.

    The through-variable at each port represents the force exerted by the
    spring *on the external body* connected at that port::

        force_b = k * (pos_a - pos_b)   -- pulls B toward A
        force_a = -force_b              -- pulls A toward B

    Ports:
        flange_a, flange_b -- translational.
    """

    def __init__(self, name: str, stiffness: NonNegativeFloat = 1.0):
        super().__init__(name=name)
        TRANSLATIONAL.coupling(self, "stiffness", stiffness, TRANSLATIONAL.coordinate)


@component(
    "LinearDamper",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsLinearDamper": {
            "category": "Forces",
            "icon": "Gauge",
            "domain": MECHANICS_DOMAIN_ID,
        }
    },
)
@dataclass
class LinearDamper(AcausalElement):
    """Viscous damper: ``F = b * (vel_a - vel_b)``.

    Ports:
        flange_a, flange_b -- translational.
    """

    def __init__(self, name: str, damping: NonNegativeFloat = 1.0):
        super().__init__(name=name)
        TRANSLATIONAL.coupling(self, "damping", damping, TRANSLATIONAL.velocity)


@component(
    "ForceSource",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsForceSource": {
            "category": "Boundary",
            "icon": "LogIn",
            "domain": MECHANICS_DOMAIN_ID,
        }
    },
)
@dataclass
class ForceSource(AcausalElement):
    """External force from the causal world.

    Port:
        flange -- translational.  The through variable is set directly
        from the causal input array.
    """

    source_payload = FORCE_PAYLOAD

    def __init__(self, name: str):
        super().__init__(name=name)
        TRANSLATIONAL.one_port(self, "force_source")


@component(
    "Ground",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsGround": {"category": "Boundary", "icon": "Anchor", "domain": MECHANICS_DOMAIN_ID}
    },
)
@dataclass
class Ground(AcausalElement):
    """Fixed translational reference (pos = 0, vel = 0).

    Port:
        flange -- translational.  All across variables connected to this
        port become grounded (fixed at zero).
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        TRANSLATIONAL.one_port(self, "ground")


@component(
    "PrescribedMotion",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
)
@dataclass
class PrescribedMotion(AcausalElement):
    """Position driven from a causal input signal.

    Port:
        flange -- translational.  The connected ``pos`` (and ``vel``)
        across variables become causal inputs (removed from the state
        vector).
    """

    source_payload = PRESCRIBED_MOTION_PAYLOAD

    def __init__(self, name: str):
        super().__init__(name=name)
        TRANSLATIONAL.one_port(self, "prescribed_motion")


@component(
    "PositionSensor",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsPositionSensor": {
            "category": "Sensors",
            "icon": "Ruler",
            "domain": MECHANICS_DOMAIN_ID,
        }
    },
)
@dataclass
class PositionSensor(AcausalElement):
    """Reads position at a connection node.

    Port:
        flange -- translational.  This sensor contributes zero force.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        TRANSLATIONAL.sensor(self, TRANSLATIONAL.coordinate)


@component(
    "VelocitySensor",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsVelocitySensor": {
            "category": "Sensors",
            "icon": "Activity",
            "domain": MECHANICS_DOMAIN_ID,
        }
    },
)
@dataclass
class VelocitySensor(AcausalElement):
    """Reads velocity at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        TRANSLATIONAL.sensor(self, TRANSLATIONAL.velocity)


@component(
    "ForceSensor",
    configure=element_options,
    category="Acausal / Translational",
    icon="MoveHorizontal",
    domain=ACAUSAL_DOMAIN_ID,
    aliases={
        "MechanicsForceSensor": {
            "category": "Sensors",
            "icon": "Gauge",
            "domain": MECHANICS_DOMAIN_ID,
        }
    },
)
@dataclass
class ForceSensor(AcausalElement):
    """Reads net force at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        TRANSLATIONAL.sensor(self, TRANSLATIONAL.effort)
