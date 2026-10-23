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

from feedbax.acausal.base import (
    AcausalElement,
    AcausalEquation,
    AcausalPort,
    Domain,
)


def _fqn(element: str, port: str, slot: str) -> str:
    """Fully-qualified variable name."""
    return f"{element}.{port}.{slot}"


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------

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

    def __init__(self, name: str, mass: float = 1.0):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.params[f"{name}.mass"] = mass

        pos = _fqn(name, "flange", "pos")
        vel = _fqn(name, "flange", "vel")

        # d(pos)/dt = vel
        self.equations.append(AcausalEquation(
            lhs_var=pos,
            rhs_fn=lambda vals, _v=vel: vals[_v],
            depends_on=(vel,),
        ))
        # d(vel)/dt = net_force / mass  (net_force injected by assembly)
        # We leave this as a placeholder; the assembly algorithm builds the
        # actual RHS that sums all through-var contributions at this node.
        # We encode the *mass equation* so assembly knows this is an inertia.
        self.equations.append(AcausalEquation(
            lhs_var=vel,
            rhs_fn=None,  # filled by assembly
            depends_on=(),
        ))
        self.element_type = "mass"


# ---------------------------------------------------------------------------
# LinearSpring
# ---------------------------------------------------------------------------

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

    def __init__(self, name: str, stiffness: float = 1.0):
        super().__init__(name=name)
        for port_name in ("flange_a", "flange_b"):
            self.ports[port_name] = AcausalPort(
                name=port_name,
                domain=Domain.TRANSLATIONAL,
                across_vars=("pos", "vel"),
                through_var="force",
            )
        self.params[f"{name}.stiffness"] = stiffness

        pos_a = _fqn(name, "flange_a", "pos")
        pos_b = _fqn(name, "flange_b", "pos")
        force_a = _fqn(name, "flange_a", "force")
        force_b = _fqn(name, "flange_b", "force")
        k_key = f"{name}.stiffness"

        # force_b = k * (pos_a - pos_b)  -- spring pulls B toward A
        self.equations.append(AcausalEquation(
            lhs_var=force_b,
            rhs_fn=lambda vals, _pa=pos_a, _pb=pos_b, _k=k_key: (
                vals[_k] * (vals[_pa] - vals[_pb])
            ),
            depends_on=(pos_a, pos_b),
            is_through_def=True,
        ))
        # force_a = -force_b  -- spring pulls A toward B
        self.equations.append(AcausalEquation(
            lhs_var=force_a,
            rhs_fn=lambda vals, _fb=force_b: -vals[_fb],
            depends_on=(force_b,),
            is_through_def=True,
        ))


# ---------------------------------------------------------------------------
# LinearDamper
# ---------------------------------------------------------------------------

@dataclass
class LinearDamper(AcausalElement):
    """Viscous damper: ``F = b * (vel_a - vel_b)``.

    Ports:
        flange_a, flange_b -- translational.
    """

    def __init__(self, name: str, damping: float = 1.0):
        super().__init__(name=name)
        for port_name in ("flange_a", "flange_b"):
            self.ports[port_name] = AcausalPort(
                name=port_name,
                domain=Domain.TRANSLATIONAL,
                across_vars=("pos", "vel"),
                through_var="force",
            )
        self.params[f"{name}.damping"] = damping

        vel_a = _fqn(name, "flange_a", "vel")
        vel_b = _fqn(name, "flange_b", "vel")
        force_a = _fqn(name, "flange_a", "force")
        force_b = _fqn(name, "flange_b", "force")
        b_key = f"{name}.damping"

        # force_b = b * (vel_a - vel_b)  -- damps B toward A's velocity
        self.equations.append(AcausalEquation(
            lhs_var=force_b,
            rhs_fn=lambda vals, _va=vel_a, _vb=vel_b, _b=b_key: (
                vals[_b] * (vals[_va] - vals[_vb])
            ),
            depends_on=(vel_a, vel_b),
            is_through_def=True,
        ))
        # force_a = -force_b
        self.equations.append(AcausalEquation(
            lhs_var=force_a,
            rhs_fn=lambda vals, _fb=force_b: -vals[_fb],
            depends_on=(force_b,),
            is_through_def=True,
        ))


# ---------------------------------------------------------------------------
# ForceSource
# ---------------------------------------------------------------------------

@dataclass
class ForceSource(AcausalElement):
    """External force from the causal world.

    Port:
        flange -- translational.  The through variable is set directly
        from the causal input array.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.element_type = "force_source"


# ---------------------------------------------------------------------------
# Ground
# ---------------------------------------------------------------------------

@dataclass
class Ground(AcausalElement):
    """Fixed translational reference (pos = 0, vel = 0).

    Port:
        flange -- translational.  All across variables connected to this
        port become grounded (fixed at zero).
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.element_type = "ground"


# ---------------------------------------------------------------------------
# PrescribedMotion
# ---------------------------------------------------------------------------

@dataclass
class PrescribedMotion(AcausalElement):
    """Position driven from a causal input signal.

    Port:
        flange -- translational.  The connected ``pos`` (and ``vel``)
        across variables become causal inputs (removed from the state
        vector).
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.element_type = "prescribed_motion"


# ---------------------------------------------------------------------------
# Sensors (read-only, zero through-variable)
# ---------------------------------------------------------------------------

@dataclass
class PositionSensor(AcausalElement):
    """Reads position at a connection node.

    Port:
        flange -- translational.  This sensor contributes zero force.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.element_type = "sensor"
        self.sensor_output = ("flange", "pos")


@dataclass
class VelocitySensor(AcausalElement):
    """Reads velocity at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.element_type = "sensor"
        self.sensor_output = ("flange", "vel")


@dataclass
class ForceSensor(AcausalElement):
    """Reads net force at a connection node."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.TRANSLATIONAL,
            across_vars=("pos", "vel"),
            through_var="force",
        )
        self.element_type = "sensor"
        self.sensor_output = ("flange", "force")
