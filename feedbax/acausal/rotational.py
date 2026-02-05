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
# Inertia
# ---------------------------------------------------------------------------

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
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.ROTATIONAL,
            across_vars=("angle", "angular_vel"),
            through_var="torque",
        )
        self.params[f"{name}.inertia"] = inertia

        angle = _fqn(name, "flange", "angle")
        omega = _fqn(name, "flange", "angular_vel")

        # d(angle)/dt = angular_vel
        self.equations.append(AcausalEquation(
            lhs_var=angle,
            rhs_fn=lambda vals, _w=omega: vals[_w],
            depends_on=(omega,),
        ))
        # d(angular_vel)/dt = net_torque / J  (filled by assembly)
        self.equations.append(AcausalEquation(
            lhs_var=omega,
            rhs_fn=None,
            depends_on=(),
        ))
        self.element_type = "mass"  # same assembly treatment as Mass


# ---------------------------------------------------------------------------
# TorsionalSpring
# ---------------------------------------------------------------------------

@dataclass
class TorsionalSpring(AcausalElement):
    """Torsional spring: ``tau = k * (angle_a - angle_b)``.

    Ports:
        flange_a, flange_b -- rotational.
    """

    def __init__(self, name: str, stiffness: float = 1.0):
        super().__init__(name=name)
        for port_name in ("flange_a", "flange_b"):
            self.ports[port_name] = AcausalPort(
                name=port_name,
                domain=Domain.ROTATIONAL,
                across_vars=("angle", "angular_vel"),
                through_var="torque",
            )
        self.params[f"{name}.stiffness"] = stiffness

        ang_a = _fqn(name, "flange_a", "angle")
        ang_b = _fqn(name, "flange_b", "angle")
        tau_a = _fqn(name, "flange_a", "torque")
        tau_b = _fqn(name, "flange_b", "torque")
        k_key = f"{name}.stiffness"

        # tau_b = k * (ang_a - ang_b)  -- spring pulls B toward A
        self.equations.append(AcausalEquation(
            lhs_var=tau_b,
            rhs_fn=lambda vals, _aa=ang_a, _ab=ang_b, _k=k_key: (
                vals[_k] * (vals[_aa] - vals[_ab])
            ),
            depends_on=(ang_a, ang_b),
            is_through_def=True,
        ))
        # tau_a = -tau_b
        self.equations.append(AcausalEquation(
            lhs_var=tau_a,
            rhs_fn=lambda vals, _tb=tau_b: -vals[_tb],
            depends_on=(tau_b,),
            is_through_def=True,
        ))


# ---------------------------------------------------------------------------
# RotationalDamper
# ---------------------------------------------------------------------------

@dataclass
class RotationalDamper(AcausalElement):
    """Rotational damper: ``tau = b * (omega_a - omega_b)``."""

    def __init__(self, name: str, damping: float = 1.0):
        super().__init__(name=name)
        for port_name in ("flange_a", "flange_b"):
            self.ports[port_name] = AcausalPort(
                name=port_name,
                domain=Domain.ROTATIONAL,
                across_vars=("angle", "angular_vel"),
                through_var="torque",
            )
        self.params[f"{name}.damping"] = damping

        omega_a = _fqn(name, "flange_a", "angular_vel")
        omega_b = _fqn(name, "flange_b", "angular_vel")
        tau_a = _fqn(name, "flange_a", "torque")
        tau_b = _fqn(name, "flange_b", "torque")
        b_key = f"{name}.damping"

        # tau_b = b * (omega_a - omega_b)  -- damps B toward A
        self.equations.append(AcausalEquation(
            lhs_var=tau_b,
            rhs_fn=lambda vals, _wa=omega_a, _wb=omega_b, _b=b_key: (
                vals[_b] * (vals[_wa] - vals[_wb])
            ),
            depends_on=(omega_a, omega_b),
            is_through_def=True,
        ))
        # tau_a = -tau_b
        self.equations.append(AcausalEquation(
            lhs_var=tau_a,
            rhs_fn=lambda vals, _tb=tau_b: -vals[_tb],
            depends_on=(tau_b,),
            is_through_def=True,
        ))


# ---------------------------------------------------------------------------
# TorqueSource
# ---------------------------------------------------------------------------

@dataclass
class TorqueSource(AcausalElement):
    """External torque from the causal world."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.ROTATIONAL,
            across_vars=("angle", "angular_vel"),
            through_var="torque",
        )
        self.element_type = "force_source"


# ---------------------------------------------------------------------------
# RotationalGround
# ---------------------------------------------------------------------------

@dataclass
class RotationalGround(AcausalElement):
    """Fixed rotational reference (angle = 0, angular_vel = 0)."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.ports["flange"] = AcausalPort(
            name="flange",
            domain=Domain.ROTATIONAL,
            across_vars=("angle", "angular_vel"),
            through_var="torque",
        )
        self.element_type = "ground"


# ---------------------------------------------------------------------------
# GearRatio
# ---------------------------------------------------------------------------

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
