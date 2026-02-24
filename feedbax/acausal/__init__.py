"""Acausal modeling framework for feedbax.

Provides equation-based (Modelica-style) component descriptions that are
assembled into JAX-traceable vector fields at construction time.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from feedbax.acausal.base import (
    AcausalConnection,
    AcausalElement,
    AcausalEquation,
    AcausalPort,
    AcausalVar,
    Domain,
    StateLayout,
)
from feedbax.acausal.system import AcausalParams, AcausalSystem, AcausalSystemState
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
from feedbax.acausal.rotational import (
    GearRatio,
    Inertia,
    RotationalDamper,
    RotationalGround,
    TorqueSource,
    TorsionalSpring,
)
