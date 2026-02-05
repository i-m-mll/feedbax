"""Pre-built effector templates combining muscles with geometry.

Each template is a feedbax Component that wraps multiple muscle models
and a geometry object, converting excitation vectors into joint torques
or Cartesian forces.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from feedbax.mechanics.templates.arm_6muscle import Arm6MuscleRigidTendon
from feedbax.mechanics.templates.pointmass_muscles import PointMass8MuscleRelu

__all__ = [
    "Arm6MuscleRigidTendon",
    "PointMass8MuscleRelu",
]
