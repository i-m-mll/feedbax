"""Pre-built native effector graphs combining muscles with geometry.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from feedbax.mechanics.templates.arm_6muscle import Arm6MuscleRigidTendon
from feedbax.mechanics.templates.pointmass_muscles import PointMass8MuscleRelu

__all__ = [
    "Arm6MuscleRigidTendon",
    "PointMass8MuscleRelu",
]
