"""Standalone muscle models as feedbax Components.

Each muscle model manages its own activation state via StateIndex and
computes force from excitation input using Euler integration.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from feedbax.mechanics.muscles.relu_muscle import ReluMuscle
from feedbax.mechanics.muscles.thelen_muscle import RigidTendonHillMuscleThelen

__all__ = [
    "ReluMuscle",
    "RigidTendonHillMuscleThelen",
]
