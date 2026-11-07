"""Exact historical template endpoints."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .muscle_templates import Arm6MuscleRigidTendon, PointMass8MuscleRelu

HISTORIES = {
    Arm6MuscleRigidTendon: History(
        legacy=(
            LegacyEndpoint(
                "Arm6MuscleRigidTendon", "1", 1, matches_unversioned=lambda params: True
            ),
        )
    ),
    PointMass8MuscleRelu: History(
        legacy=(
            LegacyEndpoint("PointMass8MuscleRelu", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
}
