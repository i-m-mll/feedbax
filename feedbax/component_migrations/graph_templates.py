"""Frozen component migration records for graph templates."""

from feedbax.compiler.mechanics_templates import (
    MassSpringDamper,
    PointMassWithMuscles,
    TwoLinkArm6Muscle,
)
from feedbax.compiler.templates import RecurrentController, SimpleFeedbackLoop
from feedbax.component_registry.history import History, LegacyEndpoint

HISTORIES = {
    RecurrentController: History(
        legacy=(
            LegacyEndpoint("Recurrent Controller", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    SimpleFeedbackLoop: History(
        legacy=(
            LegacyEndpoint("Simple Feedback Loop", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    MassSpringDamper: History(
        legacy=(
            LegacyEndpoint("MassSpringDamper", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    PointMassWithMuscles: History(
        legacy=(
            LegacyEndpoint("PointMassWithMuscles", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    TwoLinkArm6Muscle: History(
        legacy=(
            LegacyEndpoint("TwoLinkArm6Muscle", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
}
