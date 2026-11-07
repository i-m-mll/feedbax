"""Exact historical template endpoints."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .mechanics_templates import MassSpringDamper, PointMassWithMuscles, TwoLinkArm6Muscle

HISTORIES = {
    MassSpringDamper: History(legacy=(LegacyEndpoint('MassSpringDamper', "1", 1, matches_unversioned=lambda params: True),)),
    PointMassWithMuscles: History(legacy=(LegacyEndpoint('PointMassWithMuscles', "1", 1, matches_unversioned=lambda params: True),)),
    TwoLinkArm6Muscle: History(legacy=(LegacyEndpoint('TwoLinkArm6Muscle', "1", 1, matches_unversioned=lambda params: True),)),
}
