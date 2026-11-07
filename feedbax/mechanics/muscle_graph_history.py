"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .muscle_graph import VectorElements, JointMuscleGeometry, RadialMuscleGeometry


def _matches_VectorElements_v1(params):
    return set(params) <= {"size"}


def _matches_JointMuscleGeometry_v1(params):
    return set(params) <= {"paths"}


def _matches_RadialMuscleGeometry_v1(params):
    return set(params) <= {"n_pairs"}


HISTORIES = {
    VectorElements: History(
        legacy=(LegacyEndpoint("VectorElements", "1", 1, _matches_VectorElements_v1),)
    ),
    JointMuscleGeometry: History(
        legacy=(LegacyEndpoint("JointMuscleGeometry", "1", 1, _matches_JointMuscleGeometry_v1),)
    ),
    RadialMuscleGeometry: History(
        legacy=(LegacyEndpoint("RadialMuscleGeometry", "1", 1, _matches_RadialMuscleGeometry_v1),)
    ),
}
