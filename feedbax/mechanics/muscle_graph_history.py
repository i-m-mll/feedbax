"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .muscle_graph import VectorElements, JointMuscleGeometry, RadialMuscleGeometry

HISTORIES = {
    VectorElements: History(legacy=(LegacyEndpoint("VectorElements", "1", 1),)),
    JointMuscleGeometry: History(legacy=(LegacyEndpoint("JointMuscleGeometry", "1", 1),)),
    RadialMuscleGeometry: History(legacy=(LegacyEndpoint("RadialMuscleGeometry", "1", 1),)),
}
