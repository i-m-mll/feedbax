"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .direct_force_components import PointMass, TwoLinkArm

HISTORIES = {
    PointMass: History(legacy=(LegacyEndpoint("PointMass", "1", 1),)),
    TwoLinkArm: History(legacy=(LegacyEndpoint("TwoLinkArm", "1", 1),)),
}
