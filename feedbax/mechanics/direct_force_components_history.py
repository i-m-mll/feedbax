"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .direct_force_components import PointMass, TwoLinkArm


def _matches_PointMass_v1(params):
    return set(params) <= {"dt", "damping", "mass"}


def _matches_TwoLinkArm_v1(params):
    return set(params) <= {"link_lengths", "dt"}


HISTORIES = {
    PointMass: History(legacy=(LegacyEndpoint("PointMass", "1", 1, _matches_PointMass_v1),)),
    TwoLinkArm: History(legacy=(LegacyEndpoint("TwoLinkArm", "1", 1, _matches_TwoLinkArm_v1),)),
}
