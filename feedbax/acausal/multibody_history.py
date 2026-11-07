"""Exact historical identities for the physical element owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .multibody import PlanarLink, RevoluteJoint, WorldFrame, Anchor, MusclePath, PointMarker

HISTORIES = {
    PlanarLink: History(legacy=(LegacyEndpoint("PlanarLink", "1", 1, matches_unversioned=lambda params: True),)),
    RevoluteJoint: History(legacy=(LegacyEndpoint("RevoluteJoint", "1", 1, matches_unversioned=lambda params: True),)),
    WorldFrame: History(legacy=(LegacyEndpoint("WorldFrame", "1", 1, matches_unversioned=lambda params: True),)),
    Anchor: History(legacy=(LegacyEndpoint("Anchor", "1", 1, matches_unversioned=lambda params: True),)),
    MusclePath: History(legacy=(LegacyEndpoint("MusclePath", "1", 1, matches_unversioned=lambda params: True),)),
    PointMarker: History(legacy=(LegacyEndpoint("PointMarker", "1", 1, matches_unversioned=lambda params: True),)),
}
