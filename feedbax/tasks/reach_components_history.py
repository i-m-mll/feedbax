"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .reach_components import SimpleReaches, DelayedReaches, Stabilization

HISTORIES = {
    SimpleReaches: History(legacy=(LegacyEndpoint("SimpleReaches", "1", 1),)),
    DelayedReaches: History(legacy=(LegacyEndpoint("DelayedReaches", "1", 1),)),
    Stabilization: History(legacy=(LegacyEndpoint("Stabilization", "1", 1),)),
}
