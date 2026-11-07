"""Exact historical acausal assembly endpoint."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .system import AcausalSystem

HISTORIES = {
    AcausalSystem: History(
        legacy=(LegacyEndpoint("AcausalSystem", "1", 1, matches_unversioned=lambda params: True),)
    )
}
