"""Frozen component migration records for composites."""

from feedbax.acausal.system import AcausalSystem
from feedbax.component_registry.history import History, LegacyEndpoint
from feedbax.runtime.graph import Subgraph

HISTORIES = {
    Subgraph: History(
        legacy=(
            LegacyEndpoint("Subgraph", "1", 1, matches_unversioned=lambda params: True),
            LegacyEndpoint("Network", None, 1),
        )
    ),
    AcausalSystem: History(
        legacy=(LegacyEndpoint("AcausalSystem", "1", 1, matches_unversioned=lambda params: True),)
    ),
}
