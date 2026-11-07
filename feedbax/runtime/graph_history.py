"""Exact historical nested graph endpoint."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .graph import Subgraph

HISTORIES = {
    Subgraph: History(
        legacy=(LegacyEndpoint("Subgraph", "1", 1, matches_unversioned=lambda params: True),)
    )
}
