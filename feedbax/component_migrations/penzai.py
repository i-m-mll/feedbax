"""Frozen component migration records for penzai."""

from feedbax.component_registry.history import History, LegacyEndpoint
from feedbax.components.penzai import PenzaiAdapter


def _matches_PenzaiAdapter_v1(params):
    return set(params) <= {"output_port", "input_port", "builder_name"}


HISTORIES = {
    PenzaiAdapter: History(
        legacy=(
            LegacyEndpoint("PenzaiAdapter", "1", 1, _matches_PenzaiAdapter_v1),
            LegacyEndpoint("PenzaiSubgraph", "1", 1, _matches_PenzaiAdapter_v1),
        )
    ),
}
