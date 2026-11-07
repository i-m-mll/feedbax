"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .analytical_component import AnalyticalMusculoskeletalPlant

HISTORIES = {
    AnalyticalMusculoskeletalPlant: History(
        legacy=(LegacyEndpoint("AnalyticalMusculoskeletalPlant", "1", 1),)
    ),
}
