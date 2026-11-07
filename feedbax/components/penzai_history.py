"""Exact pre-revision wire endpoints for portable owners."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .penzai import PenzaiAdapter

HISTORIES = {
    PenzaiAdapter: History(legacy=(LegacyEndpoint("PenzaiAdapter", "1", 1),)),
}
