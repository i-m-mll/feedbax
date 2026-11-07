"""Exact historical graph-boundary endpoints."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .boundaries import ActuationInput, SensorOutput, BoundaryPort

HISTORIES = {
    ActuationInput: History(
        legacy=(LegacyEndpoint("ActuationInput", "1", 1, matches_unversioned=lambda params: True),)
    ),
    SensorOutput: History(
        legacy=(LegacyEndpoint("SensorOutput", "1", 1, matches_unversioned=lambda params: True),)
    ),
    BoundaryPort: History(
        legacy=(LegacyEndpoint("BoundaryPort", "1", 1, matches_unversioned=lambda params: True),)
    ),
}
