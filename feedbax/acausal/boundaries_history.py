"""Exact historical graph-boundary endpoints."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .boundaries import ActuationInput, SensorOutput, BoundaryPort

HISTORIES = {
    owner: History(legacy=(LegacyEndpoint(owner.__name__, "1", 1, matches_unversioned=lambda params: True),))
    for owner in (ActuationInput, SensorOutput, BoundaryPort)
}
