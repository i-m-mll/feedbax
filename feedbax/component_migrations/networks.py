"""Frozen component migration records for networks."""

from feedbax.component_registry.history import History, LegacyEndpoint
from feedbax.models.networks import VanillaRNN

HISTORIES = {
    VanillaRNN: History(
        legacy=(LegacyEndpoint("VanillaRNN", "1", 1), LegacyEndpoint("VanillaRNN", None, 1)),
    ),
}
