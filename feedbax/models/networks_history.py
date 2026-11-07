"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import History, LegacyEndpoint

from .networks import VanillaRNN

HISTORIES = {
    VanillaRNN: History(
        legacy=(LegacyEndpoint('VanillaRNN', '1', 1), LegacyEndpoint('VanillaRNN', None, 1)),
    ),
}
