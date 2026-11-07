"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint

from .channel import Channel

HISTORIES = {
    Channel: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("Channel", "1", 1),
            LegacyEndpoint("Channel", "2", 2),
        ),
    ),
}
