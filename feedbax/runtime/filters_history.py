"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import Change, History, LegacyEndpoint

from .filters import BandPassFilter, FirstOrderFilter, HighPassFilter

HISTORIES = {
    FirstOrderFilter: History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("FirstOrderFilter", "1", 1),
            LegacyEndpoint("FirstOrderFilter", "2", 2),
        ),
    ),
    HighPassFilter: History(
        legacy=(LegacyEndpoint('HighPassFilter', '1', 1), LegacyEndpoint('HighPassFilter', None, 1)),
    ),
    BandPassFilter: History(
        legacy=(LegacyEndpoint('BandPassFilter', '1', 1), LegacyEndpoint('BandPassFilter', None, 1)),
    ),
}
