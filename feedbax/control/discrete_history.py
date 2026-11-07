"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import History, LegacyEndpoint

from .discrete import IntegratorDiscrete, UnitDelay, ZeroOrderHold

HISTORIES = {
    IntegratorDiscrete: History(
        legacy=(LegacyEndpoint('IntegratorDiscrete', '1', 1), LegacyEndpoint('IntegratorDiscrete', None, 1)),
    ),
    UnitDelay: History(
        legacy=(LegacyEndpoint('UnitDelay', '1', 1), LegacyEndpoint('UnitDelay', None, 1)),
    ),
    ZeroOrderHold: History(
        legacy=(LegacyEndpoint('ZeroOrderHold', '1', 1), LegacyEndpoint('ZeroOrderHold', None, 1)),
    ),
}
