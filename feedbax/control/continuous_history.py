"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import History, LegacyEndpoint

from .continuous import Derivative, Integrator, StateSpace, TransferFunction

HISTORIES = {
    Integrator: History(
        legacy=(LegacyEndpoint('Integrator', '1', 1), LegacyEndpoint('Integrator', None, 1)),
    ),
    Derivative: History(
        legacy=(LegacyEndpoint('Derivative', '1', 1), LegacyEndpoint('Derivative', None, 1)),
    ),
    StateSpace: History(
        legacy=(LegacyEndpoint('StateSpace', '1', 1), LegacyEndpoint('StateSpace', None, 1)),
    ),
    TransferFunction: History(
        legacy=(LegacyEndpoint('TransferFunction', '1', 1), LegacyEndpoint('TransferFunction', None, 1)),
    ),
}
