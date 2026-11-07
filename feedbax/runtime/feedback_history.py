"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import History, LegacyEndpoint

from .feedback import AssembleFeedback, SelectState

HISTORIES = {
    SelectState: History(
        legacy=(LegacyEndpoint('SelectState', '1', 1), LegacyEndpoint('SelectState', None, 1)),
    ),
    AssembleFeedback: History(
        legacy=(LegacyEndpoint('AssembleFeedback', '1', 1), LegacyEndpoint('AssembleFeedback', None, 1)),
    ),
}
