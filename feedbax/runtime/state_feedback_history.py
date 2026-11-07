"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import History, LegacyEndpoint

from .state_feedback import StateFeedbackSelector

HISTORIES = {
    StateFeedbackSelector: History(
        legacy=(LegacyEndpoint('StateFeedbackSelector', '1', 1), LegacyEndpoint('StateFeedbackSelector', None, 1)),
    ),
}
