"""Frozen historical wire endpoints for native component owners."""

from feedbax.component_registry.history import History, LegacyEndpoint

from .affine import AffineFeedbackController

HISTORIES = {
    AffineFeedbackController: History(
        legacy=(LegacyEndpoint('AffineFeedbackController', '1', 1), LegacyEndpoint('AffineFeedbackController', None, 1)),
    ),
}
