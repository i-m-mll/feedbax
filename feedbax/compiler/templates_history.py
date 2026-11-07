"""Exact historical template endpoints."""

from feedbax.component_registry.history import History, LegacyEndpoint
from .templates import RecurrentController, SimpleFeedbackLoop

HISTORIES = {
    RecurrentController: History(
        legacy=(
            LegacyEndpoint("Recurrent Controller", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
    SimpleFeedbackLoop: History(
        legacy=(
            LegacyEndpoint("Simple Feedback Loop", "1", 1, matches_unversioned=lambda params: True),
        )
    ),
}
