"""Sparse curated presentation for portable component owners."""

from dataclasses import dataclass, fields

from feedbax.contracts.graph import GraphUIState
from feedbax.contracts.representation import RepresentationSpec


@dataclass(frozen=True)
class PresentationOverride:
    """Exceptional Studio metadata over the owner-derived presentation."""

    icon: str | None = None
    template_ui_state: GraphUIState | None = None
    representation: RepresentationSpec | None = None

    def as_kwargs(self):
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }
