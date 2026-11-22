"""Sparse curated presentation for portable component owners."""

from dataclasses import dataclass, field, fields
import inspect
from pathlib import Path

from feedbax.contracts.graph import GraphUIState
from feedbax.contracts.representation import RepresentationSpec


@dataclass(frozen=True)
class PresentationOverride:
    """Exceptional Studio metadata over the owner-derived presentation."""

    icon: str | None = None
    template_ui_state: GraphUIState | None = None
    representation: RepresentationSpec | None = None
    source: Path | None = field(default=None, compare=False)

    def __post_init__(self):
        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        if self.source is None and not filename.startswith("<"):
            object.__setattr__(self, "source", Path(filename).resolve())

    def as_kwargs(self):
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }
