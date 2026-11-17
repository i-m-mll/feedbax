"""Governed authored contract for Studio worker training launches."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.method_checkpoint import CheckpointLatestPointer

class PublishedTrainingCheckpoint(StrictModel):
    checkpoint_root: Path
    completed_updates: int = Field(ge=0)
    pointer: CheckpointLatestPointer
