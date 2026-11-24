"""Revision-bound choices for writing a Studio edit."""

from typing import Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.project import Sha256


type StoreKind = Literal[
    "all_readers", "selected_readers", "sweep_value", "set_entry", "row_exception",
    "variant", "other_runs", "whole_step",
]


class StudioStoreChoice(StrictModel):
    id: Sha256
    """Replay the same edit and revision with this opaque id to choose its store."""

    kind: StoreKind
    label: str
    steps: list[str] = Field(min_length=1)
    runs: list[StudioSavedTrainKey] = Field(min_length=1)
