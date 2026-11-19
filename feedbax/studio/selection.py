"""The runs a Studio edit is made for."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_plan import PlannedRow
from feedbax.contracts.studio_project import StudioProjectFilePatch


@dataclass(frozen=True)
class TrainSelection:
    """The runs a Train edit is made for, and where an edit made for them is stored.

    ``payloads`` are the selected runs' definitions, none when no run is named. ``current`` is the definition a write
    for them is made against: the named run's when they are rows or sweep values, else the step's own. ``scoped`` says
    they are not the whole step.
    ``store`` puts writes at paths in the step's definition where the scope keeps them: on the step, in a sweep value's
    patches, as an exception, or on a variant; a write through a document reference is stored in the document or as a
    difference from it.
    """

    payloads: tuple[dict[str, JsonValue], ...]
    row: PlannedRow | None
    current: dict[str, JsonValue] | None
    scoped: bool
    store: Callable[[list[OverridePatch]], list[StudioProjectFilePatch]]


def merged_files(files: list[StudioProjectFilePatch]) -> list[StudioProjectFilePatch]:
    """One patch per file, holding its patches in order."""
    merged: dict[tuple[str, str], StudioProjectFilePatch] = {}
    for file in files:
        key = (file.address.kind, file.address.id)
        prior = merged.get(key)
        merged[key] = file if prior is None else prior.model_copy(update={"patches": [*prior.patches, *file.patches]})
    return list(merged.values())
