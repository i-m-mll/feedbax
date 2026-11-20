"""The runs a Studio edit is made for."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

from pydantic import JsonValue

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_plan import PlannedRow
from feedbax.contracts.project_sweep import AxisCondition
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch


@dataclass(frozen=True)
class StoredEdit:
    """The files an edit writes: the documents it creates, then the patches of files, which may name them.

    ``reached`` are the steps whose definition the edit may change.
    """

    files: list[StudioProjectFilePatch]
    created: tuple[CreateNamedDocument, ...] = ()
    reached: frozenset[str] = frozenset()

    def then(self, files: Collection[StudioProjectFilePatch]) -> StoredEdit:
        return StoredEdit(merged_files([*self.files, *files]), self.created, self.reached)

    def __add__(self, other: StoredEdit) -> StoredEdit:
        return StoredEdit(
            merged_files([*self.files, *other.files]), (*self.created, *other.created), self.reached | other.reached,
        )


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
    store: Callable[[list[OverridePatch]], StoredEdit]
    values: AxisCondition | None = None
    """The selected runs as a set of values per sweep of the step, when they are exactly those values' runs."""


def merged_files(files: list[StudioProjectFilePatch]) -> list[StudioProjectFilePatch]:
    """One patch per file, holding its patches in order."""
    merged: dict[tuple[str, str], StudioProjectFilePatch] = {}
    for file in files:
        key = (file.address.kind, file.address.id)
        prior = merged.get(key)
        merged[key] = file if prior is None else prior.model_copy(update={"patches": [*prior.patches, *file.patches]})
    return list(merged.values())
