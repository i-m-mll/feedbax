"""Authored sweep grammar and static inheritance for project steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, TypeVar

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_inputs import ProjectName

type PatchPath = Annotated[str, Field(pattern=r"^[^.]+(?:\.[^.]+)*$")]


class SetValue(StrictModel):
    value: JsonValue
    op: Literal["add", "replace"] = "replace"


class RemoveValue(StrictModel):
    op: Literal["remove"]


type PatchValue = SetValue | RemoveValue


def sweep_patches(patches: Mapping[str, PatchValue | None]) -> list[OverridePatch]:
    """Adapt authored path-keyed patches to the common patch engine."""
    return [
        OverridePatch(path=path, op=patch.op, value=patch.value)
        if isinstance(patch, SetValue)
        else OverridePatch(path=path, op="remove")
        for path, patch in patches.items()
        if patch is not None
    ]


class AxisValue(StrictModel):
    label: str | None = None
    patches: dict[PatchPath, PatchValue | None] = Field(default_factory=dict)


class Row(AxisValue):
    source: ProjectName | None = Field(default=None, alias="from")


class NamedAxis(StrictModel):
    values: dict[ProjectName, AxisValue | None]


class InputAxis(StrictModel):
    input: ProjectName


class ListedAxis(StrictModel):
    path: PatchPath
    values: list[JsonValue] = Field(min_length=1)


class RangeValues(StrictModel):
    start: float
    stop: float
    count: int = Field(ge=1)
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _check_log_range(self) -> RangeValues:
        if self.scale == "log" and min(self.start, self.stop) <= 0:
            raise ValueError("a logarithmic range requires positive bounds")
        return self


class RangeAxis(StrictModel):
    path: PatchPath
    range: RangeValues


class SampleValues(StrictModel):
    distribution: ProjectName
    count: int = Field(ge=1)
    seed: int


class SampledAxis(StrictModel):
    path: PatchPath
    sampler: SampleValues


type Axis = NamedAxis | InputAxis | ListedAxis | RangeAxis | SampledAxis


class ManualCombination(StrictModel):
    manual: list[dict[ProjectName, ProjectName]]


class AxisGroup(StrictModel):
    axes: list[ProjectName] = Field(min_length=1)
    mode: Literal["cross", "zip"]


class GroupCombination(StrictModel):
    groups: dict[ProjectName, AxisGroup]


type Combination = Literal["cross", "zip"] | ManualCombination | GroupCombination


class Sweep(StrictModel):
    axes: dict[ProjectName, Axis | None] = Field(default_factory=dict)
    rows: dict[ProjectName, Row | None] = Field(default_factory=dict)
    combination: Combination | None = None
    inherit_rows: bool = True


_T = TypeVar("_T")


def merge_named(parent: Mapping[str, _T], child: Mapping[str, _T | None]) -> dict[str, _T]:
    """Apply ordered whole-value changes to a name-keyed collection."""
    merged = dict(parent)
    for name, entry in child.items():
        if entry is None:
            merged.pop(name, None)
        else:
            merged[name] = entry
    return merged


def _merge_value(parent: AxisValue | None, child: AxisValue) -> AxisValue:
    if parent is None:
        return child.model_copy(update={"patches": merge_named({}, child.patches)})
    return child.model_copy(
        update={
            "label": child.label if "label" in child.model_fields_set else parent.label,
            "patches": merge_named(
                {path: patch for path, patch in parent.patches.items() if patch is not None},
                child.patches,
            ),
        }
    )


def _derive_row(parent: Row, child: Row) -> AxisValue:
    """Compose a source row's writes with a derived row's writes at the same path."""
    combined = _merge_value(parent, child)
    for path, write in child.patches.items():
        previous = parent.patches.get(path)
        if isinstance(previous, SetValue) and previous.op == "add":
            if isinstance(write, SetValue):
                combined.patches[path] = write.model_copy(update={"op": "add"})
            elif isinstance(write, RemoveValue):
                combined.patches.pop(path, None)
        elif isinstance(previous, RemoveValue) and isinstance(write, SetValue):
            combined.patches[path] = write.model_copy(update={"op": "replace"})
    return combined


class SweepInheritanceError(ValueError):
    def __init__(self, row: str, source: str):
        self.row = row
        self.source = source
        super().__init__(f"row {row!r} derives from missing or later row {source!r}")


def inherit_sweep(parent: Sweep | None, child: Sweep | None) -> Sweep | None:
    """Resolve by-id refinements and earlier row sources without expanding axes."""
    if child is None:
        return parent
    previous = parent or Sweep()
    axes = merge_named(
        {name: axis for name, axis in previous.axes.items() if axis is not None}, child.axes
    )
    for name, axis in child.axes.items():
        if isinstance(axis, NamedAxis):
            prior = previous.axes.get(name)
            values = (
                {key: value for key, value in prior.values.items() if value is not None}
                if isinstance(prior, NamedAxis)
                else {}
            )
            for key, value in axis.values.items():
                if value is None:
                    values.pop(key, None)
                else:
                    values[key] = _merge_value(values.get(key), value)
            axes[name] = NamedAxis(values=values)

    prior_rows = {name: row for name, row in previous.rows.items() if row is not None}
    rows = dict(prior_rows) if child.inherit_rows else {}
    sources = dict(prior_rows)
    for name, row in child.rows.items():
        if row is None:
            rows.pop(name, None)
            sources.pop(name, None)
            continue
        source = row.source
        if source is not None and source not in sources:
            raise SweepInheritanceError(name, source)
        combined = (
            _derive_row(sources[source], row)
            if source is not None
            else _merge_value(rows.get(name), row)
        )
        resolved = Row(label=combined.label, patches=combined.patches)
        rows[name] = sources[name] = resolved

    return Sweep(
        axes=axes,
        rows=rows,
        combination=(
            child.combination if "combination" in child.model_fields_set else previous.combination
        ),
    )
