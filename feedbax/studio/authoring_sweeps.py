"""Revision-bound sweep recipes for the numeric controls in a direct Train definition."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass

from feedbax.contracts.project import Step, json_pointer_parts
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import (
    Axis,
    AxisValue,
    GeneratedAxis,
    ListedAxis,
    NamedAxis,
    RangeAxis,
    SampledAxis,
    SetValue,
    axis_to_list,
)
from feedbax.contracts.run_matrix import _paths_overlap
from feedbax.contracts.studio_authoring import (
    StudioAuthoringNumber,
    StudioAuthoringSweepChange,
    StudioNameSweepValue,
    StudioSetSweepValue,
    StudioSweepEdit,
    StudioSweepList,
    StudioSweepRange,
    StudioSweepSampler,
    StudioSweepValue,
    StudioToField,
)
from feedbax.contracts.studio_project import SetProjectValue


@dataclass(frozen=True)
class SweepTarget:
    """The ordinary scalar write expressed as step-local sweep patches."""

    value: int | float
    writes: Callable[[int | float], list[SetProjectValue]]

    def entry(self, value: int | float, label: str | None = None) -> AxisValue:
        return AxisValue(
            label=label,
            value=value,
            patches={
                ".".join(json_pointer_parts(patch.path)[2:]):
                    SetValue(op=patch.op, value=patch.value)
                for patch in self.writes(value)
            },
        )

    @classmethod
    def from_field(
        cls, value: object, control: object, writes: Callable[[int | float], list[SetProjectValue]],
    ) -> SweepTarget | None:
        if not isinstance(control, StudioAuthoringNumber) or type(value) not in (int, float):
            return None
        patches = writes(value)
        if not patches or any(
            patch.value != value or "." in patch.path
            for patch in patches
        ):
            return None
        return cls(value, writes)


def _axis_paths(axis: Axis) -> set[str]:
    if isinstance(axis, GeneratedAxis):
        return {axis.path}
    if isinstance(axis, NamedAxis):
        return {path for entry in axis.values.values() if entry is not None for path in entry.patches}
    return set()


def _matches(paths: set[str], other: set[str]) -> bool:
    return any(_paths_overlap(left, right) for left in paths for right in other)


class TrainFieldSweeps:
    def __init__(self, step: Step, target: SweepTarget, label: str):
        self.step = step
        self.target = target
        self.label = label
        self.paths = set(target.entry(target.value).patches)
        self.axes = step.sweep.axes if step.sweep else {}
        self.covering = [
            (name, axis) for name, axis in self.axes.items()
            if axis is not None and _matches(self.paths, _axis_paths(axis))
        ]
        self.simple = (
            step.sweep is None or (
                step.sweep.combination in (None, "cross")
                and not step.sweep.rows
                and not step.sweep.exceptions
            )
        )

    @property
    def can_sweep(self) -> bool:
        return self.simple and not self.covering

    def _editable_list(self, axis: NamedAxis) -> bool:
        return self.simple and all(
            entry is not None
            and type(entry.value) in (int, float)
            and entry.patches == self.target.entry(entry.value).patches
            for entry in axis.values.values()
        )

    def _can_list(self, axis: GeneratedAxis) -> bool:
        return self.simple and _axis_paths(axis) == self.paths

    def view(self) -> StudioSweepList | StudioSweepRange | StudioSweepSampler | None:
        if len(self.covering) != 1:
            return None
        name, axis = self.covering[0]
        if isinstance(axis, RangeAxis):
            return StudioSweepRange(kind="range", id=name, rule=axis.range, can_list=self._can_list(axis))
        if isinstance(axis, SampledAxis):
            return StudioSweepSampler(
                kind="sampler", id=name, rule=axis.sampler, can_list=axis.draws is not None and self._can_list(axis),
            )
        editable = isinstance(axis, NamedAxis) and self._editable_list(axis)
        if isinstance(axis, ListedAxis):
            axis = axis_to_list(axis)
        if not isinstance(axis, NamedAxis) or not axis.values:
            return None
        values = []
        for value_id, entry in axis.values.items():
            if entry is None or type(entry.value) not in (int, float):
                return None
            values.append(StudioSweepValue(id=value_id, label=entry.label, value=entry.value))
        return StudioSweepList(kind="list", id=name, values=values, editable=editable)

    @staticmethod
    def _fresh_id(base: str, existing: Collection[str]) -> str:
        if base not in existing:
            return base
        counter = 2
        while f"{base}-{counter}" in existing:
            counter += 1
        return f"{base}-{counter}"

    @staticmethod
    def _axis_write(name: str, axis: NamedAxis) -> SetProjectValue:
        return SetProjectValue(
            op="replace", path=pointer(("sweep", "axes", name)),
            value=axis.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
        )

    def edit(self, command: StudioAuthoringSweepChange, validate_number: Callable[[int | float], int | float]) -> list[SetProjectValue]:
        edit = command.edit
        if edit.kind == "toSweep" and self.can_sweep:
            base = re.sub(r"[^a-z0-9]+", "-", self.label.lower()).strip("-") or "value"
            name = self._fresh_id(base, self.axes)
            axis = NamedAxis(values={"value": self.target.entry(self.target.value)})
            write = self._axis_write(name, axis)
            if self.step.sweep is None:
                return [SetProjectValue(
                    op="replace" if "sweep" in self.step.model_fields_set else "add",
                    path="/sweep", value={"axes": {name: write.value}},
                )]
            if "axes" not in self.step.sweep.model_fields_set:
                return [SetProjectValue(op="add", path="/sweep/axes", value={name: write.value})]
            return [SetProjectValue(op="add", path=write.path, value=write.value)]

        if edit.kind != "toSweep" and len(self.covering) == 1:
            name, axis = self.covering[0]
            if name == edit.sweep_id:
                return self._edit_axis(name, axis, edit, validate_number)
        raise ValueError(f"{self.label} has no supported sweep edit")

    def _edit_axis(
        self,
        name: str,
        axis: Axis,
        edit: StudioSweepEdit | StudioToField | StudioSetSweepValue | StudioNameSweepValue,
        validate_number: Callable[[int | float], int | float],
    ) -> list[SetProjectValue]:
        if edit.kind == "toList" and isinstance(axis, (RangeAxis, SampledAxis, ListedAxis)) and self._can_list(axis):
            return [self._axis_write(name, axis_to_list(axis))]
        if not isinstance(axis, NamedAxis) or not self._editable_list(axis):
            raise ValueError(f"{self.label} has no editable list")
        values = axis.values
        if edit.kind == "addValue":
            value_id = self._fresh_id("value", values)
            return [SetProjectValue(
                op="add", path=pointer(("sweep", "axes", name, "values", value_id)),
                value=self.target.entry(self.target.value).model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            )]
        if edit.kind == "toList" or (entry := values.get(edit.value_id)) is None:
            raise ValueError(f"{self.label} has no such sweep value")
        path = ("sweep", "axes", name, "values", edit.value_id)
        if edit.kind == "setSweepValue":
            number = validate_number(edit.value)
            return [
                SetProjectValue(op="replace", path=pointer((*path, "value")), value=number),
                *(
                    SetProjectValue(op="replace", path=pointer((*path, "patches", key, "value")), value=number)
                    for key in entry.patches
                ),
            ]
        if edit.kind == "nameSweepValue":
            return [SetProjectValue(
                op="replace" if "label" in entry.model_fields_set else "add",
                path=pointer((*path, "label")), value=edit.label,
            )]
        # A null axis preserves the authored position and the inverse edit.
        # It is omitted from the compiled product by the sweep grammar.
        return [
            *self.target.writes(entry.value),
            SetProjectValue(op="replace", path=pointer(("sweep", "axes", name)), value=None),
        ]
