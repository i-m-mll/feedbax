"""The sweep over a class-valued Train field: one member per class, and each parameter stored where it is shared.

A member writes its class's key at the field's tag and the parameters set for its runs. The step holds the parameters
every member's class has; a parameter only some have is stored on the members that have it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field

from pydantic import BaseModel, JsonValue

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import SetProjectValue
from feedbax.contracts.project_sweep import AxisValue, NamedAxis, RemoveValue, SetValue, _paths_overlap
from feedbax.contracts.studio_authoring import (
    StudioAuthoringOption, StudioClassField, StudioClassMember, StudioClassSweep, StudioClassSweepChange,
)
from feedbax.contracts.studio_project import StudioProjectFilePatch
from feedbax.studio.authoring_sweeps import FieldSweeps, drop_axis, patch_writes, sweep_paths
from feedbax.studio.selection import StoredEdit

type _Path = tuple[str, ...]


@dataclass(frozen=True)
class ClassField:
    """A class-valued field at ``site`` in the Train definition, its classes by key, and its sweeps on the step.

    ``sweeps`` is absent when the step holds no record of the field to sweep.
    """

    site: _Path
    label: str
    tag: str
    classes: dict[str, type[BaseModel]]
    sweeps: FieldSweeps | None

    def title(self, key: str) -> str:
        model = self.classes.get(key)
        return (model.model_config.get("title") if model else None) or key

    def parameters(self, key: str) -> set[str]:
        return set(self.classes[key].model_fields) - {self.tag}

    @property
    def covered(self) -> bool:
        return bool(self.sweeps and self.sweeps.covering)

    def members(self) -> tuple[str, NamedAxis] | None:
        """The field's sweep for the runs it is read for, when each of its values is one class and its parameters."""
        if self.sweeps is None or len(self.sweeps.covering) != 1 or len(self.sweeps.shown) != 1:
            return None
        name, axis = self.sweeps.shown[0]
        if not isinstance(axis, NamedAxis) or not axis.values or not self.sweeps.may_edit_member(axis) or any(
            entry.value not in self.classes for entry in axis.values.values() if entry is not None
        ):
            return None
        return name, axis

    def view(self, classes: list[str], selected_value: str | None) -> StudioClassField:
        members = self.members()
        sweep = None
        if members is not None:
            name, axis = members
            sweep = StudioClassSweep(id=name, members=[
                StudioClassMember(id=value_id, label=entry.label or self.title(entry.value), value=entry.value)
                for value_id, entry in axis.values.items() if entry is not None
            ])
            order = [member.value for member in sweep.members]
            classes = sorted(classes, key=lambda key: order.index(key) if key in order else len(order))
        return StudioClassField(
            id="/" + "/".join(self.site), label=self.label,
            options=[StudioAuthoringOption(value=key, label=self.title(key)) for key in self.classes],
            classes=classes, sweep=sweep,
            selected_value=selected_value if sweep is not None and selected_value is not None else None,
            can_sweep=bool(self.sweeps and self.sweeps.can_sweep),
        )


@dataclass(frozen=True)
class ClassSweepEdit:
    stored: StoredEdit
    removed: dict[str, list[str]] = dataclass_field(default_factory=dict)


def edit_class_sweep(
    field: ClassField,
    change: StudioClassSweepChange,
    stored: dict[str, JsonValue],
    *,
    scoped: bool,
    store: Callable[[list[OverridePatch]], StoredEdit],
    step_file: Callable[[list[SetProjectValue]], StudioProjectFilePatch],
) -> ClassSweepEdit:
    """The files that make ``change`` to the sweep over ``field``.

    ``stored`` is the step's record of the field. ``store`` stores writes at paths in the Train definition on the step;
    ``step_file`` writes the step's own file.
    """
    edit = change.edit
    sweeps = field.sweeps
    if scoped:
        raise ValueError(f"{field.label} has one kind for every run.")
    if sweeps is None:
        raise ValueError(f"{field.label} has no writable sweep destination.")
    if edit.kind == "to_sweep":
        if not sweeps.can_sweep:
            raise ValueError(f"{field.label} cannot be swept.")
        key = sweeps.target.value
        assert isinstance(key, str)
        entry = sweeps.target.entry(key, field.title(key))
        return ClassSweepEdit(StoredEdit([step_file(sweeps.write_axis(NamedAxis(values={key: entry})))]))
    members = field.members()
    if members is None or not sweeps.simple:
        raise ValueError(f"{field.label} has no editable sweep.")
    name, axis = members
    values = {value_id: entry for value_id, entry in axis.values.items() if entry is not None}
    if edit.kind == "to_value":
        kept = values.get(edit.value_id)
        if kept is None:
            raise ValueError(f"{field.label} has no member {edit.value_id!r}.")
        return ClassSweepEdit(
            store(patch_writes(kept.patches)).then([step_file([drop_axis(name)])]),
            {name: [value_id for value_id in values if value_id != edit.value_id]},
        )
    if edit.kind == "remove_member":
        if edit.value_id not in values or len(values) == 1:
            raise ValueError(f"{field.label} has no other member to keep.")
        rest = NamedAxis(values={value_id: entry for value_id, entry in values.items() if value_id != edit.value_id})
        return ClassSweepEdit(StoredEdit([step_file(sweeps.write_axis(rest))]), {name: [edit.value_id]})
    key = edit.key
    if key not in field.classes or any(entry.value == key for entry in values.values()):
        raise ValueError(f"{field.label} has no class {key!r} to add.")
    keeps = field.parameters(key)
    other_targets = sweep_paths(sweeps.step.sweep, besides=name)
    moved = [
        parameter for parameter in stored if parameter != field.tag and parameter not in keeps
        and not any(_paths_overlap(target, ".".join((*field.site, parameter))) for target in other_targets)
    ]
    moved_values = {
        value_id: _with_moved(field, entry, {parameter: stored[parameter] for parameter in moved})
        for value_id, entry in values.items()
    }
    added = sweeps._fresh_id(key, values)
    after = NamedAxis(values=moved_values | {added: sweeps.target.entry(key, field.title(key))})
    removals = [OverridePatch(op="remove", path=".".join((*field.site, parameter))) for parameter in moved]
    return ClassSweepEdit((store(removals) if removals else StoredEdit([])).then([step_file(sweeps.write_axis(after))]))


def _with_moved(field: ClassField, entry: AxisValue, moved: dict[str, JsonValue]) -> AxisValue:
    """``entry`` holding each of ``moved`` its class has, which the step no longer holds: its own value, or the step's."""
    own = field.parameters(entry.value)
    paths = {".".join((*field.site, parameter)): value for parameter, value in moved.items() if parameter in own}
    patches = {
        path: SetValue(op="add", value=value) for path, value in paths.items() if path not in entry.patches
    } | {
        # A member's removal of a moved parameter is its class default, which the parameter's absence now gives.
        path: patch.model_copy(update={"op": "add"}) if path in paths and isinstance(patch, SetValue) else patch
        for path, patch in entry.patches.items() if not (path in paths and isinstance(patch, RemoveValue))
    }
    return entry.model_copy(update={"patches": patches})
