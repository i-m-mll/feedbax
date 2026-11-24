"""Make runs from a selection, and move a step's axis between inline values and a Set document."""

from __future__ import annotations

import re
from typing import Any

from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.project import Project, RootDocument, RunBase, StepBase
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode
from feedbax.planner.project_plan import RunKey, _sweep_problem
from feedbax.contracts.project_sweep import (
    DEFAULT_SWEEP_ROW, CooccurringSetTrials, ListedSetTrials, ManualCombination, NamedAxis, SetAxis,
    SetEntryParts, SweepExpansionError, TrialSetField, WeightedSetTrials,
)
from feedbax.contracts.set_document import (
    SET_DOCUMENT_SCHEMA_ID, SetMisfit, axis_from_set, check_fit, entry_pointer, read_set, resolve_set_axes, set_from_axis, sets_offered, step_model,
)
from feedbax.contracts.studio_project import AuthoredFileAddress, CreateNamedDocument, StudioProjectFilePatch
from feedbax.contracts.studio_sets import (
    AddSetEntry, ApplySet, DuplicateRun, InlineSet, RemoveAxisValue, SaveSet, SetToField, SetToSweep,
    StudioAxisValue, StudioSetEdit, StudioSetEditRequest,
    StudioSetsOffered, StudioSetsOfferedRequest,
)
from feedbax.contracts.model_edit import ListedRuns
from feedbax.contracts.studio_trials import StudioSetField, StudioSetFieldEntry
from feedbax.project_members import generated_name
from feedbax.web.services.axis_values import axis_values
from feedbax.web.services.project_folder import ProjectFolderInvalid
from feedbax.web.services.trial_scope import (
    StoredEdit, SweepValuesRuns, _composed, content_edits, file_value, selected_value, selection_scope, preview_stored,
    set_document_write, set_readers,
)


def trial_set_fields(project: Project, step: str) -> list[StudioSetField]:
    fields = []
    for axis, field in project.steps[step].set_fields.items():
        try:
            entries = read_set(project.documents.get, field.axis.set).entries
        except ValueError:
            continue
        sampling = field.sampling
        rows = []
        if isinstance(sampling, WeightedSetTrials):
            rows.append(StudioSetFieldEntry(entry=None, weight=sampling.none))
        for id, entry in entries.items():
            if isinstance(sampling, ListedSetTrials):
                weight = None
            elif isinstance(sampling, CooccurringSetTrials):
                weight = sampling.weights.get(id, SetEntryParts())
            else:
                weight = sampling.weights.get(id, 1.0)
            rows.append(StudioSetFieldEntry(entry=id, label=entry.label, weight=weight))
        fields.append(StudioSetField(axis=axis, set=field.axis.set, sampling=sampling, entries=rows))
    return fields


def sets_offered_to(folders, project_id: str, body: StudioSetsOfferedRequest) -> StudioSetsOffered:
    project = folders.read(project_id, revision=body.revision).project
    if body.step not in project.steps:
        raise ProjectFolderInvalid(f"{body.step} is not a step of this project.")
    return StudioSetsOffered(sets=list(sets_offered(project, body.step)))


def set_edit(folders, project_id: str, body: StudioSetEditRequest) -> StudioSetEdit:
    project = folders.read(project_id, revision=body.revision).project
    command = body.command
    step = command.run.step if isinstance(command, DuplicateRun) and command.run else getattr(command, "step", None)
    if step not in project.steps:
        raise ProjectFolderInvalid(f"{step} is not a step of this project.")
    selected = None
    diagnoses = []
    if isinstance(command, DuplicateRun):
        stored, selected = _duplicate_run(folders, project_id, body.revision, project, command)
    elif isinstance(command, SaveSet):
        stored = _save_set(project, command)
    elif isinstance(command, InlineSet):
        stored, diagnoses = _inline_set(project, command)
    elif isinstance(command, ApplySet):
        stored = _apply_set(project, command)
    elif isinstance(command, SetToField | SetToSweep):
        stored = _move_set(project, command)
    elif isinstance(command, AddSetEntry):
        stored, selected = _add_set_entry(project, command)
    else:
        stored = _remove_value(project, command)
    if isinstance(command, AddSetEntry | ApplySet | RemoveAxisValue | SaveSet):
        preview = preview_stored(project, stored)
        sweep = preview.steps[step].sweep
        if sweep is not None:
            if isinstance(command, ApplySet):
                previous = project.steps[step].sweep
                names = sweep.axes.keys() - (previous.axes.keys() if previous is not None else set())
            else:
                names = {command.axis}
            affected = sweep.model_copy(update={"axes": {name: axis for name, axis in sweep.axes.items() if name in names}})
            try:
                resolve_set_axes(affected, preview.documents.get, step_model(preview, step))
            except SweepExpansionError as error:
                diagnoses.append(_sweep_problem(error, step=step))
    return StudioSetEdit(
        revision=body.revision, files=stored.files, created=list(stored.created), selected=selected,
        problems=[StudioProjectWindowMapProblem.from_diagnostic(problem, declaration="") for problem in diagnoses],
    )


def _step_raw(project: Project, step: str) -> tuple[AuthoredFileAddress, dict[str, Any]]:
    return (
        AuthoredFileAddress(kind="step", id=step),
        project.steps[step].model_dump(mode="json", by_alias=True, exclude_unset=True),
    )


def _authored_axis(project: Project, step: str, axis: str):
    sweep = project.steps[step].sweep
    found = sweep.axes.get(axis) if sweep is not None else None
    if found is None:
        raise ProjectFolderInvalid(f"{axis} is not an axis authored on {step}.")
    return found


def _with_axes(address, raw, axes: dict[str, Any]) -> StudioProjectFilePatch:
    sweep = raw.get("sweep") or {}
    return file_value(address, raw, ("sweep",), sweep | {"axes": axes})


def _copy_key(value: str, taken) -> str:
    """The first free ``<base>-N`` for a copy of ``value``, whose base is ``value`` without its suffix."""
    base = re.sub(r"-\d+$", "", value) or value
    return generated_name(base, lambda key: key == base or taken(key))


def _divided(row: str, axis: str, value: str) -> str:
    """The id of the row an optional axis adds beside ``row`` for ``value``."""
    parts = [] if row == DEFAULT_SWEEP_ROW else row.split("--")
    return "--".join(sorted([*parts, f"{axis}={value}"], key=lambda part: part.partition("=")[0]))


def _recolumned(row: str, axis: str, value: str, copy: str) -> str:
    """The id of ``row`` with its value ``value`` on ``axis`` replaced by ``copy``."""
    return "--".join(f"{axis}={copy}" if part == f"{axis}={value}" else part for part in row.split("--"))


def _kept_rows(rows, axis: str) -> set[str]:
    """The rows an optional axis applies to: each is kept with no value on the axis, beside its copies."""
    ids = {row.key.row for row in rows}
    return {
        row.key.row for row in rows if axis not in row.coordinates and any(
            other.coordinates.get(axis) is not None and _divided(row.key.row, axis, other.coordinates[axis].value_id)
            == other.key.row for other in rows
        )
    } & ids


def _exceptions_copied(sweep: dict[str, Any], copies: dict[str, str]) -> dict[str, Any]:
    """The step's exceptions, each that names an original row by id also naming its copy."""
    exceptions = {}
    for name, exception in sweep.get("exceptions", {}).items():
        where = (exception or {}).get("where") or {}
        if where.get("rows"):
            named = [*where["rows"], *(copies[row] for row in where["rows"] if row in copies)]
            exception = exception | {"where": where | {"rows": list(dict.fromkeys(named))}}
        exceptions[name] = exception
    return exceptions


def _entry_copied(project: Project, name: str, readers, value: str, copy: str) -> list[StudioProjectFilePatch]:
    """Each reader of the set ``name`` with its own exceptions naming the copy of entry ``value`` beside its original."""
    axes = {
        axis for step in readers if (sweep := project.steps[step].sweep) is not None
        for axis, read in sweep.axes.items() if isinstance(read, SetAxis) and read.set == name
    }
    files = []
    for step in sorted(readers):
        address, raw = _step_raw(project, step)
        sweep = raw.get("sweep") or {}
        rows = {
            row for exception in sweep.get("exceptions", {}).values()
            for row in ((exception or {}).get("where") or {}).get("rows") or ()
        }
        copies = {
            row: copied for row in rows for axis in axes
            if (copied := _recolumned(row, axis, value, copy)) != row
        }
        if copies:
            files.append(_with_sweep(address, raw, sweep, copies))
    return files


def _with_sweep(address, raw, sweep: dict[str, Any], copies: dict[str, str]) -> StudioProjectFilePatch:
    exceptions = _exceptions_copied(sweep, copies)
    return file_value(address, raw, ("sweep",), sweep | ({"exceptions": exceptions} if exceptions else {}))


def _duplicate_run(folders, project_id, revision, project: Project, command: DuplicateRun):
    """Copy each selected run: a value is added to an axis, and no existing run changes its id."""
    if command.run is None:
        raise ProjectFolderInvalid("Duplicate Run names a run of the step.")
    plan = folders.read_plan(project_id, revision=revision).plan
    key = RunKey(command.run.variant, command.run.step, command.run.row)
    if key.variant != plan.variants[0]:
        raise ProjectFolderInvalid("Duplicate Run is made in the default variant.")
    step = key.step
    rows = [row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, step)]
    ids = {row.key.row for row in rows}
    if key.row not in ids:
        raise ProjectFolderInvalid("The selected run no longer belongs to this step.")
    selected = set(command.rows) or {key.row}
    whole = not command.rows or selected == ids
    scope = None if whole else selection_scope(project, plan, key, tuple(command.rows), command.axes or None)
    named = selected_value(project, step, scope) if scope is not None else None
    address, raw = _step_raw(project, step)
    sweep = raw.get("sweep") or {}
    axes = sweep.get("axes", {})
    if named is not None and named.set is not None:
        content = _composed(project, named.set)
        entries = content["entries"]
        copy = _copy_key(named.value, entries.__contains__)
        entry = {name: item for name, item in entries[named.value].items() if name != "label"}
        edits = content_edits(content, content | {"entries": entries | {copy: entry}})
        readers = set_readers(project, named.set)
        stored = set_document_write(project, named.set, edits, readers)
        stored = stored.then(_entry_copied(project, named.set, readers, named.value, copy))
        return stored, StudioAxisValue(axis=named.axis, value=copy)
    if named is not None:
        values = axes[named.axis]["values"]
        copy = _copy_key(named.value, values.__contains__)
        value = {name: item for name, item in values[named.value].items() if name != "label"}
        copies = {row: _recolumned(row, named.axis, named.value, copy) for row in selected}
        updated = sweep | {"axes": axes | {named.axis: axes[named.axis] | {"values": values | {copy: value}}}}
        return StoredEdit([_with_sweep(address, raw, updated, copies)]), StudioAxisValue(axis=named.axis, value=copy)
    authored = project.steps[step].sweep
    if authored is not None and isinstance(authored.combination, ManualCombination) or sweep.get("rows"):
        raise ProjectFolderInvalid("Duplicate Run needs a sweep whose axes are crossed.")
    for name, axis in (authored.axes.items() if authored is not None else ()):
        if isinstance(axis, NamedAxis | SetAxis) and axis.optional and selected == _kept_rows(rows, name):
            return _optional_value_added(project, address, raw, name, axis, selected)
    name = generated_name("run", axes.__contains__)
    value = generated_name(name, name.__eq__)
    axis: dict[str, Any] = {"values": {value: {}}, "optional": True}
    if isinstance(scope, SweepValuesRuns):
        axis["where"] = {"rule": scope.rule}
        values = axis_values(rows)
        picked = {
            row.key.row for row in rows
            if all(values.get((row.key, other)) in chosen for other, chosen in scope.axes.items())
        }
    elif isinstance(scope, ListedRuns):
        axis["where"] = {"rows": sorted(set(scope.rows))}
        picked = set(scope.rows)
    elif scope is not None:
        raise ProjectFolderInvalid("Duplicate Run is made for runs of the default variant.")
    else:
        picked = ids
    copies = {row: _divided(row, name, value) for row in picked}
    updated = sweep | {"axes": axes | {name: axis}}
    return StoredEdit([_with_sweep(address, raw, updated, copies)]), StudioAxisValue(axis=name, value=value)


def _optional_value_added(project: Project, address, raw, name: str, axis, kept: set[str]):
    """Another copy of the rows an optional axis applies to: a value with no differences."""
    sweep = raw["sweep"]
    if isinstance(axis, SetAxis):
        return _insert_set_entry(project, name, axis.set)
    values = sweep["axes"][name]["values"]
    value = generated_name(name, lambda key: key == name or key in values)
    updated = sweep | {"axes": sweep["axes"] | {name: sweep["axes"][name] | {"values": values | {value: {}}}}}
    copies = {row: _divided(row, name, value) for row in kept}
    return StoredEdit([_with_sweep(address, raw, updated, copies)]), StudioAxisValue(axis=name, value=value)


def _insert_set_entry(project: Project, axis: str, name: str) -> tuple[StoredEdit, StudioAxisValue]:
    content = _composed(project, name)
    entries = content["entries"]
    value = generated_name(axis, lambda key: key == axis or key in entries)
    edits = content_edits(content, content | {"entries": entries | {value: {"differences": {}}}})
    return set_document_write(project, name, edits, set_readers(project, name)), StudioAxisValue(axis=axis, value=value)


def _add_set_entry(project: Project, command: AddSetEntry) -> tuple[StoredEdit, StudioAxisValue]:
    axis = _authored_axis(project, command.step, command.axis)
    if not isinstance(axis, SetAxis):
        raise ProjectFolderInvalid(f"{command.axis} reads no Set.")
    return _insert_set_entry(project, command.axis, axis.set)


def _save_set(project: Project, command: SaveSet) -> StoredEdit:
    axis = _authored_axis(project, command.step, command.axis)
    if not isinstance(axis, NamedAxis):
        raise ProjectFolderInvalid("Only an axis of named values is saved as a set.")
    model = step_model(project, command.step)
    if model is None:
        raise ProjectFolderInvalid(f"{command.step} reads no Model document of its own.")
    try:
        document = set_from_axis(axis, model[0])
    except ValueError as error:
        raise ProjectFolderInvalid(f"Only Model differences are saved as a set: {error}") from error
    name = generated_name("set", project.documents.__contains__)
    created = CreateNamedDocument(kind="document", id=name, document=RootDocument(
        schema=SET_DOCUMENT_SCHEMA_ID, payload=document.model_dump(mode="json", exclude_unset=True),
    ))
    address, raw = _step_raw(project, command.step)
    reading = (
        {"set": name} | ({"where": axis.where.model_dump(mode="json")} if axis.where is not None else {})
        | ({"optional": True} if axis.optional else {})
    )
    file = file_value(address, raw, ("sweep", "axes", command.axis), reading)
    return StoredEdit([file], (created,))


def _inline_set(project: Project, command: InlineSet) -> tuple[StoredEdit, list[ProjectDiagnostic]]:
    axis = _authored_axis(project, command.step, command.axis)
    if not isinstance(axis, SetAxis):
        raise ProjectFolderInvalid(f"{command.axis} reads no set.")
    model = step_model(project, command.step)
    if model is None:
        raise ProjectFolderInvalid(f"{command.step} reads no Model document of its own.")
    try:
        document = read_set(project.documents.get, axis.set)
        if not document.entries:
            raise ValueError(f"{axis.set} has no entries to inline.")
        inline = axis_from_set(document, *model, axis.where)
        inline.optional = axis.optional
    except ValueError as error:
        raise ProjectFolderInvalid(str(error)) from error
    address, raw = _step_raw(project, command.step)
    file = file_value(address, raw, ("sweep", "axes", command.axis), inline.model_dump(mode="json", exclude_unset=True))
    problems = []
    try:
        check_fit(document, model[1])
    except SetMisfit as error:
        target = next(path for path in inline.values[error.entry].patches if entry_pointer(model[0], path) == error.path)
        problems.append(ProjectDiagnostic(
            ResolutionCode.INVALID_SWEEP, f"sweep.axes.{command.axis}", str(error), step=command.step,
            authored_file=f"steps/{command.step}.json",
            authored_path=pointer(("sweep", "axes", command.axis, "values", error.entry, "patches", target)),
        ))
    return StoredEdit([file]), problems


def _apply_set(project: Project, command: ApplySet) -> StoredEdit:
    if command.set not in sets_offered(project, command.step):
        raise ProjectFolderInvalid(f"No Set {command.set!r} can be referenced by {command.step}.")
    address, raw = _step_raw(project, command.step)
    axes = (raw.get("sweep") or {}).get("axes", {})
    name = generated_name(command.set, axes.__contains__)
    return StoredEdit([_with_axes(address, raw, axes | {name: {"set": command.set}})])


def _move_set(project: Project, command: SetToField | SetToSweep) -> StoredEdit:
    step = project.steps[command.step]
    address, raw = _step_raw(project, command.step)
    field_sets = dict(raw.get("set_fields", {}))
    axes = dict((raw.get("sweep") or {}).get("axes", {}))
    if isinstance(command, SetToField):
        axis = _authored_axis(project, command.step, command.axis)
        if step.kind != "train" or not isinstance(axis, SetAxis):
            raise ProjectFolderInvalid("To Field needs a Train's Set axis.")
        if command.axis in field_sets:
            raise ProjectFolderInvalid(f"{command.axis} already has a Set field.")
        sweep = step.sweep
        assert sweep is not None
        if axis.where is not None or sweep.combination not in (None, "cross") or any(sweep.rows.values()):
            raise ProjectFolderInvalid("The Set axis is used by a row selection or grouped sweep.")
        coordinates = (f"{command.axis}=",)
        if any(
            exception is not None and (
                any(part.startswith(coordinates) for part in key.split("--"))
                or exception.where is not None
            )
            for key, exception in sweep.exceptions.items()
        ):
            raise ProjectFolderInvalid("The Set axis has run exceptions that cannot move into trials.")
        field = TrialSetField(axis=axis, sampling=WeightedSetTrials(kind="weighted"))
        field_sets[command.axis] = field.model_dump(mode="json", exclude_unset=True)
        if isinstance(step.base, StepBase | RunBase):
            axes[command.axis] = None
        else:
            axes.pop(command.axis)
    else:
        if command.axis not in field_sets:
            raise ProjectFolderInvalid(f"{command.axis} is not a Set field of {command.step}.")
        if axes.get(command.axis) is not None:
            raise ProjectFolderInvalid(f"{command.axis} already names a sweep axis.")
        axes[command.axis] = field_sets.pop(command.axis)["axis"]
    fields = file_value(address, raw, ("set_fields",), field_sets)
    sweep_file = _with_axes(address, raw, axes)
    return StoredEdit([fields.model_copy(update={"patches": [*fields.patches, *sweep_file.patches]})])


def _remove_value(project: Project, command: RemoveAxisValue) -> StoredEdit:
    axis = _authored_axis(project, command.step, command.axis)
    address, raw = _step_raw(project, command.step)
    if isinstance(axis, SetAxis):
        content = _composed(project, axis.set)
        entries = content["entries"]
        if command.value not in entries:
            raise ProjectFolderInvalid(f"{axis.set} has no entry {command.value}.")
        if len(entries) == 1 and axis.optional:
            return StoredEdit([_with_axes(address, raw, {
                name: item for name, item in raw["sweep"]["axes"].items() if name != command.axis
            })])
        kept = {name: entry for name, entry in entries.items() if name != command.value}
        return set_document_write(
            project, axis.set, content_edits(content, content | {"entries": kept}), set_readers(project, axis.set),
        )
    if not isinstance(axis, NamedAxis) or axis.values.get(command.value) is None:
        raise ProjectFolderInvalid(f"{command.value} is not a named value of {command.axis}.")
    values = {name: value for name, value in raw["sweep"]["axes"][command.axis]["values"].items() if name != command.value}
    remaining = [value for value in values.values() if value is not None]
    axes = dict(raw["sweep"]["axes"])
    if not remaining or len(remaining) == 1 and not axis.optional and not remaining[0].get("patches") and remaining[
        0
    ].get("value") is None:
        axes.pop(command.axis)
    else:
        axes[command.axis] = axes[command.axis] | {"values": values}
    return StoredEdit([_with_axes(address, raw, axes)])
