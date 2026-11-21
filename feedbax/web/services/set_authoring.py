"""Make runs from a selection, and move a step's axis between inline values and a Set document."""

from __future__ import annotations

import re
from typing import Any

from feedbax.contracts.project import Project, RootDocument
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_sweep import DEFAULT_SWEEP_ROW, ManualCombination, NamedAxis, SetAxis
from feedbax.contracts.set_document import (
    SET_DOCUMENT_SCHEMA_ID, axis_from_set, read_set, set_from_axis, sets_offered, step_model,
)
from feedbax.contracts.studio_project import AuthoredFileAddress, CreateNamedDocument, StudioProjectFilePatch
from feedbax.contracts.studio_sets import (
    ApplySet, DuplicateRun, InlineSet, RemoveAxisValue, SaveSet, StudioAxisValue, StudioSetEdit, StudioSetEditRequest,
    StudioSetsOffered, StudioSetsOfferedRequest,
)
from feedbax.contracts.model_edit import EvaluateAttachmentRows
from feedbax.project_members import generated_name
from feedbax.web.services.project_folder import ProjectFolderInvalid
from feedbax.web.services.trial_scope import (
    StoredEdit, SweepValuesScope, _composed, content_edits, file_value, selected_value, selection_scope,
    set_document_write, set_readers,
)


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
    if isinstance(command, DuplicateRun):
        stored, selected = _duplicate_run(folders, project_id, body.revision, project, command)
    elif isinstance(command, SaveSet):
        stored = _save_set(project, command)
    elif isinstance(command, InlineSet):
        stored = _inline_set(project, command)
    elif isinstance(command, ApplySet):
        stored = _apply_set(project, command)
    else:
        stored = _remove_value(project, command)
    return StudioSetEdit(
        revision=body.revision, files=stored.files, created=list(stored.created), selected=selected,
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
    return generated_name(re.sub(r"-\d+$", "", value) or value, taken)


def _divided(row: str, axis: str, value: str) -> str:
    parts = [] if row == DEFAULT_SWEEP_ROW else row.split("--")
    return "--".join(sorted([*parts, f"{axis}={value}"], key=lambda part: part.partition("=")[0]))


def _duplicate_run(folders, project_id, revision, project: Project, command: DuplicateRun):
    if command.run is None:
        raise ProjectFolderInvalid("Duplicate Run names a run of the step.")
    plan = folders.read_plan(project_id, revision=revision).plan
    key = RunKey(command.run.variant, command.run.step, command.run.row)
    if key.variant != plan.variants[0]:
        raise ProjectFolderInvalid("Duplicate Run is made in the default variant.")
    step = key.step
    rows = {row.key.row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, step)}
    if key.row not in rows:
        raise ProjectFolderInvalid("The selected run no longer belongs to this step.")
    whole = not command.rows or set(command.rows) == rows
    scope = None if whole else selection_scope(project, plan, key, tuple(command.rows), command.axes or None)
    named = selected_value(project, step, scope) if scope is not None else None
    address, raw = _step_raw(project, step)
    if named is not None and named.set is not None:
        content = _composed(project, named.set)
        entries = content["entries"]
        copy = _copy_key(named.value, entries.__contains__)
        entry = {name: item for name, item in entries[named.value].items() if name != "label"}
        edits = content_edits(content, content | {"entries": entries | {copy: entry}})
        stored = set_document_write(project, named.set, edits, set_readers(project, named.set))
        return stored, StudioAxisValue(axis=named.axis, value=copy)
    if named is not None:
        values = raw["sweep"]["axes"][named.axis]["values"]
        copy = _copy_key(named.value, values.__contains__)
        value = {name: item for name, item in values[named.value].items() if name != "label"}
        file = file_value(address, raw, ("sweep", "axes", named.axis, "values", copy), value)
        return StoredEdit([file]), StudioAxisValue(axis=named.axis, value=copy)
    sweep = raw.get("sweep") or {}
    authored = project.steps[step].sweep
    if authored is not None and isinstance(authored.combination, ManualCombination) or sweep.get("rows"):
        raise ProjectFolderInvalid("Duplicate Run needs a sweep whose axes are crossed.")
    axes = sweep.get("axes", {})
    name = generated_name("run", axes.__contains__)
    axis: dict[str, Any] = {"values": {"run": {}, "run-2": {}}}
    divided = rows
    if isinstance(scope, SweepValuesScope):
        axis["where"] = {"axes": {other: sorted(set(ids)) for other, ids in sorted(scope.axes.items())}}
        divided = {
            row.key.row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, step) and all(
                (coordinate := row.coordinates.get(other)) is not None and coordinate.value_id in ids
                for other, ids in scope.axes.items()
            )
        }
    elif isinstance(scope, EvaluateAttachmentRows):
        axis["where"] = {"rows": sorted(set(scope.rows))}
        divided = set(scope.rows)
    elif scope is not None:
        raise ProjectFolderInvalid("Duplicate Run is made for runs of the default variant.")
    exceptions = {}
    for exception_name, exception in sweep.get("exceptions", {}).items():
        where = (exception or {}).get("where") or {}
        if "rows" in where:
            where = where | {"rows": [
                renamed for row in where["rows"]
                for renamed in ((_divided(row, name, "run"), _divided(row, name, "run-2")) if row in divided else (row,))
            ]}
            exception = exception | {"where": where}
        exceptions[exception_name] = exception
    updated = sweep | {"axes": axes | {name: axis}} | ({"exceptions": exceptions} if exceptions else {})
    files = [file_value(address, raw, ("sweep",), updated), *_readers_kept(project, plan, step, divided, name)]
    return StoredEdit(files), StudioAxisValue(axis=name, value="run-2")


def _readers_kept(
    project: Project, plan, step: str, divided: set[str], axis: str,
) -> list[StudioProjectFilePatch]:
    """Keep every step that names a row of ``step`` by id on the original of that row, which becomes ``axis=run``.

    A `StepBase.row` is rewritten. A `StepReference` keeps each selected row's member key, which its reader's rows and
    requests are keyed by, so its `row` shorthand becomes the single row of its `where`.
    """
    def kept(value):
        if isinstance(value, list):
            return [kept(item) for item in value]
        if not isinstance(value, dict):
            return value
        value = {key: kept(item) for key, item in value.items()}
        if value.get("step") != step:
            return value
        if value.get("row") in divided:
            if len(plan.variants) > 1:
                raise ProjectFolderInvalid(f"A step names run {value['row']} of {step} in every variant.")
            value["where"] = {"rows": [{"key": value["row"], "variant": plan.variants[0], "row": value["row"]}]}
            del value["row"]
        where = value.get("where")
        if isinstance(where, dict) and "subset" in where and _composed(project, where["subset"]).get("rows"):
            raise ProjectFolderInvalid(f"Subset {where['subset']} names runs of {step} by id.")
        if isinstance(where, dict) and where.get("rows"):
            value["where"] = where | {"rows": [
                row | {"row": _divided(row["row"], axis, "run")} if row["row"] in divided else row
                for row in where["rows"]
            ]}
        return value

    files = []
    for name in project.steps:
        address, raw = _step_raw(project, name)
        patches = []
        base = raw.get("base")
        if isinstance(base, dict) and base.get("step") == step and base.get("row") in divided:
            patches += file_value(address, raw, ("base", "row"), _divided(base["row"], axis, "run")).patches
        if "inputs" in raw and (inputs := kept(raw["inputs"])) != raw["inputs"]:
            patches += file_value(address, raw, ("inputs",), inputs).patches
        if patches:
            files.append(StudioProjectFilePatch(address=address, patches=patches))
    return files


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
    reading = {"set": name} | ({"where": axis.where.model_dump(mode="json")} if axis.where is not None else {})
    file = file_value(address, raw, ("sweep", "axes", command.axis), reading)
    return StoredEdit([file], (created,))


def _inline_set(project: Project, command: InlineSet) -> StoredEdit:
    axis = _authored_axis(project, command.step, command.axis)
    if not isinstance(axis, SetAxis):
        raise ProjectFolderInvalid(f"{command.axis} reads no set.")
    model = step_model(project, command.step)
    if model is None:
        raise ProjectFolderInvalid(f"{command.step} reads no Model document of its own.")
    try:
        inline = axis_from_set(read_set(project.documents.get, axis.set), *model, axis.where)
    except ValueError as error:
        raise ProjectFolderInvalid(str(error)) from error
    address, raw = _step_raw(project, command.step)
    file = file_value(address, raw, ("sweep", "axes", command.axis), inline.model_dump(mode="json", exclude_unset=True))
    return StoredEdit([file])


def _apply_set(project: Project, command: ApplySet) -> StoredEdit:
    if command.set not in sets_offered(project, command.step):
        raise ProjectFolderInvalid(f"{command.set} does not apply to {command.step}'s Model.")
    address, raw = _step_raw(project, command.step)
    axes = (raw.get("sweep") or {}).get("axes", {})
    name = generated_name(command.set, axes.__contains__)
    return StoredEdit([_with_axes(address, raw, axes | {name: {"set": command.set}})])


def _remove_value(project: Project, command: RemoveAxisValue) -> StoredEdit:
    axis = _authored_axis(project, command.step, command.axis)
    address, raw = _step_raw(project, command.step)
    if isinstance(axis, SetAxis):
        content = _composed(project, axis.set)
        entries = content["entries"]
        if command.value not in entries:
            raise ProjectFolderInvalid(f"{axis.set} has no entry {command.value}.")
        if len(entries) == 1:
            raise ProjectFolderInvalid("A set keeps at least one entry.")
        kept = {name: entry for name, entry in entries.items() if name != command.value}
        return set_document_write(
            project, axis.set, content_edits(content, content | {"entries": kept}), set_readers(project, axis.set),
        )
    if not isinstance(axis, NamedAxis) or axis.values.get(command.value) is None:
        raise ProjectFolderInvalid(f"{command.value} is not a named value of {command.axis}.")
    values = {name: value for name, value in raw["sweep"]["axes"][command.axis]["values"].items() if name != command.value}
    remaining = [value for value in values.values() if value is not None]
    axes = dict(raw["sweep"]["axes"])
    if len(remaining) == 1 and not remaining[0].get("patches") and remaining[0].get("value") is None:
        axes.pop(command.axis)
    else:
        axes[command.axis] = axes[command.axis] | {"values": values}
    return StoredEdit([_with_axes(address, raw, axes)])
