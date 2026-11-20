"""Find and rewrite every authored reference to a named project member."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import JsonValue

from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.studio_project import MemberFamily

type _Json = Any
type _Rename = Callable[[str], str]


def generated_name(base: str, taken: Callable[[str], bool]) -> str:
    """The name Studio gives a member it creates: ``base`` while it is free, then the first free ``base-2``, ``base-3``.

    The rule of the interface's ``generatedName`` (``web/src/features/projectFolder/generatedName.ts``).
    """
    name, number = base, 2
    while taken(name):
        name, number = f"{base}-{number}", number + 1
    return name


def _renamed_prefix(family: MemberFamily, old: str, new: str) -> _Rename:
    """Rewrite a sweep target that addresses the member by name, `<family>.<name>...`."""
    prefix = f"{family}.{old}"

    def rename(path: str) -> str:
        if path == prefix or path.startswith(prefix + "."):
            return f"{family}.{new}" + path.removeprefix(prefix)
        return path

    return rename


def _renamed_override_target(old: str, new: str) -> _Rename:
    """Rewrite `steps.<step>.document_overrides.<name>...`, which names the document as a key."""

    def rename(path: str) -> str:
        parts = path.split(".", 4)
        if len(parts) >= 4 and parts[0] == "steps" and parts[2:4] == ["document_overrides", old]:
            parts[3] = new
            return ".".join(parts)
        return path

    return rename


def _renamed_patch_targets(patchset: _Json, rename: _Rename) -> _Json:
    if patchset is None or "patches" not in patchset:
        return patchset
    return {**patchset, "patches": {rename(path): patch for path, patch in patchset["patches"].items()}}


def _copied_patch_targets(patchset: _Json, copy: _Rename) -> _Json:
    """Add, beside each patch whose target ``copy`` rewrites, the same patch at the rewritten target."""
    if patchset is None or "patches" not in patchset:
        return patchset
    patches = patchset["patches"]
    return {**patchset, "patches": {**patches, **{copy(path): patch for path, patch in patches.items() if copy(path) != path}}}


def _mapped_sweep(sweep: _Json, patchset: Callable[[_Json], _Json], path: _Rename) -> _Json:
    """Map the patch sets of a sweep's rows, exceptions and value axes, and the paths of its path axes."""
    if sweep is None:
        return sweep
    sweep = dict(sweep)
    for section in ("rows", "exceptions"):
        if section in sweep:
            sweep[section] = {name: patchset(item) for name, item in sweep[section].items()}
    axes = {}
    for name, axis in sweep.get("axes", {}).items():
        if axis is not None and "path" in axis:
            axis = {**axis, "path": path(axis["path"])}
        elif axis is not None and isinstance(values := axis.get("values"), dict):
            axis = {**axis, "values": {key: patchset(item) for key, item in values.items()}}
        axes[name] = axis
    if "axes" in sweep:
        sweep["axes"] = axes
    return sweep


def _renamed_sweep(sweep: _Json, rename: _Rename) -> _Json:
    """Rewrite the targets of a sweep's rows, exceptions and axes, the only places targets are authored."""
    return _mapped_sweep(sweep, lambda item: _renamed_patch_targets(item, rename), rename)


def _renamed_sweeps(project: dict[str, JsonValue], rename: _Rename) -> dict[str, JsonValue]:
    """Rewrite the targets of the project's sweep (its variants) and of every step's sweep."""
    project = dict(project)
    if "sweep" in project:
        project["sweep"] = _renamed_sweep(project["sweep"], rename)
    if "steps" in project:
        project["steps"] = {
            name: {**step, "sweep": _renamed_sweep(step["sweep"], rename)} if "sweep" in step else step
            for name, step in project["steps"].items()
        }
    return project


def _renamed_document_references(value: _Json, old: str, new: str) -> _Json:
    """Rewrite every `DocumentReference` to the document, at any depth of a binding."""
    if isinstance(value, dict):
        if value.get("document") == old and value.keys() <= DocumentReference.model_fields.keys():
            return {**value, "document": new}
        return {key: _renamed_document_references(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_renamed_document_references(item, old, new) for item in value]
    return value


def _renamed_step_references(value: _Json, old: str, new: str) -> _Json:
    """Rewrite `StepBase.step` and `StepReference.step`, at any depth of a binding."""
    if isinstance(value, dict):
        return {
            key: new if key == "step" and item == old else _renamed_step_references(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_renamed_step_references(item, old, new) for item in value]
    return value


def _rekey(mapping: dict[str, _Json], old: str, new: str) -> dict[str, _Json]:
    return {new if key == old else key: item for key, item in mapping.items()}


def _rename_document(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    renamed = _renamed_document_references({key: value for key, value in project.items() if key != "documents"}, old, new)
    renamed = _renamed_sweeps(renamed, _renamed_prefix("documents", old, new))
    renamed = _renamed_sweeps(renamed, _renamed_override_target(old, new))
    renamed = {"documents": _rekey(project.get("documents", {}), old, new), **renamed}
    for step in renamed.get("steps", {}).values():
        if "document_overrides" in step:
            step["document_overrides"] = _rekey(step["document_overrides"], old, new)
    shown = dict(presentation)
    if "documents" in shown:
        shown["documents"] = _rekey(shown["documents"], old, new)
    if "steps" in shown:
        shown["steps"] = {
            name: {**step, "model_positions": _rekey(step["model_positions"], old, new)}
            if "model_positions" in step else step
            for name, step in shown["steps"].items()
        }
    return renamed, shown


def _rename_step(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    renamed = _renamed_sweeps(project, _renamed_prefix("steps", old, new))
    steps = {}
    for name, step in renamed.get("steps", {}).items():
        step = dict(step)
        for field in ("base", "inputs"):
            if field in step:
                step[field] = _renamed_step_references(step[field], old, new)
        steps[new if name == old else name] = step
    if "steps" in renamed:
        renamed["steps"] = steps
    shown = dict(presentation)
    if "steps" in shown:
        shown["steps"] = _rekey(shown["steps"], old, new)
    return renamed, shown


def rename_member(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue],
    family: MemberFamily, old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return the project and presentation with the member and its references renamed.

    A document is referred to by the project's `DocumentReference`s, `documents.<name>...`
    and `steps.<step>.document_overrides.<name>...` sweep targets, and each step's
    `document_overrides`. Documents' content is left byte for byte, Model addresses are
    relative to their document and carry no name, and `model_attachments` batches are named
    by their author. A step is referred to by `StepBase.step`, `StepReference.step` in other
    steps' inputs, and `steps.<name>...` targets in the project's and steps' sweeps. A step's group membership is its own `group`
    field, so it moves with the step. The presentation is keyed by member name.
    """
    rename = _rename_document if family == "documents" else _rename_step
    return rename(project, presentation, old, new)


def duplicate_step(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return the project and presentation with step ``old`` copied as ``new``, reading the same documents and inputs.

    A step's own sweep addresses the step relative to itself and is copied as it is. Each
    patch the project's sweep makes to ``old`` it also makes to ``new``, so that the copy
    compiles to the same requests as its source; a project sweep axis whose path addresses
    ``old`` is one path and stays on ``old``.
    """
    copied = {**project, "steps": {**project["steps"], new: project["steps"][old]}}
    if "sweep" in project:
        retarget = _renamed_prefix("steps", old, new)
        copied["sweep"] = _mapped_sweep(project["sweep"], lambda item: _copied_patch_targets(item, retarget), lambda path: path)
    shown = dict(presentation)
    if old in shown.get("steps", {}):
        shown["steps"] = {**shown["steps"], new: shown["steps"][old]}
    return copied, shown


def references_member(project: dict[str, JsonValue], family: MemberFamily, name: str) -> bool:
    """Whether any authored reference names the member, whether or not it exists."""
    others = {**project, family: {key: value for key, value in project.get(family, {}).items() if key != name}}
    return rename_member(others, {}, family, name, name + "-")[0] != others
