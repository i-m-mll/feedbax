"""Find and rewrite every authored reference to a named project member."""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue

from feedbax.contracts.studio_project import MemberFamily

type _Json = Any

_DOCUMENT_KEYED_STEP_FIELDS = ("document_overrides", "model_attachments")
_DOCUMENT_NAME_MEMBERS = ("document",)


def _renamed_document_target(text: str, old: str, new: str) -> str:
    """Rewrite a target that addresses the document by its name, `documents.<name>...`."""
    prefix = f"documents.{old}"
    if text == prefix or text.startswith(prefix + "."):
        return f"documents.{new}" + text.removeprefix(prefix)
    return text


def _renamed_document(value: _Json, old: str, new: str) -> _Json:
    if isinstance(value, dict):
        return {
            _renamed_document_target(key, old, new): (
                new if key in _DOCUMENT_NAME_MEMBERS and item == old else _renamed_document(item, old, new)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_renamed_document(item, old, new) for item in value]
    if isinstance(value, str):
        return _renamed_document_target(value, old, new)
    return value


def _renamed_step_target(text: str, old: str, new: str) -> str:
    prefix = f"steps.{old}"
    if text == prefix or text.startswith(prefix + "."):
        return f"steps.{new}" + text.removeprefix(prefix)
    return text


def _renamed_step_targets(value: _Json, old: str, new: str) -> _Json:
    """Rewrite `steps.<name>...` sweep and variant targets, which are keys or `path` values."""
    if isinstance(value, dict):
        return {
            _renamed_step_target(key, old, new): _renamed_step_targets(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_renamed_step_targets(item, old, new) for item in value]
    if isinstance(value, str):
        return _renamed_step_target(value, old, new)
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
    renamed = _renamed_document(project, old, new)
    renamed["documents"] = _rekey(renamed["documents"], old, new)
    for step in renamed.get("steps", {}).values():
        for field in _DOCUMENT_KEYED_STEP_FIELDS:
            if field in step:
                step[field] = _rekey(step[field], old, new)
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
    documents = project.get("documents")
    renamed = _renamed_step_targets({key: value for key, value in project.items() if key != "documents"}, old, new)
    if documents is not None:
        renamed = {"documents": documents, **renamed}
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

    A document is referred to by document members (`{"document": name, ...}`),
    `documents.<name>...` targets, and the maps keyed by document name. Model addresses
    are relative to their document and carry no name. A step is referred to by `StepBase.step`,
    `StepReference.step` in other steps' inputs, and `steps.<name>...` targets in the
    project's, steps' and variants' sweeps. A step's group membership is its own `group`
    field, so it moves with the step. The presentation is keyed by member name.
    """
    rename = _rename_document if family == "documents" else _rename_step
    return rename(project, presentation, old, new)


def references_member(project: dict[str, JsonValue], family: MemberFamily, name: str) -> bool:
    """Whether any authored reference names the member, whether or not it exists."""
    others = {**project, family: {key: value for key, value in project.get(family, {}).items() if key != name}}
    return rename_member(others, {}, family, name, name + "-")[0] != others
