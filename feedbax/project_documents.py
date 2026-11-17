"""Find and rewrite every authored reference to a project document by name."""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue

type _Json = Any

_DOCUMENT_KEYED_STEP_FIELDS = ("document_overrides", "model_attachments")
_NAME_MEMBERS = ("document",)


def _renamed_target(text: str, old: str, new: str) -> str:
    """Rewrite a target that addresses the document by its name, `documents.<name>...`."""
    prefix = f"documents.{old}"
    if text == prefix or text.startswith(prefix + "."):
        return f"documents.{new}" + text.removeprefix(prefix)
    return text


def _renamed(value: _Json, old: str, new: str) -> _Json:
    if isinstance(value, dict):
        return {
            _renamed_target(key, old, new): (
                new if key in _NAME_MEMBERS and item == old else _renamed(item, old, new)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_renamed(item, old, new) for item in value]
    if isinstance(value, str):
        return _renamed_target(value, old, new)
    return value


def _rekey(mapping: dict[str, _Json], old: str, new: str) -> dict[str, _Json]:
    return {new if key == old else key: item for key, item in mapping.items()}


def rename_document(
    project: dict[str, JsonValue], presentation: dict[str, JsonValue], old: str, new: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return the project and presentation with the document and its references renamed.

    References are document members (`{"document": name, ...}`), `documents.<name>...`
    targets, and the maps keyed by document name. Model addresses are relative to their
    document and carry no name.
    """
    renamed = _renamed(project, old, new)
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


def references_document(project: dict[str, JsonValue], name: str) -> bool:
    """Whether any authored reference names the document, whether or not it exists."""
    others = {**project, "documents": {key: value for key, value in project.get("documents", {}).items() if key != name}}
    return rename_document(others, {}, name, name + "-")[0] != others
