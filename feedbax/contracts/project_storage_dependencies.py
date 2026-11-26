"""Authored key presence that the legacy strict path operations can observe."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from functools import cache

from pydantic import JsonValue, ValidationError

from feedbax.contracts.matrix_core import resolve_segment
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import BasedDocument, InlineBase, Project, RemoveDocumentValue, RootDocument, StepBase
from feedbax.contracts.project import base_chain, json_pointer_parts
from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.project_resolution import compose_document, document_write_target
from feedbax.contracts.project_storage import StoredPath
from feedbax.contracts.project_sweep import GeneratedAxis, NamedAxis, Sweep
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, SetDocument
from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID

type FileAddress = tuple[str, str]

_SCHEMA_KEYS = frozenset({"schema_id", "schema_version"})
_DOCUMENT_FIELDS = DocumentReference.model_fields.keys()
_TASK_SCHEMAS = frozenset({TASK_DOCUMENT_SCHEMA_ID, PYTHON_TASK_SCHEMA_ID})


def _walk(value: JsonValue, path: StoredPath = ()) -> Iterator[tuple[StoredPath, dict[str, JsonValue]]]:
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from _walk(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))


def _position(value: JsonValue, parts: StoredPath, *, by_id: bool) -> StoredPath:
    """The longest authored prefix of a path, with a named list member located by its index."""
    resolved: list[str] = []
    path = ".".join(parts)
    for part in parts:
        try:
            key = resolve_segment(value, part, path=path, by_id=by_id)
        except ValueError:
            break
        if key is None:
            break
        resolved.append(str(key))
        value = value[key]
    return tuple(resolved)


def _sweep_paths(sweep: Sweep | None) -> Iterator[str]:
    if sweep is None:
        return
    sources = [*sweep.rows.values(), *sweep.exceptions.values()]
    for axis in sweep.axes.values():
        if isinstance(axis, GeneratedAxis):
            yield axis.path
        elif isinstance(axis, NamedAxis):
            sources.extend(axis.values.values())
    for source in sources:
        if source is not None:
            yield from (path for path, patch in source.patches.items() if patch is not None)


def protected_project_fields(project: Project) -> dict[FileAddress, set[StoredPath]]:
    """Find exact authored paths to retain before serializing the project.

    This only reads authored rules and document chains. A path with no statically
    known owner can protect just that path in compatible documents.
    """
    protected: defaultdict[FileAddress, set[StoredPath]] = defaultdict(set)
    raw = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    documents = raw.get("documents", {})
    steps = raw.get("steps", {})

    def observe(family: str, name: str, parts: StoredPath, *, by_id: bool = True, subtree: bool = False) -> None:
        root = steps.get(name) if family == "steps" else documents.get(name)
        if root is None or not (position := _position(root, parts, by_id=by_id)):
            return
        paths = protected[family, name]
        paths.add(position)
        if subtree and len(position) == len(parts):
            value = root
            for part in position:
                value = value[int(part)] if isinstance(value, list) else value[part]
            for suffix, child in _walk(value):
                paths.update((*position, *suffix, key) for key in child)

    def document(name: str, parts: StoredPath, *, by_id: bool = True, subtree: bool = False) -> None:
        for ancestor in base_chain(project.documents, name):
            observe("documents", ancestor, ("payload", *parts), by_id=by_id, subtree=subtree)

    def compatible(schemas: frozenset[str], parts: StoredPath) -> None:
        for name, entry in project.documents.items():
            if entry.schema_id in schemas:
                document(name, parts)

    def explicit(path: str) -> bool:
        family, name, *parts = path.split(".")
        if family not in {"documents", "steps"} or name.startswith("?"):
            return False
        if family == "documents" and parts[:1] == ["payload"]:
            document(name, tuple(parts[1:]))
        else:
            observe(family, name, tuple(parts))
        return True

    @cache
    def composed(name: str) -> JsonValue:
        entry = compose_document(project.documents.get, name)
        return entry.payload if isinstance(entry, RootDocument) else None

    @cache
    def lineage(name: str) -> tuple[str, ...]:
        ancestry: list[str] = []
        while name in project.steps and name not in ancestry:
            ancestry.append(name)
            base = project.steps[name].base
            if not isinstance(base, StepBase) or base.unresolved:
                break
            name = base.step
        return tuple(ancestry)

    for _, value in _walk(raw):
        key, pointer = value.get("document"), value.get("pointer")
        if isinstance(key, str) and isinstance(pointer, str) and value.keys() <= _DOCUMENT_FIELDS and not value.get("unresolved"):
            try:
                reference = DocumentReference.model_validate(value)
            except ValidationError:
                continue
            document(key, json_pointer_parts(reference.pointer), by_id=False, subtree=True)

    for name, entry in project.documents.items():
        if isinstance(entry, BasedDocument) and not entry.base.unresolved:
            for pointer, difference in entry.differences.items():
                parts = json_pointer_parts(pointer)
                document(entry.base.document, parts if isinstance(difference, RemoveDocumentValue) else parts[:-1], by_id=False)
        if entry.schema_id == SET_DOCUMENT_SCHEMA_ID:
            try:
                entries = SetDocument.model_validate(composed(name)).entries
            except ValidationError:
                continue
            for item in entries.values():
                for pointer, difference in item.differences.items():
                    parts = json_pointer_parts(pointer)
                    compatible(frozenset({MODEL_DOCUMENT_SCHEMA_ID}), parts if isinstance(difference, RemoveDocumentValue) else parts[:-1])

    for path in _sweep_paths(project.sweep):
        if path.startswith(("steps.", "documents.")):
            explicit(path)

    for name, step in project.steps.items():
        ancestors = lineage(name)
        if not ancestors:
            continue
        base = project.steps[ancestors[-1]].base
        owner = base.document if isinstance(base, DocumentReference) and not base.unresolved else None
        selection = json_pointer_parts(base.pointer) if isinstance(base, DocumentReference) else ()
        payload = base.inline if isinstance(base, InlineBase) else composed(owner) if owner is not None else None
        if owner is not None:
            for part in selection:
                key = resolve_segment(payload, part, path="/".join(selection), by_id=False)
                payload = payload[key] if key is not None else None

        if owner is not None and any(project.steps[ancestor].deltas for ancestor in ancestors):
            for path, value in _walk(payload):
                for key in value.keys() & _SCHEMA_KEYS:
                    document(owner, (*selection, *path, key))
            for ancestor in ancestors:
                for delta in project.steps[ancestor].deltas:
                    for patch in delta.patches:
                        document(owner, (*selection, *patch.path.split(".")))

        for ancestor in ancestors:
            for path in _sweep_paths(project.steps[ancestor].sweep):
                if path.startswith(("steps.", "documents.")) and explicit(path):
                    continue
                try:
                    target = document_write_target(payload, path)
                except (ValueError, ValidationError):
                    continue
                if target is not None:
                    document(target[0], tuple(target[1].split(".")))
                elif owner is not None:
                    document(owner, (*selection, *path.split(".")))
                elif step.kind == "evaluate" and path.startswith("task."):
                    compatible(_TASK_SCHEMAS, tuple(path.split(".")[1:]))

        if step.kind == "evaluate" and isinstance(payload, dict):
            params = payload.get("params")
            variation = params.get("variation") if isinstance(params, dict) else None
            edits = variation.values() if isinstance(variation, dict) else variation if isinstance(variation, list) else ()
            for edit in edits:
                path = edit.get("path") if isinstance(edit, dict) else None
                if isinstance(path, str) and path.startswith("task."):
                    parts = tuple(path.split(".")[1:])
                    try:
                        target = document_write_target(payload, "trial_documents." + path)
                    except (ValueError, ValidationError):
                        target = None
                    if target is None:
                        compatible(_TASK_SCHEMAS, parts)
                    else:
                        document(target[0], tuple(target[1].split(".")))

    return dict(protected)
