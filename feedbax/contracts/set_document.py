"""Set documents: named, shared sweep axes whose entries are differences of the Model a row reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project import (
    DifferenceNotApplied, Differences, Document, Payload, Project, RemoveDocumentValue,
    SetDocumentValue, apply_differences, check_disjoint, json_pointer_parts,
)
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document, compose_step
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import (
    AxisCondition, AxisValue, NamedAxis, RemoveValue, SetAxis, SetValue, Sweep, SweepExpansionCode,
    SweepExpansionError,
)

SET_DOCUMENT_SCHEMA_ID = "feedbax.set_document.v1"
MODEL_SLOT = "payload.trial_documents.model"
"""Where a Train or Evaluate step reads its Model document."""


class SetEntry(StrictModel):
    label: str | None = None
    differences: Differences = Field(default_factory=dict)

    @model_validator(mode="after")
    def _disjoint(self) -> SetEntry:
        check_disjoint(self.differences)
        return self


class SetDocument(StrictModel):
    entries: dict[ProjectName, SetEntry] = Field(min_length=1)


class SetMisfit(ValueError):
    """An entry whose difference at ``path`` does not apply to the Model."""

    def __init__(self, entry: str, path: str, detail: str):
        self.entry = entry
        self.path = path
        super().__init__(f"entry {entry!r} at {path}: {detail}")


def _payload_prefix(model: str) -> str:
    return f"documents.{model}.payload."


def entry_pointer(model: str, target: str) -> str:
    """The pointer into the Model payload that an inline axis patch ``target`` writes."""
    prefix = _payload_prefix(model)
    if not target.startswith(prefix):
        raise ValueError(f"{target!r} is not a patch into the payload of {model}")
    return pointer(tuple(target.removeprefix(prefix).split(".")))


def _target(model: str, path: str) -> str:
    parts = json_pointer_parts(path)
    if any(not part or "." in part for part in parts):
        raise ValueError(f"{path!r} has a key an inline axis cannot name")
    return _payload_prefix(model) + ".".join(parts)


def _present(payload: Payload, path: str) -> bool:
    value: object = payload
    for part in json_pointer_parts(path):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def check_fit(document: SetDocument, payload: Payload) -> None:
    """Raise SetMisfit for the first entry whose differences do not apply to ``payload``."""
    for key, entry in document.entries.items():
        try:
            apply_differences(payload, entry.differences)
        except DifferenceNotApplied as error:
            raise SetMisfit(key, error.path, str(error)) from error


def axis_from_set(
    document: SetDocument, model: str, payload: Payload, where: AxisCondition | None = None,
) -> NamedAxis:
    """The inline axis of patches into ``model``'s payload that applies each entry's differences to ``payload``."""
    check_fit(document, payload)
    values = {}
    for key, entry in document.entries.items():
        patches = {
            _target(model, path): (
                SetValue(value=difference.value, op="replace" if _present(payload, path) else "add")
                if isinstance(difference, SetDocumentValue) else RemoveValue(op="remove")
            )
            for path, difference in entry.differences.items()
        }
        values[key] = AxisValue(patches=patches) if entry.label is None else AxisValue(label=entry.label, patches=patches)
    return NamedAxis(values=values) if where is None else NamedAxis(values=values, where=where)


def set_from_axis(axis: NamedAxis, model: str) -> SetDocument:
    """The Set document holding an inline axis's patches into ``model``'s payload as differences."""
    entries = {}
    for key, value in axis.values.items():
        if value is None:
            continue
        if value.value is not None:
            raise ValueError(f"value {key!r} carries a coordinate value a set entry cannot hold")
        differences: Differences = {
            entry_pointer(model, target): (
                SetDocumentValue(value=write.value) if isinstance(write, SetValue) else RemoveDocumentValue(op="remove")
            )
            for target, write in value.patches.items() if write is not None
        }
        entries[key] = SetEntry(label=value.label, differences=differences)
    return SetDocument(entries=entries)


def read_set(read_document: Callable[[str], Document | None], name: str) -> SetDocument:
    """The composed Set document ``name``; raises ValueError when it is missing or not a Set."""
    composed = compose_document(read_document, name)
    if isinstance(composed, ProjectDiagnostic):
        raise ValueError(f"set {name!r}: {composed.detail}")
    if composed.schema_id != SET_DOCUMENT_SCHEMA_ID:
        raise ValueError(f"{name!r} is not a Set document")
    return SetDocument.model_validate(composed.payload)


def resolve_set_axes(
    sweep: Sweep | None, read_document: Callable[[str], Document | None], model: tuple[str, Payload] | None,
) -> dict[str, NamedAxis]:
    """Each set axis of ``sweep`` as its inline axis against the step's Model document ``model``.

    Raises SweepExpansionError on the axis whose set is missing, does not fit, or has no Model to apply to.
    """
    resolved = {}
    for name, axis in (sweep.axes if sweep is not None else {}).items():
        if not isinstance(axis, SetAxis):
            continue
        path = f"sweep.axes.{name}"
        try:
            document = read_set(read_document, axis.set)
            if model is None:
                raise ValueError("the step reads no Model document")
            resolved[name] = axis_from_set(document, *model, axis.where)
        except SetMisfit as error:
            raise SweepExpansionError(
                SweepExpansionCode.INVALID_SET, f"{path}.set", f"set {axis.set!r}, {error}",
            ) from error
        except ValueError as error:
            raise SweepExpansionError(SweepExpansionCode.INVALID_SET, f"{path}.set", str(error)) from error
    return resolved


def step_model(project: Project, step: str) -> tuple[str, Payload] | None:
    """The Model document a Train or Evaluate step's default row reads, with its composed payload."""
    if (authored := project.steps.get(step)) is None or authored.kind not in ("train", "evaluate"):
        return None
    template = compose_step(project, step)
    resolved = template if isinstance(template, ProjectDiagnostic) else template.resolve()
    if isinstance(resolved, ProjectDiagnostic):
        return None
    return model_slot(resolved.payload, resolved.document_ids, resolved.document_pointers)


def model_slot(
    payload: Payload, document_ids: Mapping[str, str], document_pointers: Mapping[str, str],
) -> tuple[str, Payload] | None:
    """The whole Model document a resolved step payload reads, with its content."""
    name = document_ids.get(MODEL_SLOT)
    slots = payload.get("trial_documents")
    content = slots.get("model") if isinstance(slots, dict) else None
    if name is None or MODEL_SLOT in document_pointers or not isinstance(content, dict):
        return None
    return name, content


def sets_offered(project: Project, step: str) -> tuple[str, ...]:
    """The Set documents whose every entry applies to the step's Model, by name."""
    model = step_model(project, step)
    if model is None:
        return ()
    offered = []
    for name, document in project.documents.items():
        if document.schema_id != SET_DOCUMENT_SCHEMA_ID:
            continue
        try:
            check_fit(read_set(project.documents.get, name), model[1])
        except ValueError:
            continue
        offered.append(name)
    return tuple(sorted(offered))

