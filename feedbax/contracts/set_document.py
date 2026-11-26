"""Set documents: named, shared sweep axes whose entries are differences of the Model a row reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import (
    DifferenceNotApplied, Differences, Document, LoadedModelDocument, Payload, Project, RemoveDocumentValue,
    SetDocumentValue, apply_differences, check_disjoint, json_pointer_parts,
)
from feedbax.contracts.project_inputs import InitialModelReference, LoadedModelReference, NonemptyText, ProjectName, StepReference
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolvedStep, compose_document, compose_step
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import (
    AxisCondition, AxisValue, NamedAxis, RemoveValue, SetAxis, SetValue, Sweep, SweepExpansionCode,
    SweepCoordinate, SweepExpansionError, sweep_value_labels,
)
from feedbax.contracts.task_document import TaskAdditions

SET_DOCUMENT_SCHEMA_ID = "feedbax.set_document.v2"
MODEL_SLOT = "payload.trial_documents.model"
"""Where a Train or Evaluate step reads its Model document."""


class SetEntry(TaskAdditions):
    label: NonemptyText
    differences: Differences = Field(default_factory=dict)

    @model_validator(mode="after")
    def _disjoint(self) -> SetEntry:
        check_disjoint(self.differences)
        return self


class SetDocument(StrictModel):
    entries: dict[ProjectName, SetEntry]


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
    values = {}
    for key, entry in document.entries.items():
        patches = {
            _target(model, path): (
                SetValue(value=difference.value, op="replace" if _present(payload, path) else "add")
                if isinstance(difference, SetDocumentValue) else RemoveValue(op="remove")
            )
            for path, difference in entry.differences.items()
        }
        values[key] = AxisValue(
            label=entry.label, patches=patches, task_variables=entry.task_variables, task_links=entry.task_links,
        )
    return NamedAxis(values=values) if where is None else NamedAxis(values=values, where=where)


def set_from_axis(axis: NamedAxis, model: str) -> SetDocument:
    """The Set document holding an inline axis's patches into ``model``'s payload as differences."""
    entries = {}
    values = ((key, value) for key, value in axis.values.items() if value is not None)
    for number, (key, value) in enumerate(values, 1):
        if value.value is not None:
            raise ValueError(f"value {key!r} carries a coordinate value a set entry cannot hold")
        differences: Differences = {
            entry_pointer(model, target): (
                SetDocumentValue(value=write.value) if isinstance(write, SetValue) else RemoveDocumentValue(op="remove")
            )
            for target, write in value.patches.items() if write is not None
        }
        coordinate = SweepCoordinate(key, value.label, number=number, explicit_null="value" in value.model_fields_set)
        _, label = sweep_value_labels(coordinate)
        entries[key] = SetEntry(
            label=label, differences=differences, task_variables=value.task_variables, task_links=value.task_links,
        )
    return SetDocument(entries=entries)


def read_set(read_document: Callable[[str], Document | None], name: str) -> SetDocument:
    """The composed Set document ``name``; raises ValueError when it is missing or not a Set."""
    composed = compose_document(read_document, name)
    if isinstance(composed, ProjectDiagnostic) or composed.schema_id != SET_DOCUMENT_SCHEMA_ID:
        if isinstance(composed, ProjectDiagnostic):
            raise ValueError(f"Set {name!r}: {composed.detail}")
        raise ValueError(f"{composed.label} is not a Set document")
    return SetDocument.model_validate(composed.payload)


class SetProblem(SweepExpansionError):
    """A Set axis blocked by content at an authored location in the Set."""

    def __init__(self, document: str, location: tuple[str, ...], path: str, detail: str):
        super().__init__(SweepExpansionCode.INVALID_SET, f"{path}.set", detail)
        self.document = document
        self.location = location


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
            if not document.entries and not axis.optional:
                raise SetProblem(axis.set, ("entries",), path, "A required Set has no entries.")
            if model is None:
                raise ValueError("the step reads no Model document")
            check_fit(document, model[1])
            authored = read_document(axis.set)
            assert authored is not None
            resolved[name] = axis_from_set(document, *model, axis.where).model_copy(
                update={"optional": axis.optional, "label": axis.label or authored.label},
            )
        except SetProblem:
            raise
        except SetMisfit as error:
            raise SetProblem(axis.set, ("entries", error.entry, "differences", error.path), path, str(error)) from error
        except ValueError as error:
            raise SweepExpansionError(SweepExpansionCode.INVALID_SET, f"{path}.set", str(error)) from error
    return resolved


def document_subject_model(
    resolved: ResolvedStep, read_document: Callable[[str], Document | None],
) -> tuple[str, Payload] | None:
    """The Model document address of an Evaluate with a document subject."""
    subject = resolved.inputs.get("subject")
    authored = resolved.input_documents.get("subject")
    if authored is None or not isinstance(subject, InitialModelReference | LoadedModelReference):
        return None
    if isinstance(subject, InitialModelReference):
        if subject.source.pointer is None and authored.schema_id == MODEL_DOCUMENT_SCHEMA_ID and isinstance(authored.payload, dict):
            return subject.source.document, authored.payload
        return None
    if authored.schema_id != LOADED_MODEL_SCHEMA_ID:
        return None
    model = LoadedModelDocument.model_validate(authored.payload).model
    if model.pointer is not None or model.unresolved:
        return None
    root = compose_document(read_document, model.document)
    if not isinstance(root, ProjectDiagnostic) and root.schema_id == MODEL_DOCUMENT_SCHEMA_ID and isinstance(root.payload, dict):
        return model.document, root.payload
    return None


def step_model(project: Project, step: str) -> tuple[str, Payload] | None:
    """The authored Model document a Train or Evaluate step's default row reads."""
    active = set()
    while step not in active:
        active.add(step)
        authored = project.steps.get(step)
        if authored is None or authored.kind not in ("train", "evaluate"):
            return None
        template = compose_step(project, step)
        resolved = template if isinstance(template, ProjectDiagnostic) else template.resolve()
        if isinstance(resolved, ProjectDiagnostic):
            return None
        model = model_slot(resolved.payload, resolved.document_ids, resolved.document_pointers)
        if model is not None or authored.kind != "evaluate":
            return model
        subject = resolved.inputs.get("subject")
        if not isinstance(subject, StepReference):
            return document_subject_model(resolved, project.documents.get)
        step = subject.step
    return None


def model_slot(
    payload: Payload, document_ids: Mapping[str, str], document_pointers: Mapping[str, str],
) -> tuple[str, Payload] | None:
    """The whole Model document a resolved step payload reads, with its content."""
    path = "payload." + ".".join(trial_model_path(payload))
    name = document_ids.get(path)
    slots = payload.get("trial_documents", payload)
    content = slots.get("model") if isinstance(slots, dict) else None
    if name is None or path in document_pointers or not isinstance(content, dict):
        return None
    return name, content


def trial_model_path(payload: Payload) -> tuple[str, ...]:
    """The Model slot of a Train or Synthesize payload."""
    return ("model",) if "trial_documents" not in payload and "model" in payload else ("trial_documents", "model")


def sets_offered(project: Project, step: str) -> tuple[str, ...]:
    """The existing Sets an addressable Train or Evaluate sweep may reference."""
    if project.steps[step].kind not in ("train", "evaluate", "synthesize"):
        return ()
    return tuple(sorted(name for name, document in project.documents.items() if document.schema_id == SET_DOCUMENT_SCHEMA_ID))
