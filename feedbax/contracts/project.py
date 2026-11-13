"""The semantic project shared by text authoring and Studio."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, StringConstraints, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.matrix_core import apply_value_patch_in_place, ensure_value_collection
from feedbax.contracts.model_attachment import ModelAttachmentDifference
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, TaskDocument,
)
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.project_inputs import InputBinding, NonemptyText, ProjectName, RowKey
from feedbax.contracts.project_sweep import PatchPath, Sweep
from feedbax.contracts.run_matrix import MatrixCompositionDelta

type Payload = dict[str, JsonValue]
type Pointer = Annotated[str, StringConstraints(pattern=r"^(?:/(?:[^~/]|~[01])*)*$")]
type DocumentOverridePath = Annotated[Pointer, StringConstraints(min_length=1)]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
type StepKind = Literal["train", "evaluate", "analyze", "report"]


def json_pointer_parts(pointer: str | None) -> tuple[str, ...]:
    """Decode an already validated JSON Pointer."""
    if not pointer:
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


class SetProjectValue(StrictModel):
    op: Literal["add", "replace"]
    path: DocumentOverridePath
    value: JsonValue


class RemoveProjectValue(StrictModel):
    op: Literal["remove"]
    path: DocumentOverridePath


type ProjectValueEdit = Annotated[SetProjectValue | RemoveProjectValue, Field(discriminator="op")]


class EnsureDocumentCollection(StrictModel):
    op: Literal["ensure"]
    path: DocumentOverridePath
    collection: Literal["object", "array"]


type DocumentEdit = Annotated[
    SetProjectValue | RemoveProjectValue | EnsureDocumentCollection, Field(discriminator="op")
]
type DocumentOverrides = dict[DocumentOverridePath, JsonValue] | list[DocumentEdit]


def document_edit_ops(edits: DocumentOverrides) -> list[DocumentEdit]:
    if isinstance(edits, dict):
        return [SetProjectValue(op="replace", path=path, value=value) for path, value in edits.items()]
    return edits


def apply_document_edit(root: dict[str, Any], edit: DocumentEdit, value: Any) -> None:
    parts = json_pointer_parts(edit.path)
    if isinstance(edit, EnsureDocumentCollection):
        ensure_value_collection(root, parts, value, path=edit.path)
    else:
        apply_value_patch_in_place(root, parts, edit.op, value, path=edit.path)


def document_edit_value(edit: DocumentEdit) -> JsonValue:
    if isinstance(edit, EnsureDocumentCollection):
        return {} if edit.collection == "object" else []
    return edit.value if isinstance(edit, SetProjectValue) else None


def apply_document_edits(root: dict[str, Any], edits: DocumentOverrides) -> None:
    for edit in document_edit_ops(edits):
        apply_document_edit(root, edit, document_edit_value(edit))


class Document(StrictModel):
    schema_id: NonemptyText = Field(alias="schema")
    payload: JsonValue

    @model_validator(mode="after")
    def _typed_payload(self) -> Document:
        if self.schema_id == MODEL_DOCUMENT_SCHEMA_ID:
            ModelDocument.model_validate(self.payload)
        elif self.schema_id == TASK_DOCUMENT_SCHEMA_ID:
            TaskDocument.model_validate(self.payload)
        elif self.schema_id == PYTHON_TASK_SCHEMA_ID:
            PythonTaskSelection.model_validate(self.payload)
        elif self.schema_id == OBJECTIVE_DOCUMENT_SCHEMA_ID:
            ObjectiveDocument.model_validate(self.payload)
        return self


class DocumentReference(StrictModel):
    document: ProjectName
    pointer: Pointer | None = None


class InlineBase(StrictModel):
    inline: Payload


class StepBase(StrictModel):
    step: ProjectName
    row: RowKey | None = None


class RunBase(StrictModel):
    run: NonemptyText


class ExternalBase(StrictModel):
    path: NonemptyText
    sha256: Sha256
    pointer: Pointer | None = None


type Base = InlineBase | DocumentReference | StepBase | RunBase | ExternalBase


class StepNote(StrictModel):
    issue: str | None = None
    reason: str | None = None
    replaces: str | None = None


class Step(StrictModel):
    kind: StepKind
    base: Base
    deltas: list[MatrixCompositionDelta] = Field(default_factory=list)
    # These differences apply when this step reads the named document.
    document_overrides: dict[ProjectName, DocumentOverrides] = Field(default_factory=dict)
    model_attachments: dict[ProjectName, list[ModelAttachmentDifference]] = Field(default_factory=dict)
    sweep: Sweep | None = None
    inputs: dict[ProjectName, InputBinding | None] = Field(default_factory=dict)
    requires: list[PatchPath] = Field(default_factory=list)
    note: StepNote | None = None
    group: ProjectName | None = None


class Project(StrictModel):
    documents: dict[ProjectName, Document] = Field(default_factory=dict)
    steps: dict[ProjectName, Step] = Field(default_factory=dict)
    sweep: Sweep | None = None

    @model_validator(mode="after")
    def _check_project_variants(self) -> Project:
        if self.sweep is not None and any(row is None for row in self.sweep.rows.values()):
            raise ValueError("project variants have no inherited rows to remove")
        return self
