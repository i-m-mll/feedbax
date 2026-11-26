"""The semantic project shared by text authoring and Studio."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, JsonValue, StringConstraints, TypeAdapter, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.display_names import member_display_name as member_display_name
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID, LOADED_MODEL_SCHEMA_VERSION, ImportedModelSource
from feedbax.contracts.matrix_core import apply_value_patch_in_place, ensure_value_collection
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, TaskDocument,
)
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_inputs import (
    DocumentReference as DocumentReference, InputBinding, NonemptyText, Pointer as Pointer, ProjectName, RowKey,
)
from feedbax.contracts.project_sweep import PatchPath, Sweep, TrialSetField
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.training_document import TRAINING_DOCUMENT_SCHEMA_ID, TrainingDocument

type Payload = dict[str, JsonValue]
type DifferencePointer = Annotated[Pointer, StringConstraints(min_length=1)]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
type StepKind = Literal["train", "synthesize", "evaluate", "analyze", "report"]
type ProjectModelArchive = Annotated[str, StringConstraints(pattern=r"^models/[A-Za-z0-9_][A-Za-z0-9_-]*\.feedbax-model$")]


class _LabelledMember(Protocol):
    @property
    def label(self) -> str | None: ...


def member_name(members: Mapping[str, _LabelledMember], key: str) -> str:
    member = members.get(key)
    return member_display_name(key, member.label if member is not None else None)


def json_pointer_parts(pointer: str | None) -> tuple[str, ...]:
    """Decode an already validated JSON Pointer."""
    if not pointer:
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


class SetProjectValue(StrictModel):
    op: Literal["add", "replace"]
    path: DifferencePointer
    value: JsonValue


class RemoveProjectValue(StrictModel):
    op: Literal["remove"]
    path: DifferencePointer


type ProjectValueEdit = Annotated[SetProjectValue | RemoveProjectValue, Field(discriminator="op")]


class EnsureDocumentCollection(StrictModel):
    op: Literal["ensure"]
    path: DifferencePointer
    collection: Literal["object", "array"]


type DocumentEdit = Annotated[
    SetProjectValue | RemoveProjectValue | EnsureDocumentCollection, Field(discriminator="op")
]


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


def apply_document_edits(root: dict[str, Any], edits: Sequence[DocumentEdit]) -> None:
    for edit in edits:
        apply_document_edit(root, edit, document_edit_value(edit))


class LoadedModelDocument(StrictModel):
    schema_version: Literal[LOADED_MODEL_SCHEMA_VERSION]
    model: DocumentReference
    """The Model's project address for navigation. Its pinned revision is in ``source``."""

    source: ImportedModelSource
    archive: ProjectModelArchive | None = Field(default=None, exclude_if=lambda value: value is None)
    """The project's portable model file, named at import."""


def check_typed_payload(schema_id: str, payload: JsonValue) -> None:
    if schema_id == MODEL_DOCUMENT_SCHEMA_ID:
        ModelDocument.model_validate(payload)
    elif schema_id == TASK_DOCUMENT_SCHEMA_ID:
        TaskDocument.model_validate(payload)
    elif schema_id == PYTHON_TASK_SCHEMA_ID:
        PythonTaskSelection.model_validate(payload)
    elif schema_id == OBJECTIVE_DOCUMENT_SCHEMA_ID:
        ObjectiveDocument.model_validate(payload)
    elif schema_id == TRAINING_DOCUMENT_SCHEMA_ID:
        TrainingDocument.model_validate(payload)
    elif schema_id == LOADED_MODEL_SCHEMA_ID:
        LoadedModelDocument.model_validate(payload)
    else:
        # A Set holds Differences, so its module imports this one.
        from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID, SetDocument

        if schema_id == SET_DOCUMENT_SCHEMA_ID:
            SetDocument.model_validate(payload)


class RootDocument(StrictModel):
    """A document whose file holds its whole payload. ``label`` is only what Studio shows, and enters no request."""

    schema_id: NonemptyText = Field(alias="schema")
    label: NonemptyText | None = Field(default=None, exclude_if=lambda value: value is None)
    payload: JsonValue

    @model_validator(mode="after")
    def _typed_payload(self) -> RootDocument:
        check_typed_payload(self.schema_id, self.payload)
        return self


class SetDocumentValue(StrictModel):
    value: JsonValue


class RemoveDocumentValue(StrictModel):
    op: Literal["remove"]


type Differences = dict[DifferencePointer, SetDocumentValue | RemoveDocumentValue]


def check_disjoint(differences: Differences) -> None:
    """Refuse a difference that lies under another."""
    paths = sorted(json_pointer_parts(path) for path in differences)
    for outer, inner in zip(paths, paths[1:]):
        if inner[: len(outer)] == outer:
            raise ValueError(f"difference {pointer(inner)!r} lies under {pointer(outer)!r}")


class DocumentBase(StrictModel):
    document: ProjectName
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)


class BasedDocument(StrictModel):
    """A document whose payload is its base's composed payload with ``differences`` applied.

    Each difference sets or removes the value at one JSON pointer into the payload. No difference
    lies under another, and none passes through a list: a list is one value. ``label`` is the document's own, not
    its base's, and enters no request.
    """

    schema_id: NonemptyText = Field(alias="schema")
    label: NonemptyText | None = Field(default=None, exclude_if=lambda value: value is None)
    base: DocumentBase
    differences: Differences = Field(default_factory=dict)

    @model_validator(mode="after")
    def _disjoint_differences(self) -> BasedDocument:
        check_disjoint(self.differences)
        return self


type Document = RootDocument | BasedDocument
DOCUMENT: TypeAdapter[Document] = TypeAdapter(Document)


class DifferenceNotApplied(ValueError):
    """A difference whose site the base no longer has."""

    def __init__(self, path: str, detail: str):
        self.path = path
        super().__init__(detail)


def diff(base: Payload, content: Payload) -> Differences:
    """The fewest disjoint differences that turn ``base`` into ``content``; a list is one value."""
    differences: Differences = {}

    def walk(before: dict[str, Any], after: dict[str, Any], parts: tuple[str, ...]) -> None:
        for key in before.keys() - after.keys():
            differences[pointer((*parts, key))] = RemoveDocumentValue(op="remove")
        for key, value in after.items():
            if key not in before:
                differences[pointer((*parts, key))] = SetDocumentValue(value=deepcopy(value))
            elif isinstance(value, dict) and isinstance(before[key], dict):
                walk(before[key], value, (*parts, key))
            elif not same_json(before[key], value):
                differences[pointer((*parts, key))] = SetDocumentValue(value=deepcopy(value))

    walk(base, content, ())
    return dict(sorted(differences.items()))


def apply_differences(base: Payload, differences: Differences) -> Payload:
    """Apply ``differences`` to a copy of ``base``.

    Raises DifferenceNotApplied when a difference's parent is not an object in ``base``, or a
    removed value is absent.
    """
    result = deepcopy(base)
    for path, difference in differences.items():
        *parents, key = json_pointer_parts(path)
        parent: Any = result
        for part in parents:
            parent = parent.get(part) if isinstance(parent, dict) else None
        if not isinstance(parent, dict):
            raise DifferenceNotApplied(path, f"the base has no object at {pointer(tuple(parents)) or '/'}")
        if isinstance(difference, SetDocumentValue):
            parent[key] = deepcopy(difference.value)
        elif key in parent:
            del parent[key]
        else:
            raise DifferenceNotApplied(path, f"the base has no value at {path} to remove")
    return result


def same_json(left: Any, right: Any) -> bool:
    """JSON equality: unlike ``==``, ``1``, ``1.0`` and ``true`` differ."""
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_json(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            same_json(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def base_chain(documents: Mapping[str, Document], name: str) -> tuple[str, ...]:
    """``name`` followed by its base, that document's base, and so on.

    The chain ends at a root document, at a missing document, or before a document it already holds;
    composing the document reports the last two.
    """
    chain: list[str] = []
    current: str | None = name
    while current is not None and current not in chain and (document := documents.get(current)) is not None:
        chain.append(current)
        current = document.base.document if isinstance(document, BasedDocument) and not document.base.unresolved else None
    return tuple(chain)


def based_on(documents: Mapping[str, Document], name: str) -> tuple[str, ...]:
    """The documents whose base is ``name``, by name."""
    return tuple(sorted(
        other for other, document in documents.items()
        if isinstance(document, BasedDocument) and not document.base.unresolved and document.base.document == name
    ))


def nearest_shared_base(documents: Mapping[str, Document], left: str, right: str) -> str | None:
    """The nearest document in both base chains, which may be either document itself."""
    shared = set(base_chain(documents, right))
    return next((name for name in base_chain(documents, left) if name in shared), None)


class InlineBase(StrictModel):
    inline: Payload


class StepBase(StrictModel):
    step: ProjectName
    row: RowKey | None = None
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)


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
    """A card. Its key (the dict key, or the file stem in a text project) is how the project refers to it;
    ``label`` is only what Studio and the CLI show, and enters no request, run key or reference."""

    kind: StepKind
    label: NonemptyText | None = Field(default=None, exclude_if=lambda value: value is None)
    base: Base
    deltas: list[MatrixCompositionDelta] = Field(default_factory=list)
    sweep: Sweep | None = None
    set_fields: dict[ProjectName, TrialSetField] = Field(default_factory=dict, exclude_if=lambda value: not value)
    inputs: dict[ProjectName, InputBinding | None] = Field(default_factory=dict)
    requires: list[PatchPath] = Field(default_factory=list)
    note: StepNote | None = None
    group: ProjectName | None = None

    @model_validator(mode="after")
    def _no_model_difference(self) -> Step:
        if isinstance(self.base, InlineBase) and isinstance(self.base.inline, dict) and MODEL_ATTACHMENTS in self.base.inline:
            raise ValueError(f"A step holds no {MODEL_ATTACHMENTS}; a Model change is a document based on the Model")
        if any(
            patch.path == MODEL_ATTACHMENTS or patch.path.startswith(MODEL_ATTACHMENTS + ".")
            for delta in self.deltas for patch in delta.patches
        ):
            raise ValueError(f"A step's deltas write no {MODEL_ATTACHMENTS}; a Model change is a document based on the Model")
        return self


class Project(StrictModel):
    documents: dict[ProjectName, Document] = Field(default_factory=dict)
    steps: dict[ProjectName, Step] = Field(default_factory=dict)
    sweep: Sweep | None = None

    @model_validator(mode="after")
    def _check_project_variants(self) -> Project:
        if self.sweep is not None and any(row is None for row in self.sweep.rows.values()):
            raise ValueError("project variants have no inherited rows to remove")
        return self
