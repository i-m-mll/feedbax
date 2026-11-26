"""The semantic project shared by text authoring and Studio."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, JsonValue, StringConstraints, TypeAdapter, field_validator, model_validator

from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA, check_analysis_labels
from feedbax.contracts.base import StrictModel
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_VERSION, ImportedModelSource
from feedbax.contracts.matrix_core import apply_value_patch_in_place, ensure_value_collection
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_inputs import (
    DocumentReference as DocumentReference, InputBinding, NonemptyText, Pointer as Pointer, ProjectName, RowKey,
)
from feedbax.contracts.project_sweep import PatchPath, Sweep, TrialSetField
from feedbax.contracts.run_matrix import MatrixCompositionDelta

type Payload = dict[str, JsonValue]
type DifferencePointer = Annotated[Pointer, StringConstraints(min_length=1)]
type KeyedCollectionPath = Callable[[tuple[str, ...]], bool]
type Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
type StepKind = Literal["train", "synthesize", "evaluate", "analyze", "report"]
type ProjectModelArchive = Annotated[str, StringConstraints(pattern=r"^models/[A-Za-z0-9_][A-Za-z0-9_-]*\.feedbax-model$")]


class _LabelledMember(Protocol):
    @property
    def label(self) -> str: ...


def member_name(members: Mapping[str, _LabelledMember], key: str) -> str:
    member = members.get(key)
    return member.label if member is not None else f"missing reference {key!r}"


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


class OrderProjectValue(StrictModel):
    op: Literal["order"]
    path: DifferencePointer
    order: list[NonemptyText]

    @field_validator("order")
    @classmethod
    def _unique_keys(cls, order: list[str]) -> list[str]:
        if len(set(order)) != len(order):
            raise ValueError("A collection order cannot name a key more than once")
        return order


type ProjectValueEdit = Annotated[
    SetProjectValue | RemoveProjectValue | OrderProjectValue, Field(discriminator="op")
]


class EnsureDocumentCollection(StrictModel):
    op: Literal["ensure"]
    path: DifferencePointer
    collection: Literal["object", "array"]


type DocumentEdit = Annotated[
    SetProjectValue | RemoveProjectValue | OrderProjectValue | EnsureDocumentCollection, Field(discriminator="op")
]


def _no_keyed_collections(_parts: tuple[str, ...]) -> bool:
    return False


def apply_document_edit(
    root: dict[str, Any], edit: DocumentEdit, value: Any, *,
    keyed_collection: KeyedCollectionPath = _no_keyed_collections,
) -> None:
    parts = json_pointer_parts(edit.path)
    if isinstance(edit, EnsureDocumentCollection):
        ensure_value_collection(root, parts, value, path=edit.path)
    elif isinstance(edit, OrderProjectValue):
        _apply_collection_order(root, parts, edit.order, path=edit.path, keyed_collection=keyed_collection)
    else:
        apply_value_patch_in_place(root, parts, edit.op, value, path=edit.path)


def document_edit_value(edit: DocumentEdit) -> JsonValue:
    if isinstance(edit, EnsureDocumentCollection):
        return {} if edit.collection == "object" else []
    return edit.value if isinstance(edit, SetProjectValue) else None


def apply_document_edits(
    root: dict[str, Any], edits: Sequence[DocumentEdit], *,
    keyed_collection: KeyedCollectionPath = _no_keyed_collections,
) -> None:
    for edit in edits:
        apply_document_edit(root, edit, document_edit_value(edit), keyed_collection=keyed_collection)


class LoadedModelDocument(StrictModel):
    schema_version: Literal[LOADED_MODEL_SCHEMA_VERSION]
    model: DocumentReference
    """The Model's project address for navigation. Its pinned revision is in ``source``."""

    source: ImportedModelSource
    archive: ProjectModelArchive | None = Field(default=None, exclude_if=lambda value: value is None)
    """The project's portable model file, named at import."""


def check_document_schema(schema_id: str) -> None:
    from feedbax.contracts.project_storage import stored_document_models

    registry = stored_document_models()
    if schema_id in registry:
        return
    family = schema_id.rsplit(".v", 1)[0]
    successor = next((name for name in registry if name.rsplit(".v", 1)[0] == family), None)
    if successor is not None:
        raise ValueError(f"Unsupported document schema {schema_id!r}. Expected {successor!r}.")
    raise ValueError(f"No registered stored document has schema {schema_id!r}.")


def check_typed_payload(schema_id: str, payload: JsonValue) -> None:
    from feedbax.contracts.figure_document import STORED_FIGURE_DOCUMENT_SCHEMA_ID
    from feedbax.contracts.project_storage import stored_document_models
    from feedbax.contracts.report_document import ReportDocument

    check_document_schema(schema_id)
    if schema_id == ANALYSIS_SCHEMA:
        check_analysis_labels(payload)
    elif schema_id not in ("subset", STORED_FIGURE_DOCUMENT_SCHEMA_ID, ReportDocument.model_fields["schema_version"].default):
        if (model := stored_document_models()[schema_id]) is not None:
            model.model_validate(payload)


class RootDocument(StrictModel):
    """A document whose file holds its whole payload. ``label`` is only what Studio shows, and enters no request."""

    schema_id: NonemptyText = Field(alias="schema")
    label: NonemptyText
    payload: JsonValue

    @model_validator(mode="after")
    def _typed_payload(self) -> RootDocument:
        check_typed_payload(self.schema_id, self.payload)
        return self


class SetDocumentValue(StrictModel):
    value: JsonValue


class RemoveDocumentValue(StrictModel):
    op: Literal["remove"]


class OrderDocumentMembers(StrictModel):
    order: list[str]


type DocumentDifference = SetDocumentValue | RemoveDocumentValue | OrderDocumentMembers
type Differences = dict[DifferencePointer, DocumentDifference]
type EditDifferences = dict[DifferencePointer, SetDocumentValue | RemoveDocumentValue]


def check_disjoint(differences: Mapping[DifferencePointer, DocumentDifference]) -> None:
    """Refuse a difference inside a value replaced or removed by another difference."""
    paths = sorted((json_pointer_parts(path), difference) for path, difference in differences.items())
    for (outer, difference), (inner, _) in zip(paths, paths[1:]):
        if not isinstance(difference, OrderDocumentMembers) and inner[: len(outer)] == outer:
            raise ValueError(f"difference {pointer(inner)!r} lies under {pointer(outer)!r}")


class DocumentBase(StrictModel):
    document: ProjectName
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)


class BasedDocument(StrictModel):
    """A document whose payload is its base's composed payload with ``differences`` applied.

    A difference sets or removes a value or orders a keyed collection at a JSON pointer. No difference
    lies under a set or remove, and none passes through a list. ``label`` belongs to this document and enters
    no request.
    """

    schema_id: NonemptyText = Field(alias="schema")
    label: NonemptyText
    base: DocumentBase
    differences: Differences = Field(default_factory=dict)

    @model_validator(mode="after")
    def _disjoint_differences(self) -> BasedDocument:
        check_document_schema(self.schema_id)
        check_disjoint(self.differences)
        return self


type Document = RootDocument | BasedDocument
DOCUMENT: TypeAdapter[Document] = TypeAdapter(Document)


class DifferenceNotApplied(ValueError):
    """A difference whose site the base no longer has."""

    def __init__(self, path: str, detail: str):
        self.path = path
        super().__init__(detail)


def diff(
    base: Payload, content: Payload, *, keyed_collection: KeyedCollectionPath = _no_keyed_collections,
) -> Differences:
    """The member edits and collection orders that turn ``base`` into ``content``."""
    differences: Differences = {}
    orders: dict[DifferencePointer, OrderDocumentMembers] = {}

    def walk(before: dict[str, Any], after: dict[str, Any], parts: tuple[str, ...]) -> None:
        if parts and keyed_collection(parts):
            edited_keys = [key for key in before if key in after] + [key for key in after if key not in before]
            if edited_keys != list(after):
                orders[pointer(parts)] = OrderDocumentMembers(order=list(after))
        for key, value in after.items():
            child = (*parts, key)
            if key not in before:
                differences[pointer(child)] = SetDocumentValue(value=deepcopy(value))
            elif isinstance(value, dict) and isinstance(before[key], dict):
                walk(before[key], value, child)
            elif not same_json(before[key], value):
                differences[pointer(child)] = SetDocumentValue(value=deepcopy(value))
        for key in before:
            if key not in after:
                differences[pointer((*parts, key))] = RemoveDocumentValue(op="remove")

    walk(base, content, ())
    differences.update(sorted(orders.items(), key=lambda item: item[0].count("/")))
    return differences


def _difference_object(payload: Payload, parts: Sequence[str], path: str) -> Payload:
    value: JsonValue = payload
    for part in parts:
        value = value.get(part) if isinstance(value, dict) else None
    if not isinstance(value, dict):
        raise DifferenceNotApplied(path, f"the base has no object at {pointer(tuple(parts)) or '/'}")
    return value


def _apply_collection_order(
    payload: Payload, parts: tuple[str, ...], order: Sequence[str], *,
    path: str, keyed_collection: KeyedCollectionPath,
) -> None:
    if not keyed_collection(parts):
        raise DifferenceNotApplied(path, f"the document has no keyed collection at {path}")
    collection = _difference_object(payload, parts, path)
    ordered = {key: collection[key] for key in order if key in collection}
    ordered.update(collection)
    collection.clear()
    collection.update(ordered)


def apply_differences(
    base: Payload, differences: Mapping[DifferencePointer, DocumentDifference], *,
    keyed_collection: KeyedCollectionPath = _no_keyed_collections,
) -> Payload:
    """Apply edits in stored order, then apply collection orders from outermost to innermost."""
    result = deepcopy(base)
    orders = []
    for path, difference in differences.items():
        if isinstance(difference, OrderDocumentMembers):
            orders.append((path, difference))
            continue
        *parents, key = json_pointer_parts(path)
        parent = _difference_object(result, parents, path)
        if isinstance(difference, SetDocumentValue):
            parent[key] = deepcopy(difference.value)
        else:
            parent.pop(key, None)
    for path, difference in sorted(orders, key=lambda item: item[0].count("/")):
        _apply_collection_order(
            result, json_pointer_parts(path), difference.order, path=path, keyed_collection=keyed_collection,
        )
    return result


def same_json(left: Any, right: Any) -> bool:
    """Authored edit equality, including the legacy distinction between integer and real literals."""
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
    label: NonemptyText
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
