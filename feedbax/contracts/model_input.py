"""Exact source documents for models consumed outside training."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, TypeAdapter

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ArtifactBlobProvider, ContentObjectRef, ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.loaded_model import IMPORTED_MODEL_MEDIA_TYPE, LOADED_MODEL_SCHEMA_ID, ImportedModelSource
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.model_document import ModelDocument, restore_model_display_names
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument, StoredObjectiveDocument
from feedbax.contracts.project_inputs import DocumentInputReference, InitialModelReference, LoadedModelReference, ObjectiveDocumentReference
from feedbax.contracts.request_refs import ContentRef, ReferenceValue
from feedbax.contracts.strict_json import strict_json_loads, strict_model_validate_json, strict_validate_json
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, STORED_TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, StoredTaskDocument, TaskDocument,
)

if TYPE_CHECKING:
    from feedbax.contracts.project import RootDocument
    from feedbax.tasks.registry import PythonTaskRegistry


INITIAL_MODEL_MEDIA_TYPE = "application/vnd.feedbax.initial-model+json"
TASK_INPUT_MEDIA_TYPE = "application/vnd.feedbax.task-input+json"
OBJECTIVE_INPUT_MEDIA_TYPE = "application/vnd.feedbax.objective-input+json"
MODEL_INITIALIZATION_POLICY = "feedbax.model.initialization.v1"


class InitialModelSource(ReferenceValue):
    kind: Literal["initial"] = "initial"
    initialization_policy: Literal[MODEL_INITIALIZATION_POLICY] = MODEL_INITIALIZATION_POLICY
    document: ModelDocument
    seed: int = Field(default=0, ge=0, lt=2**32, strict=True)

    def encode(self) -> bytes:
        projection = self.model_dump(mode="json")
        projection["document"] = self.document.scientific_projection()
        return canonical_json_v2_bytes(projection)

    @classmethod
    def decode(cls, data: bytes) -> InitialModelSource:
        projection = strict_json_loads(data)
        if not isinstance(projection, dict) or not isinstance(projection.get("document"), dict):
            raise ValueError("Invalid initial Model source")
        try:
            restore_model_display_names(projection["document"])
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("Invalid initial Model document projection") from error
        source = cls.model_validate(projection)
        if source.encode() != data:
            raise ValueError("Initial Model source must use its canonical scientific projection")
        return source


class ParameterModelSource(ReferenceValue):
    tree: ContentTreeRef
    encoding: SemanticModelEncoding
    initial: InitialModelSource


class CheckpointModelSource(ParameterModelSource):
    kind: Literal["checkpoint"] = "checkpoint"


class SynthesizedModelSource(ParameterModelSource):
    kind: Literal["synthesized"] = "synthesized"


type ModelSource = Annotated[
    InitialModelSource | CheckpointModelSource | SynthesizedModelSource, Field(discriminator="kind"),
]
type TaskInputSource = TaskDocument | PythonTaskSelection
type DocumentInput = InitialModelSource | TaskInputSource | ObjectiveDocument | ImportedModelSource

_TASK_INPUT = TypeAdapter(TaskInputSource)
DOCUMENT_INPUT_MEDIA_TYPES = frozenset((
    INITIAL_MODEL_MEDIA_TYPE, TASK_INPUT_MEDIA_TYPE, OBJECTIVE_INPUT_MEDIA_TYPE, IMPORTED_MODEL_MEDIA_TYPE,
))


def authored_document_input(
    reference: DocumentInputReference, document: RootDocument, *, python_tasks: PythonTaskRegistry | None = None,
) -> DocumentInput:
    if isinstance(reference, InitialModelReference):
        if document.schema_id != STORED_MODEL_DOCUMENT_SCHEMA_ID:
            raise ValueError("Initial model input requires a Model document")
        return InitialModelSource(
            document=StoredModelDocument.model_validate(document.payload).lower(), seed=reference.seed,
        )
    if isinstance(reference, LoadedModelReference):
        from feedbax.contracts.project import LoadedModelDocument

        if document.schema_id != LOADED_MODEL_SCHEMA_ID:
            raise ValueError("Loaded model input requires a Loaded Model document")
        return LoadedModelDocument.model_validate(document.payload).lower()
    if isinstance(reference, ObjectiveDocumentReference):
        if document.schema_id != STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID:
            raise ValueError("Objective input requires an Objective document")
        return StoredObjectiveDocument.model_validate(document.payload).lower()
    if document.schema_id == STORED_TASK_DOCUMENT_SCHEMA_ID:
        return StoredTaskDocument.model_validate(document.payload).lower()
    if document.schema_id == PYTHON_TASK_SCHEMA_ID:
        from feedbax.tasks.registry import resolve_python_task_instance

        selection = PythonTaskSelection.model_validate(document.payload)
        if python_tasks is None:
            raise ValueError(f"Python task {selection.type!r} requires a task registry")
        task = resolve_python_task_instance(selection, python_tasks)
        if isinstance(task, tuple):
            raise ValueError(task[0].reason)
        return task.selection()
    raise ValueError("Task input requires a Task document or Python task selection")


def encode_document_input(value: DocumentInput) -> tuple[ContentRef, bytes]:
    if isinstance(value, InitialModelSource):
        media_type, raw = INITIAL_MODEL_MEDIA_TYPE, value.encode()
    elif isinstance(value, ImportedModelSource):
        media_type = IMPORTED_MODEL_MEDIA_TYPE
        raw = canonical_json_v2_bytes(value.model_dump(mode="json"))
    else:
        media_type = OBJECTIVE_INPUT_MEDIA_TYPE if isinstance(value, ObjectiveDocument) else TASK_INPUT_MEDIA_TYPE
        if isinstance(value, TaskDocument):
            value = value.without_markers()
        raw = canonical_json_v2_bytes(value.model_dump(mode="json"))
    return ContentRef(sha256=sha256(raw).hexdigest(), size=len(raw), media_type=media_type), raw


def document_input_object(content: ContentRef) -> ContentObjectRef:
    if content.tree or content.media_type not in DOCUMENT_INPUT_MEDIA_TYPES:
        raise ValueError("Expected an immutable Model, Task or Objective document input")
    return ContentObjectRef(
        object_id=CONTENT_OBJECT_REF_PREFIX + content.sha256, sha256=content.sha256, size_bytes=content.size,
    )


def read_document_input(content: ContentRef, artifacts: ArtifactBlobProvider) -> DocumentInput:
    return decode_document_input(content, artifacts.get_object(document_input_object(content)))


def decode_document_input(content: ContentRef, raw: bytes) -> DocumentInput:
    document_input_object(content)
    if content.media_type == INITIAL_MODEL_MEDIA_TYPE:
        value = InitialModelSource.decode(raw)
    elif content.media_type == IMPORTED_MODEL_MEDIA_TYPE:
        value = strict_model_validate_json(ImportedModelSource, raw)
    elif content.media_type == TASK_INPUT_MEDIA_TYPE:
        value = strict_validate_json(_TASK_INPUT, raw)
    else:
        value = strict_model_validate_json(ObjectiveDocument, raw)
    if encode_document_input(value)[0] != content:
        raise ValueError("Document input does not match its canonical content reference")
    return value
