"""Authored and immutable identity of a model loaded from native weights."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.artifact_custody import CONTENT_OBJECT_REF_PREFIX, ContentObjectRef, ContentTreeRef
from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.model_document import ModelDocument, restore_model_display_names
from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256
from feedbax.contracts.strict_json import strict_json_loads


LOADED_MODEL_SCHEMA_ID = "feedbax.spec.loaded_model.v2"
IMPORTED_MODEL_MEDIA_TYPE = "application/vnd.feedbax.imported-model+json"


def encode_imported_model_document(model: ModelDocument) -> bytes:
    return canonical_json_v2_bytes(model.scientific_projection())


def decode_imported_model_document(raw: bytes) -> ModelDocument:
    projection = strict_json_loads(raw)
    if not isinstance(projection, dict):
        raise ValueError("Imported Model projection must be an object")
    try:
        restore_model_display_names(projection)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("Invalid imported Model projection") from error
    model = ModelDocument.model_validate(projection)
    if encode_imported_model_document(model) != raw:
        raise ValueError("Imported Model must use its canonical scientific projection")
    return model


class LoadedModelAsset(StrictModel):
    """The authored pin of an immutable object."""

    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0)]

    def lower(self) -> ContentObjectRef:
        return ContentObjectRef(
            object_id=f"{CONTENT_OBJECT_REF_PREFIX}{self.sha256}", sha256=self.sha256, size_bytes=self.size_bytes,
        )

    @classmethod
    def from_ref(cls, ref: ContentObjectRef) -> "LoadedModelAsset":
        return cls(sha256=ref.sha256, size_bytes=ref.size_bytes)


class ImportedModelSource(ReferenceValue):
    kind: Literal["imported"]
    model: ContentObjectRef
    """The canonical scientific projection of the composed Model at import."""

    weights: ContentTreeRef
    encoding: SemanticModelEncoding


class ImportedModelProvenance(ReferenceValue):
    kind: Literal["imported"]
    model_revision: Sha256
    weights_sha256: Sha256
    """The descriptor digest of the accepted native weights tree."""

    replicas: tuple[Name, ...] = Field(min_length=1)


class ModelSlotFitProblem(StrictModel):
    kind: Literal["missing_slot", "unexpected_slot", "mismatched_slot"]
    replica: Name
    address: Name
    detail: Name


class ModelGraphFitProblem(StrictModel):
    kind: Literal["unsupported_owner", "unresolved_model", "graph_mismatch"]
    address: Name
    detail: Name


type ModelFitProblem = Annotated[
    ModelSlotFitProblem | ModelGraphFitProblem, Field(discriminator="kind"),
]


class ModelFitReport(StrictModel):
    problems: list[ModelFitProblem] = Field(min_length=1)
