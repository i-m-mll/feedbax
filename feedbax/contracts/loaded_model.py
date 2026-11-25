"""Authored and immutable identity of a model loaded from native weights."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.request_refs import Name, ReferenceValue, Sha256


LOADED_MODEL_SCHEMA_ID = "feedbax.spec.loaded_model"
LOADED_MODEL_SCHEMA_VERSION = "feedbax.spec.loaded_model.v1"
IMPORTED_MODEL_MEDIA_TYPE = "application/vnd.feedbax.imported-model+json"


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
