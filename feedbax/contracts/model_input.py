"""Exact source documents for models consumed outside training."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.artifact_custody import ContentTreeRef
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.model_document import ModelDocument, restore_model_display_names
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.request_refs import ReferenceValue
from feedbax.contracts.strict_json import strict_json_loads


INITIAL_MODEL_MEDIA_TYPE = "application/vnd.feedbax.initial-model+json"
MODEL_INITIALIZATION_POLICY = "feedbax.model.initialization.v1"


class InitialModelSource(ReferenceValue):
    kind: Literal["initial"] = "initial"
    initialization_policy: Literal[MODEL_INITIALIZATION_POLICY] = MODEL_INITIALIZATION_POLICY
    model_name: ProjectName
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


class CheckpointModelSource(ReferenceValue):
    kind: Literal["checkpoint"] = "checkpoint"
    tree: ContentTreeRef
    encoding: SemanticModelEncoding
    initial: InitialModelSource | None = None


type ModelSource = Annotated[InitialModelSource | CheckpointModelSource, Field(discriminator="kind")]
