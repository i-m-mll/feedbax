"""Stored training document: a Train step's method, its settings and the run controls."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from feedbax.contracts.request_specs import DocumentTrainingConfig
from feedbax.plugins.declarations import validate_declaration_key


TRAINING_DOCUMENT_SCHEMA_ID = "feedbax.spec.training_document"
TRAINING_DOCUMENT_KEYS = ("method", "settings", "training_config")
"""The Train payload keys a training document supplies, each read at its own pointer."""


class TrainingDocument(BaseModel):
    """Settings several Train steps share; a step keeps its seed and starting checkpoint."""

    model_config = ConfigDict(extra="forbid")

    method: str
    settings: dict[str, JsonValue]
    training_config: DocumentTrainingConfig

    @field_validator("method")
    @classmethod
    def _method_key(cls, key: str) -> str:
        return validate_declaration_key(key)
