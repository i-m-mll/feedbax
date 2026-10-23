"""Portable logical bindings for staged analysis execution resources."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import field_validator

from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.base import StrictModel


STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID = "feedbax.spec.staged_execution"
STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION = "feedbax.spec.staged_execution.v1"
STAGED_CHECKPOINT_CUSTODY_BACKEND = "feedbax-checkpoint-transaction-tree"

_BINDING_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*")


def validate_staged_binding_name(name: str) -> str:
    """Validate one portable/runtime resource binding name."""
    if not isinstance(name, str) or _BINDING_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"staged execution binding names must match ^[a-z][a-z0-9_.-]*$; got {name!r}"
        )
    return name


class StagedCheckpointCustodySpec(StrictModel):
    """Root-free logical selection of the trusted checkpoint transaction backend."""

    backend: Literal["feedbax-checkpoint-transaction-tree"]


class StagedExecutionDescriptor(StrictModel):
    """Versioned portable resource requirements for one staged execution."""

    schema_id: Literal["feedbax.spec.staged_execution"]
    schema_version: Literal["feedbax.spec.staged_execution.v1"]
    artifact_providers: dict[str, ImmutableArtifactBlobProviderSpec]
    checkpoint_custody: dict[str, StagedCheckpointCustodySpec]

    @field_validator("artifact_providers", "checkpoint_custody")
    @classmethod
    def _validate_binding_names(cls, value: dict[str, object]) -> dict[str, object]:
        for name in value:
            validate_staged_binding_name(name)
        return value


__all__ = [
    "STAGED_CHECKPOINT_CUSTODY_BACKEND",
    "STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID",
    "STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION",
    "StagedCheckpointCustodySpec",
    "StagedExecutionDescriptor",
    "validate_staged_binding_name",
]
