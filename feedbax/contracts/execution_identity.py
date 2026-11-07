"""Immutable artifact references and canonical execution identity for a row."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.execution_schema import (
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID,
    EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION,
)
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
    validate_sha256,
)


class SchemaArtifactRef(StrictModel):
    """Immutable registry reference to a governed structured payload."""

    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    sha256: str
    uri: str | None = None

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        validate_sha256(value, field_name="sha256")
        return value

    @field_validator("artifact_id")
    @classmethod
    def _reject_mutable_locator(cls, value: str) -> str:
        if "latest.json" in value:
            raise ValueError("artifact_id must be an immutable registry-resolvable locator")
        return value


class IdentityArtifactRef(SchemaArtifactRef):
    """Base immutable artifact reference carrying a separate semantic hash."""


class AuthoredIntentRef(IdentityArtifactRef):
    """Authored document reference and its family-defined intent identity."""

    intent_hash: str

    @field_validator("intent_hash")
    @classmethod
    def _validate_intent_hash(cls, value: str) -> str:
        validate_sha256(value, field_name="intent_hash")
        return value


class ResolvedSnapshotRef(IdentityArtifactRef):
    """Resolved-semantics snapshot reference and decoded root identity."""

    root_hash: str

    @field_validator("root_hash")
    @classmethod
    def _validate_root_hash(cls, value: str) -> str:
        validate_sha256(value, field_name="root_hash")
        return value


class ExecutionCapsuleRef(IdentityArtifactRef):
    """Execution capsule reference and derived execution identity."""

    execution_hash: str

    @field_validator("execution_hash")
    @classmethod
    def _validate_execution_hash(cls, value: str) -> str:
        validate_sha256(value, field_name="execution_hash")
        return value


class ImmutableInputDigest(StrictModel):
    """Digest identifying immutable external input bytes."""

    algorithm: Literal["sha256"] = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        validate_sha256(value, field_name="digest.value")
        return value


class ImmutableInputIdentity(StrictModel):
    """Canonical identity for one external execution input."""

    role: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    digest: ImmutableInputDigest
    schema_id: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_schema_pair(self) -> "ImmutableInputIdentity":
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError("schema_id and schema_version must be supplied together")
        if "latest.json" in self.identifier:
            raise ValueError("immutable input identifier must not reference latest.json")
        return self


class ExecutionIdentityEnvelope(StrictModel):
    """Immutable scientific identity committed by ASSEMBLE for one row."""

    schema_id: Literal["feedbax.spec.execution_identity_envelope"] = (
        EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.execution_identity_envelope.v2"] = (
        EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION
    )
    payload: SchemaArtifactRef
    authored_intent: AuthoredIntentRef
    resolved_snapshot: ResolvedSnapshotRef
    execution_capsule: ExecutionCapsuleRef
    immutable_inputs: list[ImmutableInputIdentity]
    row_provenance: TrainingRowProvenance | None = None

    @model_validator(mode="after")
    def _validate_execution_identity(self) -> "ExecutionIdentityEnvelope":
        canonical = canonicalize_immutable_input_identities(self.immutable_inputs)
        supplied = [
            identity.model_dump(mode="json", exclude_none=True)
            for identity in self.immutable_inputs
        ]
        if supplied != canonical:
            raise ValueError(
                "immutable_inputs must be in canonical (role, kind, identifier, digest) order"
            )
        expected = training_run_execution_hash(self.resolved_snapshot.root_hash, canonical)
        if self.execution_capsule.execution_hash != expected:
            raise ValueError(
                "execution_hash does not match resolved root and canonical immutable inputs; "
                f"envelope={self.execution_capsule.execution_hash!r}, computed={expected!r}"
            )
        if (
            self.row_provenance is not None
            and self.row_provenance.lowered_execution_payload_hash != self.payload.sha256
        ):
            raise ValueError(
                "row provenance lowered_execution_payload_hash does not match the "
                "custodied execution payload"
            )
        return self


def execution_identity_projection(
    envelope: ExecutionIdentityEnvelope,
) -> dict[str, Any]:
    """Project one execution envelope onto durable manifest identity fields."""
    return {
        "intent_hash": envelope.authored_intent.intent_hash,
        "resolved_semantics_root_hash": envelope.resolved_snapshot.root_hash,
        "execution_hash": envelope.execution_capsule.execution_hash,
        "input_data_identities": canonicalize_immutable_input_identities(envelope.immutable_inputs),
    }
