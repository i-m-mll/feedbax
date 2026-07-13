"""Durable run-bundle contract for Feedbax orchestration."""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
    validate_sha256,
)


RUN_BUNDLE_SCHEMA_ID = "feedbax.orchestration.run_bundle"
RUN_BUNDLE_SCHEMA_VERSION_V1 = "feedbax.orchestration.run_bundle.v1"
RUN_BUNDLE_SCHEMA_VERSION_V2 = "feedbax.orchestration.run_bundle.v2"
RUN_BUNDLE_SCHEMA_VERSION_V3 = "feedbax.orchestration.run_bundle.v3"
RUN_BUNDLE_SCHEMA_VERSION_V4 = "feedbax.orchestration.run_bundle.v4"
RUN_BUNDLE_SCHEMA_VERSION = "feedbax.orchestration.run_bundle.v5"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_ID = "feedbax.spec.execution_identity_envelope"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION_V1 = "feedbax.spec.execution_identity_envelope.v1"
EXECUTION_IDENTITY_ENVELOPE_SCHEMA_VERSION = "feedbax.spec.execution_identity_envelope.v2"
ROW_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def mint_run_set_id(now: datetime | None = None) -> str:
    """Mint ``<utc-date>-<8-hex>`` run-set identity."""
    timestamp = now or datetime.now(timezone.utc)
    return f"{timestamp.astimezone(timezone.utc).date().isoformat()}-{secrets.token_hex(4)}"


def default_orchestration_root(run_set_id: str) -> Path:
    """Return the default orchestration root for a run set."""
    configured = os.environ.get("FEEDBAX_ORCHESTRATION_ROOT")
    if configured:
        return Path(configured).expanduser() / run_set_id
    return Path.home() / ".cache" / "feedbax" / "orchestration" / run_set_id


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


class RowLaunchSpec(StrictModel):
    """Operational launch instructions, excluded from scientific identity."""

    command: list[str] = Field(default_factory=list)
    entry: str | None = None
    collect: list[str] = Field(default_factory=list)
    payload_routing: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_launch_entry(self) -> "RowLaunchSpec":
        if not self.command and not self.entry:
            raise ValueError("row launch requires command or entry")
        return self


class RunRowSpec(StrictModel):
    """One compiled row in a run bundle."""

    row_id: str = Field(min_length=1)
    execution: ExecutionIdentityEnvelope
    launch: RowLaunchSpec

    @field_validator("row_id")
    @classmethod
    def _validate_row_id(cls, value: str) -> str:
        if not ROW_ID_RE.fullmatch(value):
            raise ValueError("row_id must match ^[A-Za-z0-9_.-]+$")
        return value

    @model_validator(mode="after")
    def _validate_provenance_row(self) -> "RunRowSpec":
        provenance = self.execution.row_provenance
        if provenance is not None and provenance.row_id != self.row_id:
            raise ValueError("run-row provenance row_id must match row_id")
        return self


class RepoRevision(StrictModel):
    """Repository revision declaration used in environment fingerprints."""

    path: str = "."
    revision: str
    dirty_allowed: bool = False


class EnvironmentDeclaration(StrictModel):
    """Declared execution environment for a run bundle."""

    python_version: str | None = None
    repo_revisions: list[RepoRevision] = Field(default_factory=list)
    lockfile_hashes: dict[str, str] = Field(default_factory=dict)
    overlay_steps: list[str] = Field(default_factory=list)
    image_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LaunchPolicy(StrictModel):
    """Row launch concurrency policy."""

    max_parallel_rows: int = Field(default=1, ge=1)
    warm_first: bool = False
    stagger_seconds: float = Field(default=0.0, ge=0.0)


class BudgetPolicy(StrictModel):
    """Run-set budget guards enforced by MONITOR."""

    max_wall_clock_seconds: float = Field(gt=0.0)
    max_spend_usd: float | None = Field(default=None, ge=0.0)


class InputCustodyPin(StrictModel):
    """Immutable input custody pin."""

    role: str = Field(min_length=1)
    checkpoint_transaction_id: str = Field(min_length=1)

    @field_validator("checkpoint_transaction_id")
    @classmethod
    def _reject_mutable_latest(cls, value: str) -> str:
        if "latest.json" in value:
            raise ValueError(
                "input custody pins must name checkpoint transactions, not latest.json"
            )
        return value


class RunBundle(StrictModel):
    """Schema-versioned orchestration request for a run set."""

    schema_id: Literal["feedbax.orchestration.run_bundle"] = RUN_BUNDLE_SCHEMA_ID
    schema_version: Literal["feedbax.orchestration.run_bundle.v5"] = RUN_BUNDLE_SCHEMA_VERSION
    run_set_id: str = Field(default_factory=mint_run_set_id)
    driver: str = "local"
    rows: list[RunRowSpec] = Field(min_length=1)
    environment: EnvironmentDeclaration
    launch_policy: LaunchPolicy = Field(default_factory=LaunchPolicy)
    budget: BudgetPolicy
    input_custody_pins: list[InputCustodyPin] = Field(default_factory=list)
    orchestration_root: str | None = None
    keep_alive: bool = False
    deadman_enabled: bool = False
    deadman_silence_seconds: int = Field(default=1800, ge=60)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bundle(self) -> "RunBundle":
        seen: set[str] = set()
        for row in self.rows:
            if row.row_id in seen:
                raise ValueError(f"duplicate row_id: {row.row_id!r}")
            seen.add(row.row_id)
        return self

    @property
    def run_set_dir(self) -> Path:
        """Return the local directory where orchestration state is stored."""
        if self.orchestration_root:
            root = Path(self.orchestration_root).expanduser()
            return root if root.name == self.run_set_id else root / self.run_set_id
        return default_orchestration_root(self.run_set_id)

    def row(self, row_id: str) -> RunRowSpec:
        """Return one row spec by id."""
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise KeyError(row_id)
