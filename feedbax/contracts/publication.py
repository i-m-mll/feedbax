"""Exact-byte custody, artifact, provenance, publication, and checkpoint contracts.

These records describe meaning and identity without selecting a storage engine.
Every reference names exact bytes; paths and provider locators are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel


PUBLICATION_SCHEMA_ID = "feedbax.publication"
PUBLICATION_SCHEMA_VERSION = "feedbax.publication.v1"
ARTIFACT_RECORD_SCHEMA_ID = "feedbax.artifact_record"
ARTIFACT_RECORD_SCHEMA_VERSION = "feedbax.artifact_record.v1"
CHECKPOINT_SET_SCHEMA_ID = "feedbax.checkpoint_set"
CHECKPOINT_SET_SCHEMA_VERSION = "feedbax.checkpoint_set.v1"
PROVENANCE_EDGE_SCHEMA_ID = "feedbax.provenance_edge"
PROVENANCE_EDGE_SCHEMA_VERSION = "feedbax.provenance_edge.v1"
PUBLICATION_RECEIPT_SCHEMA_ID = "feedbax.publication_receipt"
PUBLICATION_RECEIPT_SCHEMA_VERSION = "feedbax.publication_receipt.v1"
SHA256_ALGORITHM = "sha256"


def canonical_bytes(value: object) -> bytes:
    """Encode one protocol value canonically for identity and persistence."""
    if isinstance(value, StrictModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_digest(value: str, *, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


class BlobRef(StrictModel):
    """The complete portable identity of one immutable byte string."""

    algorithm: Literal["sha256"] = SHA256_ALGORITHM
    digest: str
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_identity(self) -> "BlobRef":
        _validate_digest(self.digest, field="blob digest")
        return self

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlobRef":
        return cls(digest=sha256(data), size_bytes=len(data))


@runtime_checkable
class BlobStore(Protocol):
    """Storage-neutral immutable byte custody boundary."""

    def stage(self, data: bytes) -> BlobRef: ...

    def verify(self, ref: BlobRef) -> None: ...

    def read(self, ref: BlobRef) -> bytes: ...

    def enumerate(self) -> Sequence[BlobRef]: ...


ReferenceDomain = Literal[
    "document_revision",
    "semantic_ir",
    "workflow_plan",
    "invocation",
    "attempt",
    "artifact_version",
    "checkpoint_transaction",
    "checkpoint_set",
    "publication",
]


class ExactRef(StrictModel):
    """A typed identity coupled to the exact canonical bytes that define it."""

    domain: ReferenceDomain
    identity: str = Field(min_length=1)
    bytes: BlobRef


class ArtifactRecord(StrictModel):
    """Typed meaning assigned to immutable payload blobs."""

    schema_id: Literal["feedbax.artifact_record"] = ARTIFACT_RECORD_SCHEMA_ID
    schema_version: Literal["feedbax.artifact_record.v1"] = ARTIFACT_RECORD_SCHEMA_VERSION
    logical_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    payload_schema_id: str = Field(min_length=1)
    payload_schema_version: str = Field(min_length=1)
    blobs: tuple[BlobRef, ...]
    dimensions: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_record(self) -> "ArtifactRecord":
        if not self.blobs:
            raise ValueError("artifact record must identify at least one exact blob")
        if len({(blob.algorithm, blob.digest, blob.size_bytes) for blob in self.blobs}) != len(
            self.blobs
        ):
            raise ValueError("artifact record must not repeat a blob")
        expected = artifact_version_id(
            logical_id=self.logical_id,
            role=self.role,
            media_type=self.media_type,
            payload_schema_id=self.payload_schema_id,
            payload_schema_version=self.payload_schema_version,
            blobs=self.blobs,
            dimensions=self.dimensions,
        )
        if self.version_id != expected:
            raise ValueError(
                f"artifact version_id does not identify its canonical meaning: expected {expected}"
            )
        return self

    @property
    def exact_ref(self) -> ExactRef:
        raw = canonical_bytes(self)
        return ExactRef(
            domain="artifact_version",
            identity=self.version_id,
            bytes=BlobRef.from_bytes(raw),
        )


def artifact_version_id(
    *,
    logical_id: str,
    role: str,
    media_type: str,
    payload_schema_id: str,
    payload_schema_version: str,
    blobs: Sequence[BlobRef],
    dimensions: Mapping[str, str | int | float | bool] | None = None,
) -> str:
    identity = {
        "logical_id": logical_id,
        "role": role,
        "media_type": media_type,
        "payload_schema_id": payload_schema_id,
        "payload_schema_version": payload_schema_version,
        "blobs": [blob.model_dump(mode="json") for blob in blobs],
        "dimensions": dict(dimensions or {}),
    }
    return f"artifact-version:sha256:{sha256(canonical_bytes(identity))}"


def artifact_record(
    *,
    logical_id: str,
    role: str,
    media_type: str,
    payload_schema_id: str,
    payload_schema_version: str,
    blobs: Sequence[BlobRef],
    dimensions: Mapping[str, str | int | float | bool] | None = None,
) -> ArtifactRecord:
    blob_tuple = tuple(blobs)
    dimensions_dict = dict(dimensions or {})
    return ArtifactRecord(
        logical_id=logical_id,
        version_id=artifact_version_id(
            logical_id=logical_id,
            role=role,
            media_type=media_type,
            payload_schema_id=payload_schema_id,
            payload_schema_version=payload_schema_version,
            blobs=blob_tuple,
            dimensions=dimensions_dict,
        ),
        role=role,
        media_type=media_type,
        payload_schema_id=payload_schema_id,
        payload_schema_version=payload_schema_version,
        blobs=blob_tuple,
        dimensions=dimensions_dict,
    )


ProvenanceRelation = Literal[
    "produced_by",
    "consumed_by",
    "derived_from",
    "resumed_from",
    "supersedes",
]


class ProvenanceEdge(StrictModel):
    """One append-only exact lineage fact."""

    schema_id: Literal["feedbax.provenance_edge"] = PROVENANCE_EDGE_SCHEMA_ID
    schema_version: Literal["feedbax.provenance_edge.v1"] = PROVENANCE_EDGE_SCHEMA_VERSION
    relation: ProvenanceRelation
    subject: ExactRef
    object: ExactRef

    @property
    def identity(self) -> str:
        return f"provenance:sha256:{sha256(canonical_bytes(self))}"


class CheckpointSlot(StrictModel):
    """One named, typed checkpoint state slot."""

    name: str = Field(min_length=1)
    state_type: str = Field(min_length=1)
    array_structure_id: str = Field(min_length=1)
    codec_schema_id: str = Field(min_length=1)
    codec_schema_version: str = Field(min_length=1)
    blob: BlobRef


class CheckpointSet(StrictModel):
    """A jointly validated resumable state artifact."""

    schema_id: Literal["feedbax.checkpoint_set"] = CHECKPOINT_SET_SCHEMA_ID
    schema_version: Literal["feedbax.checkpoint_set.v1"] = CHECKPOINT_SET_SCHEMA_VERSION
    checkpoint_id: str = Field(min_length=1)
    transaction: ExactRef
    training_program_id: str = Field(min_length=1)
    graph: ExactRef
    experiment: ExactRef
    progress: dict[str, int | float | str]
    prng_state: BlobRef | None = None
    slots: tuple[CheckpointSlot, ...]
    continuation: Literal["resume", "fork"]
    parent: ExactRef | None = None

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> "CheckpointSet":
        if not self.progress:
            raise ValueError("checkpoint progress must be explicit")
        if not self.slots:
            raise ValueError("checkpoint must contain at least one state slot")
        names = [slot.name for slot in self.slots]
        if len(set(names)) != len(names):
            raise ValueError("checkpoint state slot names must be unique")
        if self.continuation == "resume" and self.parent is None:
            raise ValueError("a resume checkpoint must identify its exact parent checkpoint")
        if self.parent is not None and self.parent.domain != "checkpoint_set":
            raise ValueError("checkpoint parent must be an exact checkpoint_set reference")
        if self.transaction.domain != "checkpoint_transaction":
            raise ValueError(
                "checkpoint transaction must be an exact checkpoint_transaction reference"
            )
        if self.graph.domain != "semantic_ir":
            raise ValueError("checkpoint graph must be an exact semantic_ir reference")
        if self.experiment.domain != "document_revision":
            raise ValueError("checkpoint experiment must be an exact document_revision reference")
        expected = checkpoint_set_id(
            training_program_id=self.training_program_id,
            transaction=self.transaction,
            graph=self.graph,
            experiment=self.experiment,
            progress=self.progress,
            prng_state=self.prng_state,
            slots=self.slots,
            continuation=self.continuation,
            parent=self.parent,
        )
        if self.checkpoint_id != expected:
            raise ValueError(
                f"checkpoint_id does not identify its canonical meaning: expected {expected}"
            )
        return self

    @property
    def exact_ref(self) -> ExactRef:
        raw = canonical_bytes(self)
        return ExactRef(
            domain="checkpoint_set",
            identity=self.checkpoint_id,
            bytes=BlobRef.from_bytes(raw),
        )


def checkpoint_set_id(**values: Any) -> str:
    encoded: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, StrictModel):
            encoded[key] = value.model_dump(mode="json")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            encoded[key] = [
                item.model_dump(mode="json") if isinstance(item, StrictModel) else item
                for item in value
            ]
        else:
            encoded[key] = value
    return f"checkpoint:sha256:{sha256(canonical_bytes(encoded))}"


class PublicationRequest(StrictModel):
    """Complete all-or-nothing logical publication intent."""

    schema_id: Literal["feedbax.publication"] = PUBLICATION_SCHEMA_ID
    schema_version: Literal["feedbax.publication.v1"] = PUBLICATION_SCHEMA_VERSION
    idempotency_key: str = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...]
    provenance: tuple[ProvenanceEdge, ...] = ()
    checkpoints: tuple[CheckpointSet, ...] = ()

    @model_validator(mode="after")
    def _validate_transaction(self) -> "PublicationRequest":
        if not self.artifacts and not self.checkpoints:
            raise ValueError("publication must contain an artifact or checkpoint")
        version_ids = [record.version_id for record in self.artifacts]
        if len(set(version_ids)) != len(version_ids):
            raise ValueError("publication must not repeat an artifact version")
        checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in self.checkpoints]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ValueError("publication must not repeat a checkpoint set")
        edge_ids = [edge.identity for edge in self.provenance]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("publication must not repeat a provenance edge")
        return self

    @property
    def request_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"idempotency_key"})
        return sha256(canonical_bytes(payload))

    @property
    def publication_id(self) -> str:
        return "publication:sha256:" + sha256(
            canonical_bytes(
                {"idempotency_key": self.idempotency_key, "request_sha256": self.request_sha256}
            )
        )


class PublicationReceipt(StrictModel):
    """Committed transaction identity and its exact visible records."""

    schema_id: Literal["feedbax.publication_receipt"] = PUBLICATION_RECEIPT_SCHEMA_ID
    schema_version: Literal["feedbax.publication_receipt.v1"] = PUBLICATION_RECEIPT_SCHEMA_VERSION
    publication_id: str
    idempotency_key: str
    request_sha256: str
    artifact_refs: tuple[ExactRef, ...]
    checkpoint_refs: tuple[ExactRef, ...]


@runtime_checkable
class PublicationCatalog(Protocol):
    """Atomic logical publication and query boundary."""

    def commit(self, request: PublicationRequest) -> PublicationReceipt: ...

    def receipt(self, publication_id: str) -> PublicationReceipt | None: ...

    def artifact(self, version_id: str) -> ArtifactRecord | None: ...

    def checkpoint(self, checkpoint_id: str) -> CheckpointSet | None: ...


class PublicationService:
    """Validate blob custody before atomically exposing logical records."""

    def __init__(self, blobs: BlobStore, catalog: PublicationCatalog) -> None:
        self._blobs = blobs
        self._catalog = catalog

    def stage(self, data: bytes) -> BlobRef:
        """Stage bytes without making a logical artifact visible."""
        return self._blobs.stage(data)

    def publish(self, request: PublicationRequest) -> PublicationReceipt:
        refs = [blob for record in request.artifacts for blob in record.blobs]
        for checkpoint in request.checkpoints:
            refs.append(checkpoint.prng_state)
            refs.extend(slot.blob for slot in checkpoint.slots)
        for ref in refs:
            self._blobs.verify(ref)
        return self._catalog.commit(request)


__all__ = [
    "ARTIFACT_RECORD_SCHEMA_ID",
    "ARTIFACT_RECORD_SCHEMA_VERSION",
    "ArtifactRecord",
    "BlobRef",
    "BlobStore",
    "CHECKPOINT_SET_SCHEMA_ID",
    "CHECKPOINT_SET_SCHEMA_VERSION",
    "CheckpointSet",
    "CheckpointSlot",
    "ExactRef",
    "PUBLICATION_SCHEMA_ID",
    "PUBLICATION_SCHEMA_VERSION",
    "PROVENANCE_EDGE_SCHEMA_ID",
    "PROVENANCE_EDGE_SCHEMA_VERSION",
    "PUBLICATION_RECEIPT_SCHEMA_ID",
    "PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "ProvenanceEdge",
    "PublicationCatalog",
    "PublicationReceipt",
    "PublicationRequest",
    "PublicationService",
    "artifact_record",
    "artifact_version_id",
    "canonical_bytes",
    "checkpoint_set_id",
]
