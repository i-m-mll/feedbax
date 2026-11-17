"""Exact persisted state and progress of a managed training method."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.request_refs import Name, ReferenceValue, RequestId, Sha256

METHOD_EXTRA_MEDIA_TYPE = "application/vnd.feedbax.method-extra+safetensors"
TransactionId = Annotated[str, Field(pattern=r"^tx-[0-9a-f]{32}$")]


class MethodCheckpointIdentity(ReferenceValue):
    request_id: RequestId
    method_key: Name

    @model_validator(mode="after")
    def _train(self) -> "MethodCheckpointIdentity":
        if not self.request_id.startswith("train:"):
            raise ValueError("A training checkpoint requires a Train request")
        return self


class MethodCheckpointClock(ReferenceValue):
    completed_updates: int = Field(ge=0, strict=True)
    segment_start: int = Field(ge=0, strict=True)
    schedule_origin: int = Field(ge=0, strict=True)
    seed: int = Field(ge=0, lt=2**32, strict=True)

    @model_validator(mode="after")
    def _order(self) -> "MethodCheckpointClock":
        if not self.schedule_origin <= self.segment_start <= self.completed_updates:
            raise ValueError("Checkpoint clocks must follow schedule origin, segment start, completed updates")
        return self


class MethodCheckpointModel(ReferenceValue):
    tree: ContentTreeRef
    encoding: SemanticModelEncoding


class MethodCheckpointExtra(ReferenceValue):
    object: ContentObjectRef
    descriptor_sha256: Sha256


class MethodCheckpointManifest(ReferenceValue):
    schema_version: Literal["feedbax.method_checkpoint.v1"] = "feedbax.method_checkpoint.v1"
    transaction_id: TransactionId
    run_id: Name
    final: bool
    identity: MethodCheckpointIdentity
    clock: MethodCheckpointClock
    model: MethodCheckpointModel
    extra: MethodCheckpointExtra
    validity: ContentTreeRef | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def state_sha256(self) -> str:
        state = self.model_dump(mode="json", exclude={"transaction_id", "run_id", "final", "validity"})
        return sha256_bytes(canonical_json_v2_bytes(state))


class CheckpointLatestPointer(ReferenceValue):
    """Published occurrence and authenticated state at one completed update."""

    schema_version: Literal["feedbax.method_checkpoint_pointer.v1"] = "feedbax.method_checkpoint_pointer.v1"
    run_id: Name
    transaction_id: TransactionId
    manifest_sha256: Sha256
    transaction_root_sha256: Sha256
    completed_updates: int = Field(ge=0, strict=True)
    final: bool

    @property
    def manifest_relative_path(self) -> str:
        return method_checkpoint_manifest_path(self.transaction_id)


def method_checkpoint_manifest_path(transaction_id: str) -> str:
    return f"transactions/{transaction_id}/manifest.json"


def method_checkpoint_extra_path(extra: MethodCheckpointExtra) -> str:
    return f"blobs/{extra.object.sha256}.safetensors"
