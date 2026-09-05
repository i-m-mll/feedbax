"""Portable contract for opening the immutable local artifact blob provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
import unicodedata
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.base import ArtifactRef, StrictModel


IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID = "feedbax.spec.immutable_artifact_blob_provider"
IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION = "feedbax.spec.immutable_artifact_blob_provider.v1"
IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND = "feedbax-local-sha256-cas"
IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND = "feedbax-local"
CONTENT_OBJECT_REF_SCHEMA_ID = "feedbax.spec.content_object_ref"
CONTENT_OBJECT_REF_SCHEMA_VERSION = "feedbax.spec.content_object_ref.v1"
CONTENT_TREE_SCHEMA_ID = "feedbax.spec.content_tree"
CONTENT_TREE_SCHEMA_VERSION = "feedbax.spec.content_tree.v1"
CONTENT_TREE_REF_SCHEMA_ID = "feedbax.spec.content_tree_ref"
CONTENT_TREE_REF_SCHEMA_VERSION = "feedbax.spec.content_tree_ref.v1"
CONTENT_MANIFEST_REF_SCHEMA_ID = "feedbax.spec.content_manifest_ref"
CONTENT_MANIFEST_REF_SCHEMA_VERSION = "feedbax.spec.content_manifest_ref.v1"
CONTENT_BINDING_SCHEMA_ID = "feedbax.spec.content_binding"
CONTENT_BINDING_SCHEMA_VERSION = "feedbax.spec.content_binding.v1"
CONTENT_MIGRATION_RECEIPT_SCHEMA_ID = "feedbax.spec.content_migration_receipt"
CONTENT_MIGRATION_RECEIPT_SCHEMA_VERSION = "feedbax.spec.content_migration_receipt.v1"
CONTENT_OBJECT_REF_PREFIX = "content-object://sha256/"
CONTENT_TREE_REF_PREFIX = "content-tree://sha256/"
CONTENT_MANIFEST_REF_PREFIX = "content-manifest://"
LEGACY_CONTENT_MIGRATION_COMMAND = "python -m feedbax migrate-content-store"


class ArtifactBlobCustodyError(ValueError):
    """Base error exposed by an immutable artifact custody provider."""


class ArtifactBlobReferenceError(ArtifactBlobCustodyError):
    """An artifact identifier or reference is invalid."""


class ArtifactBlobIntegrityError(ArtifactBlobCustodyError):
    """Immutable artifact bytes fail identity validation."""


class ArtifactBlobContainmentError(ArtifactBlobCustodyError):
    """A custody path escapes or aliases its canonical store."""


class ContentManifestConflictError(ArtifactBlobCustodyError):
    """One manifest kind/id was already pinned to different exact bytes."""


class LegacyContentFormatError(ArtifactBlobCustodyError):
    """A successor runtime was given input accepted only by the offline migrator."""


def _canonical_content_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("content-tree members require canonical POSIX relative paths")
    canonical = path.as_posix()
    if unicodedata.normalize("NFC", canonical) != canonical:
        raise ValueError("content-tree member paths must use NFC Unicode normalization")
    return canonical


class ContentObjectRef(StrictModel):
    """Role-free reference to one exact immutable byte object."""

    schema_id: Literal[CONTENT_OBJECT_REF_SCHEMA_ID] = CONTENT_OBJECT_REF_SCHEMA_ID
    schema_version: Literal[CONTENT_OBJECT_REF_SCHEMA_VERSION] = CONTENT_OBJECT_REF_SCHEMA_VERSION
    object_id: str = Field(pattern=r"^content-object://sha256/[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_identity(self) -> "ContentObjectRef":
        if self.object_id != f"{CONTENT_OBJECT_REF_PREFIX}{self.sha256}":
            raise ValueError("content object id does not match its SHA-256")
        return self


class ContentTreeFile(StrictModel):
    """One regular-file member of a role-free content tree."""

    relative_path: str
    object: ContentObjectRef

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        return _canonical_content_path(value)


class ContentTree(StrictModel):
    """Canonical role-free descriptor for a directory tree."""

    schema_id: Literal[CONTENT_TREE_SCHEMA_ID] = CONTENT_TREE_SCHEMA_ID
    schema_version: Literal[CONTENT_TREE_SCHEMA_VERSION] = CONTENT_TREE_SCHEMA_VERSION
    directories: tuple[str, ...] = ()
    files: tuple[ContentTreeFile, ...] = ()

    @field_validator("directories")
    @classmethod
    def _validate_directories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_canonical_content_path(path) for path in value)

    @model_validator(mode="after")
    def _validate_paths(self) -> "ContentTree":
        directories = list(self.directories)
        files = [entry.relative_path for entry in self.files]
        if directories != sorted(directories) or len(directories) != len(set(directories)):
            raise ValueError("content-tree directories must be unique and canonically ordered")
        if files != sorted(files) or len(files) != len(set(files)):
            raise ValueError("content-tree files must be unique and canonically ordered")
        all_paths = directories + files
        collision_keys = [unicodedata.normalize("NFC", path).casefold() for path in all_paths]
        if len(collision_keys) != len(set(collision_keys)):
            raise ValueError("content-tree paths collide after target-path normalization")
        file_set = set(files)
        for path in directories:
            if path in file_set:
                raise ValueError("content-tree path is both a directory and a file")
        folded_paths = [unicodedata.normalize("NFC", path).casefold() for path in all_paths]
        folded_files = {unicodedata.normalize("NFC", path).casefold() for path in file_set}
        if any(
            other.startswith(f"{file_path}/")
            for file_path in folded_files
            for other in folded_paths
        ):
            raise ValueError("content-tree file path collides with a descendant path")
        return self


class ContentTreeRef(StrictModel):
    """Pinned identity of one canonical content-tree descriptor."""

    schema_id: Literal[CONTENT_TREE_REF_SCHEMA_ID] = CONTENT_TREE_REF_SCHEMA_ID
    schema_version: Literal[CONTENT_TREE_REF_SCHEMA_VERSION] = CONTENT_TREE_REF_SCHEMA_VERSION
    tree_id: str = Field(pattern=r"^content-tree://sha256/[0-9a-f]{64}$")
    descriptor: ContentObjectRef

    @model_validator(mode="after")
    def _validate_identity(self) -> "ContentTreeRef":
        if self.tree_id != f"{CONTENT_TREE_REF_PREFIX}{self.descriptor.sha256}":
            raise ValueError("content tree id does not match its descriptor SHA-256")
        return self


class ContentManifestRef(StrictModel):
    """One logical manifest kind/id pinned to exact immutable bytes."""

    schema_id: Literal[CONTENT_MANIFEST_REF_SCHEMA_ID] = CONTENT_MANIFEST_REF_SCHEMA_ID
    schema_version: Literal[CONTENT_MANIFEST_REF_SCHEMA_VERSION] = (
        CONTENT_MANIFEST_REF_SCHEMA_VERSION
    )
    kind: str = Field(min_length=1)
    id: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    object: ContentObjectRef

    @model_validator(mode="after")
    def _validate_ref(self) -> "ContentManifestRef":
        expected = f"{CONTENT_MANIFEST_REF_PREFIX}{self.kind}/{self.id}"
        if self.ref != expected:
            raise ValueError("content manifest ref does not match its kind and id")
        if any("\x00" in value for value in (self.kind, self.id)):
            raise ValueError("content manifest kind/id must be NUL-free")
        return self


class ContentBinding(StrictModel):
    """A runtime role attached separately to role-free tree identity."""

    schema_id: Literal[CONTENT_BINDING_SCHEMA_ID] = CONTENT_BINDING_SCHEMA_ID
    schema_version: Literal[CONTENT_BINDING_SCHEMA_VERSION] = CONTENT_BINDING_SCHEMA_VERSION
    binding_name: str = Field(min_length=1)
    root_kind: str = Field(min_length=1)
    tree: ContentTreeRef


class ContentMigrationMapping(StrictModel):
    """One old locator mapped to its canonical successor reference."""

    source_type: Literal["staged-root-v1", "manifest-packet-v1", "path-backed-import"]
    source_ref: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_ref: str
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ContentMigrationReceipt(StrictModel):
    """Complete deterministic receipt emitted by an explicit offline migration."""

    schema_id: Literal[CONTENT_MIGRATION_RECEIPT_SCHEMA_ID] = CONTENT_MIGRATION_RECEIPT_SCHEMA_ID
    schema_version: Literal[CONTENT_MIGRATION_RECEIPT_SCHEMA_VERSION] = (
        CONTENT_MIGRATION_RECEIPT_SCHEMA_VERSION
    )
    source_schema_version: str
    destination_store: str
    mappings: tuple[ContentMigrationMapping, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactBlobProvider(Protocol):
    """Scientific-core byte custody dependency supplied by a persistence adapter."""

    def store_bytes(
        self,
        data: bytes,
        *,
        role: str,
        logical_name: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef: ...

    def get_bytes(
        self,
        artifact: ArtifactRef | str,
        *,
        size_bytes: int | None = None,
    ) -> bytes: ...

    def store_object(self, data: bytes) -> ContentObjectRef: ...

    def get_object(
        self,
        object_ref: ContentObjectRef | str,
        *,
        size_bytes: int | None = None,
    ) -> bytes: ...

    def bind_manifest(
        self,
        kind: str,
        manifest_id: str,
        raw_bytes: bytes,
    ) -> ContentManifestRef: ...

    def resolve_manifest(
        self,
        kind: str,
        manifest_id: str,
    ) -> tuple[ContentManifestRef, bytes]: ...


class ImmutableArtifactBlobProviderConfig(StrictModel):
    """Portable configuration for the fixed local SHA-256 CAS provider."""

    storage_backend: Literal["feedbax-local"] = IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND


class ImmutableArtifactBlobProviderSpec(StrictModel):
    """Root-free portable specification for immutable artifact blob custody."""

    schema_id: Literal["feedbax.spec.immutable_artifact_blob_provider"] = (
        IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.immutable_artifact_blob_provider.v1"] = (
        IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION
    )
    kind: Literal["feedbax-local-sha256-cas"] = IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND
    config: ImmutableArtifactBlobProviderConfig = Field(
        default_factory=ImmutableArtifactBlobProviderConfig
    )


__all__ = [
    "CONTENT_BINDING_SCHEMA_ID",
    "CONTENT_BINDING_SCHEMA_VERSION",
    "CONTENT_MANIFEST_REF_PREFIX",
    "CONTENT_MANIFEST_REF_SCHEMA_ID",
    "CONTENT_MANIFEST_REF_SCHEMA_VERSION",
    "CONTENT_MIGRATION_RECEIPT_SCHEMA_ID",
    "CONTENT_MIGRATION_RECEIPT_SCHEMA_VERSION",
    "CONTENT_OBJECT_REF_PREFIX",
    "CONTENT_OBJECT_REF_SCHEMA_ID",
    "CONTENT_OBJECT_REF_SCHEMA_VERSION",
    "CONTENT_TREE_REF_PREFIX",
    "CONTENT_TREE_REF_SCHEMA_ID",
    "CONTENT_TREE_REF_SCHEMA_VERSION",
    "CONTENT_TREE_SCHEMA_ID",
    "CONTENT_TREE_SCHEMA_VERSION",
    "LEGACY_CONTENT_MIGRATION_COMMAND",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION",
    "IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND",
    "ArtifactBlobContainmentError",
    "ArtifactBlobCustodyError",
    "ArtifactBlobIntegrityError",
    "ArtifactBlobProvider",
    "ArtifactBlobReferenceError",
    "ContentBinding",
    "ContentManifestConflictError",
    "ContentManifestRef",
    "ContentMigrationMapping",
    "ContentMigrationReceipt",
    "ContentObjectRef",
    "ContentTree",
    "ContentTreeFile",
    "ContentTreeRef",
    "ImmutableArtifactBlobProviderConfig",
    "ImmutableArtifactBlobProviderSpec",
    "LegacyContentFormatError",
]
