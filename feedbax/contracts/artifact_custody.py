"""Portable contract for opening the immutable local artifact blob provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.base import ArtifactRef


IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID = "feedbax.spec.immutable_artifact_blob_provider"
IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION = "feedbax.spec.immutable_artifact_blob_provider.v1"
IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND = "feedbax-local-sha256-cas"
IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND = "feedbax-local"


class ArtifactBlobCustodyError(ValueError):
    """Base error exposed by an immutable artifact custody provider."""


class ArtifactBlobReferenceError(ArtifactBlobCustodyError):
    """An artifact identifier or reference is invalid."""


class ArtifactBlobIntegrityError(ArtifactBlobCustodyError):
    """Immutable artifact bytes fail identity validation."""


class ArtifactBlobContainmentError(ArtifactBlobCustodyError):
    """A custody path escapes or aliases its canonical store."""


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
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION",
    "IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND",
    "ArtifactBlobContainmentError",
    "ArtifactBlobCustodyError",
    "ArtifactBlobIntegrityError",
    "ArtifactBlobProvider",
    "ArtifactBlobReferenceError",
    "ImmutableArtifactBlobProviderConfig",
    "ImmutableArtifactBlobProviderSpec",
]
