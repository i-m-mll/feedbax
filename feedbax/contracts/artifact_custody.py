"""Portable contract for opening the immutable local artifact blob provider."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from feedbax.contracts.manifest import StrictModel


IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID = "feedbax.spec.immutable_artifact_blob_provider"
IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION = "feedbax.spec.immutable_artifact_blob_provider.v1"
IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND = "feedbax-local-sha256-cas"
IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND = "feedbax-local"


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
    "ImmutableArtifactBlobProviderConfig",
    "ImmutableArtifactBlobProviderSpec",
]
