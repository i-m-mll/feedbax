"""Persistence and indexing helpers for Feedbax artifacts."""

from feedbax.contracts.artifact_custody import (
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID,
    IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION,
    IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND,
    ImmutableArtifactBlobProviderConfig,
    ImmutableArtifactBlobProviderSpec,
)
from feedbax.persistence.artifact_custody import (
    ArtifactBlobContainmentError,
    ArtifactBlobCustodyError,
    ArtifactBlobIntegrityError,
    ArtifactBlobReferenceError,
    ImmutableArtifactBlobProvider,
    open_immutable_artifact_blob_provider,
)
from feedbax.persistence.publication import (
    LocalBlobStore,
    PublicationConflictError,
    PublicationError,
    SQLitePublicationCatalog,
    UnsupportedPublicationSchemaError,
)

__all__ = [
    "ArtifactBlobContainmentError",
    "ArtifactBlobCustodyError",
    "ArtifactBlobIntegrityError",
    "ArtifactBlobReferenceError",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_KIND",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_ID",
    "IMMUTABLE_ARTIFACT_BLOB_PROVIDER_SCHEMA_VERSION",
    "IMMUTABLE_ARTIFACT_BLOB_STORAGE_BACKEND",
    "ImmutableArtifactBlobProvider",
    "ImmutableArtifactBlobProviderConfig",
    "ImmutableArtifactBlobProviderSpec",
    "LocalBlobStore",
    "PublicationConflictError",
    "PublicationError",
    "SQLitePublicationCatalog",
    "UnsupportedPublicationSchemaError",
    "open_immutable_artifact_blob_provider",
]
