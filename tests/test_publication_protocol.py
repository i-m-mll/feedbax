"""Boundary laws for exact-byte transactional publication."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.contracts.publication import (
    ArtifactRecord,
    BlobRef,
    CheckpointSet,
    CheckpointSlot,
    ExactRef,
    ProvenanceEdge,
    PublicationRequest,
    PublicationService,
    artifact_record,
    checkpoint_set_id,
)
from feedbax.persistence.publication import (
    LocalBlobStore,
    PublicationConflictError,
    SQLitePublicationCatalog,
    UnsupportedPublicationSchemaError,
)


def _stack(tmp_path: Path) -> tuple[LocalBlobStore, SQLitePublicationCatalog, PublicationService]:
    blobs = LocalBlobStore(tmp_path / "custody")
    catalog = SQLitePublicationCatalog(tmp_path / "publication.sqlite")
    return blobs, catalog, PublicationService(blobs, catalog)


def _ref(domain: str, identity: str, data: bytes) -> ExactRef:
    return ExactRef(domain=domain, identity=identity, bytes=BlobRef.from_bytes(data))


def test_exact_references_cannot_omit_digest_size_or_identity() -> None:
    with pytest.raises(ValidationError):
        BlobRef.model_validate({"digest": "a" * 64})
    with pytest.raises(ValidationError):
        ExactRef.model_validate(
            {"domain": "artifact_version", "bytes": {"digest": "a" * 64, "size_bytes": 1}}
        )
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(
            {
                "logical_id": "example.report",
                "version_id": "artifact-version:unverified",
                "role": "report",
                "media_type": "application/pdf",
                "payload_schema_id": "example.report",
                "payload_schema_version": "example.report.v1",
                "blobs": [],
            }
        )


def test_local_blob_store_is_content_addressed_verified_and_idempotent(tmp_path: Path) -> None:
    blobs, _, _ = _stack(tmp_path)
    first = blobs.stage(b"exact bytes")
    second = blobs.stage(b"exact bytes")
    assert first == second
    assert blobs.read(first) == b"exact bytes"
    assert blobs.enumerate() == (first,)


def test_catalog_rejects_an_unsupported_durable_schema_without_migration(tmp_path: Path) -> None:
    path = tmp_path / "publication.sqlite"
    SQLitePublicationCatalog(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE publication_metadata SET schema_version = ? WHERE singleton = 1",
            ("feedbax.publication.v0",),
        )
    with pytest.raises(UnsupportedPublicationSchemaError, match="no implicit migration"):
        SQLitePublicationCatalog(path)


def test_catalog_rejects_an_old_schema_before_creating_current_tables(tmp_path: Path) -> None:
    path = tmp_path / "old-publication.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE publication_metadata "
            "(singleton INTEGER PRIMARY KEY, schema_id TEXT, schema_version TEXT)"
        )
        connection.execute(
            "INSERT INTO publication_metadata VALUES (1, ?, ?)",
            ("feedbax.publication", "feedbax.publication.v0"),
        )
    with pytest.raises(UnsupportedPublicationSchemaError, match="no implicit migration"):
        SQLitePublicationCatalog(path)
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert tables == {"publication_metadata"}


def test_publication_is_idempotent_and_conflicting_replay_fails_closed(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    blob = service.stage(b"analysis")
    record = artifact_record(
        logical_id="example.analysis",
        role="analysis",
        media_type="application/json",
        payload_schema_id="example.analysis",
        payload_schema_version="example.analysis.v1",
        blobs=(blob,),
    )
    request = PublicationRequest(idempotency_key="example-publication", artifacts=(record,))
    first = service.publish(request)
    assert service.publish(request) == first
    assert catalog.receipt(first.publication_id) == first
    assert catalog.artifact(record.version_id) == record

    changed = artifact_record(
        logical_id="example.analysis",
        role="analysis",
        media_type="application/json",
        payload_schema_id="example.analysis",
        payload_schema_version="example.analysis.v1",
        blobs=(service.stage(b"changed"),),
    )
    with pytest.raises(PublicationConflictError, match="different canonical content"):
        service.publish(
            PublicationRequest(idempotency_key="example-publication", artifacts=(changed,))
        )


def test_publication_commits_one_complete_checkpoint_set(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    slot = CheckpointSlot(
        name="model",
        state_type="model_state",
        array_structure_id="example.model.structure.v1",
        codec_schema_id="feedbax.array_codec",
        codec_schema_version="feedbax.array_codec.v1",
        blob=service.stage(b"model state"),
    )
    values = {
        "training_program_id": "example.training",
        "graph": _ref("semantic_ir", "example-graph", b"graph"),
        "experiment": _ref("document_revision", "example-experiment", b"experiment"),
        "progress": {"step": 10},
        "prng_state": service.stage(b"prng state"),
        "slots": (slot,),
        "continuation": "fork",
        "parent": None,
    }
    checkpoint = CheckpointSet(checkpoint_id=checkpoint_set_id(**values), **values)
    receipt = service.publish(
        PublicationRequest(
            idempotency_key="example-checkpoint", artifacts=(), checkpoints=(checkpoint,)
        )
    )
    assert receipt.checkpoint_refs == (checkpoint.exact_ref,)
    assert catalog.checkpoint(checkpoint.checkpoint_id) == checkpoint


def test_publication_rolls_back_every_logical_record_on_late_failure(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    record = artifact_record(
        logical_id="example.figure",
        role="figure",
        media_type="image/svg+xml",
        payload_schema_id="example.figure",
        payload_schema_version="example.figure.v1",
        blobs=(service.stage(b"<svg/>"),),
    )
    edge = ProvenanceEdge(
        relation="produced_by",
        subject=record.exact_ref,
        object=_ref("workflow_plan", "example-plan", b"plan"),
    )
    request = PublicationRequest(
        idempotency_key="rollback",
        artifacts=(record,),
        provenance=(edge,),
    )
    with sqlite3.connect(catalog.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_provenance
            BEFORE INSERT ON provenance_edges
            BEGIN SELECT RAISE(ABORT, 'injected late failure'); END
            """
        )
    with pytest.raises(PublicationConflictError, match="conflicts"):
        service.publish(request)
    assert catalog.receipt(request.publication_id) is None
    assert catalog.artifact(record.version_id) is None
    with sqlite3.connect(catalog.path) as connection:
        assert connection.execute("SELECT count(*) FROM provenance_edges").fetchone() == (0,)


def test_catalog_reads_fail_closed_when_exact_record_bytes_are_corrupted(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    record = artifact_record(
        logical_id="example.analysis",
        role="analysis",
        media_type="application/json",
        payload_schema_id="example.analysis",
        payload_schema_version="example.analysis.v1",
        blobs=(service.stage(b"analysis"),),
    )
    service.publish(PublicationRequest(idempotency_key="corruption", artifacts=(record,)))
    with sqlite3.connect(catalog.path) as connection:
        connection.execute(
            "UPDATE artifacts SET record_size_bytes = record_size_bytes + 1 "
            "WHERE version_id = ?",
            (record.version_id,),
        )
    with pytest.raises(PublicationConflictError, match="record integrity failed"):
        catalog.artifact(record.version_id)


def test_publication_refuses_lineage_to_unpublished_logical_records(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    record = artifact_record(
        logical_id="example.report",
        role="report",
        media_type="application/pdf",
        payload_schema_id="example.report",
        payload_schema_version="example.report.v1",
        blobs=(service.stage(b"report"),),
    )
    unknown = ExactRef(
        domain="artifact_version",
        identity="artifact-version:sha256:" + "f" * 64,
        bytes=BlobRef(digest="e" * 64, size_bytes=12),
    )
    request = PublicationRequest(
        idempotency_key="false-lineage",
        artifacts=(record,),
        provenance=(
            ProvenanceEdge(relation="derived_from", subject=record.exact_ref, object=unknown),
        ),
    )
    with pytest.raises(PublicationConflictError, match="unknown or mismatched artifact"):
        service.publish(request)
    assert catalog.artifact(record.version_id) is None
