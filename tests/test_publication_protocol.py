"""Boundary laws for exact-byte transactional publication."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.contracts.publication import (
    ArtifactRecord,
    BlobRef,
    ExactRef,
    ProvenanceEdge,
    PublicationRequest,
    PublicationService,
    artifact_record,
)
from feedbax.persistence.publication import (
    LocalBlobStore,
    PublicationConflictError,
    SQLitePublicationCatalog,
    UnsupportedPublicationSchemaError,
)
from feedbax.workflow.publication import (
    ArtifactPayload,
    CheckpointPayload,
    SISU_ARTIFACT_CHAIN,
    publish_sisu_artifact_chain,
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
                "logical_id": "sisu.report",
                "version_id": "artifact-version:unverified",
                "role": "report",
                "media_type": "application/pdf",
                "payload_schema_id": "sisu.report",
                "payload_schema_version": "sisu.report.v1",
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
        logical_id="sisu.analysis",
        role="analysis",
        media_type="application/json",
        payload_schema_id="sisu.analysis",
        payload_schema_version="sisu.analysis.v1",
        blobs=(blob,),
    )
    request = PublicationRequest(idempotency_key="sisu-publication", artifacts=(record,))
    first = service.publish(request)
    assert service.publish(request) == first
    assert catalog.receipt(first.publication_id) == first
    assert catalog.artifact(record.version_id) == record

    changed = artifact_record(
        logical_id="sisu.analysis",
        role="analysis",
        media_type="application/json",
        payload_schema_id="sisu.analysis",
        payload_schema_version="sisu.analysis.v1",
        blobs=(service.stage(b"changed"),),
    )
    with pytest.raises(PublicationConflictError, match="different canonical content"):
        service.publish(
            PublicationRequest(idempotency_key="sisu-publication", artifacts=(changed,))
        )


def test_publication_rolls_back_every_logical_record_on_late_failure(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    record = artifact_record(
        logical_id="sisu.figure",
        role="figure",
        media_type="image/svg+xml",
        payload_schema_id="sisu.figure",
        payload_schema_version="sisu.figure.v1",
        blobs=(service.stage(b"<svg/>"),),
    )
    edge = ProvenanceEdge(
        relation="produced_by",
        subject=record.exact_ref,
        object=_ref("workflow_plan", "sisu-plan", b"plan"),
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
        logical_id="sisu.analysis",
        role="analysis",
        media_type="application/json",
        payload_schema_id="sisu.analysis",
        payload_schema_version="sisu.analysis.v1",
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
        logical_id="sisu.report",
        role="report",
        media_type="application/pdf",
        payload_schema_id="sisu.report",
        payload_schema_version="sisu.report.v1",
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


def _checkpoint_payload(step: int) -> CheckpointPayload:
    return CheckpointPayload(
        progress={"step": step},
        prng_state=f"rng-{step}".encode(),
        slots={
            "controller": (
                "controller_state",
                "sisu.controller.structure.v1",
                "feedbax.array_codec",
                "feedbax.array_codec.v1",
                f"params-{step}".encode(),
            ),
            "optimizer": (
                "optimizer_state",
                "sisu.optimizer.structure.v1",
                "feedbax.array_codec",
                "feedbax.array_codec.v1",
                f"optimizer-{step}".encode(),
            ),
        },
    )


def test_sisu_exemplar_publishes_complete_chain_and_checkpoints_once(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    payloads = {
        role: ArtifactPayload(
            data=f"{role} bytes".encode(),
            media_type="application/json",
            schema_id=f"sisu.{role}",
            schema_version=f"sisu.{role}.v1",
        )
        for role in SISU_ARTIFACT_CHAIN
    }
    arguments = {
        "idempotency_key": "sisu-complete-chain",
        "study_id": "sisu-continuous-conditioning",
        "training_program_id": "sisu.training.continuous",
        "workflow_plan": _ref("workflow_plan", "sisu-plan", b"plan"),
        "graph": _ref("semantic_ir", "sisu-graph", b"graph"),
        "experiment": _ref("document_revision", "sisu-experiment", b"experiment"),
        "payloads": payloads,
        "trained_checkpoint": _checkpoint_payload(100),
        "continued_checkpoint": _checkpoint_payload(200),
    }
    first = publish_sisu_artifact_chain(service, **arguments)
    assert publish_sisu_artifact_chain(service, **arguments) == first
    assert len(first.artifact_refs) == 6
    assert len(first.checkpoint_refs) == 2
    assert [catalog.artifact(ref.identity).role for ref in first.artifact_refs] == list(
        SISU_ARTIFACT_CHAIN
    )
    continued = catalog.checkpoint(first.checkpoint_refs[1].identity)
    assert continued is not None
    assert continued.parent == first.checkpoint_refs[0]


def test_sisu_exemplar_refuses_a_partial_chain_before_publication(tmp_path: Path) -> None:
    _, catalog, service = _stack(tmp_path)
    with pytest.raises(ValueError, match="complete artifact chain"):
        publish_sisu_artifact_chain(
            service,
            idempotency_key="partial",
            study_id="sisu",
            training_program_id="sisu.training",
            workflow_plan=_ref("workflow_plan", "plan", b"plan"),
            graph=_ref("semantic_ir", "graph", b"graph"),
            experiment=_ref("document_revision", "experiment", b"experiment"),
            payloads={},
            trained_checkpoint=_checkpoint_payload(1),
            continued_checkpoint=_checkpoint_payload(2),
        )
    assert catalog.receipt("publication:missing") is None
