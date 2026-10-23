"""Local implementations of the storage-neutral publication protocols."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feedbax.contracts.publication import (
    PUBLICATION_SCHEMA_ID,
    PUBLICATION_SCHEMA_VERSION,
    ArtifactRecord,
    BlobRef,
    CheckpointSet,
    ExactRef,
    PublicationReceipt,
    PublicationRequest,
    canonical_bytes,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


class PublicationError(RuntimeError):
    """Base failure at the logical publication boundary."""


class PublicationConflictError(PublicationError):
    """An identity or idempotency key was replayed with different bytes."""


class UnsupportedPublicationSchemaError(PublicationError):
    """The catalog declares a schema this implementation cannot interpret."""


class LocalBlobStore:
    """The sole local BlobStore, backed by the hardened suffixless SHA-256 CAS."""

    def __init__(self, root: Path | str) -> None:
        self._provider = ImmutableArtifactBlobProvider(root=Path(root).absolute())

    @property
    def root(self) -> Path:
        return Path(self._provider.root)

    def stage(self, data: bytes) -> BlobRef:
        stored = self._provider.store_bytes(
            data,
            role="blob",
            logical_name="immutable-bytes",
        )
        return BlobRef(digest=stored.sha256, size_bytes=stored.size_bytes)

    def verify(self, ref: BlobRef) -> None:
        self.read(ref)

    def read(self, ref: BlobRef) -> bytes:
        return self._provider.get_bytes(
            f"artifact://sha256/{ref.digest}",
            size_bytes=ref.size_bytes,
        )

    def enumerate(self) -> tuple[BlobRef, ...]:
        store = self.root / "artifacts" / "sha256"
        if not store.exists():
            return ()
        refs: list[BlobRef] = []
        for path in sorted(store.glob("[0-9a-f][0-9a-f]/*")):
            if (
                len(path.name) != 64
                or any(character not in "0123456789abcdef" for character in path.name)
                or path.parent.name != path.name[:2]
            ):
                continue
            ref = BlobRef(digest=path.name, size_bytes=path.stat().st_size)
            self.verify(ref)
            refs.append(ref)
        return tuple(refs)


class SQLitePublicationCatalog:
    """Transactional, replay-safe local publication catalog.

    Immutable blobs may be staged before this transaction. Artifact, checkpoint,
    provenance, and publication rows become visible in one SQLite commit.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            metadata_exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'publication_metadata'"
            ).fetchone()
            if metadata_exists is not None:
                row = connection.execute(
                    "SELECT schema_id, schema_version FROM publication_metadata "
                    "WHERE singleton = 1"
                ).fetchone()
                if row != (PUBLICATION_SCHEMA_ID, PUBLICATION_SCHEMA_VERSION):
                    observed = "missing metadata row" if row is None else f"{row[0]} {row[1]}"
                    raise UnsupportedPublicationSchemaError(
                        "publication catalog schema is unsupported; no implicit migration exists: "
                        + observed
                    )
            connection.execute("BEGIN IMMEDIATE")
            statements = (
                """
                CREATE TABLE IF NOT EXISTS publication_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS publications (
                    publication_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_sha256 TEXT NOT NULL,
                    receipt_json TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    version_id TEXT PRIMARY KEY,
                    logical_id TEXT NOT NULL,
                    publication_id TEXT NOT NULL REFERENCES publications(publication_id),
                    record_sha256 TEXT NOT NULL,
                    record_size_bytes INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS artifact_blobs (
                    version_id TEXT NOT NULL REFERENCES artifacts(version_id),
                    ordinal INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    PRIMARY KEY (version_id, ordinal)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    publication_id TEXT NOT NULL REFERENCES publications(publication_id),
                    record_sha256 TEXT NOT NULL,
                    record_size_bytes INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS provenance_edges (
                    edge_id TEXT PRIMARY KEY,
                    publication_id TEXT NOT NULL REFERENCES publications(publication_id),
                    edge_json TEXT NOT NULL
                )
                """,
            )
            for statement in statements:
                connection.execute(statement)
            if metadata_exists is None:
                connection.execute(
                    "INSERT INTO publication_metadata VALUES (1, ?, ?)",
                    (PUBLICATION_SCHEMA_ID, PUBLICATION_SCHEMA_VERSION),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def commit(self, request: PublicationRequest) -> PublicationReceipt:
        artifact_refs = tuple(record.exact_ref for record in request.artifacts)
        checkpoint_refs = tuple(checkpoint.exact_ref for checkpoint in request.checkpoints)
        receipt = PublicationReceipt(
            publication_id=request.publication_id,
            idempotency_key=request.idempotency_key,
            request_sha256=request.request_sha256,
            artifact_refs=artifact_refs,
            checkpoint_refs=checkpoint_refs,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_sha256, receipt_json FROM publications WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing[0] != request.request_sha256:
                    raise PublicationConflictError(
                        "publication idempotency key already names different canonical content"
                    )
                persisted = PublicationReceipt.model_validate_json(existing[1])
                if persisted != receipt:
                    raise PublicationConflictError(
                        "publication receipt differs from the replayed canonical content"
                    )
                connection.commit()
                return persisted

            self._validate_exact_lineage(connection, request)

            connection.execute(
                "INSERT INTO publications VALUES (?, ?, ?, ?)",
                (
                    request.publication_id,
                    request.idempotency_key,
                    request.request_sha256,
                    receipt.model_dump_json(),
                ),
            )
            for record, exact_ref in zip(request.artifacts, artifact_refs, strict=True):
                connection.execute(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.version_id,
                        record.logical_id,
                        request.publication_id,
                        exact_ref.bytes.digest,
                        exact_ref.bytes.size_bytes,
                        canonical_bytes(record).decode("utf-8"),
                    ),
                )
                connection.executemany(
                    "INSERT INTO artifact_blobs VALUES (?, ?, ?, ?)",
                    [
                        (record.version_id, ordinal, blob.digest, blob.size_bytes)
                        for ordinal, blob in enumerate(record.blobs)
                    ],
                )
            for checkpoint, exact_ref in zip(request.checkpoints, checkpoint_refs, strict=True):
                connection.execute(
                    "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                    (
                        checkpoint.checkpoint_id,
                        request.publication_id,
                        exact_ref.bytes.digest,
                        exact_ref.bytes.size_bytes,
                        canonical_bytes(checkpoint).decode("utf-8"),
                    ),
                )
            connection.executemany(
                "INSERT INTO provenance_edges VALUES (?, ?, ?)",
                [
                    (
                        edge.identity,
                        request.publication_id,
                        canonical_bytes(edge).decode("utf-8"),
                    )
                    for edge in request.provenance
                ],
            )
            connection.commit()
            return receipt
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PublicationConflictError(
                "publication identity conflicts with already committed canonical content"
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_exact_lineage(
        connection: sqlite3.Connection, request: PublicationRequest
    ) -> None:
        request_artifacts = {record.version_id: record.exact_ref for record in request.artifacts}
        request_checkpoints = {
            checkpoint.checkpoint_id: checkpoint.exact_ref for checkpoint in request.checkpoints
        }
        for edge in request.provenance:
            for ref in (edge.subject, edge.object):
                if ref.domain == "artifact_version":
                    expected = request_artifacts.get(ref.identity)
                    if expected is None:
                        row = connection.execute(
                            "SELECT record_sha256, record_size_bytes FROM artifacts "
                            "WHERE version_id = ?",
                            (ref.identity,),
                        ).fetchone()
                        if row is not None:
                            expected = ExactRef(
                                domain="artifact_version",
                                identity=ref.identity,
                                bytes=BlobRef(digest=row[0], size_bytes=row[1]),
                            )
                    if expected != ref:
                        raise PublicationConflictError(
                            f"provenance references unknown or mismatched artifact {ref.identity}"
                        )
                if ref.domain == "checkpoint_set":
                    expected = request_checkpoints.get(ref.identity)
                    if expected is None:
                        row = connection.execute(
                            "SELECT record_sha256, record_size_bytes FROM checkpoints "
                            "WHERE checkpoint_id = ?",
                            (ref.identity,),
                        ).fetchone()
                        if row is not None:
                            expected = ExactRef(
                                domain="checkpoint_set",
                                identity=ref.identity,
                                bytes=BlobRef(digest=row[0], size_bytes=row[1]),
                            )
                    if expected != ref:
                        raise PublicationConflictError(
                            f"provenance references unknown or mismatched checkpoint {ref.identity}"
                        )

    def receipt(self, publication_id: str) -> PublicationReceipt | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT idempotency_key, request_sha256, receipt_json "
                "FROM publications WHERE publication_id = ?",
                (publication_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        receipt = PublicationReceipt.model_validate_json(row[2])
        if (
            receipt.publication_id != publication_id
            or receipt.idempotency_key != row[0]
            or receipt.request_sha256 != row[1]
        ):
            raise PublicationConflictError(f"publication receipt integrity failed: {publication_id}")
        return receipt

    def artifact(self, version_id: str) -> ArtifactRecord | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT record_sha256, record_size_bytes, record_json "
                "FROM artifacts WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        record = ArtifactRecord.model_validate_json(row[2])
        if record.version_id != version_id or BlobRef.from_bytes(canonical_bytes(record)) != BlobRef(
            digest=row[0], size_bytes=row[1]
        ):
            raise PublicationConflictError(f"artifact record integrity failed: {version_id}")
        return record

    def checkpoint(self, checkpoint_id: str) -> CheckpointSet | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT record_sha256, record_size_bytes, record_json "
                "FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        checkpoint = CheckpointSet.model_validate_json(row[2])
        if checkpoint.checkpoint_id != checkpoint_id or BlobRef.from_bytes(
            canonical_bytes(checkpoint)
        ) != BlobRef(digest=row[0], size_bytes=row[1]):
            raise PublicationConflictError(f"checkpoint record integrity failed: {checkpoint_id}")
        return checkpoint


__all__ = [
    "LocalBlobStore",
    "PublicationConflictError",
    "PublicationError",
    "SQLitePublicationCatalog",
    "UnsupportedPublicationSchemaError",
]
