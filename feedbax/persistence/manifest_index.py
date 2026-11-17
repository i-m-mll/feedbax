"""SQLite indexing for Feedbax manifest files."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from feedbax.contracts.base import (
    ArtifactRef,
    default_manifest_root,
)
from feedbax.contracts.manifest import (
    BaseManifest,
    load_manifest,
)


def default_index_path(root: Path | str | None = None) -> Path:
    root_path = Path(root) if root is not None else default_manifest_root()
    return root_path / "index" / "feedbax.sqlite"


def connect_index(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    if conn.execute("PRAGMA user_version").fetchone()[0] >= 2:
        return
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 2:
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS manifests (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                provider_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT,
                path TEXT NOT NULL,
                source_repo TEXT,
                source_branch TEXT,
                source_commit TEXT,
                dirty INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                manifest_id TEXT NOT NULL,
                role TEXT NOT NULL,
                logical_name TEXT NOT NULL,
                artifact_id TEXT,
                sha256 TEXT,
                media_type TEXT NOT NULL,
                size_bytes INTEGER,
                storage_backend TEXT NOT NULL,
                uri TEXT,
                PRIMARY KEY (manifest_id, role, logical_name, artifact_id, uri),
                FOREIGN KEY (manifest_id) REFERENCES manifests(id) ON DELETE CASCADE
            )
            """
        )
        # Older releases still write this derived table when sharing the index.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage_edges (
                child_id TEXT NOT NULL,
                parent_kind TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                role TEXT,
                PRIMARY KEY (child_id, parent_kind, parent_id, role),
                FOREIGN KEY (child_id) REFERENCES manifests(id) ON DELETE CASCADE
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(manifests)")}
        if "payload_json" in columns:
            conn.execute("ALTER TABLE manifests DROP COLUMN payload_json")
        conn.execute("PRAGMA user_version = 2")


def _artifact_tuple(manifest_id: str, artifact: ArtifactRef) -> tuple:
    return (
        manifest_id,
        artifact.role,
        artifact.logical_name,
        artifact.artifact_id,
        artifact.sha256,
        artifact.media_type,
        artifact.size_bytes,
        artifact.storage_backend,
        artifact.uri,
    )


def index_manifest(
    conn: sqlite3.Connection,
    manifest: BaseManifest,
    *,
    path: Path | str,
) -> None:
    """Update one manifest projection in the caller's transaction."""
    prov = manifest.provenance
    conn.execute(
        """
        INSERT OR REPLACE INTO manifests (
            id, kind, schema_version, provider_version, created_at, status, path,
            source_repo, source_branch, source_commit, dirty
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            manifest.id,
            manifest.kind,
            manifest.schema_version,
            manifest.provider_version,
            manifest.created_at.isoformat(),
            manifest.status,
            str(path),
            prov.source_repo,
            prov.source_branch,
            prov.source_commit,
            int(prov.dirty) if prov.dirty is not None else None,
        ),
    )
    conn.execute("DELETE FROM artifacts WHERE manifest_id = ?", (manifest.id,))
    conn.executemany(
        """
        INSERT OR REPLACE INTO artifacts (
            manifest_id, role, logical_name, artifact_id, sha256, media_type,
            size_bytes, storage_backend, uri
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [_artifact_tuple(manifest.id, artifact) for artifact in manifest.artifacts],
    )


def index_manifest_file(
    manifest_path: Path | str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> None:
    path = Path(manifest_path)
    conn_path = Path(db_path) if db_path is not None else default_index_path(root)
    with closing(connect_index(conn_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        index_manifest(conn, load_manifest(path), path=path)


def find_manifest_paths_by_id(
    manifest_id: str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> list[Path]:
    """Return indexed manifest paths for a deterministic manifest ID."""
    root_path = Path(root) if root is not None else default_manifest_root()
    conn_path = Path(db_path) if db_path is not None else default_index_path(root_path)
    if not conn_path.exists():
        return []
    with closing(connect_index(conn_path)) as conn:
        return query_manifest_paths_by_id(conn, manifest_id)


def query_manifest_paths_by_id(conn: sqlite3.Connection, manifest_id: str) -> list[Path]:
    """Return locator paths using the caller's connection and snapshot."""
    rows = conn.execute(
        "SELECT path FROM manifests WHERE id = ? ORDER BY path",
        (manifest_id,),
    ).fetchall()
    return [Path(row[0]) for row in rows]


def _record_with_payload(row: tuple[str, str, str]) -> dict[str, Any]:
    manifest_id, kind, path = row
    manifest = load_manifest(Path(path))
    if (manifest.id, manifest.kind) != (manifest_id, kind):
        raise ValueError(
            "manifest index identity disagrees with canonical file: "
            f"index={(kind, manifest_id)!r}, file={(manifest.kind, manifest.id)!r}, path={path}"
        )
    return {
        "id": manifest.id,
        "kind": manifest.kind,
        "schema_version": manifest.schema_version,
        "provider_version": manifest.provider_version,
        "created_at": manifest.created_at.isoformat(),
        "status": manifest.status,
        "path": path,
        "payload": manifest.model_dump(mode="json", exclude_none=True),
    }


def get_indexed_manifest_record(
    manifest_id: str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return one indexed manifest row with its JSON payload."""
    root_path = Path(root) if root is not None else default_manifest_root()
    conn_path = Path(db_path) if db_path is not None else default_index_path(root_path)
    if not conn_path.exists():
        return None
    conn = connect_index(conn_path)
    try:
        row = conn.execute(
            """
            SELECT id, kind, path
            FROM manifests
            WHERE id = ?
            """,
            (manifest_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _record_with_payload(row)


def iter_indexed_manifest_records_by_kind(
    manifest_kind: str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return indexed manifest rows for one manifest kind with JSON payloads."""
    root_path = Path(root) if root is not None else default_manifest_root()
    conn_path = Path(db_path) if db_path is not None else default_index_path(root_path)
    if not conn_path.exists():
        return []
    conn = connect_index(conn_path)
    try:
        rows = conn.execute(
            """
            SELECT id, kind, path
            FROM manifests
            WHERE kind = ?
            """,
            (manifest_kind,),
        ).fetchall()
    finally:
        conn.close()
    return sorted(
        (_record_with_payload(row) for row in rows),
        key=lambda record: (record["created_at"], record["path"]),
        reverse=True,
    )


def remove_manifest_from_index(
    manifest_id: str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Remove a manifest and dependent index rows without deleting bytes."""
    root_path = Path(root) if root is not None else default_manifest_root()
    conn_path = Path(db_path) if db_path is not None else default_index_path(root_path)
    if not conn_path.exists():
        return
    conn = connect_index(conn_path)
    try:
        with conn:
            conn.execute("DELETE FROM manifests WHERE id = ?", (manifest_id,))
    finally:
        conn.close()


def iter_indexed_manifest_paths_by_kind(
    manifest_kind: str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> list[Path]:
    """Return indexed manifest paths for one manifest kind."""
    root_path = Path(root) if root is not None else default_manifest_root()
    conn_path = Path(db_path) if db_path is not None else default_index_path(root_path)
    if not conn_path.exists():
        return []
    conn = connect_index(conn_path)
    try:
        rows = conn.execute(
            "SELECT path FROM manifests WHERE kind = ? ORDER BY created_at, path",
            (manifest_kind,),
        ).fetchall()
    finally:
        conn.close()
    return [Path(row[0]) for row in rows]


def iter_manifest_files(root: Path | str | None = None) -> list[Path]:
    root_path = Path(root) if root is not None else default_manifest_root()
    manifests_dir = root_path / "manifests"
    if not manifests_dir.exists():
        return []
    return sorted(manifests_dir.glob("**/*.json"))


def rebuild_manifest_index(
    root: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
) -> Path:
    root_path = Path(root) if root is not None else default_manifest_root()
    conn_path = Path(db_path) if db_path is not None else default_index_path(root_path)
    with closing(connect_index(conn_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM manifests")
        seen: dict[str, Path] = {}
        for path in iter_manifest_files(root_path):
            manifest = load_manifest(path)
            if manifest.id in seen:
                raise ValueError(
                    f"duplicate manifest ID {manifest.id!r}: {seen[manifest.id]} and {path}"
                )
            seen[manifest.id] = path
            index_manifest(conn, manifest, path=path)
    return conn_path
