"""SQLite indexing for Feedbax manifest files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feedbax.contracts.manifest import ArtifactRef, BaseManifest, default_manifest_root, load_manifest


def default_index_path(root: Path | str | None = None) -> Path:
    root_path = Path(root) if root is not None else default_manifest_root()
    return root_path / "index" / "feedbax.sqlite"


def connect_index(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
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
            dirty INTEGER,
            payload_json TEXT NOT NULL
        );

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
        );

        CREATE TABLE IF NOT EXISTS lineage_edges (
            child_id TEXT NOT NULL,
            parent_kind TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            role TEXT,
            PRIMARY KEY (child_id, parent_kind, parent_id, role),
            FOREIGN KEY (child_id) REFERENCES manifests(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


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
    payload = manifest.model_dump_json(exclude_none=True)
    prov = manifest.provenance
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO manifests (
                id, kind, schema_version, provider_version, created_at, status, path,
                source_repo, source_branch, source_commit, dirty, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                payload,
            ),
        )
        conn.execute("DELETE FROM artifacts WHERE manifest_id = ?", (manifest.id,))
        conn.execute("DELETE FROM lineage_edges WHERE child_id = ?", (manifest.id,))
        conn.executemany(
            """
            INSERT OR REPLACE INTO artifacts (
                manifest_id, role, logical_name, artifact_id, sha256, media_type,
                size_bytes, storage_backend, uri
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_artifact_tuple(manifest.id, artifact) for artifact in manifest.artifacts],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO lineage_edges (
                child_id, parent_kind, parent_id, role
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (manifest.id, parent.kind, parent.id, parent.role)
                for parent in manifest.provenance.parents
            ],
        )


def index_manifest_file(
    manifest_path: Path | str,
    *,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> None:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    conn_path = Path(db_path) if db_path is not None else default_index_path(root)
    conn = connect_index(conn_path)
    try:
        index_manifest(conn, manifest, path=path)
    finally:
        conn.close()


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
    conn = connect_index(conn_path)
    try:
        rows = conn.execute(
            "SELECT path FROM manifests WHERE id = ? ORDER BY path",
            (manifest_id,),
        ).fetchall()
    finally:
        conn.close()
    return [Path(row[0]) for row in rows]


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
    if conn_path.exists():
        conn_path.unlink()
    conn = connect_index(conn_path)
    try:
        for manifest_path in iter_manifest_files(root_path):
            index_manifest(conn, load_manifest(manifest_path), path=manifest_path)
    finally:
        conn.close()
    return conn_path
