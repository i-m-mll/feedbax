from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

from sqlalchemy import inspect

from feedbax.config import STRINGS
from feedbax.persistence.database import (
    ModelRecord,
    init_db_session,
    query_model_records,
    update_table_schema,
)
from feedbax.persistence.manifest_index import connect_index, default_index_path


def test_root_persistence_modules_are_not_importable() -> None:
    assert importlib.util.find_spec("feedbax.database") is None
    assert importlib.util.find_spec("feedbax.manifest_index") is None


def test_database_session_preserves_dynamic_schema_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "models.db"
    session = init_db_session(f"sqlite:///{db_path}")
    try:
        engine = session.get_bind()
        update_table_schema(
            engine,
            STRINGS.db_table_names.models,
            {"issue_a911c56_dynamic_column": 1},
        )

        columns = {column["name"] for column in inspect(engine).get_columns(ModelRecord.__tablename__)}
        assert "issue_a911c56_dynamic_column" in columns
        assert query_model_records(session, {"missing_column": "value"}) == []
    finally:
        session.close()


def test_manifest_index_schema_initializes_under_persistence(tmp_path: Path) -> None:
    index_path = default_index_path(tmp_path)

    conn = connect_index(index_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()

    assert index_path == tmp_path / "index" / "feedbax.sqlite"
    assert {name for (name,) in rows} == {"artifacts", "lineage_edges", "manifests"}

    with sqlite3.connect(index_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
