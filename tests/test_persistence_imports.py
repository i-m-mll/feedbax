from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import plotly.graph_objects as go
import pytest
from sqlalchemy import inspect

import feedbax.persistence.database as database
from feedbax.config import STRINGS
from feedbax.persistence.database import (
    CURRENT_MODEL_HASH_VERSION,
    EvaluationRecord,
    FigureRecord,
    ModelRecord,
    add_evaluation_figure,
    clear_db_session_cache,
    db_session,
    generate_figure_hash,
    init_db_session,
    query_model_records,
    query_records,
    update_table_schema,
)
from feedbax.persistence.manifest_index import connect_index, default_index_path


def test_root_persistence_modules_are_not_importable() -> None:
    assert importlib.util.find_spec("feedbax.database") is None
    assert importlib.util.find_spec("feedbax.manifest_index") is None


def test_database_session_preserves_dynamic_schema_updates(tmp_path: Path) -> None:
    clear_db_session_cache()
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
        clear_db_session_cache()


def test_database_session_factory_is_cached_per_path(tmp_path: Path) -> None:
    clear_db_session_cache()
    db_path = tmp_path / "models.db"
    first = init_db_session(f"sqlite:///{db_path}")
    second = init_db_session(f"sqlite:///{db_path}")
    other = init_db_session(f"sqlite:///{tmp_path / 'other.db'}")
    try:
        assert first.get_bind() is second.get_bind()
        assert first.get_bind() is not other.get_bind()
    finally:
        first.close()
        second.close()
        other.close()
        clear_db_session_cache()


def test_db_session_does_not_verify_model_files_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_db_session_cache()
    monkeypatch.setattr(database.PATHS, "db", tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("check_model_files should be explicit")

    monkeypatch.setattr(database, "check_model_files", fail_if_called)
    try:
        with db_session("default_no_verify"):
            pass

        with pytest.raises(AssertionError, match="explicit"):
            with db_session("explicit_verify", verify_files=True):
                pass
    finally:
        clear_db_session_cache()


def test_database_declares_hot_filter_indices(tmp_path: Path) -> None:
    clear_db_session_cache()
    session = init_db_session(f"sqlite:///{tmp_path / 'models.db'}")
    try:
        inspector = inspect(session.get_bind())
        model_indexes = {
            column
            for index in inspector.get_indexes(ModelRecord.__tablename__)
            for column in index["column_names"]
        }
        eval_indexes = {
            column
            for index in inspector.get_indexes(EvaluationRecord.__tablename__)
            for column in index["column_names"]
        }
        figure_indexes = {
            column
            for index in inspector.get_indexes(FigureRecord.__tablename__)
            for column in index["column_names"]
        }
    finally:
        session.close()
        clear_db_session_cache()

    assert {"is_path_defunct", "expt_name", "postprocessed", "pert__type", "pert__std"} <= (
        model_indexes
    )
    assert {"archived", "expt_name", "created_at"} <= eval_indexes
    assert {"archived", "evaluation_hash", "identifier", "figure_type", "pert__type"} <= (
        figure_indexes
    )


def test_model_record_rejects_unsupported_hash_version() -> None:
    supported = ModelRecord(hash="supported", hash_version=CURRENT_MODEL_HASH_VERSION)
    assert supported.path.name == "supported.eqx"

    unsupported = ModelRecord(hash="legacy", hash_version="v1")
    with pytest.raises(ValueError, match="Unsupported model hash_version"):
        _ = unsupported.path


def test_json_filters_use_canonical_serializer(tmp_path: Path) -> None:
    clear_db_session_cache()
    session = init_db_session(f"sqlite:///{tmp_path / 'models.db'}")
    try:
        record = EvaluationRecord(
            hash="eval-json",
            perturbation_config={"b": 2, "a": 1},
            archived=False,
        )
        session.add(record)
        session.commit()

        matches = query_records(
            session,
            EvaluationRecord,
            {"perturbation_config": {"a": 1, "b": 2}},
        )
        assert [match.hash for match in matches] == ["eval-json"]
    finally:
        session.close()
        clear_db_session_cache()


def test_add_evaluation_figure_rolls_back_and_cleans_file_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_db_session_cache()
    monkeypatch.setattr(database.PATHS, "figures", tmp_path / "figures")
    session = init_db_session(f"sqlite:///{tmp_path / 'models.db'}")
    try:
        eval_record = EvaluationRecord(hash="eval-cleanup", archived=False)
        existing_hash = generate_figure_hash(eval_record.hash, "loss", {})
        existing = FigureRecord(
            hash=existing_hash,
            evaluation_hash=eval_record.hash,
            identifier="loss",
            figure_type="plotly",
            saved_formats=["json"],
            archived=False,
        )
        session.add_all([eval_record, existing])
        session.commit()

        def fail_commit():
            raise RuntimeError("forced commit failure")

        monkeypatch.setattr(session, "commit", fail_commit)

        with pytest.raises(RuntimeError, match="forced commit failure"):
            add_evaluation_figure(
                session,
                eval_record,
                go.Figure(),
                "loss",
                save_formats=[],
            )

        assert not (tmp_path / "figures" / eval_record.hash / f"{existing_hash}.json").exists()
        assert session.query(FigureRecord).filter(FigureRecord.hash == existing_hash).one()
    finally:
        session.close()
        clear_db_session_cache()


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
