import sqlite3

import pytest

from feedbax.bin import db_merge


def test_table_schema_quotes_sqlite_identifiers(tmp_path) -> None:
    db_path = tmp_path / "quoted.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE "select" ("where" INTEGER, "has space" TEXT)')

    schema = db_merge.get_table_schema(str(db_path), "select")

    assert [column["name"] for column in schema] == ["where", "has space"]


def test_add_missing_columns_quotes_sqlite_identifiers(tmp_path) -> None:
    db_path = tmp_path / "quoted.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE "select" ("where" INTEGER)')

    db_merge.add_missing_columns(
        str(db_path),
        "select",
        [{"name": "group", "type": "TEXT"}],
    )

    schema = db_merge.get_table_schema(str(db_path), "select")
    assert [column["name"] for column in schema] == ["where", "group"]


def test_declared_type_rejects_statement_breakout() -> None:
    with pytest.raises(ValueError, match="Unsafe SQLite declared type"):
        db_merge.quote_declared_type("TEXT; DROP TABLE models")
