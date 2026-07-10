"""Data migrations for the legacy SQLAlchemy persistence database.

The persistence database predates the structured manifest store, so its schema
is reconciled when a session is opened. Historical project-specific columns
must be migrated here before they are removed from the declared ORM schema.
"""

from __future__ import annotations

import json
from typing import Any

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy.engine import Engine


class PersistenceSchemaMigrationError(RuntimeError):
    """Raised when an existing persistence database cannot be safely reconciled."""


_LEGACY_CONDITION_COLUMN = "sisu_params"


def _decode_json_object(value: Any, *, field: str, row_id: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PersistenceSchemaMigrationError(
                f"Cannot migrate evaluations row {row_id}: {field} contains invalid JSON."
            ) from exc
    if not isinstance(value, dict):
        raise PersistenceSchemaMigrationError(
            f"Cannot migrate evaluations row {row_id}: {field} must contain a JSON object."
        )
    return value


def migrate_legacy_evaluation_condition_metadata(
    engine: Engine,
    *,
    table_name: str,
    metadata_column: str,
) -> None:
    """Move a removed project-specific condition column into generic metadata.

    Existing values are preserved under their historical key inside
    ``condition_metadata``. Keeping the key in caller-owned JSON retains the
    original query meaning without retaining project vocabulary in the ORM
    schema or public persistence API. The migration fails closed on malformed
    JSON or a conflicting pre-existing metadata key, then drops the legacy
    column after every row has been migrated.
    """

    columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
    if _LEGACY_CONDITION_COLUMN not in columns:
        return
    if metadata_column not in columns:
        raise PersistenceSchemaMigrationError(
            f"Cannot migrate {table_name}.{_LEGACY_CONDITION_COLUMN}: "
            f"missing replacement column {metadata_column}."
        )

    table = Table(table_name, MetaData(), autoload_with=engine)
    id_column = table.c.id
    legacy_column = table.c[_LEGACY_CONDITION_COLUMN]
    generic_column = table.c[metadata_column]

    with engine.begin() as connection:
        rows = connection.execute(select(id_column, legacy_column, generic_column)).all()
        for row_id, legacy_value, generic_value in rows:
            legacy_object = _decode_json_object(
                legacy_value,
                field=_LEGACY_CONDITION_COLUMN,
                row_id=row_id,
            )
            if legacy_object is None:
                continue

            generic_object = (
                _decode_json_object(
                    generic_value,
                    field=metadata_column,
                    row_id=row_id,
                )
                or {}
            )
            existing_value = generic_object.get(_LEGACY_CONDITION_COLUMN)
            if existing_value is not None and existing_value != legacy_object:
                raise PersistenceSchemaMigrationError(
                    f"Cannot migrate evaluations row {row_id}: {metadata_column} already "
                    f"contains conflicting {_LEGACY_CONDITION_COLUMN!r} metadata."
                )

            migrated = dict(generic_object)
            migrated[_LEGACY_CONDITION_COLUMN] = legacy_object
            connection.execute(
                table.update().where(id_column == row_id).values({metadata_column: migrated})
            )

        Operations(MigrationContext.configure(connection)).drop_column(
            table_name,
            _LEGACY_CONDITION_COLUMN,
        )
