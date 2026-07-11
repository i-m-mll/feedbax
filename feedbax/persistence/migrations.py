"""Data migrations for the legacy SQLAlchemy persistence database.

The persistence database predates the structured manifest store, so its schema
is reconciled when a session is opened. Historical project-specific columns
must be migrated here before they are removed from the declared ORM schema.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol, Sequence

from alembic.migration import MigrationContext
from alembic.operations import Operations
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy.engine import Engine

from feedbax.contracts.manifest import (
    EntrypointRef,
    EvaluationRunManifest,
    EvaluationRunSpec,
    ParentRef,
    Provenance,
    spec_payload,
)


class PersistenceSchemaMigrationError(RuntimeError):
    """Raised when an existing persistence database cannot be safely reconciled."""


_LEGACY_CONDITION_COLUMN = "sisu_params"
LEGACY_EVALUATION_RECORD_SCHEMA_VERSION = "feedbax.persistence.evaluation_record.v1"
LEGACY_EVALUATION_RECORD_MIGRATION_VERSION = (
    "feedbax.migration.evaluation_record_to_manifest.v1"
)


class LegacyEvaluationRecordSnapshot(BaseModel):
    """Versioned portable snapshot of one row from the legacy evaluations table."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    schema_version: Literal[LEGACY_EVALUATION_RECORD_SCHEMA_VERSION] = (
        LEGACY_EVALUATION_RECORD_SCHEMA_VERSION
    )
    hash: str
    created_at: datetime | None = None
    expt_name: str | None = None
    model_hashes: list[str] = Field(default_factory=list)
    perturbation_config: dict[str, Any] | None = None
    condition_metadata: dict[str, Any] | None = None
    task_variants: dict[str, Any] | None = None
    eval_setup_params: dict[str, Any] | None = None


class LegacyEvaluationRecordLike(Protocol):
    """Structural subset of the legacy ORM row consumed by the migration."""

    hash: str
    created_at: datetime | None
    expt_name: str | None
    model_hashes: Sequence[str] | None
    perturbation_config: dict[str, Any] | None
    condition_metadata: dict[str, Any] | None
    task_variants: dict[str, Any] | None
    eval_setup_params: dict[str, Any] | None


def migrate_legacy_evaluation_record(
    snapshot: LegacyEvaluationRecordSnapshot | Mapping[str, Any] | LegacyEvaluationRecordLike,
) -> EvaluationRunManifest:
    """Convert one legacy evaluation row snapshot to the canonical manifest schema.

    The legacy row hash is retained as the manifest ID so existing Studio and
    analysis references remain resolvable. Legacy list projections treated
    these rows as completed, so the transition preserves that status while
    recording the source and migration schema identities explicitly.
    """

    source = (
        snapshot
        if isinstance(snapshot, LegacyEvaluationRecordSnapshot)
        else LegacyEvaluationRecordSnapshot.model_validate(snapshot)
    )
    training_refs = [
        ParentRef(kind="TrainingRunManifest", id=run_id, role="training_run")
        for run_id in source.model_hashes
    ]
    params = dict(source.perturbation_config or {})
    for field_name in ("condition_metadata", "task_variants", "eval_setup_params"):
        value = getattr(source, field_name)
        if value is not None:
            params[field_name] = value
    if source.expt_name:
        params["label"] = source.expt_name

    run_spec = EvaluationRunSpec(
        evaluation_type="feedbax.studio.default_eval",
        training_run_ids=list(source.model_hashes),
        inputs=training_refs,
        params=params,
    )
    manifest_fields: dict[str, Any] = {}
    if source.created_at is not None:
        manifest_fields["created_at"] = source.created_at
    return EvaluationRunManifest(
        id=source.hash,
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            run_spec.model_dump(mode="json", exclude_none=True),
        ),
        input_training_runs=training_refs,
        provenance=Provenance(
            entrypoint=EntrypointRef(
                kind="feedbax-persistence-migration",
                name="migrate_legacy_evaluation_record",
            ),
            parents=training_refs,
        ),
        metadata={
            "name": source.expt_name or source.hash,
            "legacy_evaluation_record": {
                "source_schema_version": source.schema_version,
                "migration_schema_version": LEGACY_EVALUATION_RECORD_MIGRATION_VERSION,
                "source_id": source.hash,
            },
        },
        **manifest_fields,
    )


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
