"""Public emission API for three-layer training-run specification storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from feedbax.contracts.manifest import (
    ArtifactRef,
    StrictModel,
    TrainingRunManifest,
    sha256_file,
)
from feedbax.contracts.run_matrix import InlineMatrixBaseSpec, TrainingRunMatrixSpec
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.spec_storage import (
    TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
    TrainingRunExecutionCapsule,
    build_resolved_semantics_snapshot,
    store_canonical_json_artifact,
    training_run_execution_hash,
    training_run_authored_envelope_hash,
    training_run_composed_intent_hash,
    training_run_intent_hash,
    training_spec_canonical_bytes,
)
from feedbax.training.run_matrix import (
    MaterializedRunMatrix,
    RowPayloadValidator,
    materialize_adapted_run_matrix,
    materialize_run_matrix,
)


class TrainingSpecStorageResult(StrictModel):
    """Identity and custody pointers returned by the registered emitter."""

    intent_hash: str
    authored_envelope_hash: str
    composed_intent_hash: str
    execution_hash: str
    resolved_root_hash: str
    authored_path: str
    snapshot_artifact: ArtifactRef
    capsule_artifact: ArtifactRef
    capsule: TrainingRunExecutionCapsule


def stamp_training_run_manifest_identities(
    manifest: TrainingRunManifest,
    storage: TrainingSpecStorageResult,
) -> TrainingRunManifest:
    """Stamp dual identity without permitting archived identity redefinition."""
    if manifest.intent_hash is not None and manifest.intent_hash != storage.intent_hash:
        raise ValueError(
            "archived TrainingRunManifest intent_hash must never be overwritten"
        )
    expected_execution = training_run_execution_hash(
        storage.resolved_root_hash,
        storage.capsule.input_data_identities,
    )
    if manifest.execution_hash is not None and manifest.execution_hash != expected_execution:
        raise ValueError(
            "materializer drift: archived TrainingRunManifest execution_hash must never "
            "be overwritten"
        )
    payload = manifest.model_dump(mode="json", exclude_none=True)
    payload.update(
        {
            "intent_hash": storage.intent_hash,
            "execution_hash": expected_execution,
            "resolved_semantics_root_hash": storage.resolved_root_hash,
            "input_data_identities": storage.capsule.input_data_identities,
        }
    )
    return TrainingRunManifest.model_validate(payload)


def emit_training_run_spec_storage(
    authored: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    authored_path: Path,
    custody_root: Path,
    materializer_commit: str,
    dependency_lock_path: Path,
    input_data_identities: list[dict[str, Any]] | None = None,
    environment_digest: str | None = None,
    allow_inline_base: bool = False,
    row_validator: RowPayloadValidator | None = None,
    _test_materializer_transform: Callable[[MaterializedRunMatrix], MaterializedRunMatrix]
    | None = None,
) -> TrainingSpecStorageResult:
    """Write layer 1 canonically and layers 2/3 through local custody.

    Inline bases are fixture-only and rejected unless ``allow_inline_base`` is
    deliberately enabled by a test or fixture producer.
    """
    if isinstance(authored, TrainingRunMatrixSpec):
        matrix = authored
    else:
        migrated = default_spec_registry.migrate("TrainingRunMatrixSpec", authored)
        matrix = TrainingRunMatrixSpec.model_validate(migrated.payload)
    if isinstance(matrix.base, InlineMatrixBaseSpec) and not allow_inline_base:
        raise ValueError(
            "inline matrix bases are tests/fixtures only; production emission requires "
            "a content-pinned authored_intent or resolved_output reference"
        )
    authored_document = matrix.model_dump(mode="json", exclude_none=True)
    intent_hash = training_run_intent_hash(authored_document)
    authored_envelope_hash = training_run_authored_envelope_hash(authored_document)
    if row_validator is None:
        materialized = materialize_run_matrix(matrix, repo_root=repo_root)
    else:
        materialized = materialize_adapted_run_matrix(
            matrix,
            repo_root=repo_root,
            row_validator=row_validator,
        )
    if _test_materializer_transform is not None:
        materialized = _test_materializer_transform(materialized)
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_bytes(training_spec_canonical_bytes(authored_document) + b"\n")

    resolved_tree = {
        "run_set_id": materialized.run_set_id,
        "rows": [
            {
                "row_id": row.row_id,
                "planned_run_id": row.planned_run_id,
                "seed": row.seed,
                "axis_coordinates": (
                    _row_coordinate(materialized, row).model_dump(
                        mode="json", exclude_none=True
                    )
                    if _row_coordinate(materialized, row) is not None
                    else None
                ),
                "overrides": [
                    override.model_dump(mode="json", exclude_none=True)
                    if hasattr(override, "model_dump")
                    else override
                    for override in row.overrides
                ],
                "payload": row.payload,
            }
            for row in materialized.rows
        ],
    }
    composed_intent_hash = training_run_composed_intent_hash(resolved_tree)
    snapshot = build_resolved_semantics_snapshot(resolved_tree)
    snapshot_artifact = store_canonical_json_artifact(
        snapshot,
        root=custody_root,
        role="training_run_resolved_semantics",
        logical_name=f"{authored_path.stem}.resolved.json",
    )
    inputs = list(input_data_identities or [])
    relevant_versions = {
        "training_run_matrix": matrix.schema_version,
        "resolved_semantics": snapshot["schema_version"],
        "execution_capsule": TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
    }
    for row in materialized.rows:
        _record_payload_schema_versions(row.payload, relevant_versions)
    capsule = TrainingRunExecutionCapsule(
        materializer_commit=materializer_commit,
        relevant_schema_versions=relevant_versions,
        dependency_lock_digest=sha256_file(dependency_lock_path),
        environment_digest=environment_digest,
        input_data_identities=inputs,
        intent_hash=intent_hash,
        resolved_root_hash=snapshot["root_hash"],
    )
    capsule_artifact = store_canonical_json_artifact(
        capsule,
        root=custody_root,
        role="training_run_execution_capsule",
        logical_name=f"{authored_path.stem}.execution.json",
    )
    assert capsule.execution_hash is not None
    return TrainingSpecStorageResult(
        intent_hash=intent_hash,
        authored_envelope_hash=authored_envelope_hash,
        composed_intent_hash=composed_intent_hash,
        execution_hash=capsule.execution_hash,
        resolved_root_hash=snapshot["root_hash"],
        authored_path=str(authored_path),
        snapshot_artifact=snapshot_artifact,
        capsule_artifact=capsule_artifact,
        capsule=capsule,
    )


def _record_payload_schema_versions(
    payload: Mapping[str, Any],
    versions: dict[str, str],
) -> None:
    schema_id = payload.get("schema_id")
    schema_version = payload.get("schema_version")
    if isinstance(schema_id, str) and isinstance(schema_version, str):
        versions[schema_id] = schema_version
    graph = payload.get("graph")
    if isinstance(graph, Mapping):
        inline = graph.get("inline")
        if isinstance(inline, Mapping):
            graph_id = inline.get("schema_id")
            graph_version = inline.get("schema_version")
            if isinstance(graph_id, str) and isinstance(graph_version, str):
                versions[graph_id] = graph_version


def _row_coordinate(materialized: MaterializedRunMatrix, row: Any) -> Any:
    if row.coordinate is not None:
        return row.coordinate
    return next(
        (
            coordinate
            for coordinate in materialized.run_set_manifest.axes.runs
            if coordinate.run_id == row.planned_run_id
        ),
        None,
    )
