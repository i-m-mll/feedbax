from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from feedbax.analysis.harness import (
    MatrixMaterializerHarness,
    convert_evaluation_manifest_storage,
    resolve_manifest_regeneration,
)
from feedbax.contracts.base import canonical_json_bytes
from feedbax.contracts.artifact_store import store_json_artifact
from feedbax.contracts.manifest import (
    EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,
    EVALUATION_RUN_MANIFEST_SCHEMA_VERSION_V1,
    EvaluationRunManifest,
    EvaluationRunSpec,
    RegenerationCommand,
    RegenerationSpec,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.manifest_packet import export_manifest_packet, import_manifest_packet
from feedbax.persistence.artifact_custody import ArtifactBlobIntegrityError
from feedbax.persistence.manifest_index import (
    connect_index,
    default_index_path,
    get_indexed_manifest_record,
    rebuild_manifest_index,
)


def _manifest(row_id: str) -> EvaluationRunManifest:
    spec = EvaluationRunSpec(
        evaluation_type="testpkg.shared_storage",
        params={"row_id": row_id},
    )
    return EvaluationRunManifest(
        id=f"feedbax-evaluation-run:{row_id}",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec", spec.model_dump(mode="json", exclude_none=True)
        ),
    )


def _materialize(root: Path, count: int):
    rows = [(f"row-{index}", {"value": index}) for index in range(count)]

    def execute(row_id: str, _resolved: dict[str, int], row_root: Path):
        manifest = _manifest(row_id)
        return manifest, write_manifest(manifest, root=row_root, index=False)

    return MatrixMaterializerHarness(root=root, shared_store=True).materialize(
        rows,
        execute=execute,
        command=["python", "-m", "feedbax", "matrix-harness"],
        title="Shared storage",
        source="EvaluationRunMatrixSpec",
        matrix_metadata={
            "axis_expansion": {
                "coordinates": list(range(4_000)),
                "canonical_row_order": [row_id for row_id, _ in rows],
            }
        },
        regeneration_parameters={"executable_spec": {"padding": "x" * 100_000}},
    )


def _stored_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def test_shared_storage_is_linear_resolvable_portable_and_rebuildable(tmp_path: Path) -> None:
    one_root = tmp_path / "one"
    many_root = tmp_path / "many"
    _materialize(one_root, 1)
    result = _materialize(many_root, 20)

    per_extra_row = (_stored_bytes(many_root) - _stored_bytes(one_root)) / 19
    assert per_extra_row < 10_000
    shared_refs = {
        artifact.artifact_id
        for row in result.rows
        for artifact in row.result.artifacts
        if artifact.role == "shared_regeneration_context"
    }
    assert len(shared_refs) == 1
    assert all(b"xxxxxxxxxxxxxxxx" not in row.manifest_path.read_bytes() for row in result.rows)

    first = result.rows[0]
    assert resolve_manifest_regeneration(first.result, root=many_root) == first.regeneration
    before_repeat = _stored_bytes(many_root)
    repeated = _materialize(many_root, 20)
    assert _stored_bytes(many_root) == before_repeat
    assert resolve_manifest_regeneration(repeated.rows[0].result, root=many_root) == (
        repeated.rows[0].regeneration
    )

    index_path = rebuild_manifest_index(many_root)
    connection = sqlite3.connect(index_path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(manifests)")}
    finally:
        connection.close()
    assert "payload_json" not in columns
    index_path.unlink()
    rebuild_manifest_index(many_root)
    record = get_indexed_manifest_record(first.result.id, root=many_root)
    assert record is not None
    assert record["payload"]["evaluation_spec"]["inline"] == first.result.evaluation_spec.inline

    packet = tmp_path / "packet"
    exported = export_manifest_packet(
        [first.result.id], root=many_root, dest=packet, include_artifacts="include"
    )
    assert exported.included_artifact_count == len(first.result.artifacts)
    imported_root = tmp_path / "imported"
    import_manifest_packet(packet, root=imported_root)
    imported = load_manifest(next((imported_root / "manifests" / "evaluation_runs").glob("*.json")))
    assert resolve_manifest_regeneration(imported, root=imported_root) == first.regeneration

    shared = next(
        artifact
        for artifact in first.result.artifacts
        if artifact.role == "shared_regeneration_context"
    )
    shared_path = many_root / "artifacts" / "sha256" / shared.sha256[:2] / shared.sha256
    shared_path.write_bytes(b"corrupt")
    with pytest.raises(ArtifactBlobIntegrityError, match="size|sha256"):
        resolve_manifest_regeneration(first.result, root=many_root)


def test_historical_conversion_preserves_exact_source_and_target_versions(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    manifest = _manifest("legacy")
    regeneration = RegenerationSpec(
        command=RegenerationCommand(argv=["python", "-m", "feedbax", "matrix-harness"]),
        parameters={
            "row_id": "legacy",
            "resolved": {"value": 1},
            "executable_spec": {"matrix": [1, 2, 3]},
        },
        metadata={
            "source": "EvaluationRunMatrixSpec",
            "custody": "content-addressed",
            "axis_expansion": {"coordinates": [0, 1, 2]},
        },
    )
    legacy = manifest.model_dump(mode="json", exclude_none=True)
    legacy["schema_version"] = EVALUATION_RUN_MANIFEST_SCHEMA_VERSION_V1
    legacy["metadata"] = {
        "matrix_harness": {
            "row_id": "legacy",
            "regeneration_spec": regeneration.model_dump(mode="json", exclude_none=True),
            "axis_expansion": {"coordinates": [0, 1, 2]},
        }
    }
    source_bytes = canonical_json_bytes(legacy)
    source.write_bytes(source_bytes)

    destination = tmp_path / "normalized"
    conversion = convert_evaluation_manifest_storage(source, destination_root=destination)
    assert source.read_bytes() == source_bytes
    assert conversion.source_schema_version == EVALUATION_RUN_MANIFEST_SCHEMA_VERSION_V1
    assert conversion.target_schema_version == EVALUATION_RUN_MANIFEST_SCHEMA_VERSION
    assert conversion.source_sha256 != conversion.target_sha256
    assert conversion.target_path.read_bytes()
    normalized = load_manifest(conversion.target_path)
    assert normalized.id == manifest.id
    resolved = resolve_manifest_regeneration(normalized, root=destination)
    assert resolved.parameters == regeneration.parameters
    assert resolved.metadata == regeneration.metadata
    assert normalized.migration_records[0].metadata["source_sha256"] == conversion.source_sha256
    assert default_index_path(destination).exists() is False
    assert (
        convert_evaluation_manifest_storage(source, destination_root=destination).target_sha256
        == conversion.target_sha256
    )

    conflicting_source = tmp_path / "legacy-conflict.json"
    conflicting = dict(legacy)
    conflicting["metadata"] = dict(legacy["metadata"])
    conflicting_harness = dict(conflicting["metadata"]["matrix_harness"])
    conflicting_regeneration = dict(conflicting_harness["regeneration_spec"])
    conflicting_parameters = dict(conflicting_regeneration["parameters"])
    conflicting_parameters["resolved"] = {"value": 2}
    conflicting_regeneration["parameters"] = conflicting_parameters
    conflicting_harness["regeneration_spec"] = conflicting_regeneration
    conflicting["metadata"]["matrix_harness"] = conflicting_harness
    conflicting_source.write_bytes(canonical_json_bytes(conflicting))
    with pytest.raises(ValueError, match="separate explicit custody domains"):
        convert_evaluation_manifest_storage(conflicting_source, destination_root=destination)


def test_historical_conversion_rejects_corrupt_or_conflicting_replay_artifact(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "historical"
    source = source_root / "manifests" / "evaluation_runs" / "legacy.json"
    source.parent.mkdir(parents=True)
    manifest = _manifest("legacy-artifact")
    regeneration = RegenerationSpec(
        command=RegenerationCommand(argv=["python", "-m", "feedbax", "matrix-harness"]),
        parameters={"row_id": "legacy-artifact", "resolved": {"value": 1}},
        metadata={"source": "EvaluationRunMatrixSpec", "custody": "content-addressed"},
    )
    artifact = store_json_artifact(
        regeneration.model_dump(mode="json", exclude_none=True),
        root=source_root,
        role="regeneration_spec",
        logical_name="legacy.regeneration.json",
    )
    legacy = manifest.model_dump(mode="json", exclude_none=True)
    legacy["schema_version"] = EVALUATION_RUN_MANIFEST_SCHEMA_VERSION_V1
    legacy["artifacts"] = [artifact.model_dump(mode="json", exclude_none=True)]
    legacy["metadata"] = {
        "matrix_harness": {
            "row_id": "legacy-artifact",
            "regeneration_spec": regeneration.model_dump(mode="json", exclude_none=True),
        }
    }
    source.write_bytes(canonical_json_bytes(legacy))

    conflicting = dict(legacy)
    conflicting["metadata"] = dict(legacy["metadata"])
    conflicting_harness = dict(conflicting["metadata"]["matrix_harness"])
    conflicting_inline = dict(conflicting_harness["regeneration_spec"])
    conflicting_parameters = dict(conflicting_inline["parameters"])
    conflicting_parameters["resolved"] = {"value": 2}
    conflicting_inline["parameters"] = conflicting_parameters
    conflicting_harness["regeneration_spec"] = conflicting_inline
    conflicting["metadata"]["matrix_harness"] = conflicting_harness
    source.write_bytes(canonical_json_bytes(conflicting))
    with pytest.raises(ValueError, match="inline and referenced regeneration specs conflict"):
        convert_evaluation_manifest_storage(source, destination_root=tmp_path / "normalized")

    source.write_bytes(canonical_json_bytes(legacy))
    artifact_path = source_root / artifact.metadata["relative_path"]
    artifact_path.write_bytes(b"x" * artifact.size_bytes)
    with pytest.raises(ValueError, match="sha256 disagrees"):
        convert_evaluation_manifest_storage(source, destination_root=tmp_path / "normalized")


def test_index_schema_migrates_away_from_payload_copy(tmp_path: Path) -> None:
    index_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(index_path)
    try:
        connection.execute(
            """
            CREATE TABLE manifests (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                provider_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT,
                path TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                source_repo TEXT,
                source_branch TEXT,
                source_commit TEXT,
                dirty INTEGER
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    migrated = connect_index(index_path)
    try:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(manifests)")}
        version = migrated.execute("PRAGMA user_version").fetchone()[0]
    finally:
        migrated.close()
    assert "payload_json" not in columns
    assert version == 2
