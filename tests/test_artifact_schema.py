from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feedbax.contracts.artifact_schema import (
    ArrayStoreValidationError,
    array_store_ref,
    read_npz_array_store,
    validate_role_coverage,
    write_npz_array_store,
)
from feedbax.contracts.manifest import ModelArtifactManifest, spec_payload
from feedbax.contracts.migrations import (
    MigrationRegistry,
    SchemaMigration,
    UnsupportedMigrationPath,
)


def test_npz_array_store_roundtrip_preserves_roles_dtype_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "model.params.npz"
    arrays = {
        "node:network/subgraph/node:cell/param:weight_hh": np.eye(3, dtype=np.float32),
        "node:network/subgraph/node:readout/param:bias": np.arange(2, dtype=np.float64),
    }

    payload = write_npz_array_store(
        path,
        arrays,
        store_role="params",
        graph_spec_ref="graph://demo",
        graph_spec_sha256="abc123",
    )
    loaded = read_npz_array_store(path)

    assert loaded.payload == payload
    assert loaded.payload.storage_backend == "npz.v1"
    assert loaded.payload.graph_spec_ref == "graph://demo"
    assert set(loaded.arrays) == set(arrays)
    for role, expected in arrays.items():
        actual = loaded.arrays[role]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_array_store_role_validation_reports_missing_and_unexpected_roles(tmp_path: Path) -> None:
    path = tmp_path / "model.state.npz"
    write_npz_array_store(
        path,
        {"state:mechanics/position": np.zeros((2,), dtype=np.float32)},
        store_role="state",
    )
    loaded = read_npz_array_store(path)

    with pytest.raises(ArrayStoreValidationError, match="missing required roles"):
        loaded.validate_roles(required_roles=["state:mechanics/velocity"])

    with pytest.raises(ArrayStoreValidationError, match="unexpected"):
        validate_role_coverage(
            loaded.payload.roles,
            exact_roles=["state:mechanics/velocity"],
        )


def test_model_artifact_manifest_binds_graph_spec_and_array_store(tmp_path: Path) -> None:
    path = tmp_path / "model.params.npz"
    payload = write_npz_array_store(
        path,
        {"node:readout/param:weight": np.ones((1, 2), dtype=np.float32)},
        store_role="params",
    )
    store_ref = array_store_ref(path, payload)

    manifest = ModelArtifactManifest(
        id="feedbax-model-artifact:test",
        graph_spec=spec_payload("GraphSpec", {"nodes": {}, "wires": []}),
        parameter_store=store_ref,
    )

    assert manifest.parameter_store is not None
    assert manifest.parameter_store.array_count == 1
    assert manifest.parameter_store.roles == ["node:readout/param:weight"]


def test_schema_migration_registry_applies_synthetic_migration() -> None:
    registry = MigrationRegistry()

    def migrate_v1_to_v2(payload: dict[str, object]) -> dict[str, object]:
        payload["renamed"] = payload.pop("old")
        return payload

    registry.register(
        SchemaMigration(
            source_version="demo.v1",
            target_version="demo.v2",
            migration_id="demo-v1-to-v2",
            migrate=migrate_v1_to_v2,
        )
    )

    migrated, records = registry.migrate(
        {"schema_version": "demo.v1", "old": 3},
        source_version="demo.v1",
        target_version="demo.v2",
    )

    assert migrated == {"schema_version": "demo.v2", "renamed": 3}
    assert [record.migration_id for record in records] == ["demo-v1-to-v2"]
    assert records[0].source_schema_version == "demo.v1"
    assert records[0].target_schema_version == "demo.v2"


def test_schema_migration_registry_fails_for_missing_path() -> None:
    registry = MigrationRegistry()

    with pytest.raises(UnsupportedMigrationPath, match="'demo.v1' -> 'demo.v3'"):
        registry.migrate(
            {"schema_version": "demo.v1"},
            source_version="demo.v1",
            target_version="demo.v3",
        )
