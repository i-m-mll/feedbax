from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.contracts.base import (
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    SpecPayload,
    load_manifest,
    migrate_spec_payload,
    spec_payload,
)
from feedbax.contracts.migrations import (
    SchemaMigration,
    SpecSchemaFamily,
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
)

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.migration_contract]


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_manifest_rejects_unsupported_non_graph_embedded_spec_with_path(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "evaluation-manifest.json"
    _write_manifest(
        manifest_path,
        {
            "kind": "EvaluationRunManifest",
            "id": "feedbax-evaluation-run:test",
            "evaluation_spec": {
                "kind": "EvaluationRunSpec",
                "schema_id": "feedbax.spec.evaluation_run",
                "schema_version": "feedbax.spec.evaluation_run.v0",
                "inline": {
                    "schema_version": "feedbax.spec.evaluation_run.v0",
                    "evaluation_type": "toy_eval",
                    "params": {},
                },
            },
        },
    )

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        load_manifest(manifest_path)

    message = str(excinfo.value)
    assert "path='evaluation_spec'" in message
    assert "kind='EvaluationRunSpec'" in message
    assert "source_version='feedbax.spec.evaluation_run.v0'" in message


def test_spec_payload_writer_stamps_non_graph_family_and_hashes_current_inline() -> None:
    payload = spec_payload(
        "EvaluationRunSpec",
        {"evaluation_type": "toy_eval", "training_run_ids": [], "params": {}},
    )

    assert payload.schema_id == "feedbax.spec.evaluation_run"
    assert payload.schema_version == "feedbax.spec.evaluation_run.v1"
    assert payload.sha256 == sha256_bytes(canonical_json_bytes(payload.inline))
    assert payload.source_sha256 is None
    assert payload.migration_records == []


def test_migrate_spec_payload_rejects_unknown_external_family() -> None:
    with pytest.raises(UnknownSpecFamily, match="ExternalDemo"):
        migrate_spec_payload(
            SpecPayload(
                kind="ExternalDemo",
                schema_id="external.demo",
                schema_version="external.demo.v1",
                inline={"schema_version": "external.demo.v1"},
            )
        )


def test_migrate_spec_payload_rejects_unknown_feedbax_family_marked_external() -> None:
    with pytest.raises(UnknownSpecFamily, match="cannot be marked external"):
        migrate_spec_payload(
            SpecPayload(
                kind="FeedbaxDemo",
                schema_id="feedbax.spec.demo",
                schema_version="feedbax.spec.demo.v1",
                inline={"schema_version": "feedbax.spec.demo.v1"},
                metadata={"external": True},
            )
        )


def test_migrate_spec_payload_accepts_explicit_external_family() -> None:
    payload = migrate_spec_payload(
        SpecPayload(
            kind="ExternalDemo",
            schema_id="external.demo",
            schema_version="external.demo.v1",
            inline={"schema_version": "external.demo.v1"},
            metadata={"external": True},
        )
    )

    assert payload.schema_id == "external.demo"
    assert payload.schema_version == "external.demo.v1"
    assert payload.sha256 == sha256_bytes(canonical_json_bytes(payload.inline))


def test_migrate_spec_payload_rejects_external_inline_schema_mismatch() -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        migrate_spec_payload(
            SpecPayload(
                kind="RLRMPRunSpec",
                schema_id="rlrmp.run_spec",
                schema_version="rlrmp.run_spec.v2",
                inline={"schema_version": "rlrmp.cs_stochastic_gru.v1"},
                metadata={"external": True},
            ),
            path="training_spec",
        )

    message = str(excinfo.value)
    assert "Embedded SpecPayload schema version disagrees with inline payload" in message
    assert "path='training_spec'" in message
    assert "kind='RLRMPRunSpec'" in message
    assert "schema_version='rlrmp.run_spec.v2'" in message
    assert "inline_schema_version='rlrmp.cs_stochastic_gru.v1'" in message


def test_generic_spec_payload_migration_records_non_graph_source_hash() -> None:
    registry = SpecSchemaRegistry()
    registry.register_family(
        SpecSchemaFamily(
            kind="DemoSpec",
            schema_id="feedbax.spec.demo",
            current_version="feedbax.spec.demo.v2",
        )
    )

    def migrate_v1_to_v2(payload: dict[str, object]) -> dict[str, object]:
        payload["renamed"] = payload.pop("old")
        return payload

    registry.register_migration(
        "DemoSpec",
        SchemaMigration(
            source_version="demo.v1",
            target_version="feedbax.spec.demo.v2",
            migration_id="demo-spec-v1-to-v2",
            migrate=migrate_v1_to_v2,
        ),
    )
    legacy_inline = {"schema_version": "demo.v1", "old": 4}
    source_sha256 = sha256_bytes(canonical_json_bytes(legacy_inline))

    payload = migrate_spec_payload(
        SpecPayload(
            kind="DemoSpec",
            inline=legacy_inline,
            sha256=source_sha256,
        ),
        path="demo_spec",
        registry=registry,
    )

    assert payload.schema_id == "feedbax.spec.demo"
    assert payload.schema_version == "feedbax.spec.demo.v2"
    assert payload.inline == {"schema_version": "feedbax.spec.demo.v2", "renamed": 4}
    assert payload.sha256 == sha256_bytes(canonical_json_bytes(payload.inline))
    assert payload.source_sha256 == source_sha256
    assert [record.migration_id for record in payload.migration_records] == ["demo-spec-v1-to-v2"]
    assert payload.migration_records[0].metadata["spec_payload_kind"] == "DemoSpec"
    assert payload.migration_records[0].metadata["spec_payload_path"] == "demo_spec"
