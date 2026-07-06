from __future__ import annotations

import pytest
from pydantic import ValidationError

import feedbax.contracts.graphs.serialization as serialization
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    GRAPH_SPEC_SCHEMA_VERSION_V2,
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    ParamSchema,
    WireSpec,
)
from feedbax.contracts.manifest import (
    ArtifactMigrationRecord,
    GraphSpecManifest,
    ModelArtifactManifest,
    ParentRef,
    SpecPayload,
    load_graph_spec_from_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import SpecMigrationResult, UnsupportedSpecVersion, migrate_graph_spec
from feedbax.integrations.provider import validate_graph_spec_manifest, validate_spec

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.graph_spec_contract, pytest.mark.migration_contract]


def _legacy_metadata() -> dict[str, str]:
    return {
        "name": "legacy",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "version": LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    }


def _legacy_graph_payload() -> dict[str, object]:
    return {
        "metadata": _legacy_metadata(),
        "nodes": {},
        "wires": [],
    }


def _current_graph_payload() -> dict[str, object]:
    return GraphSpec().model_dump(mode="json")


def test_graph_spec_schema_identity_survives_json_round_trip() -> None:
    spec = GraphSpec()

    payload = spec.model_dump(mode="json")
    round_tripped = GraphSpec.model_validate_json(spec.model_dump_json())

    assert payload["schema_id"] == GRAPH_SPEC_SCHEMA_ID
    assert payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert round_tripped.schema_id == GRAPH_SPEC_SCHEMA_ID
    assert round_tripped.schema_version == GRAPH_SPEC_SCHEMA_VERSION


def test_graph_spec_strict_models_accept_generated_current_fixture() -> None:
    payload = _current_graph_payload()

    round_tripped = GraphSpec.model_validate(payload)

    assert round_tripped.model_dump(mode="json") == payload


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ParamSchema, {"name": "gain", "type": "float", "unknown": True}),
        (ComponentSpec, {"type": "Gain", "unknown": True}),
        (
            WireSpec,
            {
                "source_node": "a",
                "source_port": "output",
                "target_node": "b",
                "target_port": "input",
                "unknown": True,
            },
        ),
        (
            GraphMetadata,
            {
                "name": "g",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "unknown": True,
            },
        ),
        (GraphSpec, {"nodes": {}, "wires": [], "unknown": True}),
    ],
)
def test_graph_spec_core_models_reject_unknown_extra_fields(model, payload) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


def test_graph_spec_migration_stamps_versionless_current_payload() -> None:
    result = migrate_graph_spec({"nodes": {}, "wires": []})

    assert result.payload["schema_id"] == GRAPH_SPEC_SCHEMA_ID
    assert result.payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert result.migration_records == []


def test_legacy_graph_spec_migration_records_builtin_rewrites() -> None:
    result = migrate_graph_spec(
        {
            "metadata": _legacy_metadata(),
            "nodes": {
                "network": {
                    "type": "SimpleStagedNetwork",
                    "params": {"output_size": 2},
                    "input_ports": ["target", "feedback"],
                    "output_ports": ["output"],
                }
            },
            "wires": [
                {
                    "source_node": "network",
                    "source_port": "target",
                    "target_node": "network",
                    "target_port": "target",
                }
            ],
            "input_bindings": {"target": ("network", "target")},
        }
    )

    network = result.payload["nodes"]["network"]
    assert result.payload["schema_id"] == GRAPH_SPEC_SCHEMA_ID
    assert result.payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert network["type"] == "Network"
    assert network["params"]["out_size"] == 2
    assert network["input_ports"] == ["input", "feedback"]
    assert result.payload["wires"][0]["source_port"] == "input"
    assert result.payload["wires"][0]["target_port"] == "input"
    assert result.payload["input_bindings"]["target"] == ("network", "input")
    assert [record.migration_id for record in result.migration_records] == [
        "graph-spec-legacy-v1-to-v2",
        "graph-spec-v2-to-v3-derived-dimensions",
    ]
    assert result.migration_records[0].metadata["graph_path"] == "graph"
    assert result.payload["derived_dimensions"] == []


def test_graph_spec_v2_migration_adds_derived_dimensions_field() -> None:
    result = migrate_graph_spec(
        {
            "schema_id": GRAPH_SPEC_SCHEMA_ID,
            "schema_version": GRAPH_SPEC_SCHEMA_VERSION_V2,
            "nodes": {},
            "wires": [],
        }
    )

    assert result.payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert result.payload["derived_dimensions"] == []
    assert [record.migration_id for record in result.migration_records] == [
        "graph-spec-v2-to-v3-derived-dimensions"
    ]


def test_nested_graph_spec_migration_is_recursive_and_ordered() -> None:
    result = migrate_graph_spec(
        {
            "metadata": _legacy_metadata(),
            "nodes": {
                "network": {
                    "type": "SimpleStagedNetwork",
                    "params": {"output_size": 2},
                    "input_ports": ["target"],
                    "output_ports": ["output"],
                }
            },
            "subgraphs": {
                "network": {
                    "nodes": {
                        "feedback": {
                            "type": "FeedbackChannel",
                            "params": {},
                            "input_ports": ["input"],
                            "output_ports": ["output"],
                        }
                    }
                }
            },
        }
    )

    nested = result.payload["subgraphs"]["network"]
    assert nested["schema_id"] == GRAPH_SPEC_SCHEMA_ID
    assert nested["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert nested["nodes"]["feedback"]["type"] == "Channel"
    assert [record.metadata["graph_path"] for record in result.migration_records] == [
        "graph",
        "graph",
        "graph.subgraphs['network']",
        "graph.subgraphs['network']",
    ]


def test_unknown_graph_spec_schema_version_reports_available_migrations() -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        migrate_graph_spec(
            {
                "schema_id": GRAPH_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.spec.graph.v99",
            }
        )

    message = str(excinfo.value)
    assert "source_version='feedbax.spec.graph.v99'" in message
    assert f"current_version='{GRAPH_SPEC_SCHEMA_VERSION}'" in message
    assert "available_migrations=[" in message
    assert "graph-spec-legacy-v1-to-v2" in message
    assert "graph-spec-v2-to-v3-derived-dimensions" in message


def test_graph_spec_manifest_load_attaches_feedbax_migration_records() -> None:
    manifest = GraphSpecManifest(
        id="feedbax-graph-spec:legacy",
        graph_spec=spec_payload("GraphSpec", _legacy_graph_payload()),
    )

    result = load_graph_spec_from_manifest(manifest)

    loaded_manifest = result.manifest
    assert isinstance(loaded_manifest, GraphSpecManifest)
    assert result.custody_manifest_kind == "GraphSpecManifest"
    assert result.payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert loaded_manifest.graph_spec.inline["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert loaded_manifest.graph_spec.sha256 == spec_payload(
        "GraphSpec",
        result.payload,
    ).sha256
    assert [record.migration_id for record in result.applied_migration_records] == [
        "graph-spec-legacy-v1-to-v2",
        "graph-spec-v2-to-v3-derived-dimensions",
    ]
    assert result.migration_records == loaded_manifest.migration_records


def test_model_artifact_inline_graph_spec_preserves_downstream_migration_records() -> None:
    downstream = ArtifactMigrationRecord(
        migration_id="rlrmp-legacy-artifact-v0-to-v1",
        source_schema_version="rlrmp.model_artifact.v0",
        target_schema_version="rlrmp.model_artifact.v1",
        tool="rlrmp",
    )
    manifest = ModelArtifactManifest(
        id="feedbax-model-artifact:legacy-inline",
        graph_spec=spec_payload("GraphSpec", _legacy_graph_payload()),
        migration_records=[downstream],
    )

    result = load_graph_spec_from_manifest(manifest)

    loaded_manifest = result.manifest
    assert isinstance(loaded_manifest, ModelArtifactManifest)
    assert result.custody_manifest_kind == "ModelArtifactManifest"
    assert result.downstream_migration_records == [downstream]
    assert loaded_manifest.migration_records[0] == downstream
    assert loaded_manifest.migration_records[1].tool == "feedbax"
    assert not isinstance(loaded_manifest.graph_spec, ParentRef)
    assert loaded_manifest.graph_spec.inline["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION


def test_model_artifact_parent_graph_spec_records_are_discoverable(tmp_path) -> None:
    graph_manifest = GraphSpecManifest(
        id="feedbax-graph-spec:referenced",
        graph_spec=spec_payload("GraphSpec", _legacy_graph_payload()),
    )
    graph_manifest_path = write_manifest(graph_manifest, root=tmp_path, index=False)
    model_manifest = ModelArtifactManifest(
        id="feedbax-model-artifact:referenced",
        graph_spec=ParentRef(
            kind="GraphSpecManifest",
            id=graph_manifest.id,
            role="graph_spec",
            uri=str(graph_manifest_path),
        ),
    )

    result = load_graph_spec_from_manifest(model_manifest)

    loaded_manifest = result.manifest
    assert isinstance(loaded_manifest, GraphSpecManifest)
    assert result.custody_manifest_id == graph_manifest.id
    assert result.migration_records == loaded_manifest.migration_records
    assert loaded_manifest.graph_spec.inline["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION


def test_provider_validation_reports_manifest_migration_custody_states() -> None:
    migrated = validate_graph_spec_manifest(
        GraphSpecManifest(
            id="feedbax-graph-spec:legacy",
            graph_spec=spec_payload("GraphSpec", _legacy_graph_payload()),
        )
    )
    assert migrated.valid
    assert migrated.migration_status == "feedbax_migrated"
    assert [record.migration_id for record in migrated.migration_records] == [
        "graph-spec-legacy-v1-to-v2",
        "graph-spec-v2-to-v3-derived-dimensions",
    ]
    routed = validate_spec(
        "graph_manifest",
        GraphSpecManifest(
            id="feedbax-graph-spec:routed",
            graph_spec=spec_payload("GraphSpec", _legacy_graph_payload()),
        ).model_dump(mode="json"),
    )
    assert routed.valid
    assert routed.migration_status == "feedbax_migrated"

    current = validate_graph_spec_manifest(
        GraphSpecManifest(
            id="feedbax-graph-spec:current",
            graph_spec=spec_payload("GraphSpec", _current_graph_payload()),
        )
    )
    assert current.valid
    assert current.migration_status == "current"
    assert current.migration_records == []

    downstream_record = ArtifactMigrationRecord(
        migration_id="rlrmp-legacy-artifact-v0-to-v1",
        source_schema_version="rlrmp.model_artifact.v0",
        target_schema_version="rlrmp.model_artifact.v1",
        tool="rlrmp",
    )
    downstream = validate_graph_spec_manifest(
        ModelArtifactManifest(
            id="feedbax-model-artifact:downstream",
            graph_spec=spec_payload("GraphSpec", _current_graph_payload()),
            migration_records=[downstream_record],
        )
    )
    assert downstream.valid
    assert downstream.migration_status == "downstream_migrated"
    assert downstream.downstream_migration_records == [downstream_record]


def test_provider_validation_rejects_unsupported_manifest_graph_spec_version() -> None:
    result = validate_graph_spec_manifest(
        GraphSpecManifest(
            id="feedbax-graph-spec:unsupported",
            graph_spec=SpecPayload(
                kind="GraphSpec",
                inline={
                    "schema_id": GRAPH_SPEC_SCHEMA_ID,
                    "schema_version": "feedbax.spec.graph.v99",
                    "nodes": {},
                    "wires": [],
                },
            ),
        )
    )

    assert not result.valid
    assert result.migration_status == "rejected"
    assert result.errors[0].type == "unsupported_spec_version"
    assert "feedbax.spec.graph.v99" in result.errors[0].message


def test_spec_to_graph_invokes_public_graph_spec_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    migrated = GraphSpec(
        nodes={
            "constant": ComponentSpec(
                type="Constant",
                params={"value": 1.0},
                input_ports=[],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("constant", "output")},
    ).model_dump(mode="json")

    def fake_migrate_graph_spec(payload: GraphSpec) -> SpecMigrationResult:
        nonlocal called
        called = True
        assert isinstance(payload, GraphSpec)
        return SpecMigrationResult(
            kind="GraphSpec",
            schema_id=GRAPH_SPEC_SCHEMA_ID,
            source_version=GRAPH_SPEC_SCHEMA_VERSION,
            target_version=GRAPH_SPEC_SCHEMA_VERSION,
            payload=migrated,
            migration_records=[],
        )

    monkeypatch.setattr(serialization, "migrate_graph_spec", fake_migrate_graph_spec)

    graph = serialization.spec_to_graph(GraphSpec(), {})

    assert called
    assert "constant" in graph.nodes
