from __future__ import annotations

import pytest

import feedbax.serialization as serialization
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    ComponentSpec,
    GraphSpec,
)
from feedbax.migrations import SpecMigrationResult, UnsupportedSpecVersion, migrate_graph_spec


def _legacy_metadata() -> dict[str, str]:
    return {
        "name": "legacy",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "version": LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
    }


def test_graph_spec_schema_identity_survives_json_round_trip() -> None:
    spec = GraphSpec()

    payload = spec.model_dump(mode="json")
    round_tripped = GraphSpec.model_validate_json(spec.model_dump_json())

    assert payload["schema_id"] == GRAPH_SPEC_SCHEMA_ID
    assert payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert round_tripped.schema_id == GRAPH_SPEC_SCHEMA_ID
    assert round_tripped.schema_version == GRAPH_SPEC_SCHEMA_VERSION


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
        "graph-spec-legacy-v1-to-v2"
    ]
    assert result.migration_records[0].metadata["graph_path"] == "graph"


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
        "graph.subgraphs['network']",
    ]


def test_unknown_graph_spec_schema_version_reports_available_migrations() -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        migrate_graph_spec(
            {
                "schema_id": GRAPH_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.graph_spec.v99",
            }
        )

    message = str(excinfo.value)
    assert "source_version='feedbax.graph_spec.v99'" in message
    assert f"current_version='{GRAPH_SPEC_SCHEMA_VERSION}'" in message
    assert "available_migrations=[" in message
    assert "graph-spec-legacy-v1-to-v2" in message


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
