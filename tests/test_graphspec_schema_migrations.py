from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import (
    ACAUSAL_GRAPH_SCHEMA_ID,
    ACAUSAL_GRAPH_SCHEMA_VERSION,
    AcausalConnectionSpec,
    AcausalGraphSpec,
)
from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    ParamSchema,
    WireSpec,
)
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
)
from feedbax.contracts.base import ParentRef
from feedbax.contracts.manifest import (
    GraphSpecManifest,
    ModelArtifactManifest,
    SpecPayload,
    load_graph_spec_from_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import (
    UnsupportedSpecVersion,
    migrate_graph_spec,
    migrate_structured_spec_payload,
)
from tests.support.application_registry import registry_bundle_fixture as registry_bundle_fixture

pytestmark = [
    pytest.mark.feedbax_contract,
    pytest.mark.graph_spec_contract,
    pytest.mark.migration_contract,
]

def _current_graph_payload() -> dict[str, object]:
    return GraphSpec().model_dump(mode="json")


def _acausal_graph_payload(
    *,
    schema_version: str = ACAUSAL_GRAPH_SCHEMA_VERSION,
    subgraphs: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": ACAUSAL_GRAPH_SCHEMA_ID,
        "schema_version": schema_version,
        "physical_domain": "translational",
        "nodes": {
            "mass": {
                "type": "Mass",
                "params": {"mass": 1.0},
                "input_ports": [],
                "output_ports": [],
            },
            "ground": {
                "type": "Ground",
                "params": {},
                "input_ports": [],
                "output_ports": [],
            },
        },
        "connections": [
            {"a": ("mass", "p"), "b": ("ground", "p")},
        ],
        "solver": {"solver_type": "euler", "dt": 0.001},
    }
    if subgraphs is not None:
        payload["subgraphs"] = subgraphs
    return payload


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


def test_acausal_connection_canonicalizes_undirected_endpoints() -> None:
    connection = AcausalConnectionSpec.model_validate({"a": ("mass", "p"), "b": ("ground", "p")})

    assert connection.a == ("ground", "p")
    assert connection.b == ("mass", "p")

    with pytest.raises(ValidationError, match="endpoints must be distinct"):
        AcausalConnectionSpec.model_validate({"a": ("mass", "p"), "b": ("mass", "p")})


def test_acausal_graph_spec_rejects_unknown_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AcausalGraphSpec.model_validate({**_acausal_graph_payload(), "unknown": True})






def test_nested_acausal_reader_preserves_component_revisions_through_containers(tmp_path):
    from feedbax.contracts.component_history import Change, History
    from tests.support.component_owners import register_owners
    from feedbax.contracts.portable import Ports
    from feedbax.runtime.graph import Component

    class ReaderOwner(Component):
        value: int = 0
        ports: ClassVar = Ports({}, {})

        def __call__(self, inputs, state, *, key):
            return {}, state

    from feedbax.builtins.acausal.translational import Mass

    class ReaderMass(Mass):
        def __init__(self, name: str, value: int = 0):
            super().__init__(name=name)
            self.value = value

    history = History(Change(2, migrate=lambda params: {"value": params["value"] + 1}))
    ReaderOwner.component_history = ReaderMass.component_history = history
    registry = ComponentRegistry()
    register_owners(registry, "reader", ReaderOwner, ReaderMass)
    node = {"type": "reader.ReaderOwner", "params": {"value": 0}, "revision": 1}
    physical_node = {"type": "reader.ReaderMass", "params": {"value": 0}, "revision": 1}
    inner = {
        "schema_id": ACAUSAL_GRAPH_SCHEMA_ID,
        "physical_domain": "translational",
        "nodes": {"owner": physical_node},
        "solver": {"solver_type": "euler", "dt": 0.001},
    }
    acausal = {
        **inner,
        "schema_version": ACAUSAL_GRAPH_SCHEMA_VERSION,
        "nodes": {"owner": physical_node, "child": {"type": "feedbax.AcausalSystem"}},
        "subgraphs": {"child": inner},
    }
    direct = migrate_structured_spec_payload(
        "AcausalGraphSpec", acausal, component_registry=registry
    )
    causal = {
        "schema_id": GRAPH_SPEC_SCHEMA_ID,
        "schema_version": GRAPH_SPEC_SCHEMA_VERSION,
        "nodes": {
            "owner": node,
            "plant": {"type": "feedbax.AcausalSystem"},
        },
        "subgraphs": {"plant": acausal},
    }
    manifest = GraphSpecManifest(
        id="feedbax-graph-spec:reader",
        graph_spec=SpecPayload(kind="GraphSpec", inline=causal),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    loaded_manifest = load_graph_spec_from_manifest(manifest_path, component_registry=registry)
    expected = {"type": "reader.ReaderMass", "params": {"value": 1}, "revision": 2}
    for graph in (
        direct.payload,
        loaded_manifest.payload["subgraphs"]["plant"],
    ):
        assert graph["schema_version"] == ACAUSAL_GRAPH_SCHEMA_VERSION
        assert {key: graph["nodes"]["owner"][key] for key in expected} == expected
        nested = graph["subgraphs"]["child"]
        assert nested["schema_version"] == ACAUSAL_GRAPH_SCHEMA_VERSION
        assert {key: nested["nodes"]["owner"][key] for key in expected} == expected
    assert loaded_manifest.payload["nodes"]["owner"]["params"] == {"value": 1}
    assert node == {"type": "reader.ReaderOwner", "params": {"value": 0}, "revision": 1}


@pytest.mark.parametrize("version", ["feedbax.spec.acausal_graph.v0", "feedbax.spec.acausal_graph.v1"])
def test_older_acausal_graph_versions_are_refused_in_nested_graphs(version) -> None:
    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        migrate_graph_spec(
            {
                "schema_id": GRAPH_SPEC_SCHEMA_ID,
                "schema_version": GRAPH_SPEC_SCHEMA_VERSION,
                "nodes": {},
                "wires": [],
                "subgraphs": {
                    "plant": _acausal_graph_payload(schema_version=version)
                },
            }
        )

    message = str(excinfo.value)
    assert "family='AcausalGraphSpec'" in message
    assert f"source_version={version!r}" in message


def test_nested_acausal_subgraphs_validate_recursively() -> None:
    nested = AcausalGraphSpec.model_validate(
        _acausal_graph_payload(subgraphs={"inner": _acausal_graph_payload()})
    )

    assert isinstance(nested.subgraphs["inner"], AcausalGraphSpec)


def test_spec_to_graph_requires_acausal_system_interior() -> None:
    graph = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="feedbax.AcausalSystem",
            )
        }
    )

    with pytest.raises(ValueError, match="feedbax.domain.acausal"):
        compile_graph(
            GraphDocument(graph=graph),
            ComponentRegistry(),
        )


def test_graph_spec_migration_stamps_versionless_current_payload() -> None:
    result = migrate_graph_spec({"nodes": {}, "wires": []})

    assert result.payload["schema_id"] == GRAPH_SPEC_SCHEMA_ID
    assert result.payload["schema_version"] == GRAPH_SPEC_SCHEMA_VERSION
    assert result.migration_records == []


def test_acausal_interior_rejects_component_param_array_values_at_typed_boundary() -> None:
    payload = _acausal_graph_payload()
    payload["nodes"]["mass"]["params"]["matrix"] = {
        "schema_id": ARRAY_VALUE_SCHEMA_ID,
        "schema_version": ARRAY_VALUE_SCHEMA_VERSION,
        "encoding": "constant",
        "shape": [2, 2],
        "dtype": "float32",
        "nonfinite": "forbid",
        "value": 0.0,
    }

    with pytest.raises(ValidationError, match="AcausalGraphSpec does not support"):
        AcausalGraphSpec.model_validate(payload)


def test_model_artifact_parent_graph_spec_records_are_discoverable(tmp_path) -> None:
    graph_manifest = GraphSpecManifest(
        id="feedbax-graph-spec:referenced",
        graph_spec=spec_payload("GraphSpec", _current_graph_payload()),
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
