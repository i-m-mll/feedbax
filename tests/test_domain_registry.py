from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.component_registry import ComponentRegistry, DomainRegistry, builtin_domain_registry
from feedbax.contracts.domain import (
    ACAUSAL_DOMAIN_ID,
    CAUSAL_DOMAIN_ID,
    DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID,
    DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION,
    PENZAI_DOMAIN_ID,
    DomainMeta,
    EditorCapability,
    DomainTheme,
)
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from tests.graph_compiler_test_support import spec_to_graph
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.web.api import domains as domains_api


def _domain(domain_id: str) -> DomainMeta:
    return DomainMeta(
        id=domain_id,
        display_name="Test",
        interior_schema_id=None,
        edge_semantics="directed",
        allows_multi_edge_per_port=False,
        nestable_domains=[],
        editor=EditorCapability(kind="none", editable=False),
        theme=DomainTheme(color="test", icon="Box", edge_style="directed"),
        compiler_id=None,
    )


def test_builtin_domain_registry_lists_four_domains() -> None:
    registry = builtin_domain_registry()

    assert [domain.id for domain in registry.list_all()] == [
        "feedbax.domain.causal",
        "feedbax.domain.acausal",
        "feedbax.domain.mechanics",
        "feedbax.domain.penzai",
    ]
    assert registry.get(ACAUSAL_DOMAIN_ID).allows_multi_edge_per_port is True  # type: ignore[union-attr]
    assert registry.get(PENZAI_DOMAIN_ID).editor.kind == "inspector"  # type: ignore[union-attr]


def test_domain_registry_rejects_duplicate_and_invalid_ids() -> None:
    registry = DomainRegistry()
    registry.register(_domain("feedbax.domain.test"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_domain("feedbax.domain.test"))

    with pytest.raises(ValueError, match=r"feedbax\.domain\.\*"):
        _domain("feedbax.spec.not-a-domain")


def test_domains_api_returns_versioned_builtin_payload() -> None:
    app = FastAPI()
    app.include_router(domains_api.router, prefix="/api/domains")
    client = TestClient(app)

    response = client.get("/api/domains")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["schema_id"] == DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID
    assert payload["schema_version"] == DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION
    assert [domain["id"] for domain in payload["domains"]] == [
        "feedbax.domain.causal",
        "feedbax.domain.acausal",
        "feedbax.domain.mechanics",
        "feedbax.domain.penzai",
    ]


def test_component_definitions_carry_domain_metadata() -> None:
    registry = ComponentRegistry(load_user_components=False)

    for definition in registry.list_all():
        assert definition.domain.startswith("feedbax.domain.")

    assert registry.get("Subgraph").interior_domain == CAUSAL_DOMAIN_ID  # type: ignore[union-attr]
    assert registry.get("AcausalSystem").interior_domain == ACAUSAL_DOMAIN_ID  # type: ignore[union-attr]
    assert registry.get("PenzaiAdapter").interior_domain == PENZAI_DOMAIN_ID  # type: ignore[union-attr]


def test_acausal_missing_interior_error_names_domain() -> None:
    graph = GraphSpec(
        nodes={
            "arm": ComponentSpec(
                type="AcausalSystem",
                params={"dt": 0.001},
                input_ports=["input"],
                output_ports=["state"],
            )
        }
    )

    with pytest.raises(ValueError) as exc_info:
        spec_to_graph(graph, ComponentRegistry(load_user_components=False))

    message = str(exc_info.value)
    assert "requires an acausal interior" in message
    assert ACAUSAL_DOMAIN_ID in message


def test_subgraph_missing_interior_behavior_remains_unchanged() -> None:
    graph = GraphSpec(
        nodes={
            "nested": ComponentSpec(
                type="Subgraph",
                params={},
                input_ports=[],
                output_ports=[],
            )
        }
    )

    with pytest.raises(ValueError, match="Missing subgraph spec for 'nested'"):
        spec_to_graph(graph, ComponentRegistry(load_user_components=False))


def test_legacy_network_missing_interior_behavior_remains_unchanged() -> None:
    graph = GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                params={},
                input_ports=[],
                output_ports=[],
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Network node 'network' has no subgraph. Open it in Studio to generate "
            "the internal architecture, then save again."
        ),
    ):
        spec_to_graph(graph, ComponentRegistry(load_user_components=False))


def test_domain_payload_schema_rejects_unknown_old_version() -> None:
    family = default_spec_registry.resolve("DomainRegistryPayload")
    assert family.schema_id == DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID
    assert family.current_version == DOMAIN_REGISTRY_PAYLOAD_SCHEMA_VERSION
    assert family.policy is not None
    assert family.policy.stance == "reject"

    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        default_spec_registry.migrate(
            "DomainRegistryPayload",
            {
                "schema_id": DOMAIN_REGISTRY_PAYLOAD_SCHEMA_ID,
                "schema_version": "feedbax.spec.domain.v0",
                "domains": [],
            },
        )
