from dataclasses import replace

import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.migration_routes import validate_migration_routes
from feedbax.contracts.migrations import ComponentMigration, UnsupportedComponentMigration


def test_component_migration_chain_validates_only_at_current_endpoint():
    registry = ComponentRegistry(load_user_components=False)
    first = ComponentMigration(
        source_type="Gain",
        target_type="Gain",
        migration_id="old-gain-stage-1",
        owner="feedbax",
        source_param_schema_version="legacy",
        target_param_schema_version="intermediate",
        migrate_params=lambda params: {"intermediate_value": params["scale"] * 2},
        matches_unversioned=lambda params: "scale" in params,
    )
    last = ComponentMigration(
        source_type="Gain",
        target_type="Gain",
        migration_id="old-gain-stage-2",
        owner="feedbax",
        source_param_schema_version="intermediate",
        target_param_schema_version="1",
        migrate_params=lambda params: {"gain": params["intermediate_value"] + 1},
    )
    registry.register_migration(first)
    registry.register_migration(last)

    resolution = registry.resolve_component_spec(
        "Gain", {"scale": 4.0}, historical_unversioned=True
    )
    assert resolution.params == {"gain": 9.0}
    assert resolution.param_schema_version == "1"
    assert [edge.migration_id for edge in resolution.migrations] == [
        "old-gain-stage-1",
        "old-gain-stage-2",
    ]
    with pytest.raises(UnsupportedComponentMigration, match="Ambiguous unversioned"):
        registry.resolve_component_spec("Gain", {"gain": 4.0}, historical_unversioned=True)


def test_migration_graph_rejects_cycles_duplicate_ids_and_missing_terminal():
    edge = ComponentMigration(
        source_type="Gain",
        target_type="Gain",
        migration_id="old-gain",
        owner="feedbax",
        source_param_schema_version="legacy",
        target_param_schema_version="intermediate",
    )
    routes = {("Gain", "legacy"): edge}
    with pytest.raises(UnsupportedComponentMigration, match="without a current declaration"):
        validate_migration_routes(routes, {"Gain": "1"}, require_current_terminal=True)

    returning = replace(
        edge,
        migration_id="returning",
        source_param_schema_version="intermediate",
        target_param_schema_version="legacy",
    )
    with pytest.raises(UnsupportedComponentMigration, match="cycle"):
        validate_migration_routes({**routes, ("Gain", "intermediate"): returning}, {"Gain": "1"})

    duplicate = replace(edge, source_param_schema_version="other", target_param_schema_version="1")
    with pytest.raises(UnsupportedComponentMigration, match="Duplicate component migration ID"):
        validate_migration_routes({**routes, ("Gain", "other"): duplicate}, {"Gain": "1"})
