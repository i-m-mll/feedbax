from dataclasses import replace

import pytest

from feedbax.component_registry import ComponentPack, ComponentRegistry
from feedbax.component_registry.migration_routes import validate_migration_routes
from feedbax.contracts.migrations import ComponentMigration, UnsupportedComponentMigration
from feedbax.component_registry.history import Change, History, LegacyEndpoint
from feedbax.contracts.portable import Ports
from feedbax.contracts.component import PortType
from feedbax.runtime.graph import Component
from types import ModuleType
from typing import ClassVar

AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1 = "feedbax.component.affine_value_composer.v1"
AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2 = "feedbax.component.affine_value_composer.v2"


def test_component_migration_chain_validates_only_at_current_endpoint():
    class ChainGain(Component):
        gain: float
        ports: ClassVar = Ports({"input": PortType()}, {"output": PortType()})

        def __call__(self, inputs, state, *, key):
            return {"output": self.gain * inputs["input"]}, state

    history = History(
        Change(2, migrate=lambda params: {"intermediate_value": params["scale"] * 2}),
        Change(3, migrate=lambda params: {"gain": params["intermediate_value"] + 1}),
        legacy=(LegacyEndpoint("ChainGain", "legacy", 1,
                               matches_unversioned=lambda params: "scale" in params),),
    )
    module = ModuleType("test_chain_owners")
    ChainGain.__module__ = module.__name__
    module.ChainGain = ChainGain
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="example", modules=(module,), histories={ChainGain: history}))
    resolution = registry.resolve_component_spec(
        "ChainGain", {"scale": 4.0}, historical_unversioned=True
    )
    assert resolution.params == {"gain": 9.0}
    assert resolution.revision == 3
    with pytest.raises(UnsupportedComponentMigration, match="Ambiguous unversioned"):
        registry.resolve_component_spec("ChainGain", {"gain": 4.0}, historical_unversioned=True)


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


@pytest.mark.parametrize(
    ("source", "historical_unversioned"),
    [
        (AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1, False),
        (AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V2, False),
        (None, True),
    ],
)
def test_affine_composer_historical_parameters_reach_outer_v3(
    source: str | None, historical_unversioned: bool
) -> None:
    registry = ComponentRegistry()
    current = registry.get("feedbax.AffineValueComposer")
    assert current is not None
    params = dict(current.default_params)
    if source is None:
        params["label"] = "legacy"
    else:
        params["schema_version"] = source
        if source == AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1:
            params["label"] = "legacy"
    resolved = registry.resolve_component_spec(
        "AffineValueComposer",
        params,
        param_schema_version=source,
        historical_unversioned=historical_unversioned,
    )
    assert resolved.revision == 3
    assert "schema_version" not in resolved.params
    assert "label" not in resolved.params


def test_threshold_historical_selector_reaches_current_contract() -> None:
    registry = ComponentRegistry()
    current = registry.get("feedbax.ThresholdLatchedForce")
    assert current is not None
    params = {
        **current.default_params,
        "state_selector": {"path": ["pos", 0]},
    }
    params.pop("lateral_force")
    resolved = registry.resolve_component_spec(
        "ThresholdLatchedForce", params, historical_unversioned=True
    )
    assert resolved.revision == current.revision
    assert resolved.params["state_selector"] == {"kind": "fixed", "path": ["pos", 0]}
    assert resolved.params["lateral_force"] == 0.0
