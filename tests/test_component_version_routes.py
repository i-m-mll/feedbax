import pytest

from feedbax.component_registry import ComponentPack, ComponentRegistry
from feedbax.contracts.migrations import UnsupportedComponentMigration
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
        legacy=(
            LegacyEndpoint(
                "ChainGain", "legacy", 1, matches_unversioned=lambda params: "scale" in params
            ),
        ),
    )
    module = ModuleType("test_chain_owners")
    ChainGain.__module__ = module.__name__
    module.ChainGain = ChainGain
    registry = ComponentRegistry()
    registry.ingest(
        ComponentPack(namespace="example", modules=(module,), histories={ChainGain: history})
    )
    resolution = registry.resolve_component_spec(
        "ChainGain", {"scale": 4.0}, historical_unversioned=True
    )
    assert resolution.params == {"gain": 9.0}
    assert resolution.revision == 3
    with pytest.raises(UnsupportedComponentMigration, match="Ambiguous unversioned"):
        registry.resolve_component_spec("ChainGain", {"gain": 4.0}, historical_unversioned=True)


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
    params = {}
    if source is None:
        params["label"] = "legacy"
    else:
        params["schema_version"] = source
        if source == AFFINE_VALUE_COMPOSER_SCHEMA_VERSION_V1:
            params["label"] = "legacy"
    resolved = registry.resolve_component_spec(
        "AffineValueComposer",
        params,
        legacy_token=source,
        historical_unversioned=historical_unversioned,
    )
    assert resolved.revision == 3
    assert "schema_version" not in resolved.params
    assert "label" not in resolved.params


def test_threshold_historical_selector_reaches_current_contract() -> None:
    registry = ComponentRegistry()
    current = registry.get("feedbax.ThresholdLatchedForce")
    assert current is not None
    params = {"state_selector": {"path": ["pos", 0]}}
    resolved = registry.resolve_component_spec(
        "ThresholdLatchedForce", params, historical_unversioned=True
    )
    assert resolved.revision == current.revision
    assert resolved.params["state_selector"] == {"kind": "fixed", "path": ["pos", 0]}
    assert resolved.params["lateral_force"] == 0.0


def test_integer_history_rename_and_namespace_reservations():
    class CurrentName(Component):
        gain: float = 2.0
        ports: ClassVar = Ports({"input": PortType()}, {"output": PortType()})

        def __call__(self, inputs, state, *, key):
            return {"output": inputs["input"] * self.gain}, state

    registry = ComponentRegistry()
    history = History(
        Change(2, renamed_from="PreviousName"), legacy=(LegacyEndpoint("OldWire", "frozen.v1", 1),)
    )
    registry.ingest(
        ComponentPack(
            namespace="rename", components=(CurrentName,), histories={CurrentName: history}
        )
    )
    old = registry.resolve_component_spec("OldWire", {}, legacy_token="frozen.v1")
    assert (old.type_id, old.revision, old.params) == ("rename.CurrentName", 2, {"gain": 2.0})
    assert registry.serialize_component(CurrentName()) == ("rename.CurrentName", {"gain": 2.0}, 2)
    for invalid in (True, "2", 0):
        with pytest.raises(ValueError, match="strictly positive integer"):
            registry.resolve_component_spec("rename.CurrentName", {}, revision=invalid)
    for invalid in (True, 1, 1.0):
        with pytest.raises(ValueError, match="token must be a string"):
            registry.resolve_component_spec("rename.PreviousName", {}, legacy_token=invalid)
    unrelated = type("PreviousName", (CurrentName,), {})
    with pytest.raises(ValueError, match="reserved"):
        registry.ingest(ComponentPack(namespace="rename", components=(unrelated,)))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("single_batch", [False, True])
def test_legacy_identity_reservation_is_atomic_across_registration_order(reverse, single_batch):
    class CurrentOwner(Component):
        gain: float = 1.0
        ports: ClassVar = Ports({}, {})

        def __call__(self, inputs, state, *, key):
            return {}, state

    class ReusedAlias(CurrentOwner):
        pass

    history = History(legacy=(
        LegacyEndpoint("reserved.ReusedAlias", "1", 1),
        LegacyEndpoint("reserved.ReusedAlias", None, 1),
    ))
    owners = (ReusedAlias, CurrentOwner) if reverse else (CurrentOwner, ReusedAlias)
    registry = ComponentRegistry()
    if single_batch:
        before = registry.names()
        with pytest.raises(ValueError, match="reserved"):
            registry.ingest(ComponentPack(
                namespace="reserved", components=owners, histories={CurrentOwner: history}
            ))
    else:
        first, second = owners
        registry.ingest(ComponentPack(
            namespace="reserved", components=(first,),
            histories={CurrentOwner: history} if first is CurrentOwner else {},
        ))
        before = registry.names()
        with pytest.raises(ValueError, match="reserved"):
            registry.ingest(ComponentPack(
                namespace="reserved", components=(second,),
                histories={CurrentOwner: history} if second is CurrentOwner else {},
            ))
    assert registry.names() == before


@pytest.mark.parametrize(
    "changes", [(Change(3, identity=True),), (Change(2, identity=True), Change(2, identity=True))]
)
def test_published_history_destinations_cannot_be_reindexed(changes):
    with pytest.raises(ValueError, match="consecutive"):
        History(*changes)


def test_unversioned_evidence_requires_a_unique_endpoint():
    from feedbax.mechanics.force_fields import CurlField

    class EvidenceComponent(CurlField):
        pass

    history = History(
        Change(2, identity=True),
        legacy=(
            LegacyEndpoint("Ambiguous", "one", 1, lambda params: True),
            LegacyEndpoint("Ambiguous", "two", 2, lambda params: True),
        ),
    )
    registry = ComponentRegistry()
    registry.ingest(
        ComponentPack(
            namespace="evidence",
            components=(EvidenceComponent,),
            histories={EvidenceComponent: history},
        )
    )
    with pytest.raises(UnsupportedComponentMigration, match="Ambiguous unversioned"):
        registry.resolve_component_spec("Ambiguous", {}, historical_unversioned=True)
