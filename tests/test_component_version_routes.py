"""Current component revisions and rename histories."""

from typing import ClassVar

import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.component_history import Change, History
from feedbax.contracts.component import PortType
from feedbax.contracts.portable import Ports
from feedbax.runtime.graph import Component
from tests.support.component_owners import register_owners


def test_component_migration_chain_validates_only_at_current_endpoint():
    class ChainGain(Component):
        gain: float
        ports: ClassVar = Ports({"input": PortType()}, {"output": PortType()})
        component_history = History(
            Change(2, migrate=lambda params: {"intermediate_value": params["scale"] * 2}),
            Change(3, migrate=lambda params: {"gain": params["intermediate_value"] + 1}),
        )

        def __call__(self, inputs, state, *, key):
            return {"output": self.gain * inputs["input"]}, state

    registry = ComponentRegistry()
    register_owners(registry, "example", ChainGain)
    resolution = registry.resolve_component_spec("example.ChainGain", {"scale": 4.0}, revision=1)
    assert resolution.params == {"gain": 9.0}
    assert resolution.revision == 3


@pytest.mark.parametrize("revision", [1, 2])
def test_affine_composer_integer_revisions_reach_current_contract(revision):
    registry = ComponentRegistry()
    params = {"label": "migration"} if revision == 1 else {}
    resolved = registry.resolve_component_spec("feedbax.AffineValueComposer", params, revision=revision)
    assert resolved.revision == 3
    assert "label" not in resolved.params


def test_integer_history_rename_and_namespace_reservations():
    class CurrentName(Component):
        gain: float = 2.0
        ports: ClassVar = Ports({"input": PortType()}, {"output": PortType()})
        component_history = History(Change(2, renamed_from="PreviousName"))

        def __call__(self, inputs, state, *, key):
            return {"output": inputs["input"] * self.gain}, state

    registry = ComponentRegistry()
    register_owners(registry, "rename", CurrentName)
    old = registry.resolve_component_spec("rename.PreviousName", {}, revision=1)
    assert (old.type_id, old.revision, old.params) == ("rename.CurrentName", 2, {"gain": 2.0})
    assert registry.serialize_component(CurrentName()) == ("rename.CurrentName", {"gain": 2.0}, 2)
    for invalid in (True, "2", 0):
        with pytest.raises(ValueError, match="strictly positive integer"):
            registry.resolve_component_spec("rename.CurrentName", {}, revision=invalid)

    class PreviousName(CurrentName):
        component_history = None

    before = registry.names()
    with pytest.raises(ValueError, match="reserved"):
        register_owners(registry, "rename", PreviousName)
    assert registry.names() == before


@pytest.mark.parametrize(
    "changes", [(Change(3, identity=True),), (Change(2, identity=True), Change(2, identity=True))]
)
def test_published_history_destinations_cannot_be_reindexed(changes):
    with pytest.raises(ValueError, match="consecutive"):
        History(*changes)
