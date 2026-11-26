"""The authored Model address identifies exactly one typed element or field."""

from hypothesis import given, strategies as st
from pydantic import ValidationError
import pytest

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project_sweep import (
    CHECKPOINT_RULE_TARGET, CheckpointRuleAxis, ListedAxis, SetValue, model_field_path, model_field_target, sweep_patch,
)


def _record(tag, value, **fields):
    return st.fixed_dictionaries({tag: st.just(value), **fields})


_IDS = st.one_of(
    st.sampled_from(("inputs", "outputs", "valve-left", "01")),
    st.from_regex(r"[A-Za-z_][A-Za-z0-9_-]{0,10}", fullmatch=True),
)
_PORTS = _record("kind", "port", end=st.one_of(
    _record("owner", "component", component=_IDS, port=_IDS, direction=st.sampled_from(("input", "output", "physical"))),
    *(_record("owner", boundary, port=_IDS) for boundary in ("inputs", "outputs", "boundary")),
))
_WIRES = _record("kind", "wire", id=_IDS)
_ATTACHMENTS = _record("kind", "attachment", id=_IDS, parent=st.one_of(_PORTS, _WIRES))
_PARENTS = st.one_of(
    _record("kind", "graph"), _record("kind", "component", id=_IDS),
    _record("kind", "physical_connection", id=_IDS), _PORTS, _WIRES, _ATTACHMENTS,
)
_PARTS = st.one_of(
    st.integers(min_value=0, max_value=2**53 - 1),
    st.sampled_from(("0", "001", "quote\"", "path/part", "a.b#c", "a..b", "_", "line\nbreak")),
    st.text(st.characters(exclude_categories=("Cs",)), max_size=12),
)
_ELEMENTS = st.one_of(
    _PARENTS, _record("kind", "inputs"), _record("kind", "outputs"),
    _record("kind", "field", parent=_PARENTS, path=st.lists(_PARTS, min_size=1, max_size=4)),
)


@given(composites=st.lists(_IDS, max_size=3), element=_ELEMENTS)
def test_authored_model_address_round_trips_every_element_shape(composites, element):
    address = ModelAddress.model_validate({"composites": composites, "element": element})
    stored = address.to_authored_text()
    assert ModelAddress.from_authored_text(stored) == address
    assert ModelAddress.from_authored_text(stored).to_authored_text() == stored


def test_compact_spelling_distinguishes_boundaries_positions_and_quoted_keys():
    for boundary, direction in (("inputs", "in"), ("outputs", "out")):
        port = ModelAddress.from_authored_text(f"{boundary}.x")
        component = ModelAddress.from_authored_text(f"{direction}.{boundary}.x")
        assert port.label() == component.label()
        assert port != component
        assert port.element.end.owner == boundary
        assert component.element.end.component == boundary

    text = r'outer-left/inner/wire.drive#delay.state.0."0"."a/b.#\"\\"'
    address = ModelAddress.from_authored_text(text)
    assert address.composites == ("outer-left", "inner")
    assert address.element.path == ("state", 0, "0", 'a/b.#"\\')
    assert address.element.parent.id == "delay"
    assert address.element.parent.parent.id == "drive"
    assert address.to_authored_text() == text

    runtime = '{"composites":[],"element":{"end":{"owner":"outputs","port":"x"},"kind":"port"}}'
    assert port.to_text() == runtime
    assert ModelAddress.from_text(runtime) == port


@pytest.mark.parametrize("text", [
    "", " graph", "graph ", "outer//graph", '"outer"/graph', "out.plant", "inputs.", "outputs#probe",
    "component.plant#probe", "wire.drive#", "physical_connection.joint", 'inputs."load"',
    "component.plant.-1", "component.plant.01", "component.plant.9007199254740992",
    'component.plant."plain"', r'component.plant."\u0031"', r'component.plant."a\/b"', 'graph."unclosed',
])
def test_authored_model_address_refuses_malformed_and_noncanonical_spelling(text):
    with pytest.raises(ValueError):
        ModelAddress.from_authored_text(text)


@given(composites=st.lists(_IDS, max_size=3), parent=_PARENTS, parts=st.lists(_PARTS, min_size=1, max_size=4))
def test_sweep_targets_round_trip_compact_typed_fields(composites, parent, parts):
    address = ModelAddress.model_validate({
        "composites": composites, "element": {"kind": "field", "parent": parent, "path": parts},
    })
    for scope in (None, ("documents", "reach-model"), ("steps", "train")):
        text = model_field_path(address, scope)
        axis = ListedAxis(path=text, values=[1])
        target = model_field_target(axis.path)
        assert target.address == address
        assert target.scope == scope
        write = OverridePatch(path=text, value=1)
        assert sweep_patch(write.path, SetValue(value=1)).address == address


def test_checkpoint_rule_target_is_disjoint_and_legacy_sweep_addresses_are_refused():
    axis = CheckpointRuleAxis(checkpoint_rule={"every": 2})
    assert axis.path == CHECKPOINT_RULE_TARGET == "{@subject.checkpoint}"
    assert model_field_target(axis.path) is None
    assert model_field_target("trainer.settings.seed") is None
    address = ModelAddress.from_authored_text("component.plant.fields.mass")
    for target in (address.to_text(), f"documents.reach-model.{address.to_text()}"):
        with pytest.raises(ValidationError):
            ListedAxis(path=target, values=[2])
