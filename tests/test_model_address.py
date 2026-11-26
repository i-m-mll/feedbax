"""The authored Model address identifies exactly one typed element or field."""

from hypothesis import given, strategies as st
import pytest

from feedbax.contracts.model_address import ModelAddress


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
    st.sampled_from(("0", "001", "quote\"", "path/part", "a.b#c", "_", "line\nbreak")),
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
