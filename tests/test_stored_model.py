"""P2 Model member shape, ordering, and projection to the executable contract."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from feedbax.compiler.model_document import lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_VERSION
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.examples import load_shipped_projects


def _stored_model() -> dict:
    return {
        "step_seconds": 0.02,
        "ports": {"inputs": {
            "right": {"label": "Right", "type": {}},
            "left": {"label": "Left", "type": {}},
        }, "outputs": {"out": {"label": "Out", "type": {}}}},
        "interior": {
            "components": {
                "second": {"kind": "leaf", "label": "Second", "type": "feedbax.Scale", "attachments": {
                    "sources": {"output": {
                        "probe": {"kind": "probe", "label": "Probe"},
                        "gain": {"kind": "intervention", "operation": "scale"},
                    }},
                    "sinks": {"input": {"incoming": {"kind": "probe", "label": "Incoming"}}},
                }},
                "first": {
                    "kind": "composite",
                    "label": "First",
                    "ports": {"physical": {"mount": {"label": "Mount", "type": "translational"}}},
                    "interior": {
                        "physical_domain": "translational",
                        "components": {"mass": {"kind": "leaf", "label": "Mass", "type": "feedbax.Mass"}},
                        "connections": {"joint": {
                            "a": {"owner": "boundary", "port": "mount"},
                            "b": {"owner": "component", "component": "mass", "port": "port"},
                        }},
                    },
                },
            },
            "wires": {"drive": {
                "source": {"owner": "inputs", "port": "right"},
                "target": {"owner": "component", "component": "second", "port": "input"},
                "attachments": {
                    "probe": {"kind": "probe", "label": "After drive"},
                    "delay": {"kind": "delay", "delay": {"value": 1, "unit": "st"}},
                    "hold": {"kind": "hold"},
                },
            }},
        },
    }


def test_lowering_restores_member_addresses_and_attachment_order():
    stored = StoredModelDocument.model_validate(_stored_model())
    runtime = stored.lower()
    assert runtime.schema_version == MODEL_DOCUMENT_SCHEMA_VERSION
    assert [port.id for port in runtime.ports.inputs] == ["right", "left"]
    assert list(runtime.interior.components) == ["second", "first"]
    chain = runtime.interior.wires["drive"].attachments
    assert [item.id for item in chain] == ["probe", "delay", "hold"]
    assert chain[0].name == "After drive"
    leaf = runtime.interior.components["second"]
    assert [item.id for item in leaf.attachments.sources["output"]] == ["probe", "gain"]
    assert leaf.attachments.sinks["input"][0].name == "Incoming"
    physical = runtime.interior.components["first"]
    assert physical.ports.physical[0].id == "mount"
    assert physical.interior.connections["joint"].a.port == "mount"

    projected = StoredModelDocument.from_lowered(runtime)
    payload = projected.model_dump(mode="json")
    assert "schema_version" not in payload
    assert list(payload["interior"]["wires"]["drive"]["attachments"]) == ["probe", "delay", "hold"]
    assert list(payload["ports"]["inputs"]) == ["right", "left"]
    assert projected == stored


@pytest.mark.parametrize(("path", "value", "problem"), [
    (("ports", "inputs"), [], "dict_type"),
    (("ports", "inputs", "right"), {"name": "Right", "type": {}}, "label"),
    (("ports", "inputs", "right", "id"), "right", "extra_forbidden"),
    (("ports", "outputs", "2bad"), {"label": "Bad", "type": {}}, "string_pattern_mismatch"),
    (("ports", "outputs", "right"), {"label": "Duplicate", "type": {}}, "port IDs"),
    (("interior", "wires", "drive", "attachments"), [], "dict_type"),
    (("interior", "wires", "drive", "attachments", "probe", "id"), "probe", "extra_forbidden"),
    (("interior", "wires", "second"), {"source": {"owner": "inputs", "port": "right"},
                                     "target": {"owner": "outputs", "port": "out"}}, "element IDs"),
    (("schema_version",), MODEL_DOCUMENT_SCHEMA_VERSION, "extra_forbidden"),
])
def test_stored_model_rejects_nonkeyed_or_ambiguous_members(path, value, problem):
    raw = _stored_model()
    at = raw
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = deepcopy(value)
    with pytest.raises(ValidationError, match=problem):
        StoredModelDocument.model_validate(raw)


def test_shipped_model_projection_preserves_the_executable_graph():
    document = load_shipped_projects()["point-mass-reaching"].documents["point-mass"]
    original = StoredModelDocument.model_validate(document.payload).lower()
    restored = StoredModelDocument.model_validate_json(
        StoredModelDocument.from_lowered(original).model_dump_json(),
    ).lower()
    assert canonical_json_v2_bytes(restored.scientific_projection()) == canonical_json_v2_bytes(
        original.scientific_projection(),
    )
    registry = ComponentRegistry()
    assert lower_model_document(restored, registry).graph == lower_model_document(original, registry).graph
