"""P2 Model member shape, ordering, and projection to the executable contract."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from feedbax.compiler.model_document import lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_VERSION
from feedbax.contracts.model_input import authored_document_input
from feedbax.contracts.project import BasedDocument, DocumentBase, Project, RootDocument, SetDocumentValue
from feedbax.contracts.project_inputs import DocumentReference, InitialModelReference
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.examples import load_shipped_projects
from feedbax.web.app import create_app


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


def test_attachment_address_names_its_keyed_member():
    path = ("interior", "components", "first", "interior", "wires", "drive", "attachments", "gain", "operands")
    address = ModelAddress.at_stored_path(path)
    assert address.composites == ("first",)
    assert address.element.parent.id == "gain"
    assert address.stored_path() == path


def test_initial_input_uses_the_lowered_schema_and_order():
    stored = RootDocument(schema=STORED_MODEL_DOCUMENT_SCHEMA_ID, label="Model", payload=_stored_model())
    source = authored_document_input(
        InitialModelReference(kind="initial_model", source=DocumentReference(document="model"), seed=4), stored,
    )
    assert source.document.schema_version == MODEL_DOCUMENT_SCHEMA_VERSION
    assert [port.id for port in source.document.ports.inputs] == ["right", "left"]
    assert source.seed == 4


@pytest.mark.parametrize("inherited", [True, False])
def test_based_model_port_order_survives_reopen_and_undo(tmp_path, inherited):
    outputs = {"right": {"label": "Right", "type": {}}, "left": {"label": "Left", "type": {}}}
    differences = {} if inherited else {"/ports/outputs": SetDocumentValue(value=outputs)}
    project = Project(documents={
        "base": RootDocument(
            schema=STORED_MODEL_DOCUMENT_SCHEMA_ID, label="Base",
            payload={"step_seconds": 0.01, "ports": {"outputs": outputs} if inherited else {}, "interior": {}},
        ),
        "tuned": BasedDocument(
            schema=STORED_MODEL_DOCUMENT_SCHEMA_ID, label="Tuned", base=DocumentBase(document="base"),
            differences=differences,
        ),
    })
    for name, data in dump_text_project(project, ProjectConfiguration(project="model", store="results")).items():
        path = tmp_path / "model" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        url = "/api/projects/model"

        def opened():
            return client.get(url + "/open").json()["snapshot"]

        def field(snapshot, key):
            return snapshot["project"]["documents"]["tuned"].get(key, {})

        def port_order(snapshot):
            response = client.post(url + "/model/projection", json={
                "revision": snapshot["revision"], "document": "tuned",
            })
            assert response.is_success, response.text
            return list(response.json()["resolved_model"]["ports"]["outputs"])

        snapshot = opened()
        assert port_order(snapshot) == ["right", "left"]
        response = client.post(url + "/model/edit", json={
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": "tuned", "command": {"operation": {
                "op": "move_port", "index": 0, "address": {
                    "composites": [], "element": {"kind": "port", "end": {"owner": "outputs", "port": "left"}},
                },
            }},
        })
        assert response.is_success, response.text
        token = response.json()["result"]["undo_token"]
        saved = opened()
        ordered = field(saved, "differences")["/ports/outputs"]
        if inherited:
            assert ordered == {"order": ["left", "right"]}
        else:
            assert list(ordered["value"]) == ["left", "right"]
        assert port_order(saved) == ["left", "right"]

        response = client.post(url + "/model/undo", json={
            "revision": saved["revision"], "presentation_revision": saved["presentation_revision"], "token": token,
        })
        assert response.is_success, response.text
        restored = opened()
        assert field(restored, "differences") == {
            path: value.model_dump(mode="json") for path, value in differences.items()
        }
        assert port_order(restored) == ["right", "left"]
