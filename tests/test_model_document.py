"""The stored model and address contracts through the shared project format."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.project import Document, Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project


def _leaf(name: str = "Gain") -> dict:
    return {"kind": "leaf", "name": name, "type": "feedbax.Scale", "fields": {"name": "scientific"}}


def _three_levels() -> dict:
    interior = {"components": {"same": _leaf()}}
    for label in ("inner", "outer"):
        interior = {"components": {
            "same": _leaf(),
            label: {
                "kind": "composite", "name": label,
                "ports": {"outputs": [{"id": "out", "name": "nested", "type": {}}]},
                "interior": interior,
            },
        }}
    return {
        "step_seconds": 0.02,
        "ports": {"inputs": [{"id": "root-in", "name": "task", "type": {}}]},
        "interior": interior,
    }


def test_text_project_preserves_one_recursive_model(tmp_path: Path) -> None:
    payload = _three_levels()
    document = Document(schema=MODEL_DOCUMENT_SCHEMA_ID, payload=payload)
    project = Project(documents={"model": document})
    for name, data in dump_text_project(project, ProjectConfiguration(project="model", store="store")).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert load_text_project(tmp_path).project.documents["model"].payload == payload
    del payload["interior"]["components"]["outer"]["interior"]
    with pytest.raises(ValidationError, match="interior"):
        Document(schema=MODEL_DOCUMENT_SCHEMA_ID, payload=payload)


def test_physical_ports_and_connections_are_confined_to_physical_interiors() -> None:
    payload = _three_levels()
    physical = {"id": "flange", "name": "Flange", "type": "translational"}
    outer = payload["interior"]["components"]["outer"]
    outer["ports"]["physical"] = [physical]
    with pytest.raises(ValidationError, match="require an Equations or Mechanics interior"):
        ModelDocument.model_validate(payload)

    outer["interior"] = {"physical_domain": "translational", "connections": {
        "link": {"a": {"owner": "boundary", "port": "flange"},
                 "b": {"owner": "component", "component": "mass", "port": "flange"}},
    }}
    model = ModelDocument.model_validate(payload)
    assert model.interior.components["outer"].interior.kind == "equations"
    payload["interior"]["connections"] = outer["interior"]["connections"]
    with pytest.raises(ValidationError, match="connections"):
        ModelDocument.model_validate(payload)


def test_scientific_projection_preserves_declared_meaning_on_rename() -> None:
    payload = _three_levels()
    original = ModelDocument.model_validate(payload).scientific_projection()
    outer = payload["interior"]["components"]["outer"]
    outer["name"] = "Renamed"
    outer["ports"]["outputs"][0]["name"] = "renamed"
    outer["interior"]["components"]["inner"]["interior"]["components"]["same"]["name"] = "Same"
    assert ModelDocument.model_validate(payload).scientific_projection() == original
    outer["interior"]["components"]["inner"]["interior"]["components"]["same"]["fields"]["name"] = "new"
    assert ModelDocument.model_validate(payload).scientific_projection() != original

    payload = _three_levels()
    payload["ports"]["inputs"][0]["name"] = "other task signal"
    assert ModelDocument.model_validate(payload).scientific_projection() != original


def test_model_address_canonical_text_distinguishes_a_port_and_field_path() -> None:
    address = ModelAddress.model_validate({
        "document": "model", "composites": ["outer", "inner"],
        "element": {
            "kind": "field",
            "parent": {"kind": "port", "end": {
                "owner": "component", "component": "same", "port": "value", "direction": "input",
            }},
            "path": ["fields", "array/length", 0],
        },
    })
    assert address.to_text() == (
        '{"composites":["outer","inner"],"document":"model","element":{"kind":"field",'
        '"parent":{"end":{"component":"same","direction":"input","owner":"component",'
        '"port":"value"},"kind":"port"},"path":["fields","array/length",0]}}'
    )
    assert ModelAddress.from_text(address.to_text()) == address
    with pytest.raises(ValueError, match="canonical JSON"):
        ModelAddress.from_text(address.to_text() + " ")
    with pytest.raises(ValidationError, match="path"):
        ModelAddress.from_text(address.to_text().replace(",0]", ",true]"))
