"""Apply saved edits directly and through the existing Studio HTTP routes."""

import json
import subprocess
import sys
import os
from itertools import count
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import TypeAdapter
import pytest

from feedbax.contracts.model_edit import ModelEditCommand, ModelEditRequest
from feedbax.contracts.expression_values import LiteralValue, ValueBase, ValueType
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.contracts.studio_project import (
    PresentationDocument, StudioPresentationWrite, StudioProjectCreateFile, StudioProjectPatch,
    StudioProjectPatchBatch, StudioProjectRename,
)
from feedbax.contracts.task_document import Cell, Signal
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.project_folder import AppliedModelEdit, ProjectFolder, write_new_project_folder
from feedbax.studio import model_edit
from tests.test_document_train_pair import authored_documents, document_project


def _address(kind="graph", *, scope=(), **fields):
    return {"document": "reach-model", "composites": scope, "element": {"kind": kind, **fields}}


def _end(component, port):
    return {"owner": "component", "component": component, "port": port}


def _gain():
    return {"kind": "leaf", "name": "Gain", "type": "feedbax.Gain", "fields": {"gain": 1.0}}


def _port(owner, port):
    return {"owner": owner, "port": port}


def _model_project():
    functions, authored_model, task, objective = authored_documents()
    signal = Signal(
        id="test_signal", name="test_signal", type=ValueType(base=ValueBase.REAL),
        cells=(Cell(first_epoch="hold", last_epoch="move", value=LiteralValue(value=1.0)),),
    )
    task = task.model_copy(update={"signals": (*task.signals, signal)})
    raw = document_project((functions, authored_model, task, objective))
    model = raw["documents"]["reach-model"]["payload"]
    interior = model["interior"]
    interior["components"].update({name: _gain() for name in ("a", "b", "c", "d")})
    interior["components"]["physics"] = {
        "kind": "composite", "name": "Physics",
        "ports": {"physical": [{"id": "frame", "name": "Frame", "type": "translational"}]},
        "interior": {
            "physical_domain": "translational",
            "components": {"mass": {"kind": "leaf", "name": "Mass", "type": "feedbax.Mass"}},
        },
    }
    interior["wires"].update({
        "ab": {"source": _end("a", "output"), "target": _end("b", "input"), "attachments": [
            {"kind": "intervention", "id": "scale", "operation": "Scale", "fields": {"factor": 0.5}},
        ]},
        "bc": {"source": _end("b", "output"), "target": _end("c", "input")},
    })
    model["ports"]["inputs"].extend([
        {"id": "one", "name": "One", "type": {}, "default": {"value": 0}},
        {"id": "two", "name": "Two", "type": {}, "default": {"value": 0}},
    ])
    return Project.model_validate(raw)


def _edit_cases():
    graph = _address()
    a = _address("component", id="a")
    ab = _address("wire", id="ab")
    outer = _address("component", id="wrapper")
    point = {"x": 2, "y": 3}
    port = {"id": "fresh", "name": "Fresh", "type": {}}
    return {
        "add_component": {"graph": graph, "component": _gain(), "position": point},
        "move_components": {"placements": [{"address": a, "position": point}]},
        "flip_component": {"address": a},
        "collapse_graph": {"graph": _address(scope=("wrapper",)), "collapsed": True},
        "route_wire": {"address": ab, "waypoints": [point]},
        "delete": {"selection": [a]},
        "connect": {"graph": graph, "source": _end("c", "output"), "target": _end("d", "input")},
        "connect_task_signal": {"graph": graph, "run": {"variant": "default", "step": "train", "row": "default"},
                                "signal": "test_signal", "target": _end("d", "input")},
        "connect_physical": {
            "graph": _address(scope=("physics",)),
            "connection": {"a": _end("mass", "flange"), "b": _port("boundary", "frame")},
        },
        "add_port": {"graph": graph, "direction": "input", "port": port},
        "remove_port": {"address": _address("port", end=_port("inputs", "one"))},
        "rename": {"address": a, "name": "Renamed"},
        "move_port": {"address": _address("port", end=_port("inputs", "one")), "index": 0},
        "set_field": {"address": _address("field", parent=a["element"], path=["fields", "gain"]), "value": 2.0},
        "set_attachment": {"difference": {"op": "add", "owner": ab, "after": "scale", "attachment": {
            "kind": "intervention", "id": "second", "operation": "Scale", "fields": {"factor": 2.0},
        }}},
        "move_wire_attachment": {
            "address": _address("attachment", parent=ab["element"], id="scale"),
            "destination": _address("wire", id="bc"),
        },
        "insert_component": {"address": ab, "component": _gain(), "input_port": "input",
                             "output_port": "output", "position": point},
        "bring_signal_in": {"source_graph": graph, "source": _end("a", "output"),
                            "destination": _address(scope=("wrapper",))},
        "copy": {"selection": [a]},
        "cut": {"selection": [a]},
        "duplicate": {"selection": [a]},
        "paste": {"graph": graph, "clipboard": {"components": {"a": _gain()}, "wires": {}, "connections": {}}},
        "group": {"selection": [a], "name": "Group"},
        "ungroup": {"address": outer},
    }


@pytest.fixture(scope="module")
def model_project():
    return _model_project()


def _setup(root, project, name):
    root.mkdir(exist_ok=True)
    write_new_project_folder(
        root / name, project, ProjectConfiguration(project=name, store="results"), PresentationDocument(),
    )


def _files(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*.json")}


def test_core_import_has_no_web_dependency():
    subprocess.run([sys.executable, "-c", """
import ast
from pathlib import Path
import sys

from feedbax import project_folder, project_model_selection

for module in (project_folder, project_model_selection):
    tree = ast.parse(Path(module.__file__).read_bytes())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or '')
            names.extend(f'{node.module}.{item.name}' for item in node.names)
    assert not any(name == 'feedbax.web' or name.startswith('feedbax.web.') for name in names)
assert not any(name == 'feedbax.web' or name.startswith('feedbax.web.') for name in sys.modules)
"""], check=True, timeout=10, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})


def test_cases_cover_the_model_command_contract():
    variants = TypeAdapter(ModelEditCommand).json_schema()["properties"]["operation"]["discriminator"]["mapping"]
    assert set(_edit_cases()) == set(variants)


def test_each_named_operation_has_core_and_http_file_parity(tmp_path, monkeypatch, model_project):
    from feedbax.web.app import create_app

    direct, served = tmp_path / "direct", tmp_path / "served"
    registry = new_application_registry_bundle()
    core = ProjectFolder(direct, components=registry.components, functions=registry.expression_functions)
    cases = {}
    for operation, fields in _edit_cases().items():
        for root in (direct, served):
            _setup(root, model_project, operation)
        before = core.read(operation)
        original = _files(direct / operation)
        body = ModelEditRequest.model_validate({
            "revision": before.revision, "presentation_revision": before.presentation_revision,
            "document": "reach-model", "command": {"operation": {"op": operation, **fields}},
        })
        ids = count()
        monkeypatch.setattr(model_edit, "uuid4", lambda: UUID(int=next(ids)))
        result = core.apply_model_command(operation, body, registry.components)
        assert isinstance(result, AppliedModelEdit), (operation, result)
        changed = (result.snapshot.revision, result.snapshot.presentation_revision) != (
            before.revision, before.presentation_revision,
        )
        assert changed == (operation != "copy"), operation
        assert (_files(direct / operation) != original) == changed, operation
        cases[operation] = body, result

    with TestClient(create_app(projects_dir=served)) as client:
        for operation, (body, result) in cases.items():
            ids = count()
            response = client.post(
                f"/api/projects/{operation}/model/edit", json=body.model_dump(mode="json", exclude_unset=True),
            )
            assert response.status_code == 200, (operation, response.text)
            reply = response.json()["result"]
            assert reply["status"] == "applied", (operation, reply)
            assert (reply.get("undo_token") is not None) == (result.change is not None), operation
            snapshot = reply["snapshot"]
            assert snapshot["revision"] == result.snapshot.revision, operation
            assert snapshot["presentation_revision"] == result.snapshot.presentation_revision, operation
            assert reply["changed"] == [address.model_dump(mode="json") for address in result.changed], operation
            assert reply.get("clipboard") == (
                result.clipboard.model_dump(mode="json", exclude_unset=True) if result.clipboard else None
            ), operation
            assert _files(direct / operation) == _files(served / operation), operation


def test_named_files_and_patch_batches_have_core_and_http_parity(tmp_path, model_project):
    from feedbax.web.app import create_app

    direct, served = tmp_path / "direct", tmp_path / "served"
    for root in (direct, served):
        _setup(root, model_project, "project")
    core = ProjectFolder(direct)
    snapshot = core.read("project")
    with TestClient(create_app(projects_dir=served)) as client:
        url = "/api/projects/project"

        def compare(method, suffix, body, result, *, envelope=False):
            response = client.request(
                method, url + suffix, json=body.model_dump(mode="json", by_alias=True, exclude_unset=True),
            )
            assert response.is_success, response.text
            saved = response.json()
            if envelope:
                saved, result = saved["snapshot"], result.snapshot
            assert saved["revision"] == result.revision
            assert saved["presentation_revision"] == result.presentation_revision
            assert _files(direct) == _files(served)
            return result

        for file in (
            {"kind": "document", "id": "settings", "document": {"schema": "custom", "payload": {"x": 1}}},
            {"kind": "step", "id": "next", "step": {"kind": "train", "base": {"step": "train"}}},
        ):
            body = StudioProjectCreateFile.model_validate({"revision": snapshot.revision, "file": file})
            snapshot = compare("POST", "/files", body, core.create_file("project", body), envelope=True)
        patch = {"address": {"kind": "document", "id": "settings"},
                 "patches": [{"op": "replace", "path": "/payload/x", "value": 2}]}
        body = StudioProjectPatch.model_validate({"revision": snapshot.revision, **patch})
        snapshot = compare("PATCH", "", body, core.patch("project", body))
        patch["patches"][0]["value"] = 3
        body = StudioProjectPatchBatch.model_validate({"revision": snapshot.revision, "files": [
            patch, {"address": {"kind": "step", "id": "next"}, "patches": [
                {"op": "add", "path": "/group", "value": "experiment"},
            ]},
        ]})
        snapshot = compare("PATCH", "", body, core.patch("project", body))
        body = StudioPresentationWrite(
            revision=snapshot.presentation_revision, presentation=PresentationDocument(visible_groups=["experiment"]),
        )
        snapshot = compare("PUT", "/presentation", body, core.write_presentation("project", body))
        rename = StudioProjectRename(revision=snapshot.revision, name="renamed")
        compare("POST", "/rename", rename, core.rename("project", rename), envelope=True)
        assert "project" not in json.loads((direct / "project" / "renamed.feedbax").read_bytes())
