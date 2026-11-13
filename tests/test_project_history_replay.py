"""Guarded replay of exact authored project snapshots."""

from base64 import b64decode, b64encode
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from feedbax import project_folder
from feedbax.contracts.jobs import StudioJobEventHistory
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project_experiment import PROJECT_DECLARATION_FILENAME
from feedbax.contracts.project_io import STUDIO_PATCH_JOURNAL_FILENAME, load_text_project
from feedbax.contracts.studio_project import StudioProjectReplay
from feedbax.host import HostError
from feedbax.web.api import jobs
from feedbax.web.app import create_app


URL = "/api/projects/study"
REPLAY = f"{URL}/history/replay"
MODEL = ("document", "model")
DECLARATION = ("declaration", "declaration")
STEP = ("step", "fit")
VARIANTS = [("variant", name) for name in ("10", "2")]
MANIFEST = {"schema_version": "feedbax.project.v1", "project": "study", "store": "results"}


def _write(root, relative, value, *, encoding="utf-8"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False) + "\n\n").encode(encoding)
    path.write_bytes(data)
    return data


def _model(seconds):
    return {
        "schema": MODEL_DOCUMENT_SCHEMA_ID,
        "payload": {"step_seconds": seconds, "ports": {}, "interior": {}},
    }


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    _write(root, PROJECT_DECLARATION_FILENAME, MANIFEST)
    _write(root, "documents/model.json", _model(0.01))
    _write(root, "documents/notes.json", {"schema": "notes", "payload": "café"})
    _write(root, "presentation.json", {})
    return root, TestClient(create_app(projects_dir=tmp_path))


def _snapshot(client):
    response = client.get(URL)
    assert response.status_code == 200, response.text
    return response.json()


def _files(snapshot):
    return {
        DECLARATION: snapshot["declaration"],
        **{(file["address"]["kind"], file["address"]["id"]): file for file in snapshot["file_revisions"]},
    }


def _replay_body(current, desired, *addresses):
    current_files, desired_files = _files(current), _files(desired)
    return {
        "revision": current["revision"],
        "files": [
            {
                "address": {"kind": kind, "id": name},
                "before": current_files.get((kind, name), {}).get("revision"),
                "after": desired_files.get((kind, name), {}).get("data"),
            }
            for kind, name in addresses
        ],
    }


def _replay(client, current, desired, *addresses):
    response = client.post(REPLAY, json=_replay_body(current, desired, *addresses))
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def test_replay_restores_exact_model_bytes_and_emits_one_revision(study, monkeypatch):
    root, client = study
    path = root / "documents/model.json"
    original = path.read_bytes()
    before = _snapshot(client)
    changed = _write(root, "documents/model.json", _model(0.02), encoding="utf-16")
    outside = _snapshot(client)
    assert b64decode(_files(outside)[MODEL]["data"]) == changed
    patch = client.patch(URL, json={
        "revision": outside["revision"], "address": {"kind": "document", "id": "model"},
        "patches": [{"op": "replace", "path": "/payload/step_seconds", "value": 0.03}],
    })
    assert patch.status_code == 422

    monkeypatch.setattr(jobs, "_FOLDER_OBSERVATION_INTERVAL", 0)

    class JobsStream:
        calls = 0

        def events(self, *, after, projects, wait, machine):
            self.calls += 1
            if self.calls == 2:
                body = _replay_body(outside, before, MODEL)
                restored = project_folder.ProjectFolder(root.parent).replay(
                    root.name, StudioProjectReplay.model_validate(body),
                )
                assert restored.revision == before["revision"]
            if self.calls > 2:
                raise HostError("unavailable", "Stream ended")
            return 0, StudioJobEventHistory(cursor=0, oldest_cursor=0, events=[])

    client.app.state.job_service = JobsStream()
    response = client.get("/api/events?folder=study")
    assert response.status_code == 200
    notices = [
        json.loads(frame.split("\ndata: ", 1)[1])
        for frame in response.text.split("\n\n") if frame.startswith("event: project_revision\n")
    ]
    assert [notice["revision"] for notice in notices] == [outside["revision"], before["revision"]]
    assert path.read_bytes() == original
    redo = _replay(client, before, outside, MODEL)
    assert redo == outside
    assert path.read_bytes() == changed


def test_replay_validates_every_target_before_publication(study):
    root, client = study
    before = _snapshot(client)
    body = _replay_body(before, before, MODEL, STEP)
    model, step = body["files"]
    step["after"] = b64encode(b'{"kind":"train","base":{"inline":{}}}').decode()
    model["after"] = b64encode(json.dumps(_model(-1)).encode()).decode()
    assert client.post(REPLAY, json=body).status_code == 422
    step["after"] = b64encode(b'{"kind":"train","base":{"inline":{"x":1e999}}}').decode()
    model["after"] = _files(before)[MODEL]["data"]
    assert client.post(REPLAY, json=body).status_code == 422
    model["before"] = "0" * 64
    conflict = client.post(REPLAY, json=body)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["current"] == before["revision"]
    model["before"] = _files(before)[MODEL]["revision"]
    body["revision"] = "0" * 64
    assert client.post(REPLAY, json=body).status_code == 409
    body["revision"] = before["revision"]
    for kind, name in (("step", "../fit"), ("presentation", "presentation")):
        step["address"] = {"kind": kind, "id": name}
        assert client.post(REPLAY, json=body).status_code == 422
    assert _snapshot(client) == before
    assert not (root / "steps").exists()
    assert not (root / STUDIO_PATCH_JOURNAL_FILENAME).exists()


def test_replay_recovers_variant_creation_and_removal_and_retains_configuration(study, monkeypatch):
    root, client = study
    before = _snapshot(client)
    step_data = _write(root, "steps/fit.json", {
        "kind": "train", "base": {"inline": {}},
        "sweep": {"axes": {"sample": {"values": {"10": {"value": 10}, "2": {"value": 2}}}}},
    })
    _write(root, "variants/10.json", {"label": "ten"})
    _write(root, "variants/2.json", {"label": "two"})
    _write(root, PROJECT_DECLARATION_FILENAME, {**MANIFEST, "sweep": {}, "variants": ["10", "2"]})
    outside = _snapshot(client)
    renamed = {**MANIFEST, "project": "renamed", "store": "other", "settings": {"foo": "bar"}}
    _write(root, PROJECT_DECLARATION_FILENAME, {**renamed, "sweep": {}, "variants": ["10", "2"]})
    current = _snapshot(client)
    assert current["declaration"]["authored_revision"] == outside["declaration"]["authored_revision"]
    assert current["declaration"]["revision"] != outside["declaration"]["revision"]
    invalid = client.post(REPLAY, json=_replay_body(current, before, VARIANTS[0]))
    assert invalid.status_code == 422

    untouched = {name: (root / name).read_bytes() for name in ("documents/notes.json", "presentation.json")}
    writer = project_folder.write_bytes_atomic
    interrupt = True

    def interrupt_declaration(path, data, **kwargs):
        nonlocal interrupt
        if path.name == PROJECT_DECLARATION_FILENAME and interrupt:
            interrupt = False
            raise OSError("interrupted before declaration publication")
        return writer(path, data, **kwargs)

    monkeypatch.setattr(project_folder, "write_bytes_atomic", interrupt_declaration)
    addresses = (STEP, *VARIANTS, DECLARATION)
    journal = root / STUDIO_PATCH_JOURNAL_FILENAME
    response = client.post(REPLAY, json=_replay_body(current, before, *addresses))
    assert response.status_code == 503
    assert journal.is_file() and not (root / "variants/10.json").exists()
    removed = _snapshot(client)
    assert not journal.exists()
    assert json.loads((root / PROJECT_DECLARATION_FILENAME).read_bytes()) == renamed
    assert removed["declaration"]["authored_revision"] == before["declaration"]["authored_revision"]
    assert not (root / "steps/fit.json").exists()

    (root / "variants").rmdir()
    (root / "steps").rmdir()
    interrupt = True
    response = client.post(REPLAY, json=_replay_body(removed, outside, *addresses))
    assert response.status_code == 503
    assert journal.is_file() and (root / "variants/10.json").is_file()
    restored = _snapshot(client)
    assert not journal.exists()
    assert restored["declaration"]["authored_revision"] == outside["declaration"]["authored_revision"]
    assert json.loads((root / PROJECT_DECLARATION_FILENAME).read_bytes()) == {
        **renamed, "sweep": {}, "variants": ["10", "2"],
    }
    assert list(load_text_project(root).project.sweep.rows) == ["10", "2"]
    assert (root / "steps/fit.json").read_bytes() == step_data
    assert {name: (root / name).read_bytes() for name in untouched} == untouched


def test_replay_refuses_a_symlinked_family(study, tmp_path: Path):
    root, client = study
    before = _snapshot(client)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "steps").symlink_to(outside, target_is_directory=True)
    body = _replay_body(before, before, STEP)
    body["files"][0]["after"] = b64encode(b'{"kind":"train","base":{"inline":{}}}').decode()
    assert client.post(REPLAY, json=body).status_code == 422
    assert list(outside.iterdir()) == []
    assert not (root / STUDIO_PATCH_JOURNAL_FILENAME).exists()


def test_replay_case_only_rename_is_order_independent(study):
    root, client = study
    original = root / "documents/model.json"
    renamed = original.with_name("Model.json")
    before = _snapshot(client)
    if renamed.exists():
        add_alias = _replay_body(before, before, ("document", "Model"))
        add_alias["files"][0]["after"] = _files(before)[MODEL]["data"]
        assert client.post(REPLAY, json=add_alias).status_code == 422
        assert _snapshot(client) == before

    original.rename(renamed)
    outside = _snapshot(client)
    assert _replay(client, outside, before, MODEL, ("document", "Model")) == before
    assert {path.name for path in original.parent.iterdir()} == {"model.json", "notes.json"}
    assert _replay(client, before, outside, ("document", "Model"), MODEL) == outside
    assert {path.name for path in original.parent.iterdir()} == {"Model.json", "notes.json"}
