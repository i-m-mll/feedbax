"""Studio edits the same durable authored files the text reader opens."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

from fastapi.testclient import TestClient
import pytest

from feedbax._atomic import write_bytes_atomic
from feedbax.contracts.jobs import StudioJobEventHistory
from feedbax.contracts.project import Project
from feedbax.contracts.project_experiment import PROJECT_DECLARATION_FILENAME
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from feedbax.contracts.project_plan import compile_project
from feedbax.host import HostError
from feedbax.web.api import jobs
from feedbax.web.app import create_app
from feedbax.web.services import project_folder


_URL = "/api/projects/reach"
_GAIN = "/payload/nodes/gain/params/gain"


@pytest.fixture
def study(tmp_path: Path):
    corpus = json.loads(
        (Path(__file__).parent / "fixtures" / "request_identity_corpus.json").read_bytes()
    )
    spec = next(row["spec"] for row in corpus["requests"] if row["kind"] == "train")
    graph = spec.pop("graph")
    project = Project.model_validate(
        {
            "documents": {"model": {"schema": graph["schema_id"], "payload": graph}},
            "steps": {
                "fit": {
                    "kind": "train",
                    "base": {"inline": {"graph": {"document": "model"}, **spec}},
                    "sweep": {
                        "axes": {
                            "seed": {
                                "values": {
                                    "10": {"patches": {"seed": {"value": 10}}},
                                    "2": {"patches": {"seed": {"value": 2}}},
                                }
                            }
                        },
                        "exceptions": {
                            "z_first": {"where": {"rows": []}},
                            "a_second": {"where": {"rows": []}},
                        },
                    },
                }
            },
        }
    )
    root = tmp_path / "reach"
    for name, data in dump_text_project(
        project, ProjectConfiguration(project="reach", store="results")
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root, TestClient(create_app(projects_dir=tmp_path))


def _read(client: TestClient):
    response = client.get(_URL)
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def _patch(
    client: TestClient,
    revision: str,
    value: int | float,
    *,
    kind="document",
    name="model",
    path=_GAIN,
):
    return client.patch(
        _URL,
        json={
            "revision": revision,
            "address": {"kind": kind, "id": name},
            "patches": [{"op": "replace", "path": path, "value": value}],
        },
    )


def _gain(root: Path):
    return (
        load_text_project(root)
        .project.documents["model"]
        .payload["nodes"]["gain"]["params"]["gain"]
    )


def test_new_projects_are_durable_openable_and_distinct(tmp_path):
    projects = tmp_path / "projects"
    app = create_app(projects_dir=projects)

    def create():
        client = TestClient(app)
        try:
            response = client.post("/api/projects")
            assert response.status_code == 201, response.text
            assert response.headers["Cache-Control"] == "no-store"
            document = response.json()
            project_id = document["snapshot"]["project_id"]
            assert response.headers["Location"] == f"/api/projects/{project_id}/open"
            return document
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        creates = [pool.submit(create) for _ in range(2)]
        documents = [creation.result() for creation in creates]

    ids = {document["snapshot"]["project_id"] for document in documents}
    assert len(ids) == 2
    assert {entry.name for entry in projects.iterdir()} == ids
    reopened = TestClient(create_app(projects_dir=projects))
    for document in documents:
        snapshot = document["snapshot"]
        project_id = snapshot["project_id"]
        folder = projects / project_id
        text = load_text_project(folder)
        assert text.project == Project()
        assert text.configuration.project == project_id
        assert text.store_path == folder / "results"
        assert text.revision == snapshot["revision"]
        assert json.loads((folder / "presentation.json").read_bytes()) == {}
        response = reopened.get(f"/api/projects/{project_id}/open")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json() == document
        assert reopened.get(f"/api/projects/{project_id}").json() == snapshot


def test_rename_changes_only_the_declaration_and_checks_authored_revision(study):
    root, client = study
    initial = client.get(f"{_URL}/open").json()
    presentation = {"documents": {"model": {"camera": {"x": 1, "y": 2, "zoom": 1.5}}}}
    response = client.put(f"{_URL}/presentation", json={
        "revision": initial["snapshot"]["presentation_revision"],
        "presentation": presentation,
    })
    assert response.status_code == 200, response.text
    saved = response.json()
    store = root / "results"
    store.mkdir()
    (store / "retained").write_bytes(b"existing store contents")
    before = load_text_project(root)
    plan = compile_project(before.project)
    assert not plan.diagnostics
    declaration = root / PROJECT_DECLARATION_FILENAME
    original_declaration = json.loads(declaration.read_bytes())
    untouched = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path != declaration
    }

    response = client.post(f"{_URL}/rename", json={"revision": saved["revision"], "name": "renamed"})
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    renamed = response.json()
    assert renamed["configuration"] == {**initial["configuration"], "project": "renamed"}
    current = renamed["snapshot"]
    assert current["revision"] != saved["revision"]
    assert current == {**saved, "revision": current["revision"]}
    assert json.loads(declaration.read_bytes()) == {**original_declaration, "project": "renamed"}
    assert all(path.read_bytes() == data for path, data in untouched.items())
    after = load_text_project(root)
    assert after.store_path == before.store_path
    assert compile_project(after.project) == plan

    stale = client.post(f"{_URL}/rename", json={"revision": saved["revision"], "name": "stale"})
    assert stale.status_code == 409
    assert stale.json()["detail"]["current"] == current["revision"]
    assert _patch(client, saved["revision"], 9).status_code == 409
    assert client.post(
        f"{_URL}/rename", json={"revision": current["revision"], "name": "renamed"}
    ).json() == renamed
    reopened = TestClient(create_app(projects_dir=root.parent))
    assert reopened.get(f"{_URL}/open").json() == renamed
    assert reopened.get("/api/projects/renamed/open").status_code == 404


def test_open_uses_managed_folder_ids_and_refuses_directory_symlinks(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    for name, content in dump_text_project(
        Project(), ProjectConfiguration(project="external", store="results")
    ).items():
        (external / name).write_bytes(content)
    (projects / "alias").symlink_to(external, target_is_directory=True)
    client = TestClient(create_app(projects_dir=projects))
    assert client.get("/api/projects/alias/open").status_code == 422
    assert client.get("/api/projects/external/open").status_code == 404
    assert client.get("/api/projects/external.json/open").status_code == 422


def test_rename_and_patch_from_one_revision_have_one_winner(study):
    root, client = study
    original = _read(client)
    start = Barrier(2)

    def rename():
        start.wait(timeout=4)
        return client.post(f"{_URL}/rename", json={
            "revision": original["revision"], "name": "renamed",
        })

    def patch():
        start.wait(timeout=4)
        return _patch(client, original["revision"], 3)

    with ThreadPoolExecutor(max_workers=2) as pool:
        rename_request, patch_request = pool.submit(rename), pool.submit(patch)
        renamed, patched = rename_request.result(), patch_request.result()

    assert sorted([renamed.status_code, patched.status_code]) == [200, 409]
    opened = client.get(f"{_URL}/open").json()
    if renamed.is_success:
        assert opened == renamed.json()
        assert patched.json()["detail"]["current"] == opened["snapshot"]["revision"]
        assert opened["snapshot"]["semantic_root"] == original["semantic_root"]
    else:
        assert opened["snapshot"] == patched.json()
        assert renamed.json()["detail"]["current"] == opened["snapshot"]["revision"]
        assert opened["configuration"]["project"] == "reach"
        assert _gain(root) == 3


def test_named_document_edit_reopens_and_presentation_leaves_authored_identity(study):
    root, client = study
    before = _read(client)
    assert "note" not in before["project"]["steps"]["fit"]
    file_revisions = {entry["address"]["id"]: entry["revision"] for entry in before["file_revisions"]}
    authored_file = root / "documents/model.json"
    authored_file.chmod(0o644)
    response = _patch(client, before["revision"], 3)
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["revision"] != before["revision"]
    assert saved["semantic_root"] != before["semantic_root"]
    current_files = {entry["address"]["id"]: entry["revision"] for entry in saved["file_revisions"]}
    assert current_files["fit"] == file_revisions["fit"]
    assert current_files["model"] != file_revisions["model"]
    assert _gain(root) == 3
    assert authored_file.stat().st_mode & 0o777 == 0o644
    assert _read(TestClient(create_app(projects_dir=root.parent))) == saved
    plan_before = compile_project(load_text_project(root).project)
    assert len(plan_before.runs) == 2 and not plan_before.diagnostics
    authored_bytes = authored_file.read_bytes()

    presentation = {
        "documents": {
            "model": {
                "positions": {"gain": {"x": 40, "y": 20}},
                "camera": {"x": 2, "y": 3, "zoom": 1.5},
            }
        },
        "steps": {"fit": {"sweep_order": ["seed"], "folded_sweep_id": "seed"}},
    }
    response = client.put(
        f"{_URL}/presentation",
        json={
            "revision": saved["presentation_revision"],
            "presentation": presentation,
        },
    )
    assert response.status_code == 200, response.text
    placed = response.json()
    assert placed["presentation"]["documents"] == presentation["documents"]
    assert placed["presentation"]["steps"] == presentation["steps"]
    assert placed["presentation_revision"] != saved["presentation_revision"]
    assert placed["revision"] == saved["revision"]
    assert placed["semantic_root"] == saved["semantic_root"]
    assert authored_file.read_bytes() == authored_bytes
    assert compile_project(load_text_project(root).project) == plan_before
    assert _read(TestClient(create_app(projects_dir=root.parent))) == placed

    presentation_bytes = (root / "presentation.json").read_bytes()
    invalid = client.put(
        f"{_URL}/presentation",
        json={
            "revision": placed["presentation_revision"],
            "presentation": {
                **presentation,
                "window": {"drawer_open": True},
                "playback_position": 0.3,
            },
        },
    )
    assert invalid.status_code == 422
    assert (root / "presentation.json").read_bytes() == presentation_bytes
    stale = client.put(
        f"{_URL}/presentation",
        json={
            "revision": saved["presentation_revision"],
            "presentation": {},
        },
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["current"] == placed["presentation_revision"]
    assert stale.json()["detail"]["resource"] == "presentation"


def test_named_sweep_edit_preserves_value_ids_order_and_other_request(study):
    root, client = study
    before = _read(client)
    previous = compile_project(load_text_project(root).project)
    response = _patch(
        client,
        before["revision"],
        7,
        kind="step",
        name="fit",
        path="/sweep/axes/seed/values/10/patches/seed/value",
    )
    assert response.status_code == 200, response.text
    reopened = load_text_project(root)
    sweep = reopened.project.steps["fit"].sweep
    assert list(sweep.axes["seed"].values) == ["10", "2"]
    assert list(sweep.exceptions) == ["z_first", "a_second"]
    current = compile_project(reopened.project)
    assert [run.key for run in current.runs] == [run.key for run in previous.runs]
    assert [run.coordinates for run in current.runs] == [run.coordinates for run in previous.runs]
    assert current.runs[0].request_id != previous.runs[0].request_id
    assert current.runs[1].request_id == previous.runs[1].request_id


def test_competing_writers_from_one_revision_have_one_winner(study):
    root, client = study
    revision = _read(client)["revision"]
    start = Barrier(2, timeout=5)

    def save(value):
        writer = TestClient(create_app(projects_dir=root.parent))
        start.wait()
        return value, _patch(writer, revision, value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(save, [4, 5]))
    assert sorted(response.status_code for response in results.values()) == [200, 409]
    (winner,) = (value for value, response in results.items() if response.status_code == 200)
    (loser,) = (response for response in results.values() if response.status_code == 409)
    assert _gain(root) == winner
    assert loser.json()["detail"]["current"] == load_text_project(root).revision


def test_external_revision_and_invalid_batch_preserve_external_bytes(study):
    root, client = study
    baseline = _read(client)
    path = root / "documents/model.json"
    path.write_bytes(path.read_bytes() + b"\n")
    reformatted = _read(client)
    assert reformatted["revision"] != baseline["revision"]
    assert reformatted["semantic_root"] == baseline["semantic_root"]
    assert _patch(client, baseline["revision"], 3).status_code == 409

    body = json.loads(path.read_bytes())
    body["payload"]["nodes"]["gain"]["params"]["gain"] = 9
    data = json.dumps(body, indent=4).encode()
    write_bytes_atomic(path, data)
    assert _gain(root) == 9
    assert _patch(client, reformatted["revision"], 3).status_code == 409
    latest = _read(client)
    assert latest["semantic_root"] != reformatted["semantic_root"]
    response = client.patch(
        _URL,
        json={
            "revision": latest["revision"],
            "address": {"kind": "document", "id": "model"},
            "patches": [
                {"op": "replace", "path": _GAIN, "value": 3},
                {"op": "remove", "path": "/schema"},
            ],
        },
    )
    assert response.status_code == 422
    assert path.read_bytes() == data
    assert load_text_project(root).revision == latest["revision"]


def test_external_change_while_staging_is_detected_before_replace(study, monkeypatch):
    root, client = study
    before = _read(client)
    path = root / "documents/model.json"
    raw = path.read_bytes() + b"\n"

    def inject_edit(target, data, **kwargs):
        precondition = kwargs.pop("before_publish")

        def competing_edit():
            write_bytes_atomic(target, raw)
            precondition()

        return write_bytes_atomic(target, data, before_publish=competing_edit, **kwargs)

    monkeypatch.setattr(project_folder, "write_bytes_atomic", inject_edit)
    response = _patch(client, before["revision"], 3)
    assert response.status_code == 409
    assert path.read_bytes() == raw
    assert not list(path.parent.glob(".*.tmp"))


def test_unreadable_presentation_does_not_hide_the_authored_project(study):
    root, client = study
    before = _read(client)
    (root / "presentation.json").write_bytes(b"{")
    broken = _read(client)
    assert broken["presentation"] is None
    assert broken["presentation_error"] == "unreadable_presentation"
    assert broken["revision"] == before["revision"]
    assert broken["presentation_revision"] != before["presentation_revision"]
    assert _patch(client, broken["revision"], 3).status_code == 200
    repaired = client.put(
        f"{_URL}/presentation",
        json={
            "revision": broken["presentation_revision"],
            "presentation": {},
        },
    )
    assert repaired.status_code == 200
    assert repaired.json()["presentation_error"] is None
    assert _gain(root) == 3


def test_project_and_authored_paths_cannot_point_outside_the_root(study, tmp_path):
    root, client = study
    (tmp_path / "elsewhere").symlink_to(root, target_is_directory=True)
    assert client.get("/api/projects/elsewhere").status_code == 422
    original = root / "documents/model.json"
    outside = tmp_path / "outside.json"
    original.rename(outside)
    original.symlink_to(outside)
    assert client.get(_URL).status_code == 422


def test_existing_event_stream_observes_an_actual_external_file_edit(study, monkeypatch):
    root, client = study
    before = load_text_project(root).revision
    monkeypatch.setattr(jobs, "_FOLDER_OBSERVATION_INTERVAL", 0)

    class JobsStream:
        calls = 0

        def events(self, *, after, projects, wait, machine):
            self.calls += 1
            if self.calls == 2:
                path = root / "documents/model.json"
                document = json.loads(path.read_bytes())
                document["payload"]["nodes"]["gain"]["params"]["gain"] = 11
                write_bytes_atomic(path, json.dumps(document).encode())
            if self.calls > 2:
                raise HostError("unavailable", "Stream ended")
            return 0, StudioJobEventHistory(cursor=0, oldest_cursor=0, events=[])

    client.app.state.job_service = JobsStream()
    response = client.get("/api/events?folder=reach")
    assert response.status_code == 200
    observations = [
        json.loads(frame.split("\ndata: ", 1)[1])
        for frame in response.text.split("\n\n")
        if frame.startswith("event: project_revision\n")
    ]
    assert [notice["revision"] for notice in observations] == [
        before,
        load_text_project(root).revision,
    ]
    assert observations[0]["revision"] != observations[1]["revision"]
    assert _gain(root) == 11
