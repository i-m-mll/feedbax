"""Exercise native project tickets through the private and browser transports."""

from base64 import b64encode
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.project import ExternalBase, InlineBase
from feedbax.contracts.project_io import (
    ProjectConfiguration, STUDIO_WRITE_LOCK_FILENAME, dump_text_project, read_text_project_snapshot,
)
from feedbax.contracts.studio_project import StudioProjectRename
from feedbax import project_folder
from feedbax.desktop import transport as desktop_transport
from feedbax.desktop.control import Endpoint
from feedbax.desktop.preferences import register_preferences
from feedbax.jobs import PlanJobAttempts
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_train_project


def _desktop_clients(app):
    endpoint = Endpoint.create(41234)
    transport = desktop_transport.DesktopTransport(app, endpoint)
    transport.register_ticket_route(lambda: False)

    async def unix_socket(scope, receive, send):
        await transport({**scope, "server": None}, receive, send)

    async def loopback_socket(scope, receive, send):
        if scope["type"] == "http":
            # TestClient uses a list, while Uvicorn supplies the bound socket address.
            scope = {**scope, "server": tuple(scope["server"])}
        await transport(scope, receive, send)

    control = TestClient(unix_socket, base_url="http://localhost")
    browser = TestClient(loopback_socket, base_url=endpoint.origin)
    return transport, control, browser


def test_native_entry_creates_and_opens_project_windows(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    app = create_app(projects_dir=projects)
    folders = app.state.project_folders
    first = folders.new().snapshot
    second = folders.new().snapshot
    for project in (first, second):
        folders.rename(
            project.project_id,
            StudioProjectRename(name="same_title", revision=project.revision),
        )
    (projects / "alias").symlink_to(projects / first.project_id, target_is_directory=True)
    (projects / "empty").mkdir()
    external = tmp_path / "external"
    external.mkdir()

    transport, control, browser = _desktop_clients(app)
    endpoint = transport.endpoint
    ticket_url = "/_desktop/ticket"
    exchange_url = "/_desktop/exchange"

    def exchange(token):
        return browser.post(
            exchange_url,
            headers={"Origin": endpoint.origin, "X-Feedbax-Ticket": token},
        )

    for path, expected in (
        (Path(first.project_id), 422),
        (projects, 422),
        (external, 422),
        (projects / "alias", 422),
        (projects / "empty", 422),
        (projects / "missing", 404),
    ):
        assert control.post(ticket_url, json={"project_folder": str(path)}).status_code == expected
    assert control.post(ticket_url, json={
        "project_id": second.project_id,
        "project_folder": str(projects / first.project_id),
    }).status_code == 422

    default_ticket = control.post(ticket_url, json={}).json()
    created_id = default_ticket["project"]["id"]
    assert created_id not in (first.project_id, second.project_id)
    default = exchange(default_ticket["ticket"])
    assert default.status_code == 204
    assert default.headers["x-feedbax-project"] == created_id
    assert "HttpOnly" in default.headers["set-cookie"]
    created = browser.get(f"/api/projects/{created_id}/open")
    assert created.status_code == 200
    assert created.json()["snapshot"]["project_id"] == created_id
    assert browser.post(ticket_url, json={}, headers={"Origin": endpoint.origin}).status_code == 403
    browser.cookies.clear()

    for project in (first, second):
        open_url = f"/api/projects/{project.project_id}/open"
        browser.cookies.clear()
        assert browser.get(open_url).status_code == 403
        issued = control.post(ticket_url, json={"project_folder": str(projects / project.project_id)})
        assert issued.status_code == 200, issued.text
        ticket = issued.json()
        assert ticket["project"] == {"id": project.project_id, "title": "same_title"}
        authenticated = exchange(ticket["ticket"])
        assert authenticated.status_code == 204
        assert authenticated.headers["x-feedbax-project"] == project.project_id
        assert exchange(ticket["ticket"]).status_code == 403
        opened = browser.get(open_url)
        assert opened.status_code == 200, opened.text
        assert opened.json()["snapshot"]["project_id"] == project.project_id
        assert opened.json()["configuration"]["project"] == "same_title"

    renewed = control.post(ticket_url, json={"project_id": first.project_id})
    assert renewed.status_code == 200, renewed.text
    renewed_ticket = renewed.json()
    assert renewed_ticket["project"] == {"id": first.project_id, "title": "same_title"}
    ticket = transport.tickets[renewed_ticket["ticket"]]
    monkeypatch.setattr(desktop_transport, "time", SimpleNamespace(monotonic=lambda: ticket.expires))
    assert exchange(renewed_ticket["ticket"]).status_code == 403


def _write_train(root: Path, *, pinned_base: Path | None = None):
    root.mkdir(parents=True)
    project = managed_train_project()
    if pinned_base is not None:
        base = project.steps["train"].base
        assert isinstance(base, InlineBase)
        raw = base.model_dump_json(include={"inline"}).encode()
        path = root / pinned_base
        path.write_bytes(raw)
        project.steps["train"].base = ExternalBase(path=str(pinned_base), sha256=sha256_bytes(raw), pointer="/inline")
    files = dump_text_project(project, ProjectConfiguration(project="same_title", store="local-results"))
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)


def _cli_request_id(root: Path) -> str:
    cli = subprocess.run(
        [sys.executable, "-m", "feedbax.cli", "plan"],
        cwd=root, capture_output=True, text=True, check=False, timeout=20,
    )
    assert cli.returncode == 0, cli.stderr
    address, request_id = cli.stdout.split()
    assert address == "default/train/default"
    return request_id


def test_external_git_project_opens_saves_and_reopens_in_place(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    repo = tmp_path / "agent checkout"
    _write_train(repo)
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True, timeout=8)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, timeout=8)
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    app = create_app(projects_dir=projects)
    transport, control, browser = _desktop_clients(app)
    origin = {"Origin": transport.endpoint.origin}
    ticket_url = "/_desktop/ticket"

    def select(path):
        response = control.post(ticket_url, json={"project_folder": str(path)})
        assert response.status_code == 200, response.text
        return response.json()

    def exchange(ticket):
        response = browser.post(
            "/_desktop/exchange", headers={**origin, "X-Feedbax-Ticket": ticket["ticket"]},
        )
        assert response.status_code == 204
        assert response.headers["x-feedbax-project"] == ticket["project"]["id"]

    def studio_request_id(url, revision):
        response = browser.get(f"{url}/train-runs", params={"revision": revision})
        assert response.status_code == 200, response.text
        row, = response.json()["rows"]
        return row["request_id"]

    with browser:
        monkeypatch.setattr(
            app.state.job_service, "read_plan_attempts", lambda **_scope: PlanJobAttempts(0, {}),
        )
        created = control.post(ticket_url, json={}).json()
        exchange(created)
        created_id = created["project"]["id"]
        assert list(projects.iterdir()) == [projects / created_id]
        assert browser.post(ticket_url, json={"project_folder": str(repo)}, headers=origin).status_code == 403
        assert browser.get("/api/projects/agent%20checkout/open").status_code == 422
        assert browser.get("/api/projects/external-unknown/open").status_code == 404
        assert browser.get(f"/api/projects/{repo}/open").status_code == 404

        selected = select(repo)
        project_id = selected["project"]["id"]
        assert selected["project"]["title"] == "same_title"
        assert project_id != created_id
        exchange(selected)
        url = f"/api/projects/{project_id}"
        opened = browser.get(f"{url}/open")
        assert opened.status_code == 200, opened.text
        revision = opened.json()["snapshot"]["revision"]
        original_request = studio_request_id(url, revision)
        assert original_request == _cli_request_id(repo)
        settled = app.state.project_folders.read_plan(project_id, revision=revision)
        assert settled.store_path == repo.resolve() / "local-results"
        assert not settled.store_path.exists()

        patch = {
            "revision": revision,
            "address": {"kind": "step", "id": "train"},
            "patches": [{"op": "replace", "path": "/base/inline/training_config/batch_size", "value": 6}],
        }
        saved = browser.patch(url, json=patch, headers=origin)
        assert saved.status_code == 200, saved.text
        new_revision = saved.json()["revision"]
        assert new_revision != revision
        assert browser.patch(url, json=patch, headers=origin).status_code == 409
        changed_files = subprocess.run(
            ["git", "diff", "--name-only"], cwd=repo, check=True, capture_output=True, text=True, timeout=8,
        )
        assert changed_files.stdout.splitlines() == ["steps/train.json"]
        assert read_text_project_snapshot(repo).revision == new_revision
        new_request = studio_request_id(url, new_revision)
        assert new_request != original_request
        assert new_request == _cli_request_id(repo)

        assert select(repo)["project"] == selected["project"]
        renewed = control.post(ticket_url, json={"project_id": project_id})
        assert renewed.status_code == 200, renewed.text
        exchange(renewed.json())
        assert browser.get(f"{url}/open").json()["snapshot"]["revision"] == new_revision

        other = tmp_path / "other" / repo.name
        _write_train(other)
        assert select(other)["project"] != selected["project"]
        assert list(projects.iterdir()) == [projects / created_id]

        moved = repo.rename(tmp_path / "moved")
        repo.symlink_to(moved, target_is_directory=True)
        assert control.post(ticket_url, json={"project_folder": str(repo)}).status_code == 422
        assert browser.get(f"{url}/open").status_code in (404, 422)
        repo.unlink()
        _write_train(repo)
        assert select(repo)["project"] != selected["project"]
        assert browser.get(f"{url}/open").status_code == 404
        assert read_text_project_snapshot(moved).revision == new_revision


def test_external_pin_authority(tmp_path):
    app = create_app(projects_dir=tmp_path / "projects")
    folders = app.state.project_folders
    repo = tmp_path / "agent checkout"
    _write_train(repo, pinned_base=Path("pinned-train.json"))
    project_id = folders.open_selected_folder(repo).snapshot.project_id
    run, = folders.read_plan(project_id, revision=None).plan.runs
    assert run.request_id == _cli_request_id(repo)

    (repo / "pinned-train.json").rename(tmp_path / "pinned-train.json")
    (repo / "pinned-train.json").symlink_to(tmp_path / "pinned-train.json")
    refused = folders.read_plan(project_id, revision=None).plan
    assert not refused.runs
    assert any("outside the authorized directory" in problem.detail for problem in refused.diagnostics)


def test_replacement_after_grant_lookup_cannot_receive_a_write(tmp_path, monkeypatch):
    app = create_app(projects_dir=tmp_path / "projects")
    folders = app.state.project_folders
    repo = tmp_path / "selected"
    _write_train(repo)
    before = folders.open_selected_folder(repo).snapshot
    transport, control, browser = _desktop_clients(app)
    ticket = control.post("/_desktop/ticket", json={"project_id": before.project_id}).json()["ticket"]
    origin = {"Origin": transport.endpoint.origin}
    assert browser.post("/_desktop/exchange", headers={**origin, "X-Feedbax-Ticket": ticket}).status_code == 204
    moved = tmp_path / "moved"
    lookup = folders._folder

    def replace_after_lookup(project_id):
        grant = lookup(project_id)
        repo.rename(moved)
        _write_train(repo)
        return grant

    monkeypatch.setattr(folders, "_folder", replace_after_lookup)
    response = browser.post(
        f"/api/projects/{before.project_id}/rename",
        json={"revision": before.revision, "name": "changed"},
        headers=origin,
    )
    assert response.status_code == 404, response.text
    assert not (repo / STUDIO_WRITE_LOCK_FILENAME).exists()
    assert read_text_project_snapshot(repo).revision == read_text_project_snapshot(moved).revision == before.revision


def test_replacement_during_publish_cannot_receive_a_write(tmp_path, monkeypatch):
    folders = create_app(projects_dir=tmp_path / "projects").state.project_folders
    repo = tmp_path / "selected"
    moved = tmp_path / "moved"
    _write_train(repo)
    snapshot = folders.open_selected_folder(repo).snapshot
    write = project_folder.write_bytes_atomic

    def replace_at_publish(path, data, *, before_publish, **kwargs):
        def replace():
            before_publish()
            repo.rename(moved)
            _write_train(repo)

        write(path, data, before_publish=replace, **kwargs)

    monkeypatch.setattr(project_folder, "write_bytes_atomic", replace_at_publish)
    with pytest.raises(project_folder.ProjectFolderInvalid, match="identity changed"):
        folders.rename(snapshot.project_id, StudioProjectRename(name="changed", revision=snapshot.revision))
    assert read_text_project_snapshot(repo).revision == snapshot.revision
    assert read_text_project_snapshot(moved).configuration.project == "changed"


def test_external_replay_restores_exact_git_bytes(tmp_path):
    repo = tmp_path / "external checkout"
    _write_train(repo)
    app = create_app(projects_dir=tmp_path / "projects")
    transport, control, browser = _desktop_clients(app)
    response = control.post("/_desktop/ticket", json={"project_folder": str(repo)})
    assert response.status_code == 200, response.text
    selected = response.json()
    origin = {"Origin": transport.endpoint.origin}
    assert browser.post(
        "/_desktop/exchange", headers={**origin, "X-Feedbax-Ticket": selected["ticket"]},
    ).status_code == 204
    url = f"/api/projects/{selected['project']['id']}"
    before = browser.get(f"{url}/open").json()["snapshot"]
    step = repo / "steps" / "train.json"
    original = step.read_bytes()
    address = {"kind": "step", "id": "train"}
    outside = original.rstrip() + b"\n\n"
    step.write_bytes(outside)
    after = browser.get(f"{url}/open").json()["snapshot"]
    assert after["revision"] != before["revision"]
    for current, target in ((outside, original), (original, outside)):
        replay = browser.post(
            f"{url}/history/replay", headers=origin, json={
                "revision": after["revision"],
                "files": [{
                    "address": address, "before": sha256_bytes(current),
                    "after": b64encode(target).decode("ascii"),
                }],
            },
        )
        assert replay.status_code == 200, replay.text
        assert step.read_bytes() == target
        after = replay.json()


def test_restore_revalidates_both_windows_after_a_service_restart(tmp_path):
    projects = tmp_path / "managed"
    repo = tmp_path / "outside checkout"
    _write_train(repo)
    settings = tmp_path / "studio.json"

    def desktop():
        app = create_app(projects_dir=projects)
        register_preferences(app, settings)
        transport, control, browser = _desktop_clients(app)
        return control, browser, {"Origin": transport.endpoint.origin}

    def ticket(control, **body):
        result = control.post("/_desktop/ticket", json=body)
        assert result.status_code == 200, result.text
        return result.json()

    def exchange(browser, origin, issued):
        result = browser.post("/_desktop/exchange", headers={**origin, "X-Feedbax-Ticket": issued["ticket"]})
        assert result.status_code == 204, result.text
        assert result.headers["x-feedbax-project"] == issued["project"]["id"]

    first_control, first_browser, first_origin = desktop()
    opened = [ticket(first_control), ticket(first_control, project_folder=str(repo))]
    arrangement = {
        "mode": "details", "primaryView": "model", "secondaryView": "scene",
        "mapDrawerOpen": False, "detailsDrawerOpen": True, "inspectorOpen": False,
        "mapHeight": 390, "mapDrawerWidth": 264, "detailsDrawerWidth": 220, "inspectorWidth": 244,
        "secondaryWidth": 440, "modelPaths": {"primary": {"document": "point-mass", "path": ["nested"]}},
    }
    expected = {}
    for index, issued in enumerate(opened):
        exchange(first_browser, first_origin, issued)
        project_id = issued["project"]["id"]
        route = f"/api/desktop/preferences/windows/{project_id}"
        expected[project_id] = {**arrangement, "mapHeight": 390 + index * 100}
        saved = first_browser.put(route, json=expected[project_id], headers=first_origin)
        assert saved.status_code == 204, saved.text
        refused = first_browser.put(
            route, json={**expected[project_id], "project_folder": str(repo)}, headers=first_origin,
        )
        assert refused.status_code == 422, refused.text

    control, browser, origin = desktop()
    external_id = opened[1]["project"]["id"]
    assert control.post("/_desktop/ticket", json={"project_id": external_id}).status_code == 404
    for original in opened:
        folder = original["folder"]
        request = {"project_folder": folder["path"], "folder_identity": folder["identity"]}
        restored = ticket(control, **request)
        assert restored["project"] == original["project"]
        assert restored["folder"] == folder
        exchange(browser, origin, restored)
        project_id = restored["project"]["id"]
        route = f"/api/desktop/preferences/windows/{project_id}"
        assert browser.get(route).json() == expected[project_id]
        assert browser.post("/_desktop/ticket", json=request, headers=origin).status_code == 403

    repo.rename(tmp_path / "moved")
    _write_train(repo)
    control, _, _ = desktop()
    refused = control.post("/_desktop/ticket", json={
        "project_folder": str(repo), "folder_identity": opened[1]["folder"]["identity"],
    })
    assert refused.status_code == 404, refused.text
    assert not (repo / STUDIO_WRITE_LOCK_FILENAME).exists()
    assert control.post("/_desktop/ticket", json={"project_id": external_id}).status_code == 404

def test_recent_ticket_returns_the_latest_acknowledged_title(tmp_path):
    projects = tmp_path / "projects"
    transport, control, browser = _desktop_clients(create_app(projects_dir=projects))
    origin = {"Origin": transport.endpoint.origin}
    issued = control.post("/_desktop/ticket", json={}).json()
    project_id = issued["project"]["id"]
    assert list(projects.iterdir()) == [projects / project_id]
    assert browser.post(
        "/_desktop/exchange", headers={**origin, "X-Feedbax-Ticket": issued["ticket"]},
    ).status_code == 204
    url = f"/api/projects/{project_id}"
    revision = browser.get(f"{url}/open").json()["snapshot"]["revision"]
    renamed = browser.post(f"{url}/rename", json={"revision": revision, "name": "renamed"}, headers=origin)
    assert renamed.status_code == 200, renamed.text
    location = issued["folder"]
    request = {"project_folder": location["path"], "folder_identity": location["identity"]}
    assert browser.post("/_desktop/ticket", json=request, headers=origin).status_code == 403
    reopened = control.post("/_desktop/ticket", json=request)
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["project"] == {"id": project_id, "title": "renamed"}
    assert reopened.json()["folder"] == location
    assert list(projects.iterdir()) == [projects / project_id]
