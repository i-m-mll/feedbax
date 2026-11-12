"""Exercise native project tickets through the private and browser transports."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from feedbax.contracts.studio_project import StudioProjectRename
from feedbax.desktop import transport as desktop_transport
from feedbax.desktop.control import Endpoint
from feedbax.web.app import create_app


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

    endpoint = Endpoint.create(41234)
    transport = desktop_transport.DesktopTransport(app, endpoint)
    transport.register_ticket_route(lambda: False)

    async def unix_socket(scope, receive, send):
        await transport({**scope, "server": None}, receive, send)

    async def loopback_socket(scope, receive, send):
        # TestClient uses a list, while Uvicorn supplies the bound socket address.
        await transport({**scope, "server": tuple(scope["server"])}, receive, send)

    control = TestClient(unix_socket, base_url="http://localhost")
    browser = TestClient(loopback_socket, base_url=endpoint.origin)
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
