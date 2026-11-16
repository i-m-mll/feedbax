"""Exercise native preference rebinding through the private project ticket."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from feedbax.desktop.control import Endpoint, _read_private_json, write_private_json
from feedbax.desktop.preferences import register_preferences
from feedbax.desktop.transport import DesktopTransport
from feedbax.project_folder import ProjectFolderLocation


@pytest.mark.parametrize(
    ("history", "has_canonical", "migrates"),
    [("restore", False, True), ("recent", False, True), ("inconsistent", False, False), ("restore", True, True)],
)
def test_native_ticket_rebinds_arrangements_only_for_the_same_folder(tmp_path, history, has_canonical, migrates):
    path = tmp_path / "studio.json"
    managed = tmp_path / "projects"
    folder = managed / "legacy"
    folder.mkdir(parents=True)
    location = ProjectFolderLocation.selected(folder)
    canonical = "opened-project"
    previous, current = {"mapHeight": 310}, {"mapHeight": 430}
    preferences = {"legacy": previous}
    if has_canonical:
        preferences[canonical] = current
    write_private_json(path, {"windows": preferences})

    binding = {"path": str(location.path), "identity": list(location.identity)}
    if history != "recent":
        write_private_json(tmp_path / "studio-windows.json", {"legacy": {"folder": binding}})
    if history != "restore":
        recent = binding if history == "recent" else {**binding, "identity": [0, 0]}
        write_private_json(tmp_path / "studio-recents.json", [{"id": "legacy", "folder": recent}])

    def open_folder(selected, expected_identity):
        assert selected == folder and expected_identity is None
        return SimpleNamespace(
            snapshot=SimpleNamespace(project_id=canonical),
            configuration=SimpleNamespace(project="Reaching"),
        ), location

    app = FastAPI()
    app.state.project_folders = SimpleNamespace(root=managed, open_native_folder=open_folder)
    transport = DesktopTransport(app, Endpoint.create(41234))
    transport.register_ticket_route(lambda: False, rebind_arrangement=register_preferences(app, path))

    async def unix_socket(scope, receive, send):
        await transport({**scope, "server": None}, receive, send)

    with TestClient(unix_socket, base_url="http://localhost") as client:
        opened = client.post("/_desktop/ticket", json={"project_folder": str(folder)})

    assert opened.status_code == 200, opened.text
    assert opened.json()["project"]["id"] == canonical
    expected = {canonical: current if has_canonical else previous} if migrates else preferences
    assert _read_private_json(path) == {"windows": expected}
