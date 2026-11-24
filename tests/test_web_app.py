import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from feedbax.host import HostClient
from feedbax.web.app import create_app


def test_studio_vite_origin_is_allowed_for_api_requests() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/api/provider/manifest",
        headers={
            "Origin": "http://localhost:3008",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3008"


def test_studio_rejects_retired_training_and_inert_controller_operations(tmp_path) -> None:
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        for method, path in (
            ("POST", "/api/training"),
            ("POST", "/api/training/worker/connect"),
            ("DELETE", "/api/training/old-job"),
            ("GET", "/api/training/old-job/checkpoint/download"),
            ("POST", "/api/execution/simulate"),
            ("POST", "/api/orchestration/launch"),
        ):
            assert client.request(method, path).status_code == 404
        for path in ("/ws/training/old-job", "/ws/simulation"):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(path):
                    pytest.fail(f"Retired WebSocket accepted {path}")

        targets = client.get("/api/orchestration/targets")
        assert targets.status_code == 200
        assert targets.json()["default"] == "local"


def test_target_catalog_reads_inventory_without_starting_the_local_host(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    connected = False

    def exchange(_host, message, *, timeout, start_host):
        assert not start_host
        request = json.loads(message)
        assert request["op"] == "ping"
        assert request["include_runtime"] is True
        if not connected:
            raise FileNotFoundError()
        return {"value": {"runtime": {"backend": "cpu", "devices": ["Test CPU"]}}}

    monkeypatch.setattr(HostClient, "_exchange", exchange)
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
        absent = client.get("/api/orchestration/targets")
        assert absent.status_code == 200, absent.text
        assert absent.json()["targets"][0]["profile"] == {"id": "local", "kind": "local", "label": "This Mac"}
        connected = True
        present = client.get("/api/orchestration/targets")
        assert present.status_code == 200, present.text
        assert present.json()["targets"][0]["profile"] == absent.json()["targets"][0]["profile"] | {
            "runtime": {"backend": "cpu", "devices": ["Test CPU"]},
        }
