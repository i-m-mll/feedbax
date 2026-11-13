import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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


def test_studio_rejects_retired_training_and_inert_controller_operations() -> None:
    client = TestClient(create_app())

    for method, path in (
        ("POST", "/api/training"),
        ("POST", "/api/training/worker/connect"),
        ("DELETE", "/api/training/old-job"),
        ("GET", "/api/training/old-job/checkpoint/download"),
        ("POST", "/api/execution/simulate"),
        ("POST", "/api/orchestration/launch"),
        ("GET", "/api/orchestration/targets"),
    ):
        assert client.request(method, path).status_code == 404
    for path in ("/ws/training/old-job", "/ws/simulation"):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(path):
                pytest.fail(f"Retired WebSocket accepted {path}")
