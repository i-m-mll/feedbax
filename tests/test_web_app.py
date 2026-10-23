from fastapi.testclient import TestClient

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
