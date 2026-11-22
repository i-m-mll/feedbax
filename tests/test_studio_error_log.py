import logging

from fastapi.testclient import TestClient

from feedbax.web.app import create_app


def test_an_error_studio_shows_is_logged_whole_under_its_ref(tmp_path, caplog):
    """Catches an error Studio shows that the service's log cannot find by the ref on its element."""
    report = {
        "ref": "0a1b2c3d", "kind": "request", "message": "StudioProjectPatch: patches.0 Invalid input",
        "path": "/api/projects/reaches", "method": "PATCH", "contract": "StudioProjectPatch",
        "issues": ["patches.0.value.where Required"], "sent": {"revision": "a" * 64}, "stack": "ApiRequestError: …",
    }
    client = TestClient(create_app(projects_dir=tmp_path / "projects"))
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        assert client.post("/api/studio/errors", json=report).status_code == 204
    [record] = [record.getMessage() for record in caplog.records if "0a1b2c3d" in record.getMessage()]
    assert "patches.0.value.where Required" in record and "a" * 64 in record and "StudioProjectPatch" in record
