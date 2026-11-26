"""The stored project budget applies to authored arrays in typed and open payloads."""

import json

from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.training_document import TRAINING_DOCUMENT_SCHEMA_ID
from feedbax.web.app import create_app


_URL = "/api/projects/study"


def _matrix(size, diagonal=1.0):
    return [[diagonal if row == column else 0.0 for column in range(size)] for row in range(size)]


def _budget_problem(detail, file, path):
    assert file in detail
    assert path in detail
    assert "289 scalars" in detail and "256" in detail
    assert "rule" in detail and "content reference" in detail


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    (root / "documents").mkdir(parents=True)
    (root / "study.feedbax").write_text(json.dumps({"schema_version": "feedbax.project.v1", "store": "results"}))
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        yield root, client


def test_model_edit_accepts_256_scalars_and_refuses_289_at_the_matrix_field(study):
    root, client = study
    model = root / "documents/model.json"
    model.write_text(json.dumps({
        "schema": STORED_MODEL_DOCUMENT_SCHEMA_ID,
        "label": "Model",
        "payload": {
            "step_seconds": "10 ms",
            "ports": {},
            "interior": {"components": {"plant": {
                "kind": "leaf", "label": "Plant", "type": "feedbax.LinearStateSpace",
                "fields": {"A": _matrix(16), "B": [[1.0, 0.0]] * 16},
            }}},
        },
    }))
    address = ModelAddress.at_stored_path(("interior", "components", "plant", "fields", "A")).model_dump(mode="json")

    def edit(snapshot, value):
        response = client.post(f"{_URL}/model/edit", json={
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": "model", "command": {"operation": {"op": "set_field", "address": address, "value": value}},
        })
        assert response.status_code == 200, response.text
        return response.json()["result"]

    opened = client.get(_URL)
    assert opened.status_code == 200, opened.text
    accepted = edit(opened.json(), _matrix(16, 0.5))
    assert accepted["status"] == "applied"
    saved = model.read_bytes()
    assert json.loads(saved)["payload"]["interior"]["components"]["plant"]["fields"]["A"] == _matrix(16, 0.5)

    refused = edit(accepted["snapshot"], _matrix(17))
    assert refused["status"] == "refused"
    assert refused["problem"]["address"] == address
    _budget_problem(refused["problem"]["detail"], "documents/model.json", "/payload/interior/components/plant/fields/A")
    assert model.read_bytes() == saved


def test_untyped_document_settings_use_the_same_budget_on_patch_and_read(study):
    root, client = study
    file = "documents/training.json"
    training = root / file
    raw = {
        "schema": TRAINING_DOCUMENT_SCHEMA_ID,
        "label": "Calibrated training",
        "payload": {
            "method": "test.calibrated_training",
            "settings": {"calibration": {"matrix": [[1.0]], "target": [0.0] * 256}},
            "training_config": {},
        },
    }
    training.write_text(json.dumps(raw))
    path = "/payload/settings/calibration/matrix"

    def patch(snapshot, matrix):
        return client.patch(_URL, json={
            "revision": snapshot["revision"], "address": {"kind": "document", "id": "training"},
            "patches": [{"op": "replace", "path": path, "value": matrix}],
        })

    opened = client.get(_URL)
    assert opened.status_code == 200, opened.text
    accepted = patch(opened.json(), _matrix(16))
    assert accepted.status_code == 200, accepted.text
    saved = training.read_bytes()
    assert json.loads(saved)["payload"]["settings"]["calibration"]["matrix"] == _matrix(16)

    refused = patch(accepted.json(), _matrix(17))
    assert refused.status_code == 422, refused.text
    _budget_problem(refused.json()["detail"], file, path)
    assert training.read_bytes() == saved

    raw["payload"]["settings"]["calibration"]["matrix"] = _matrix(17)
    training.write_text(json.dumps(raw))
    malformed = client.get(_URL)
    assert malformed.status_code == 422, malformed.text
    _budget_problem(malformed.json()["detail"], file, path)
