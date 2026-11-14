"""The real folder boundary serves compiler-derived geometry for one revision."""

from fastapi.testclient import TestClient
import pytest

from feedbax.web.app import create_app
from tests.fixtures.scene_geometry import native_scene_samples, seed_scene_project


_PROJECT = "/api/projects/scene"
_SCENE = f"{_PROJECT}/scene/planar"


@pytest.fixture
def project(tmp_path):
    pytest.importorskip("mujoco")
    root = tmp_path / "scene"
    seed_scene_project(root)
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        yield root, client


def _source(client, document):
    snapshot = client.get(_PROJECT)
    assert snapshot.is_success, snapshot.text
    return {
        "revision": snapshot.json()["revision"],
        "document_id": document,
        "node_path": ["mechanics"],
    }


def _read(client, source):
    response = client.post(_SCENE, json=source)
    assert response.is_success, response.text
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()


def _edit(client, document, component, path, value):
    snapshot = client.get(_PROJECT).json()
    response = client.post(f"{_PROJECT}/model/edit", json={
        "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
        "document": document,
        "command": {"operation": {
            "op": "set_field", "value": value,
            "address": {"document": document, "composites": ["mechanics"], "element": {
                "kind": "field", "parent": {"kind": "component", "id": component}, "path": ["fields", *path],
            }},
        }},
    })
    assert response.is_success, response.text
    assert response.json()["result"]["status"] == "applied", response.text


def test_scene_binds_the_exact_revision_and_preserves_authored_coordinates(project):
    root, client = project
    source = _source(client, "chain")
    model = _read(client, source)["model"]
    assert model["model_identity"] == native_scene_samples(root, "chain")["identity"]
    assert [(link["joint_id"], link["qpos_index"]) for link in model["links"]] == [
        ("shoulder", 0),
        ("elbow", 1),
        ("wrist", 2),
    ]
    assert model["reference_qpos"] == [0, 0, 0]
    assert model["links"][1]["child_frame"] == "forearm.distal"
    assert model["muscles"][0]["points"][1]["frame"] == "forearm.distal"
    assert model["muscles"][0]["points"][1]["position"] == pytest.approx([0.03, -0.09])

    _edit(client, "chain", "forearm", ["length"], 0.37)
    _edit(client, "chain", "cross_chain", ["path_points", 1, "offset"], [0.05, -0.07])
    assert client.post(_SCENE, json=source).status_code == 409
    updated = _read(client, _source(client, "chain"))["model"]
    assert updated["model_identity"] != model["model_identity"]
    assert updated["links"][1]["length"] == 0.37
    assert updated["muscles"][0]["points"][1]["frame"] == "forearm.distal"
    assert updated["muscles"][0]["points"][1]["position"] == pytest.approx([0.05, -0.07])


def test_scene_reports_missing_frames_and_unresolved_occurrences(project):
    _, client = project
    source = _source(client, "chain")
    wrong_node = _read(client, {**source, "node_path": ["unknown", "mechanics"]})
    assert wrong_node["model"] is None
    assert "unknown" in wrong_node["diagnostics"][0]["message"]
    _edit(client, "chain", "cross_chain", ["path_points", 1, "frame"], "missing.distal")
    invalid = _read(client, _source(client, "chain"))
    assert invalid["model"] is None
    assert [(entry["code"], entry["node_ids"]) for entry in invalid["diagnostics"]] == [
        ("mechanics.muscle_path_missing_frame", ["cross_chain"]),
    ]
    assert "missing.distal" in invalid["diagnostics"][0]["message"]
