"""The real folder boundary serves compiler-derived geometry for one revision."""

from fastapi.testclient import TestClient
import pytest

from feedbax.web.app import create_app
from tests.fixtures.scene_geometry import native_scene_samples, seed_scene_project


_PROJECT = "/api/projects/scene"
_SCENE = f"{_PROJECT}/scene/planar"
_PARAMS = "/payload/subgraphs/mechanics/nodes"


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

    response = client.patch(
        _PROJECT,
        json={
            "revision": source["revision"],
            "address": {"kind": "document", "id": "chain"},
            "patches": [
                {"op": "replace", "path": f"{_PARAMS}/forearm/params/length", "value": 0.37},
                {
                    "op": "replace",
                    "path": f"{_PARAMS}/cross_chain/params/path_points/1/offset",
                    "value": [0.05, -0.07],
                },
            ],
        },
    )
    assert response.is_success, response.text
    assert client.post(_SCENE, json=source).status_code == 409
    updated = _read(client, _source(client, "chain"))["model"]
    assert updated["model_identity"] != model["model_identity"]
    assert updated["links"][1]["length"] == 0.37
    assert updated["muscles"][0]["points"][1]["frame"] == "forearm.distal"
    assert updated["muscles"][0]["points"][1]["position"] == pytest.approx([0.05, -0.07])


def test_scene_reports_missing_frames_and_omits_only_unsupported_wraps(project):
    _, client = project
    wrapped = _read(client, _source(client, "wrapped"))
    assert len(wrapped["model"]["links"]) == 3
    assert wrapped["model"]["muscles"] == [{"node_id": "cross_chain", "points": None}]
    assert [(entry["code"], entry["node_ids"]) for entry in wrapped["diagnostics"]] == [
        ("scene.unsupported_wrap", ["cross_chain"]),
    ]
    source = _source(client, "chain")
    wrong_node = _read(client, {**source, "node_path": ["unknown", "mechanics"]})
    assert wrong_node["model"] is None
    assert "unknown" in wrong_node["diagnostics"][0]["message"]
    response = client.patch(
        _PROJECT,
        json={
            "revision": source["revision"],
            "address": {"kind": "document", "id": "chain"},
            "patches": [
                {
                    "op": "replace",
                    "path": f"{_PARAMS}/cross_chain/params/path_points/1/frame",
                    "value": "missing.distal",
                }
            ],
        },
    )
    assert response.is_success, response.text
    invalid = _read(client, _source(client, "chain"))
    assert invalid["model"] is None
    assert [(entry["code"], entry["node_ids"]) for entry in invalid["diagnostics"]] == [
        ("mechanics.muscle_path_missing_frame", ["cross_chain"]),
    ]
    assert "missing.distal" in invalid["diagnostics"][0]["message"]
