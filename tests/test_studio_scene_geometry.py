"""The project route serves Model-addressed geometry for one revision."""

from fastapi.testclient import TestClient
import jax
import jax.numpy as jnp
import pytest

from feedbax.builtins.mechanics.direct_force_components import TwoLinkArm
from feedbax.contracts.model_address import ModelAddress
from feedbax.runtime.graph import init_state_from_component
from feedbax.web.app import create_app
from tests.fixtures.scene_geometry import native_scene_samples, scene_files, seed_scene_project


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
            "address": {"composites": ["mechanics"], "element": {
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


def test_project_geometry_matches_the_recorded_arm_tip_and_poses_every_mechanism(tmp_path):
    pytest.importorskip("mujoco")
    files = scene_files()
    components = files[0]["document"]["payload"]["interior"]["components"]
    components["mechanics"]["interior"]["components"]["upper"]["name"] = "Upper arm"
    components["plants"] = {
        "kind": "composite", "name": "Plants", "ports": {},
        "interior": {
            "components": {
                "arm": {
                    "kind": "leaf", "name": "Arm", "type": "feedbax.TwoLinkArm",
                    "fields": {"link_lengths": [0.4, 0.5]},
                },
                "mass": {"kind": "leaf", "name": "Mass", "type": "feedbax.PointMass"},
            },
        },
    }
    seed_scene_project(tmp_path / "scene", files)

    arm = TwoLinkArm(link_lengths=[0.4, 0.5])
    state = init_state_from_component(arm)
    state = arm.replace_state_value(state, ("effector", "pos"), jnp.array([0.3, 0.2]))
    outputs, state = arm({"force": jnp.array([1.0, -1.0])}, state, key=jax.random.PRNGKey(0))
    recorded = arm.state_view(state)
    angles = recorded.plant.skeleton.angle.tolist()
    effector = outputs["effector"].pos.tolist()
    arm_angles = ModelAddress.at_state(("plants",), "arm", "skeleton/angle").to_text()
    arm_position = ModelAddress.at_state(("plants",), "arm", "effector/position").to_text()
    mass_position = ModelAddress.at_state(("plants",), "mass", "effector/position").to_text()
    mass_skeleton = ModelAddress.at_state(("plants",), "mass", "skeleton/position").to_text()

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        source = {"revision": client.get(_PROJECT).json()["revision"], "document_id": "single"}
        route = f"{_PROJECT}/scene/geometry"
        response = client.post(route, json={
            **source, "state_values": {arm_angles: angles, mass_position: [0.2, -0.1]},
        })
        assert response.is_success, response.text
        assert response.headers["Cache-Control"] == "no-store"
        result = response.json()
        assert result["diagnostics"] == []
        geometries = {item["address"]["element"]["id"]: item for item in result["mechanisms"]}
        assert geometries.keys() == {"mechanics", "arm", "mass"}
        assert geometries["arm"]["links"][-1]["end"] == pytest.approx(effector)
        assert geometries["arm"]["links"][-1]["joint"]["element"]["path"] == ["geometry", "joint", 1]
        assert geometries["mass"]["name"] == "Mass"
        assert geometries["mass"]["position"] == pytest.approx([0.2, -0.1])
        assert geometries["mass"]["point"]["element"]["path"] == ["geometry", "point"]
        assert geometries["mechanics"]["links"][0]["body"]["composites"] == ["mechanics"]
        assert geometries["mechanics"]["links"][0]["body_name"] == "Upper arm"

        preview = client.post(route, json={
            **source, "state_values": {arm_position: [0.3, 0.2], mass_skeleton: 0.25},
        })
        assert preview.is_success, preview.text
        sampled = {item["address"]["element"]["id"]: item for item in preview.json()["mechanisms"]}
        assert sampled["arm"]["links"][-1]["end"] == pytest.approx([0.3, 0.2])
        assert sampled["mass"]["position"] == pytest.approx([0.25, 0.25])
