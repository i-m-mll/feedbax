"""The project route serves Model-addressed geometry for one revision."""

from fastapi.testclient import TestClient
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.builtins.mechanics.direct_force_components import TwoLinkArm
from feedbax.compiler.model_document import lower_planar_scene_interior
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.contracts.project_io import load_text_project
from feedbax.mechanics.authored_model import compile_planar_model
from feedbax.runtime.graph import init_state_from_component
from feedbax.studio import scene_geometry
from feedbax.web.app import create_app
from tests.fixtures.scene_geometry import scene_files, seed_scene_project


_PROJECT = "/api/projects/scene"
_SCENE = f"{_PROJECT}/scene/geometry"


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
        "poses": [{}],
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


def test_scene_batches_native_poses_at_one_revision(project, monkeypatch):
    import mujoco

    root, client = project
    saved = load_text_project(root)
    model = StoredModelDocument.model_validate(saved.project.documents["chain"].payload).lower()
    topology = lower_planar_scene_interior(model, ("mechanics",), ComponentRegistry())
    compiled = compile_planar_model(topology)
    data = mujoco.MjData(compiled.model)
    effector = compiled.site_ids[compiled.effector_site]
    joint_coordinates = ([0.43, -0.67, 0.26], compiled.model.qpos0, [-0.31, 0.58, -0.48])
    tips = []
    for qpos in joint_coordinates:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(compiled.model, data)
        tips.append(data.site_xpos[effector, :2].tolist())

    compile_count = 0

    def count_compiles(graph):
        nonlocal compile_count
        compile_count += 1
        return compile_planar_model(graph)

    monkeypatch.setattr(scene_geometry, "compile_planar_model", count_compiles)
    source = _source(client, "chain")
    configuration = ModelAddress.at_state((), "mechanics", "skeleton/qpos").to_text()
    request = {**source, "poses": [
        {configuration: joint_coordinates[0]}, {configuration: [0.1]},
        {}, {configuration: joint_coordinates[2]},
    ]}
    frames = _read(client, request)["frames"]
    assert compile_count == 1
    assert len(frames) == 4
    assert frames[1]["mechanisms"] == []
    assert "3 finite joint coordinates" in frames[1]["diagnostics"][0]["message"]
    for index, expected in zip((0, 2, 3), tips, strict=True):
        frame = frames[index]
        assert frame["diagnostics"] == []
        mechanism, = frame["mechanisms"]
        assert mechanism["links"][-1]["end"] == pytest.approx(expected)
        assert [link["joint"]["element"]["id"] for link in mechanism["links"]] == [
            "shoulder", "elbow", "wrist",
        ]

    _edit(client, "chain", "forearm", ["length"], 0.37)
    _edit(client, "chain", "cross_chain", ["path_points", 1, "offset"], [0.05, -0.07])
    assert client.post(_SCENE, json=request).status_code == 409
    updated, = _read(client, _source(client, "chain"))["frames"]
    mechanism, = updated["mechanisms"]
    link = mechanism["links"][1]
    assert np.linalg.norm(np.subtract(link["end"], link["start"])) == pytest.approx(0.37)
    muscle, = mechanism["muscles"]
    assert np.linalg.norm(np.subtract(muscle["points"][1], link["end"])) == pytest.approx(
        np.hypot(0.05, -0.07),
    )


def test_scene_reports_missing_frames(project):
    _, client = project
    _edit(client, "chain", "cross_chain", ["path_points", 1, "frame"], "missing.distal")
    invalid, = _read(client, _source(client, "chain"))["frames"]
    assert invalid["mechanisms"] == []
    assert [(entry["code"], entry["node_ids"]) for entry in invalid["diagnostics"]] == [
        ("mechanics.muscle_path_missing_frame", ["cross_chain"]),
    ]
    assert "missing.distal" in invalid["diagnostics"][0]["message"]


def test_project_geometry_matches_the_recorded_arm_tip_and_poses_every_mechanism(tmp_path):
    pytest.importorskip("mujoco")
    files = scene_files()
    components = files[0]["document"]["payload"]["interior"]["components"]
    components["mechanics"]["interior"]["components"]["upper"]["label"] = "Upper arm"
    components["plants"] = {
        "kind": "composite", "label": "Plants", "ports": {},
        "interior": {
            "components": {
                "arm": {
                    "kind": "leaf", "label": "Arm", "type": "feedbax.TwoLinkArm",
                    "fields": {"link_lengths": [0.4, 0.5]},
                },
                "mass": {"kind": "leaf", "label": "Mass", "type": "feedbax.PointMass"},
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
            **source, "poses": [{arm_angles: angles, mass_position: [0.2, -0.1]}],
        })
        assert response.is_success, response.text
        assert response.headers["Cache-Control"] == "no-store"
        result = response.json()["frames"][0]
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
            **source, "poses": [{arm_position: [0.3, 0.2], mass_skeleton: 0.25}],
        })
        assert preview.is_success, preview.text
        sampled = {item["address"]["element"]["id"]: item for item in preview.json()["frames"][0]["mechanisms"]}
        assert sampled["arm"]["links"][-1]["end"] == pytest.approx([0.3, 0.2])
        assert sampled["mass"]["position"] == pytest.approx([0.25, 0.25])
