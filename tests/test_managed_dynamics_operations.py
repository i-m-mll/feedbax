"""Read PCA and equilibrium products of a trained saved example in Studio."""

from base64 import b64decode

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA
from feedbax.contracts.project import Project
from feedbax.contracts.semantic_values import semantic_key
from feedbax.examples import load_shipped_projects
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.host import job_row, token, wait_until
from tests.test_canonical_evaluate_admission import _save_project


def _dynamics_project():
    project = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task_name = "simple-reaches-task"
    task = project["documents"][task_name]["payload"]
    task["fixed_steps"] = 8
    task["epochs"]["move"]["end"]["length"] = "8 st"
    task["validation_trials"] = dict(list(task["validation_trials"].items())[:2])
    trained = {"step": "train", "checkpoint": "last"}
    task_binding = {"kind": "task", "source": {"document": task_name}}
    project["steps"] = {
        "train": project["steps"]["simple-reaches"],
        "rollout": {
            "label": "Rollout",
            "kind": "evaluate",
            "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
                "seed": 7, "trial_policy": "paired", "trial_set": {"kind": "every_condition"},
            }}},
            "inputs": {"subject": trained},
        },
        "analyze": {
            "label": "Analyze",
            "kind": "analyze",
            "base": {"inline": {
                "inputs": {"states": {"kind": "dataset"}, "model": {"kind": "model"}, "task": {"kind": "task"}},
                "operations": {
                    "pca": {
                        "label": "Pca",
                        "operation": "feedbax.hidden_state_pca",
                        "inputs": {"states": {"input": "states"}},
                        "params": {"state": semantic_key("model", "state", ("cell",), "hidden")},
                    },
                    "roots": {
                        "label": "Roots",
                        "operation": "feedbax.fixed_points",
                        "inputs": {"model": {"input": "model"}, "task": {"input": "task"}},
                        "params": {"loss_tolerance": 1e-6, "max_updates": 1600},
                    },
                },
                "views": {
                    "pc-plane": {
                        "label": "Pc plane",
                        "inputs": {"trajectory": {"element": "pca", "product": "pca"}},
                        "figure": {
                            "data": ["trajectory"],
                            "layout": {"pc": {"row": 1, "col": 1}},
                            "layers": {"pc-plane": {
                                "role": "trajectory", "panel": "pc", "mark": "line",
                                "group": ["trial"],
                                "select": {"checkpoint": {"index": 0}, "condition": {"index": 0}},
                                "encode": {
                                    "x": {"variable": "scores", "select": {"component": 1}},
                                    "y": {"variable": "scores", "select": {"component": 2}},
                                },
                            }},
                        },
                    },
                },
            }},
            "inputs": {"states": {"step": "rollout"}, "model": trained, "task": task_binding},
        },
    }
    train_config = project["steps"]["train"]["base"]["inline"]["training_config"]
    train_config.update(batch_size=1, replicates=1)
    analysis = project["steps"]["analyze"]["base"]["inline"]
    project["documents"]["dynamics-analysis"] = {"label": "Dynamics analysis", "schema": ANALYSIS_SCHEMA, "payload": analysis}
    project["steps"]["analyze"]["base"] = {"document": "dynamics-analysis"}
    return project


@pytest.mark.parametrize(("axis", "checkpoint"), [("checkpoint", "last"), ("update", {"every": 1})])
def test_pca_authoring_offers_only_the_bound_rollout_clock(host, tmp_path, monkeypatch, axis, checkpoint):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    project = _dynamics_project()
    project["steps"]["rollout"]["inputs"]["subject"]["checkpoint"] = checkpoint
    layer = project["documents"]["dynamics-analysis"]["payload"]["views"]["pc-plane"]["figure"]["layers"]["pc-plane"]
    layer["select"] = {axis: {"index": 0}, "condition": {"index": 0}}
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "dynamics", Project.model_validate(project))
    with TestClient(create_app(projects_dir=folders.root, automatic_run="off")) as client:
        key = {"variant": "default", "step": "analyze", "row": "default"}
        response = client.post("/api/projects/dynamics/authoring/analyze", json={
            "revision": folders.read("dynamics").revision, "key": key,
        })
        assert response.status_code == 200, response.text
        catalog = response.json()
    pca = catalog["operation_outputs"]["pca"]["pca"]
    assert pca["variables"]["scores"]["dims"] == [axis, "condition", "trial", "state_time", "component"]
    assert {"update", "checkpoint"} & pca["coordinates"].keys() == {axis}


def test_analyze_saved_activity_and_fixed_points_of_trained_example(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    project = _dynamics_project()
    _save_project(ProjectFolderService(tmp_path), "dynamics", Project.model_validate(project))
    url = "/api/projects/dynamics"

    def expect(response, code=200):
        assert response.status_code == code, response.text
        return response.json()

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        revision = expect(client.get(url))["revision"]

        def window():
            return expect(client.get(f"{url}/train-runs", params={"revision": revision, "window_map": True}))["window_map"]

        def run(*steps):
            submission = expect(client.post(f"{url}/train-runs", json={
                "revision": revision, "token": token("trained-dynamics-" + "-".join(steps)),
                "run_keys": [{"variant": "default", "step": step, "row": "default"} for step in steps],
            }), 202)
            for item in submission["items"]:
                assert item["disposition"] == "admitted", item
                job = wait_until(lambda: (
                    observed if (observed := job_row(host, item["job_id"])) and observed["outcome"] is not None else None
                ))
                assert job["state"] == "succeeded", job["outcome"]

        run("train")
        assert not window()["problems"], window()["problems"]
        run("rollout", "analyze")
        products = {}
        for element in window()["analysis_elements"]:
            (request,) = element["requests"]
            products.update(request["result"]["products"])
        pca = expect(client.get(products["pca"]))
        roots = expect(client.get(products["roots"]))

        def values(product, field):
            return expect(client.get(f"{products[product]}/values", params={"field": field}))["values"]

        basis = np.asarray(values("pca", "basis")).reshape(3, 4)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=5e-6)
        scores = pca["variables"]["scores"]["dims"]
        assert scores[-4:] == ["condition", "trial", "state_time", "component"]
        assert pca["dimensions"]["trial"] == 2
        assert values("pca", "component") == [1, 2, 3]
        assert len(values("pca", "state_time")) == 9
        assert roots["variables"]["fixed_points"]["dims"][-3:] == ["condition", "candidate", "state_component"]
        accepted = np.asarray(values("roots", "accepted"), dtype=bool).reshape(2, 2)
        residuals = np.asarray(values("roots", "residual")).reshape(2, 2)
        assert np.all(accepted.any(axis=1))
        assert np.all(residuals[accepted] < 1e-6)
        trajectories = np.asarray(values("pca", "scores")).reshape(2, 9, 3)
        traces = expect(client.get(products["figure"]))["data"]
        assert len(traces) == len(trajectories)
        for trace, points in zip(traces, trajectories, strict=True):
            for axis, projected in zip(("x", "y"), points[:, :2].T, strict=True):
                encoded = trace[axis]
                np.testing.assert_allclose(np.frombuffer(b64decode(encoded["bdata"]), dtype=encoded["dtype"]), projected)
