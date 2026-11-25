"""A native model remains executable after its exporting project is gone."""

from copy import deepcopy
from shutil import rmtree

import numpy as np
from fastapi.testclient import TestClient

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.host import token, wait_until
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_saved_light_run import _saved_light_definition


def test_loaded_model_evaluates_and_initializes_train_in_a_new_project(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    folders = ProjectFolderService(projects)
    raw = _saved_light_definition()
    train = raw["steps"]["train"]
    train_payload = train["base"]["inline"]
    baseline = raw["steps"]["baseline"]
    source = Project.model_validate({"documents": raw["documents"], "steps": {"train": train, "baseline": baseline}})
    _save_project(folders, "exporter", source)

    def key(step):
        return {"variant": "default", "step": step, "row": "default"}

    def expect(response, code=200):
        assert response.status_code == code, response.text
        return response.json()

    def run(client, path, revision, *names):
        started = expect(client.post(f"{path}/train-runs", json={
            "revision": revision, "token": token(path.rsplit("/", 1)[-1]), "run_keys": [key(name) for name in names],
        }), 202)["items"]
        assert all(item["disposition"] == "admitted" for item in started), started
        jobs = []
        for item in started:
            def finished():
                (job,) = host.jobs(job_id=item["job_id"])
                return job if job["outcome"] is not None else None

            job = wait_until(finished)
            assert job["state"] == "succeeded", job["outcome"]
            jobs.append(job)
        return jobs

    def receipt(project, job):
        store = ImmutableArtifactBlobProvider(projects / project / "results")
        return store, SuccessView(store).read_success(ContentObjectRef.model_validate(job["outcome"]["store_success"])).receipt

    def rollout(store, accepted, *, checkpoint=None):
        with open_stored_dataset(store, dataset_member(accepted.products["states"][0])) as data:
            result = {}
            for name, variable in {**data.descriptor.coords, **data.descriptor.variables}.items():
                if name == "checkpoint":
                    assert data.data(name).tolist() == [checkpoint]
                    continue
                value = np.asarray(data.data(name)).copy()
                if "checkpoint" in variable.dims:
                    value = np.take(value, 0, axis=variable.dims.index("checkpoint"))
                result[name] = value
            return result

    with TestClient(create_app(projects_dir=projects)) as client:
        path = "/api/projects/exporter"
        revision = expect(client.get(path))["revision"]
        trained, evaluated = run(client, path, revision, "train", "baseline")
        coordinate = train_payload["training_config"]["n_batches"]
        output = client.post(f"{path}/loaded-models/export", json={
            "revision": revision, "key": key("train"), "job_id": trained["id"], "completed_updates": coordinate,
        })
        assert output.status_code == 200, output.text
        assert output.headers["Content-Disposition"].endswith(f'train-{coordinate}.feedbax-model"')
        expected = rollout(*receipt("exporter", evaluated), checkpoint=coordinate)

        inputs = {
            "subject": {"kind": "loaded_model", "source": {"document": "fitted"}},
            "task": {"kind": "task", "source": train_payload["trial_documents"]["task"]},
            "objective": {"kind": "objective", "source": train_payload["trial_documents"]["objective"]},
        }
        loaded_baseline = deepcopy(baseline) | {"inputs": inputs}
        loaded_baseline["base"]["inline"]["params"]["input_bindings"] = train_payload["trial_documents"]["input_bindings"]
        shorter = deepcopy(loaded_baseline)
        shorter["inputs"]["task"]["source"] = {"document": "shorter"}
        raw["documents"]["shorter"] = deepcopy(raw["documents"]["task"])
        short_task = raw["documents"]["shorter"]["payload"]
        short_task["fixed_steps"] = 7
        short_task["epochs"][0]["end"]["length"]["value"] = 7
        next_train = deepcopy(train)
        next_train["base"]["inline"]["training_config"]["n_batches"] = 1
        raw["steps"]["analyze"]["inputs"]["states"] = {"step": "baseline"}
        for layer in raw["documents"]["velocity-figure"]["payload"]["layers"]:
            layer.get("select", {}).pop("checkpoint", None)
        raw["steps"] = {
            "train": next_train, "baseline": loaded_baseline, "shorter": shorter, "analyze": raw["steps"]["analyze"],
        }
        _save_project(folders, "receiver", Project.model_validate(raw))
        rmtree(projects / "exporter")
        path = "/api/projects/receiver"
        revision = expect(client.get(path))["revision"]
        imported = expect(client.post(f"{path}/loaded-models/import", params={
            "revision": revision, "name": "fitted", "model": train_payload["trial_documents"]["model"]["document"],
        }, content=output.content, headers={"Content-Type": "application/octet-stream"}), 201)
        revision = imported["snapshot"]["revision"]
        edit = expect(client.post(f"{path}/authoring/train/train/start", json={
            "revision": revision, "source": inputs["subject"],
        }))
        assert not edit["problems"] and edit["files"]
        revision = expect(client.patch(path, json={"revision": revision, "files": edit["files"]}))["revision"]

        baseline_job, short_job, initialized, *analysis = run(
            client, path, revision, "baseline", "shorter", "train", "analyze",
        )
        assert analysis and all(job["request"].startswith(("analyze:", "render:")) for job in analysis)
        actual = rollout(*receipt("receiver", baseline_job))
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        assert len(rollout(*receipt("receiver", short_job))["state_time"]) == 8
        _, next_result = receipt("receiver", initialized)
        assert next_result.continuation.clock.completed_updates == 1
        assert (next_result.continuation.clock.segment_start, next_result.continuation.clock.schedule_origin) == (0, 0)
        assert next_result.inputs["checkpoint"][0].selection is None
        window = expect(client.get(f"{path}/train-runs", params={"revision": revision, "window_map": True}))["window_map"]
        (selected,) = [row for row in window["evaluations"] if row["key"]["step"] == "baseline"]
        assert selected["result"]["loaded_model"]["kind"] == "imported"
        assert selected["result"].get("checkpoint_source") is None
        replay = expect(client.post(f"{path}/evaluate-replay", json={
            "revision": revision, "job_id": baseline_job["id"], "key": key("baseline"), "trial": "0",
        }))
        np.testing.assert_array_equal(replay["state_time"], actual["state_time"])
