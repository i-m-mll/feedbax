"""A native model remains executable after its exporting project is gone."""

from copy import deepcopy
from shutil import rmtree

import numpy as np
from fastapi.testclient import TestClient

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import decode_request
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
    train_payload["training_config"]["snapshot_interval"] = 1
    baseline = raw["steps"]["baseline"]
    model_name = train_payload["trial_documents"]["model"]["document"]
    model = raw["documents"][model_name]
    interior = model["payload"]["interior"]
    interior["components"].update({
        "motor-noise": {"kind": "leaf", "name": "Motor noise", "type": "feedbax.Noise", "fields": {"shape": [2]}},
        "actuate": {"kind": "leaf", "name": "Actuate", "type": "feedbax.Sum"},
    })

    def port(component, name):
        return {"owner": "component", "component": component, "port": name}

    interior["wires"].update({
        "control-wire": {"source": interior["wires"]["plant-wire"]["source"], "target": port("actuate", "a")},
        "noise-wire": {"source": port("motor-noise", "output"), "target": port("actuate", "b")},
    })
    interior["wires"]["plant-wire"]["source"] = port("actuate", "output")
    raw["documents"]["quiet-model"] = {
        "label": "Quiet model",
        "schema": model["schema"], "base": {"document": model_name},
        "differences": {"/interior/components/motor-noise/fields/std": {"value": 0.0}},
    }

    def quiet(step):
        varied = deepcopy(step)
        varied["base"]["inline"]["trial_documents"] = {"model": {"document": "quiet-model"}}
        return varied

    source = Project.model_validate({"documents": raw["documents"], "steps": {
        "train": train, "baseline": baseline, "quiet": quiet(baseline),
    }})
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
        trained, evaluated, quiet_trained = run(client, path, revision, "train", "baseline", "quiet")
        coordinate = train_payload["training_config"]["n_batches"]
        export = {
            "revision": revision, "key": key("train"), "job_id": trained["id"], "completed_updates": coordinate,
        }
        assert client.post(f"{path}/loaded-models/export", json=export | {"key": key("baseline")}).status_code == 404
        output = client.post(f"{path}/loaded-models/export", json=export)
        assert output.status_code == 200, output.text
        assert output.headers["Content-Disposition"].endswith(f'train-{coordinate}.feedbax-model"')
        early = client.post(f"{path}/loaded-models/export", json=export | {"completed_updates": 1})
        assert early.status_code == 200 and early.content != output.content
        expected = rollout(*receipt("exporter", evaluated), checkpoint=coordinate)
        expected_quiet = rollout(*receipt("exporter", quiet_trained), checkpoint=coordinate)

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
        raw["documents"]["loads"] = {"label": "Loads", "schema": "feedbax.set_document.v2", "payload": {"entries": {
            "pulse": {"label": "Pulse", "task_variables": [{
                "id": "amplitude", "name": "Amplitude", "type": {"base": "real", "shape": [2]},
                "value": {"kind": "call", "function": "vector", "args": [
                    {"kind": "literal", "value": 1.0}, {"kind": "literal", "value": 0.0},
                ]},
            }], "differences": {"/interior/wires/plant-wire/attachments": {"value": [{
                "id": "pulse", "kind": "intervention", "operation": "Add",
                "operands": {"value": {"kind": "task", "variable": "amplitude"}},
                "window": {"spans": [{
                    "kind": "interval",
                    "start": {"anchor": {"kind": "trial", "edge": "start"}, "offset": {"value": 2, "unit": "st"}},
                    "end": {"offset": {"value": 2, "unit": "st"}},
                }]},
            }]}}},
        }}}
        next_train = deepcopy(train)
        next_train["base"]["inline"]["training_config"]["n_batches"] = 1
        invalid_continue = deepcopy(next_train)
        invalid_continue["inputs"] = {"checkpoint": inputs["subject"]}
        invalid_continue["base"]["inline"]["checkpoint_initialization"] = {
            "mode": "continue_from", "source_completed_updates": coordinate,
            "state": "preserve", "rng": "carry", "schedule": "continue",
        }
        raw["steps"]["analyze"]["inputs"]["states"] = {"step": "baseline"}
        for layer in raw["documents"]["velocity-figure"]["payload"]["layers"]:
            layer.get("select", {}).pop("checkpoint", None)
        raw["steps"] = {
            "train": next_train, "baseline": loaded_baseline, "shorter": shorter,
            "perturbed": deepcopy(loaded_baseline), "quiet": quiet(loaded_baseline),
            "analyze": raw["steps"]["analyze"], "invalid-continue": invalid_continue,
        }
        _save_project(folders, "receiver", Project.model_validate(raw))
        rmtree(projects / "exporter")
        path = "/api/projects/receiver"
        revision = expect(client.get(path))["revision"]
        imported = expect(client.post(f"{path}/loaded-models/import", params={
            "revision": revision, "name": "fitted", "model": model_name,
        }, content=output.content, headers={"Content-Type": "application/octet-stream"}), 201)
        revision = imported["snapshot"]["revision"]
        edit = expect(client.post(f"{path}/authoring/train/train/start", json={
            "revision": revision, "source": inputs["subject"],
        }))
        assert not edit["problems"] and edit["files"]
        revision = expect(client.patch(path, json={"revision": revision, "files": edit["files"]}))["revision"]

        def set_edit(command):
            nonlocal revision
            edit = expect(client.post(f"{path}/authoring/sets", json={"revision": revision, "command": command}))
            assert not edit["problems"] and edit["files"], edit
            revision = expect(client.patch(path, json={"revision": revision, "files": edit["files"]}))["revision"]

        set_edit({"kind": "apply_set", "step": "perturbed", "set": "loads"})
        swept = folders.read_plan("receiver", revision=revision).plan
        assert [(row.key.row, row.problems) for row in swept.runs if row.key.step == "perturbed"] == [("loads=pulse", ())]
        scoped = expect(client.post(f"{path}/model/evaluate-model", json={
            "revision": revision, "run": key("perturbed") | {"row": "loads=pulse"}, "scope": {"kind": "step"},
        }))
        assert scoped["editable"] and not scoped["problems"], scoped["problems"]
        assert list(scoped["resolved_model"]["interior"]["wires"]["plant-wire"]["attachments"]) == ["pulse"]
        set_edit({"kind": "set_to_field", "step": "perturbed", "axis": "loads"})
        set_edit({"kind": "set_field_weight", "step": "perturbed", "axis": "loads", "entry": None, "weight": 0})
        pairing = expect(client.post(f"{path}/authoring/trials", json={
            "revision": revision, "run": key("perturbed"),
        }))
        assert not pairing["problems"], pairing["problems"]
        (origin,) = [item["set_entry"] for item in pairing["variables"] if item.get("set_entry")]
        assert origin == {"step": "perturbed", "axis": "loads", "set": "loads", "entry": "pulse", "variable": "amplitude"}
        plan = folders.read_plan("receiver", revision=revision).plan
        paired_steps = {"baseline", "perturbed", "quiet"}
        members = {
            row.key.step: decode_request(plan.requests[row.request_id], plan.documents.__getitem__).inputs["subject"]
            for row in plan.runs if row.key.step in paired_steps
        }
        assert members.keys() == paired_steps
        assert members["baseline"] == members["perturbed"] == members["quiet"]

        baseline_job, short_job, perturbed_job, quiet_job, initialized, *analysis = run(
            client, path, revision, "baseline", "shorter", "perturbed", "quiet", "train", "analyze",
        )
        assert analysis and all(job["request"].startswith(("analyze:", "render:")) for job in analysis)
        actual = rollout(*receipt("receiver", baseline_job))
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        assert len(rollout(*receipt("receiver", short_job))["state_time"]) == 8
        perturbed = rollout(*receipt("receiver", perturbed_job))
        address = ModelAddress.model_validate({"composites": [], "element": {
            "kind": "attachment", "parent": {"kind": "wire", "id": "plant-wire"}, "id": "pulse",
        }}).to_text()
        active = perturbed[pointer(("v1", "task", "intervention_window", address))]
        np.testing.assert_array_equal(active, np.broadcast_to([False, False, True, True, False, False, False, False], active.shape))
        velocity = "/v1/model/state/plant/effector~1velocity"
        np.testing.assert_array_equal(perturbed[velocity][..., :3, :], actual[velocity][..., :3, :])
        assert np.all(perturbed[velocity][..., 4, 0] > actual[velocity][..., 4, 0])
        quiet_loaded = rollout(*receipt("receiver", quiet_job))
        np.testing.assert_array_equal(quiet_loaded[velocity], expected_quiet[velocity])
        assert np.any(np.abs(quiet_loaded[velocity] - actual[velocity]) > 1e-6)
        _, next_result = receipt("receiver", initialized)
        assert next_result.continuation.clock.completed_updates == 1
        assert (next_result.continuation.clock.segment_start, next_result.continuation.clock.schedule_origin) == (0, 0)
        assert next_result.inputs["checkpoint"][0].selection is None
        window = expect(client.get(f"{path}/train-runs", params={"revision": revision, "window_map": True}))["window_map"]
        assert any("a Loaded Model supports only initialize_from" in item["detail"] for item in window["problems"])
        (selected,) = [row for row in window["evaluations"] if row["key"]["step"] == "baseline"]
        assert selected["result"]["loaded_model"]["kind"] == "imported"
        assert selected["result"].get("checkpoint_source") is None
        replay = expect(client.post(f"{path}/evaluate-replay", json={
            "revision": revision, "job_id": baseline_job["id"], "key": key("baseline"), "trial": "0",
        }))
        np.testing.assert_array_equal(replay["state_time"], actual["state_time"])
