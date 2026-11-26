"""Exercise saved Run and product custody through the real local Host."""

from base64 import b64decode, b64encode
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.builtins.figure_templates import SpeedByTrial, SpeedByTrialSettings
from feedbax.cli import main
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.code_validity import SourcePath
from feedbax.contracts.figure_document import StoredFigureDocument
from feedbax.contracts.project import Project
from feedbax.contracts.receipt_validity import FileDifference, ReceiptValidity
from feedbax.planner.project_plan import ElementKey, RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_figure import FigureProduct
from feedbax.examples import load_shipped_projects
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.validity.project_admission import ProjectValidity
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests.support.host import admit, held_until_released, job_row, run_job, token, wait_until
from tests.support.managed_train_fixture import managed_train_project
from tests.test_project_analyze_plan import _operations
from tests.test_project_history_replay import _replay_body
from tests.test_project_report_plan import _report_project

_TRIALS = ["0", "1"]


def test_saved_figure_keeps_its_task_signal_after_a_studio_rename(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = raw["documents"]["simple-reaches-task"]["payload"]
    target = next(signal for signal in task["signals"] if signal["id"] == "target_pos")
    target["name"] = "Requested target"
    task["validation_trials"] = task["validation_trials"][:1]
    raw["steps"] = {
        "rollout": {
            "label": "Rollout",
            "kind": "evaluate",
            "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
                "seed": 17, "trial_policy": "paired", "variation": [],
                "trial_set": {"kind": "every_condition"},
                "input_bindings": {"target_pos": "target_pos"},
            }}},
            "inputs": {
                "subject": {"kind": "initial_model", "source": {"document": "point-mass"}},
                "task": {"kind": "task", "source": {"document": "simple-reaches-task"}},
            },
        },
        "plot": {
            "label": "Plot",
            "kind": "analyze",
            "base": {"inline": {
                "inputs": {"states": {"kind": "dataset"}},
                "views": {"target": {
                    "label": "Target",
                    "inputs": {"states": {"input": "states"}},
                    "figure": {
                        "data": ["states"],
                        "layout": {"position": {"row": 1, "col": 1}},
                        "layers": {"target-x": {
                            "role": "states", "panel": "position", "mark": "line",
                            "encode": {
                                "x": "transition_start",
                                "y": {"variable": "/v1/task/output/target_pos", "select": {"xy": "x"}},
                            },
                        }},
                    },
                }},
            }},
            "inputs": {"states": {"step": "rollout"}},
        },
    }
    project = "signal-rename"
    url = f"/api/projects/{project}"
    _save_project(ProjectFolderService(tmp_path), project, Project.model_validate(raw))
    rollout = {"variant": "default", "step": "rollout", "row": "default"}
    plot = rollout | {"step": "plot"}

    def json_object(response, status=200):
        assert response.status_code == status, response.text
        return response.json()

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        revision = json_object(client.get(url))["revision"]

        def element():
            window = json_object(client.get(f"{url}/train-runs", params={
                "revision": revision, "window_map": True,
            }))["window_map"]
            assert not window["problems"], window["problems"]
            (request,) = next(item for item in window["analysis_elements"] if item["name"] == "target")["requests"]
            return request

        def draw(name):
            assert element()["status"] in ("not_run", "waiting", "stale")
            submitted = json_object(client.post(f"{url}/train-runs", json={
                "revision": revision, "token": token(name), "run_keys": [rollout, plot],
            }), 202)
            for item in submitted["items"]:
                job = wait_until(lambda: (
                    job if (job := job_row(host, item["job_id"])) and job["outcome"] is not None else None
                ))
                assert job["state"] == "succeeded", job["outcome"]
            return json_object(client.get(element()["result"]["products"]["figure"]))

        before = draw("before")
        assert before["layout"]["yaxis"]["title"]["text"] == "Requested target (m)"
        changed = json_object(client.post(f"{url}/authoring/trials", json={
            "revision": revision, "run": rollout,
            "change": {"kind": "structure", "changes": [{
                "op": "rename", "member": "signal", "id": "target_pos", "name": "Renamed target",
            }]},
        }))
        saved = json_object(client.patch(url, json={"revision": revision, "files": changed["files"]}))
        revision = saved["revision"]
        after = draw("after")
        assert after["layout"]["yaxis"]["title"]["text"] == "Renamed target (m)"
        assert [(trace["x"], trace["y"]) for trace in after["data"]] == [
            (trace["x"], trace["y"]) for trace in before["data"]
        ]


def _saved_light_definition(*, paired=False):
    project = _report_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    evaluation = managed_evaluation_definition()
    train = evaluation["steps"]["train"]
    task = load_shipped_projects()["point-mass-reaching"].documents["simple-reaches-task"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task["payload"]["fixed_steps"] = 8
    del task["payload"]["validation_trials"]
    task["payload"]["epochs"][0]["end"]["length"]["value"] = 8
    evaluation["documents"]["task"] = task
    project["documents"] = evaluation["documents"] | {
        name: project["documents"][name]
        for name in ("velocity-analysis", "velocity-figure", "velocity-report")
    }
    project["documents"]["velocity-figure"]["payload"]["layers"]["bounds"]["style"] = {"label": "{condition}"}
    project["steps"]["train"] = train
    step = "conditions" if paired else "baseline"
    project["steps"][step] = evaluation["steps"][step]
    for evaluate in project["steps"].values():
        if evaluate.get("kind") == "evaluate":
            evaluate["base"]["inline"]["params"]["trial_set"] = {"kind": "sample", "count": len(_TRIALS)}
    project["documents"]["velocity-analysis"]["payload"]["operations"]["align"]["params"]["target"] = (
        "/v1/task/output/target_pos"
    )
    return project


def test_saved_operation_products_can_be_read_without_a_figure(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    definition = _report_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    shipped = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = shipped["documents"]["simple-reaches-task"]
    del task["payload"]["validation_trials"]
    task["payload"]["fixed_steps"] = 8
    task["payload"]["epochs"][0]["end"]["length"]["value"] = 8
    definition["documents"] = {
        "point-mass": shipped["documents"]["point-mass"], "task": task,
        "velocity-analysis": definition["documents"]["velocity-analysis"],
    }
    definition["steps"] = {
        "analyze": definition["steps"]["analyze"],
        "baseline": {
            "label": "Baseline",
            "kind": "evaluate",
            "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
                "seed": 17, "trial_policy": "paired", "variation": [],
                "trial_set": {"kind": "sample", "count": 2}, "input_bindings": {"target_pos": "target_pos"},
            }}},
            "inputs": {
                "subject": {"kind": "initial_model", "source": {"document": "point-mass"}},
                "task": {"kind": "task", "source": {"document": "task"}},
            },
        },
    }
    analysis = definition["documents"]["velocity-analysis"]["payload"]
    del analysis["views"]

    path = "/api/projects/datasets"
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "datasets", Project.model_validate(definition))

    def expect(response, status=200):
        assert response.status_code == status, response.text
        return response.json()

    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        def projection():
            revision = expect(client.get(path))["revision"]
            window = expect(client.get(f"{path}/train-runs", params={
                "revision": revision, "window_map": True,
            }))["window_map"]
            assert not window["problems"], window["problems"]
            assert all(element["kind"] == "analyze" for element in window["analysis_elements"])
            return revision, {element["name"]: element["requests"][0] for element in window["analysis_elements"]}

        revision, _ = projection()
        submission = expect(client.post(f"{path}/train-runs", json={
            "revision": revision, "token": token("operation-datasets"),
            "run_keys": [{"variant": "default", "step": step, "row": "default"} for step in ("baseline", "analyze")],
        }), 202)
        for item in submission["items"]:
            job = wait_until(lambda: (
                observed if (observed := job_row(host, item["job_id"])) and observed["outcome"] is not None else None
            ))
            assert job["state"] == "succeeded", job["outcome"]

        _, elements = projection()
        summary = elements["summarize"]["result"]
        url = summary["products"]["summary"]
        descriptor = expect(client.get(url))
        assert descriptor["variables"]["mean"]["units"] == "m/s"
        assert descriptor["variables"]["mean"]["dims"] == ["condition", "state_time"]
        total = descriptor["dimensions"]["state_time"]
        assert expect(client.get(f"{url}/values", params={"field": "count", "offset": total - 2, "limit": 3})) == {
            "field": "count", "offset": total - 2, "total": total, "values": [2, 2],
        }
        aligned = elements["align"]["result"]
        velocity = aligned["products"]["velocity"]
        assert expect(client.get(f"{velocity}/values", params={"field": "trial"}))["values"] == _TRIALS
        sample = expect(client.get(f"{velocity}/values", params={
            "field": "parallel_velocity", "offset": total - 2, "limit": 4,
        }))
        assert sample["total"] == 2 * total
        store = ImmutableArtifactBlobProvider(tmp_path / "datasets" / "results")
        success = SuccessView(store)
        ref = success.attempt_ref(aligned["job_id"])
        assert ref is not None
        receipt = success.read_receipt(ref)
        with open_stored_dataset(store, dataset_member(receipt.products["velocity"][0])) as retained:
            expected = np.asarray(retained.data("parallel_velocity")).ravel()[total - 2:total + 2]
        np.testing.assert_array_equal(sample["values"], expected)
        expect(client.get(f"{url}/values", params={"field": "absent"}), 404)
        expect(client.get(f"{url}/values", params={"field": "mean", "limit": 257}), 422)

        analysis["operations"]["summarize"]["params"]["bounds_scale"] = 2.0
        _save_project(folders, "datasets", Project.model_validate(definition))
        _, revised = projection()
        previous = revised["summarize"]["previous"]
        assert revised["summarize"]["status"] == "stale"
        assert previous["job_id"] == summary["job_id"]
        assert expect(client.get(previous["products"]["summary"])) == descriptor


def test_saved_light_chain_reuses_exact_receipts_and_retains_a_stale_figure(host, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    project = "reaching"
    definition = _saved_light_definition()
    model_key = definition["steps"]["train"]["base"]["inline"]["trial_documents"]["model"]["document"]
    definition["documents"][model_key]["payload"]["interior"]["components"]["plant"]["name"] = "Point mass"
    folders = ProjectFolderService(projects)
    _save_project(folders, project, Project.model_validate(definition))
    project_id = folders.open(project).snapshot.project_id
    path = f"/api/projects/{project}"

    def expect(response, code=200):
        assert response.status_code == code, response.text
        return response

    def terminal(count):
        def settled():
            jobs = host.jobs(project=project_id)
            return jobs if len(jobs) == count and all(job["outcome"] is not None for job in jobs) else None

        jobs = wait_until(settled)
        assert all(job["state"] == "succeeded" for job in jobs), {
            job["request"]: job["outcome"] for job in jobs if job["state"] != "succeeded"
        }
        return jobs

    with TestClient(create_app(projects_dir=projects)) as client:
        def revision():
            return expect(client.get(path)).json()["revision"]

        def run(rev, name, *steps, reuse_completed=False):
            return expect(client.post(f"{path}/train-runs", json={
                "revision": rev, "token": token(name), "reuse_completed": reuse_completed,
                "run_keys": [{"variant": "default", "step": step, "row": "default"} for step in steps],
            }), 202).json()["items"]

        def projection(rev):
            window = expect(client.get(f"{path}/train-runs", params={
                "revision": rev, "window_map": True,
            })).json()["window_map"]
            elements = {element["name"]: request for element in window["analysis_elements"]
                        for request in element["requests"]}
            (report,) = (row for row in window["rows"] if row["key"]["step"] == "report")
            return elements, report

        rev = revision()
        started = run(rev, "walkthrough", "train", "baseline", "analyze", "report")
        assert [item["disposition"] for item in started] == ["admitted"] * 6, started
        jobs = terminal(6)
        by_id = {job["id"]: job for job in jobs}
        by_kind = {job["request"].partition(":")[0]: job for job in jobs}
        light = [job for job in jobs if job["slot"] == "light"]
        assert len(light) == 4
        assert all(job["delivery"] == 1 for job in light)
        activity = expect(client.get("/api/jobs", params={"project": project_id})).json()
        assert {job["kind"] for job in activity["jobs"]} == {
            "train", "evaluate", "analyze", "render", "report",
        }
        catalog = expect(client.post(f"{path}/authoring/analyze", json={
            "revision": rev, "key": {"variant": "default", "step": "analyze", "row": "default"},
        })).json()
        assert catalog["states"] is not None
        assert {"input": "states"} in catalog["sources"]["align"]["states"]
        velocity = "/v1/model/state/plant/effector~1velocity"
        assert velocity in catalog["variables"]["feedbax.aligned_velocity"]["velocity"]
        assert catalog["variable_names"][velocity] == "Point mass · effector velocity"
        root = projects / project / "results"
        store = SuccessView(ImmutableArtifactBlobProvider(root))
        for job in light:
            ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
            retained = store.read_owned_success(ref).receipt
            assert (retained.attempt, retained.request.id) == (job["id"], job["request"])
            assert job["outcome"]["receipt"]["manifest_sha256"] == ref.sha256
            (edge,) = job["edges"]
            (role,) = retained.inputs
            (bound,) = retained.inputs[role]
            assert bound.selection.receipt == ContentObjectRef.model_validate(by_id[edge["job_id"]]["outcome"]["store_success"])
            assert edge["satisfied"]

        elements, report = projection(rev)
        assert all(element["status"] == "succeeded" for element in elements.values())
        assert report["status"] == "succeeded"
        aligned_descriptor = expect(client.get(elements["align"]["result"]["products"]["velocity"])).json()
        figure = elements["profile"]["result"]
        figure_url = figure["products"]["figure"]
        figure_response = expect(client.get(figure_url))
        assert [trace["name"] for trace in figure_response.json()["data"] if trace.get("showlegend", True)] == [
            definition["steps"]["baseline"]["label"],
        ]
        report_url = report["light"]["result"]["products"]["report"]
        rendered = expect(client.get(report_url))
        asset = re.search(r"figures/[0-9a-f]{64}\.json", rendered.text).group()
        assert expect(client.get(f"{report_url.removesuffix('report.html')}{asset}")).content == figure_response.content
        script = expect(client.get(report_url.replace("report.html", "plotly.min.js")))
        inline = re.search(r"<script>(.*?)</script>", rendered.text, re.S).group(1).encode()
        external_integrity = "sha256-" + b64encode(sha256(script.content).digest()).decode()
        inline_integrity = "sha256-" + b64encode(sha256(inline).digest()).decode()
        assert f'src="plotly.min.js" integrity="{external_integrity}"' in rendered.text
        policy = rendered.headers["Content-Security-Policy"]
        assert f"script-src '{inline_integrity}' '{external_integrity}';" in policy
        assert "connect-src 'self';" in policy
        expect(client.get(report_url.replace("report.html", "not-published.json")), 404)

        assert main(["run", str(projects / project)]) == 0
        unchanged = capsys.readouterr().out
        assert unchanged.count(" reused ") == 6
        assert {job["id"] for job in host.jobs(project=project_id)} == set(by_id)
        assert projection(rev) == (elements, report)

        definition["documents"]["velocity-analysis"]["payload"]["operations"]["summarize"]["params"]["bounds_scale"] = 2.0
        _save_project(folders, project, Project.model_validate(definition))
        edited = revision()
        stale, stale_report = projection(edited)
        stable_align, old_align = stale["align"]["result"], elements["align"]["result"]
        assert stable_align | {"products": old_align["products"]} == old_align
        assert expect(client.get(stable_align["products"]["velocity"])).json() == aligned_descriptor
        assert {stale[name]["status"] for name in ("summarize", "profile")} == {stale_report["status"]} == {"stale"}
        analysis = definition["documents"]["velocity-analysis"]["payload"]
        assert stale["summarize"]["changed"]["place"] == {
            "part": "analysis", "element": "summarize", "name": analysis["operations"]["summarize"]["label"],
            "field": "/params/bounds_scale",
        }
        assert "bounds_scale" in stale["summarize"]["changed"]["path"]
        assert stale["profile"]["changed"]["place"] == {
            "part": "analysis", "element": "profile", "name": analysis["views"]["profile"]["label"], "field": "/inputs/summary",
        }
        assert "inputs.summary" in stale["profile"]["changed"]["path"]
        assert stale_report["light"]["changed"]["place"] == {"part": "input", "role": "velocity_summary"}
        previous_url = stale["profile"]["previous"]["products"]["figure"]
        assert stale["profile"]["previous"]["job_id"] == figure["job_id"]
        assert expect(client.get(previous_url)).content == figure_response.content
        assert main(["status", str(projects / project)]) == 0
        assert "bounds_scale" in capsys.readouterr().out

        rerun = run(edited, "scoped-rerun", "analyze", "report", reuse_completed=True)
        assert [item["disposition"] for item in rerun] == ["reused", "admitted", "admitted", "admitted"], rerun
        rerun_jobs = {job["id"]: job for job in terminal(9)}
        assert rerun_jobs[by_kind["evaluate"]["id"]] == by_kind["evaluate"]
        current, current_report = projection(edited)
        assert all(element["status"] == "succeeded" for element in current.values())
        assert current_report["status"] == "succeeded"
        assert current["profile"]["previous"] == stale["profile"]["previous"]
        assert current["profile"]["result"]["job_id"] != figure["job_id"]
        assert current_report["light"]["previous"]["job_id"] == report["light"]["result"]["job_id"]
        assert expect(client.get(previous_url)).content == figure_response.content

        with monkeypatch.context() as upgraded:
            upgraded.setattr("plotly.offline.get_plotlyjs", lambda: "upgraded Plotly bytes")
            previous_report_url = current_report["light"]["previous"]["products"]["report"]
            historic = expect(client.get(previous_report_url))
            assert historic.content == rendered.content
            assert historic.headers["Content-Security-Policy"] == policy
            assert expect(client.get(previous_report_url.replace("report.html", "plotly.min.js"))).content == script.content

        definition["steps"]["baseline"]["inputs"]["subject"] = {"run": by_kind["train"]["request"]}
        _save_project(folders, project, Project.model_validate(definition))
        assert main(["run", str(projects / project)]) == 0
        assert capsys.readouterr().out.count(" reused ") == 6
        assert {job["id"] for job in host.jobs(project=project_id)} == set(rerun_jobs)

        (replacement,) = run(revision(), "replace-train", "train")
        assert replacement["disposition"] == "admitted"
        (new_train,) = [job for job in terminal(10) if job["id"] == replacement["job_id"]]
        assert new_train["request"] == by_kind["train"]["request"]
        assert main(["run", str(projects / project), "baseline"]) == 0
        (new_evaluate,) = [
            job for job in terminal(11)
            if job["request"] == by_kind["evaluate"]["request"] and job["id"] not in rerun_jobs
        ]
        assert not new_evaluate["edges"]
        ref = ContentObjectRef.model_validate(new_evaluate["outcome"]["store_success"])
        retained = SuccessView(ImmutableArtifactBlobProvider(root)).read_receipt(ref)
        subject = retained.inputs["subject"][0].evaluation_subject
        assert subject.terminal_receipt.model_dump(mode="json") == new_train["outcome"]["store_success"]



def test_model_edit_retains_exact_saved_attempts_and_waits_on_the_train(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    folders = ProjectFolderService(projects)
    definition = _saved_light_definition()
    shipped = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    documents = definition["documents"]
    shipped["documents"]["simple-reaches-task"] = documents["task"]
    definition["documents"] = shipped["documents"] | {
        name: documents[name] for name in ("velocity-analysis", "velocity-figure", "velocity-report")
    }
    definition["steps"]["train"] = shipped["steps"]["train"]
    definition["steps"]["train"]["label"] = "Fit reaching controller"
    definition["steps"]["baseline"]["label"] = "Check reaching trials"
    definition["steps"]["copy"] = deepcopy(definition["steps"]["baseline"])
    _save_project(folders, "reaching", Project.model_validate(definition))
    project_id = folders.open("reaching").snapshot.project_id
    path = "/api/projects/reaching"

    def key(step):
        return {"variant": "default", "step": step, "row": "default"}

    def expect(response, code=200):
        assert response.status_code == code, response.text
        return response.json()

    with TestClient(create_app(projects_dir=projects, quick_duration=timedelta(minutes=5))) as client:
        def plan(revision):
            return expect(client.get(f"{path}/train-runs", params={"revision": revision, "window_map": True}))

        def replay(revision, job_id, step, code=200):
            def refuse_project_wide_read(**_):
                pytest.fail("Replaying one saved attempt should not join the project's full job history.")

            with monkeypatch.context() as patch:
                patch.setattr(client.app.state.job_service, "read_plan_attempts", refuse_project_wide_read)
                return expect(client.post(f"{path}/evaluate-replay", json={
                    "revision": revision, "job_id": job_id, "key": key(step), "trial": _TRIALS[0],
                }), code)

        original_snapshot = expect(client.get(path))
        model_address = {"composites": [], "element": {
            "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
        }}
        edited = expect(client.post(f"{path}/model/edit", json={
            "revision": original_snapshot["revision"],
            "presentation_revision": original_snapshot["presentation_revision"],
            "document": "point-mass", "command": {"operation": {
                "op": "set_field", "address": model_address, "value": 2.5,
            }},
        }))
        assert edited["result"]["status"] == "applied", edited
        revision = expect(client.get(path))["revision"]
        release = tmp_path / "release"
        gate = run_job("gate", held_until_released(release), project=project_id)
        admit(host, token("gate"), gate)
        wait_until(lambda: job_row(host, gate.id, "running"))
        try:
            items = expect(client.post(f"{path}/train-runs", json={
                "revision": revision, "token": token("retained"),
                "run_keys": [key(step) for step in ("train", "baseline", "copy", "analyze")],
            }), 202)["items"]
            train, evaluate, copy, *light = items
            assert train["disposition"] == evaluate["disposition"] == "admitted"
            assert copy["disposition"] == "attached" and copy["job_id"] == evaluate["job_id"]
            window = plan(revision)["window_map"]
            waiting = {row["key"]["step"]: row for row in window["evaluations"]}
            assert waiting["baseline"]["status"] == "waiting"
            assert waiting["baseline"]["reason"] == "Waiting for Fit reaching controller."
            assert waiting["baseline"]["can_run"]
            assert not any(row.get("step") == "baseline" for row in window["problems"])
        finally:
            release.touch()
        jobs = {item["job_id"] for item in (train, evaluate, *light)}

        def finished():
            observed = [job for job in host.jobs(project=project_id) if job["id"] in jobs]
            return observed if len(observed) == len(jobs) and all(job["outcome"] is not None for job in observed) else None

        assert all(job["state"] == "succeeded" for job in wait_until(finished))
        before = plan(revision)
        replayed = replay(revision, evaluate["job_id"], "baseline")
        figure = next(
            request["result"] for element in before["window_map"]["analysis_elements"]
            if element["name"] == "profile" for request in element["requests"]
        )

        original_figure = client.get(figure["products"]["figure"])
        assert original_figure.status_code == 200
        rename = expect(client.post(f"{path}/authoring/trials", json={
            "revision": revision, "run": key("train"), "change": {"kind": "structure", "changes": [{
                "op": "rename", "member": "signal", "id": "target_pos", "name": "Aim",
            }]},
        }))
        snapshot = expect(client.patch(path, json={"revision": revision, "files": rename["files"]}))
        revision = snapshot["revision"]
        renamed = plan(revision)
        assert renamed["rows"][0]["status"] == "succeeded"
        assert renamed["rows"][0]["result"]["job_id"] == train["job_id"]
        assert all(row["can_run"] for row in renamed["window_map"]["evaluations"])
        catalog = expect(client.post(f"{path}/authoring/analyze", json={"revision": revision, "key": key("analyze")}))
        assert catalog["variable_names"]["/v1/task/output/target_pos"] == "Aim"
        projection = expect(client.post(f"{path}/model/projection", json={
            "revision": revision, "document": "point-mass", "run": key("train"),
        }))
        (binding,) = (row for row in projection["task_inputs"] if row["signal_id"] == "target_pos")
        assert binding["element"]["end"]["port"] == "target_pos"
        assert next(row["label"] for row in projection["task_signals"] if row["id"] == "target_pos") == "Aim"
        jobs_before_undo = {job["id"] for job in host.jobs(project=project_id)}
        expect(client.post(f"{path}/history/replay", json=_replay_body(
            snapshot, original_snapshot, ("document", "point-mass"),
        )))
        undone = expect(client.get(path))
        changed = undone["revision"]
        assert changed != revision
        after = plan(changed)
        assert {job["id"] for job in host.jobs(project=project_id)} == jobs_before_undo
        (old_train,) = after["rows"]
        assert old_train["status"] == "stale"
        assert old_train["request_id"] != old_train["previous"]["request_id"]
        assert old_train["previous"]["job_id"] == train["job_id"]
        assert old_train["previous"]["revision"] == before["revision"]

        window = after["window_map"]
        evaluations = {row["key"]["step"]: row for row in window["evaluations"]}
        for step in ("baseline", "copy"):
            row = evaluations[step]
            assert row["status"] == "stale" and not row["can_run"]
            assert row["reason"] == "Fit reaching controller is out of date."
            assert row["result"] is None and row["request_id"] != row["previous"]["request_id"]
            assert row["previous"]["job_id"] == evaluate["job_id"]
            assert row["previous"]["revision"] == before["revision"]
            assert row["previous"]["trials"] == _TRIALS
            assert row["previous"]["checkpoint_source"]["job_id"] == train["job_id"]
            assert replay(changed, evaluate["job_id"], step) == replayed
        replay(changed, evaluate["job_id"], "analyze", code=404)

        (analysis,) = [row for row in window["rows"] if row["key"]["step"] == "analyze"]
        assert analysis["status"] == "stale"
        assert analysis["reason"] == "Check reaching trials is out of date."
        profile = next(
            request for element in window["analysis_elements"]
            if element["name"] == "profile" for request in element["requests"]
        )
        assert profile["previous"]["job_id"] == figure["job_id"]
        assert profile["previous"]["revision"] == before["revision"]
        response = client.get(profile["previous"]["products"]["figure"])
        assert response.status_code == 200
        assert response.content == original_figure.content

        # Redo restores an exact quick receipt, which can now be stale after a source edit.
        inspect = ProjectValidity.inspect

        def inspect_with_changed_train_source(observer, view, published, **kwargs):
            observation = inspect(observer, view, published, **kwargs)
            if published.receipt.request.id != old_train["previous"]["request_id"]:
                return observation
            return replace(observation, decision=ReceiptValidity(
                receipt=published.ref, state="stale",
                reasons=(FileDifference(
                    kind="file", path=SourcePath(scope="project", path="train.py"),
                    before="a" * 64, after="b" * 64,
                ),),
            ))

        monkeypatch.setattr(ProjectValidity, "inspect", inspect_with_changed_train_source)
        expect(client.post(f"{path}/history/replay", json=_replay_body(
            undone, snapshot, ("document", "point-mass"),
        )))
        restored_revision = expect(client.get(path))["revision"]
        assert restored_revision == revision
        restored = plan(restored_revision)
        assert {job["id"] for job in host.jobs(project=project_id)} == jobs_before_undo
        (restored_train,) = restored["rows"]
        assert restored_train["status"] == "stale"
        assert restored_train["result"]["job_id"] == train["job_id"]
        for row in restored["window_map"]["evaluations"]:
            assert row["result"]["job_id"] == evaluate["job_id"]
            assert replay(restored_revision, evaluate["job_id"], row["key"]["step"]) == replayed


def test_saved_paired_evaluate_rows_make_one_figure_and_report(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    project = "paired"
    path = f"/api/projects/{project}"
    definition = _saved_light_definition(paired=True)
    analysis = definition["documents"]["velocity-analysis"]["payload"]
    analysis["operations"]["speed"] = {
        "label": "Speed",
        "operation": "feedbax.trial_speed",
        "params": {"velocity": analysis["operations"]["align"]["params"]["velocity"]},
        "inputs": {"states": {"input": "states"}},
    }
    analysis["views"]["profile"]["figure"] = {"template": "feedbax.VelocitySummary"}
    speed_figure = StoredFigureDocument.from_lowered(SpeedByTrial(SpeedByTrialSettings())).model_dump(
        mode="json", exclude_unset=True,
    )
    color = {"field": "mass.value", "scale": "load"}
    scales = {"load": {"kind": "continuous", "legend": False}}
    speed_figure["color_scales"] = scales
    speed_figure["layers"]["trial-speed"]["encode"]["color"] = color
    analysis["views"]["by_trial"] = {
        "label": "By trial",
        "figure": speed_figure,
        "inputs": {"speed": {"element": "speed", "product": "speed"}},
    }
    analysis["views"]["motion"] = {
        "label": "Motion",
        "figure": {
            "data": ["states"], "layout": {"motion": {"row": 1, "col": 1}},
            "color_scales": scales,
            "layers": {"motion": {
                "mark": "trajectory", "role": "states", "panel": "motion",
                "group": ["member", "condition", "trial"], "select": {"checkpoint": {"index": 0}},
                "encode": {
                    axis: {
                        "variable": "/v1/model/state/plant/effector~1position",
                        "select": {"xy": {"index": index}},
                    } for index, axis in enumerate(("x", "y"))
                } | {"color": color},
            }},
        },
        "inputs": {"states": {"input": "states"}},
    }
    steps = definition["steps"]
    mass_values = steps["conditions"]["sweep"]["axes"]["mass"]["values"]
    mass_values["heavy"]["label"] = "Heavy load"
    mass_values["nominal"]["label"] = "Nominal load"
    upstream = Project.model_validate(definition | {"steps": {
        name: steps[name] for name in ("train", "conditions")
    }})
    rows = {
        run.coordinates["mass"].value_id: run.key.row
        for run in compile_project(upstream).runs if run.key.step == "conditions"
    }
    order = tuple(rows)
    selected = [{"id": key, "label": mass_values[key]["label"]} for key in order]
    steps["analyze"]["inputs"]["states"] = {"step": "conditions", "where": {"clauses": [{"mass": selected}]}}
    del steps["baseline"]
    from feedbax.application.composition import load_application

    bundle = load_application().bundle
    folders = ProjectFolderService(projects, operations=bundle.operations, figure_templates=bundle.figure_templates)
    _save_project(folders, project, Project.model_validate(definition))

    def address(step, row="default"):
        return {"variant": "default", "step": step, "row": row}

    review = folders.read_plan(project, revision=None)
    plan = review.plan
    assert not plan.diagnostics
    analyze = RunKey("default", "analyze", "default")
    assert {element.name for element in plan.elements if element.run == analyze} == {
        "align", "summarize", "profile", "speed", "by_trial", "motion",
    }
    align = plan.elements[ElementKey(analyze, "align")]
    decoded = decode_request(plan.requests[align.request_id], plan.documents.__getitem__)
    request_ids = {run.key: run.request_id for run in plan.runs}
    members = [rows[key] for key in order]
    evaluation_ids = [request_ids[RunKey("default", "conditions", row)] for row in members]
    assert [(item.key, item.reference.request_id) for item in decoded.inputs["states"]] == list(
        zip(members, evaluation_ids, strict=True)
    )

    with TestClient(create_app(projects_dir=projects)) as client:
        revision = client.get(path).json()["revision"]
        response = client.post(f"{path}/train-runs", json={
            "revision": revision,
            "token": token("paired"),
            "run_keys": [
                address("train"), *(address("conditions", rows[key]) for key in order),
                address("analyze"), address("report"),
            ],
        })
        assert response.status_code == 202, response.text
        assert [item["disposition"] for item in response.json()["items"]] == ["admitted"] * 10, response.text

        def terminal():
            jobs = host.jobs(project=review.project_id)
            return jobs if len(jobs) == 10 and all(job["outcome"] is not None for job in jobs) else None

        jobs = wait_until(terminal)
        assert all(job["state"] == "succeeded" for job in jobs), [
            job["outcome"] for job in jobs if job["state"] != "succeeded"
        ]

    by_request = {job["request"]: job for job in jobs}
    store = ImmutableArtifactBlobProvider(projects / project / "results")
    view = SuccessView(store)

    def proof(request):
        ref = ContentObjectRef.model_validate(by_request[request]["outcome"]["store_success"])
        return ref, view.read_success(ref).receipt

    evaluations = [proof(request) for request in evaluation_ids]
    align_ref, aligned = proof(align.request_id)
    assert [item.key for item in aligned.inputs["states"]] == members
    assert [item.selection.receipt for item in aligned.inputs["states"]] == [ref for ref, _ in evaluations]
    assert {edge["job_id"] for edge in by_request[align.request_id]["edges"]} == {
        by_request[request]["id"] for request in evaluation_ids
    }
    train_ref, trained = proof(request_ids[RunKey("default", "train", "default")])
    assert trained.continuation is not None
    for _, receipt in evaluations:
        subject = receipt.inputs["subject"][0].evaluation_subject
        assert subject is not None and subject.terminal_receipt == train_ref
        assert subject.terminal_checkpoint == trained.continuation.provenance

    conditions, source_attrs, times = [], [], []
    for _, receipt in evaluations:
        with open_stored_dataset(store, dataset_member(receipt.products["states"][0])) as data:
            hand = "/v1/model/state/plant/effector~1position"
            producer = next(run for run in plan.runs if run.request_id == receipt.request.id)
            declared = producer.dataset_port.variables[hand]
            assert declared == data.descriptor.variables[hand]
            assert declared.dims == ("checkpoint", "condition", "trial", "state_time", "xy")
            conditions.extend(data.data("condition").tolist())
            assert data.data("trial").tolist() == _TRIALS
            assert data.data("checkpoint").tolist() == [trained.continuation.clock.completed_updates]
            times.append(np.asarray(data.data("state_time")).copy())
            source_attrs.append(data.descriptor.attrs)
    assert len(set(conditions)) == 2
    np.testing.assert_array_equal(*times)
    varied = {
        name: [attrs.get(name) for attrs in source_attrs]
        for name in ("effective_graph_sha256", "effective_task_sha256", "variation_sha256")
        if len({attrs.get(name) for attrs in source_attrs}) > 1
    }
    assert varied
    summary = plan.elements[ElementKey(analyze, "summarize")]
    summary_ref, summarized = proof(summary.request_id)
    assert summarized.inputs["velocity"][0].selection.receipt == align_ref
    for product, element in ((aligned, "velocity"), (summarized, "summary")):
        with open_stored_dataset(store, dataset_member(product.products[element][0])) as data:
            assert data.data("mass").tolist() == list(order)
            assert data.data("member").tolist() == members
            assert summary.names["source"][data.data("source").item()].key == "conditions"
            assert data.data("condition").tolist() == [[condition] for condition in conditions]
            np.testing.assert_array_equal(data.data("state_time"), times[0])
            assert {name: data.data(name).tolist() for name in varied} == varied
    with open_stored_dataset(store, dataset_member(summarized.products["summary"][0])) as data:
        mean = np.asarray(data.data("mean")).copy()
        assert np.all(data.data("count") == len(_TRIALS))
    assert mean.shape == (2, 1, 1, len(times[0]))
    assert not np.allclose(mean[0], mean[1])

    def array(values):
        if isinstance(values, dict):
            return np.frombuffer(b64decode(values["bdata"]), dtype=values["dtype"])
        return np.asarray(values)

    with TestClient(create_app(projects_dir=projects)) as reopened:
        response = reopened.get(f"{path}/train-runs", params={"revision": revision, "window_map": True})
        assert response.status_code == 200, response.text
        window = response.json()["window_map"]
        assert [edge["from_key"] for edge in window["edges"] if edge["to_key"] == address("analyze")] == [
            address("conditions", rows[key]) for key in order
        ]
        response = reopened.post(f"{path}/authoring/analyze", json={"revision": revision, "key": address("analyze")})
        assert response.status_code == 200, response.text
        catalog = response.json()
        for port in (catalog["states"], catalog["operation_outputs"]["speed"]["speed"]):
            assert {field: port["coordinates"][field]["dims"] for field in ("member", "mass", "mass.value")} == {
                "member": ["mass"], "mass": ["mass"], "mass.value": ["mass"],
            }
        assert {"input": "states"} in catalog["sources"]["motion"]["states"]
        assert {"element": "speed", "product": "speed"} in catalog["sources"]["by_trial"]["speed"]
        motion, = (element["requests"][0] for element in window["analysis_elements"] if element["name"] == "motion")
        motion_figure = reopened.get(motion["result"]["products"]["figure"])
        assert motion_figure.status_code == 200, motion_figure.text
        motion_traces = motion_figure.json()["data"]
        assert {trace["meta"]["group"]["member"] for trace in motion_traces} == set(members)
        assert len({trace["line"]["color"] for trace in motion_traces}) == 2
        (profile,) = (element["requests"][0] for element in window["analysis_elements"] if element["name"] == "profile")
        figure_ref, rendered_figure = proof(profile["request_id"])
        assert rendered_figure.inputs["summary"][0].selection.receipt == summary_ref
        figure = reopened.get(profile["result"]["products"]["figure"])
        assert figure.status_code == 200, figure.text
        traces = [trace for trace in figure.json()["data"] if trace.get("showlegend", True)]
        assert [trace["name"] for trace in traces] == [mass_values[key]["label"] for key in order]
        assert [trace["meta"]["group"]["member"] for trace in traces] == members
        for index, trace in enumerate(traces):
            np.testing.assert_allclose(array(trace["x"]), times[0])
            np.testing.assert_allclose(array(trace["y"]), mean[index, 0, 0])
        (by_trial,) = (
            element["requests"][0] for element in window["analysis_elements"] if element["name"] == "by_trial"
        )
        speed = reopened.get(by_trial["result"]["products"]["figure"])
        assert speed.status_code == 200, speed.text
        product = FigureProduct.model_validate_json(speed.content)
        assert len({trace["line"]["color"] for trace in speed.json()["data"]}) == 2
        assert product.layout.meta.panels["speed"].model_dump(exclude_none=True) == {
            "time": {}, "x_axis": "x", "y_axis": "y",
        }
        assert [run.model_dump() for run in product.layout.meta.runs["speed"]] == [
            {"job_id": by_request[request]["id"], "request_id": request}
            for request in evaluation_ids
        ]
        assert [trace.meta.model_dump() for trace in product.data] == [
            {"layer": "trial-speed", "group": {"member": member, "condition": condition, "trial": trial}}
            for member, condition in zip(members, conditions, strict=True) for trial in _TRIALS
        ]
        averaged = reopened.post(f"{path}/figures/mean", json={
            "revision": revision, "job_id": by_trial["result"]["job_id"], "mean": {"by": [], "align": "move"},
        })
        assert averaged.status_code == 200, averaged.text
        assert averaged.json()["layout"]["meta"]["panels"]["speed"]["time"] == {"zero_epoch": "move"}
        for member, condition in zip(members, conditions, strict=True):
            originals = [trace.y for trace in product.data if trace.meta.group["condition"] == condition]
            band = [
                trace for trace in averaged.json()["data"]
                if trace["meta"]["group"] == {"member": member, "condition": condition}
            ]
            assert len(band) == 3
            average = band[-1]
            np.testing.assert_allclose(average["y"], np.mean([array(samples) for samples in originals], axis=0))
        (report,) = (row for row in window["rows"] if row["key"]["step"] == "report")
        _, published_report = proof(request_ids[RunKey("default", "report", "default")])
        assert published_report.inputs["velocity_summary"][0].selection.receipt == figure_ref
        report_url = report["light"]["result"]["products"]["report"]
        rendered = reopened.get(report_url)
        assert rendered.status_code == 200, rendered.text
        asset = re.search(r"figures/[0-9a-f]{64}\.json", rendered.text).group()
        assert reopened.get(f"{report_url.removesuffix('report.html')}{asset}").content == figure.content


@pytest.mark.round(
    "A real Train step and two saved Evaluate runs are authored as one Analyze input that names the step without a row."
)
def test_a_states_input_without_a_row_reads_every_saved_run_of_its_step(host, tmp_path, monkeypatch):
    """A hand-written `states: {step: conditions}` is authored over both saved mass rows, not refused as empty."""
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    definition = _saved_light_definition(paired=True)
    definition["steps"]["analyze"]["inputs"]["states"] = {"step": "conditions"}
    del definition["steps"]["baseline"]
    folders = ProjectFolderService(projects, operations=_operations())
    _save_project(folders, "rowless", Project.model_validate(definition))
    review = folders.read_plan("rowless", revision=None)
    evaluations = [run.key for run in review.plan.runs if run.key.step == "conditions"]
    assert len(evaluations) == 2

    with TestClient(create_app(projects_dir=projects)) as client:
        response = client.post("/api/projects/rowless/train-runs", json={
            "revision": review.revision, "token": token("rowless"),
            "run_keys": [
                {"variant": key.variant, "step": key.step, "row": key.row}
                for key in (RunKey("default", "train", "default"), *evaluations)
            ],
        })
        assert response.status_code == 202, response.text
        jobs = wait_until(lambda: (
            jobs if len(jobs := host.jobs(project=review.project_id)) == 3 and all(job["outcome"] for job in jobs)
            else None
        ))
        assert all(job["state"] == "succeeded" for job in jobs), [job["outcome"] for job in jobs]
        response = client.post("/api/projects/rowless/authoring/analyze", json={
            "revision": review.revision, "key": {"variant": "default", "step": "analyze", "row": "default"},
        })
    assert response.status_code == 200, response.text
    assert "/v1/model/state/plant/effector~1velocity" in response.json()["states"]["variables"]
