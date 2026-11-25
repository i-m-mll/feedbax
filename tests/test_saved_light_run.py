"""Exercise saved Run and product custody through the real local Host."""

from base64 import b64decode, b64encode
from copy import deepcopy
from hashlib import sha256
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from feedbax.builtins.figure_templates import SpeedByTrial, SpeedByTrialSettings
from feedbax.cli import main
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import ElementKey, RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_figure import FigureProduct
from feedbax.examples import load_shipped_projects
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests.support.host import admit, held_until_released, job_row, run_job, token, wait_until
from tests.support.managed_train_fixture import managed_train_project
from tests.test_project_analyze_plan import _operations
from tests.test_project_report_plan import _report_project

_TRIALS = ["0", "1"]


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
    project["documents"]["velocity-figure"]["payload"]["layers"][0]["style"] = {"label": "{condition}"}
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


def test_saved_light_chain_reuses_exact_receipts_and_retains_a_stale_figure(host, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    project = "reaching"
    definition = _saved_light_definition()
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
        figure = elements["profile"]["result"]
        figure_url = figure["products"]["figure"]
        figure_response = expect(client.get(figure_url))
        assert [trace["name"] for trace in figure_response.json()["data"] if trace.get("showlegend", True)] == [
            "default",
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
        assert stale["align"]["result"] == elements["align"]["result"]
        assert {stale[name]["status"] for name in ("summarize", "profile")} == {stale_report["status"]} == {"stale"}
        assert stale["summarize"]["changed"]["place"] == {"part": "analysis", "element": "summarize"}
        assert "bounds_scale" in stale["summarize"]["changed"]["path"]
        assert stale["profile"]["changed"]["place"] == {"part": "analysis", "element": "profile"}
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

    with TestClient(create_app(projects_dir=projects)) as client:
        def plan(revision):
            return expect(client.get(f"{path}/train-runs", params={"revision": revision, "window_map": True}))

        def replay(revision, job_id, step, code=200):
            return expect(client.post(f"{path}/evaluate-replay", json={
                "revision": revision, "job_id": job_id, "key": key(step), "trial": _TRIALS[0],
            }), code)

        snapshot = expect(client.get(path))
        revision = snapshot["revision"]
        release = tmp_path / "release"
        gate = run_job("gate", held_until_released(release), project=project_id)
        admit(host, "gate", gate)
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
        model_address = {"composites": [], "element": {
            "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
        }}
        edited = expect(client.post(f"{path}/model/edit", json={
            "revision": revision, "presentation_revision": snapshot["presentation_revision"],
            "document": "point-mass", "command": {"operation": {
                "op": "set_field", "address": model_address, "value": 2.5,
            }},
        }))
        assert edited["result"]["status"] == "applied", edited
        changed = expect(client.get(path))["revision"]
        assert changed != revision
        after = plan(changed)
        (old_train,) = after["rows"]
        assert old_train["status"] == "stale"
        assert old_train["request_id"] != old_train["previous"]["request_id"]
        assert old_train["previous"]["job_id"] == train["job_id"]
        assert old_train["previous"]["revision"] == revision

        window = after["window_map"]
        evaluations = {row["key"]["step"]: row for row in window["evaluations"]}
        for step in ("baseline", "copy"):
            row = evaluations[step]
            assert row["status"] == "stale" and not row["can_run"]
            assert row["reason"] == "Fit reaching controller is out of date."
            assert row["result"] is None and row["request_id"] != row["previous"]["request_id"]
            assert row["previous"]["job_id"] == evaluate["job_id"]
            assert row["previous"]["revision"] == revision
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
        assert profile["previous"]["revision"] == revision
        response = client.get(profile["previous"]["products"]["figure"])
        assert response.status_code == 200
        assert response.content == original_figure.content


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
        "operation": "feedbax.trial_speed",
        "params": {"velocity": analysis["operations"]["align"]["params"]["velocity"]},
        "inputs": {"states": {"input": "states"}},
    }
    analysis["views"]["by_trial"] = {
        "figure": SpeedByTrial(SpeedByTrialSettings()).model_dump(mode="json", exclude_unset=True),
        "inputs": {"speed": {"element": "speed", "product": "speed"}},
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
    folders = ProjectFolderService(projects, operations=_operations())
    _save_project(folders, project, Project.model_validate(definition))

    def address(step, row="default"):
        return {"variant": "default", "step": step, "row": row}

    review = folders.read_plan(project, revision=None)
    plan = review.plan
    assert not plan.diagnostics
    analyze = RunKey("default", "analyze", "default")
    assert {element.name for element in plan.elements if element.run == analyze} == {
        "align", "summarize", "profile", "speed", "by_trial",
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
        assert [item["disposition"] for item in response.json()["items"]] == ["admitted"] * 9, response.text

        def terminal():
            jobs = host.jobs(project=review.project_id)
            return jobs if len(jobs) == 9 and all(job["outcome"] is not None for job in jobs) else None

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
        (profile,) = (element["requests"][0] for element in window["analysis_elements"] if element["name"] == "profile")
        figure_ref, rendered_figure = proof(profile["request_id"])
        assert rendered_figure.inputs["summary"][0].selection.receipt == summary_ref
        figure = reopened.get(profile["result"]["products"]["figure"])
        assert figure.status_code == 200, figure.text
        traces = [trace for trace in figure.json()["data"] if trace.get("showlegend", True)]
        assert [trace["name"] for trace in traces] == [f'mass={mass_values[key]["label"]}' for key in order]
        for index, trace in enumerate(traces):
            np.testing.assert_allclose(array(trace["x"]), times[0])
            np.testing.assert_allclose(array(trace["y"]), mean[index, 0, 0])
        (by_trial,) = (
            element["requests"][0] for element in window["analysis_elements"] if element["name"] == "by_trial"
        )
        speed = reopened.get(by_trial["result"]["products"]["figure"])
        assert speed.status_code == 200, speed.text
        product = FigureProduct.model_validate_json(speed.content)
        assert product.layout.meta.panels["speed"].model_dump(exclude_none=True) == {
            "time": {}, "x_axis": "x", "y_axis": "y",
        }
        assert [run.model_dump() for run in product.layout.meta.runs["speed"]] == [
            {"job_id": by_request[request]["id"], "request_id": request}
            for request in evaluation_ids
        ]
        assert [trace.meta.model_dump() for trace in product.data] == [
            {"layer": "trial-speed", "group": {"condition": condition, "trial": trial}}
            for condition in conditions for trial in _TRIALS
        ]
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
