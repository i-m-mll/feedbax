"""Exercise saved Run and product custody through the real local Host."""

from base64 import b64encode
from hashlib import sha256
import re

from fastapi.testclient import TestClient

from feedbax.cli import main
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.contracts.task_document import TaskDocument
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import host as host_fixture, wait_until
from tests.test_project_report_plan import _report_project

host = host_fixture


def test_saved_light_chain_reuses_exact_receipts_and_retains_a_stale_figure(host, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1, light=1)
    projects = tmp_path / "projects"
    project = "reaching"
    definition = _report_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = TaskDocument.model_validate(definition["documents"]["task"]["payload"])
    params = definition["steps"]["baseline"]["base"]["inline"]["params"]
    params["trial_set"] = {"kind": "conditions", "ids": list(task.condition_ids[:2])}
    folders = ProjectFolderService(projects)
    _save_project(folders, project, Project.model_validate(definition))
    path = f"/api/projects/{project}"

    def expect(response, code=200):
        assert response.status_code == code, response.text
        return response

    def terminal(count):
        def settled():
            jobs = host.jobs(project=project)
            return jobs if len(jobs) == count and all(job["outcome"] is not None for job in jobs) else None

        jobs = wait_until(settled, timeout=42)
        assert all(job["state"] == "succeeded" for job in jobs), {
            job["request"]: job["outcome"] for job in jobs if job["state"] != "succeeded"
        }
        return jobs

    with TestClient(create_app(projects_dir=projects)) as client:
        def revision():
            return expect(client.get(path)).json()["revision"]

        def run(rev, token, *steps):
            return expect(client.post(f"{path}/train-runs", json={
                "revision": rev, "token": token,
                "run_keys": [{"variant": "default", "step": step, "row": "default"} for step in steps],
            }), 202).json()["items"]

        def projection(rev):
            window = expect(client.get(f"{path}/train-runs", params={
                "revision": rev, "window_map": True,
            })).json()["window_map"]
            elements = {element["name"]: element for element in window["analysis_elements"]}
            (report,) = (row for row in window["rows"] if row["key"]["step"] == "report")
            return elements, report

        rev = revision()
        started = run(rev, "walkthrough", "train", "baseline", "analyze", "report")
        assert [item["disposition"] for item in started] == ["admitted"] * 6, started
        jobs = terminal(6)
        by_id = {job["id"]: job for job in jobs}
        by_kind = {job["request"].partition(":")[0]: job for job in jobs}
        light = [job for job in jobs if job["slot"] == "light"]
        assert len(light) == 4 and len({job["worker_pid"] for job in light}) == 1
        assert all(job["delivery"] == 1 for job in light)
        assert {job["kind"] for job in expect(client.get("/api/jobs", params={"project": project})).json()["jobs"]} == {
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
        assert figure_response.json()["data"]
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
        assert {job["id"] for job in host.jobs(project=project)} == set(by_id)
        assert projection(rev) == (elements, report)

        definition["documents"]["velocity-analysis"]["payload"]["operations"]["summarize"]["params"]["bounds_scale"] = 2.0
        _save_project(folders, project, Project.model_validate(definition))
        edited = revision()
        stale, stale_report = projection(edited)
        assert stale["align"]["result"] == elements["align"]["result"]
        assert stale["summarize"]["reason"] == "Definition changed at payload.operations.summarize.params.bounds_scale."
        assert stale["profile"]["reason"] == "Definition changed at payload.views.profile.inputs.summary."
        assert stale_report["reason"] == "Definition changed at inputs.velocity_summary."
        assert stale["profile"]["status"] == stale_report["status"] == "stale"
        previous_url = stale["profile"]["previous"]["products"]["figure"]
        assert stale["profile"]["previous"]["job_id"] == figure["job_id"]
        assert expect(client.get(previous_url)).content == figure_response.content
        assert main(["status", str(projects / project)]) == 0
        assert stale["summarize"]["reason"] in capsys.readouterr().out

        rerun = run(edited, "scoped-rerun", "analyze", "report")
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
