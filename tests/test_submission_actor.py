"""Saved CLI and Studio submissions retain their admitting clients."""

import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, replace

from fastapi.testclient import TestClient

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.jobs import StudioSubmissionActor
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.host import AdmissionItem
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_train_project
from tests.test_local_job_host import admit, job_row, restart, run_job, wait_until
from tests.test_local_job_host import host as host_fixture


host = host_fixture


def test_saved_cli_actor_and_studio_activity_survive_restart(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    projects = tmp_path / "projects"
    project = "reaching"
    for name, content in dump_text_project(
        managed_train_project(), ProjectConfiguration(project=project, store="results")
    ).items():
        path = projects / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    folders = ProjectFolderService(projects)
    reviewed = folders.read_plan(project, revision=None)
    key = RunKey("default", "train", "default")
    (planned,) = reviewed.plan.runs
    assert planned.key == key

    release = tmp_path / "release"
    host.policy(compute=1)
    gate = run_job(
        "actor-gate",
        f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)",
    )
    admit(host, "gate", gate)
    wait_until(lambda: job_row(host, gate.id, "running"))

    def cli(*flags, env_label=None):
        env = {key: value for key, value in os.environ.items() if key != "FEEDBAX_ACTOR_LABEL"}
        if env_label is not None:
            env["FEEDBAX_ACTOR_LABEL"] = env_label
        result = subprocess.run(
            [sys.executable, "-m", "feedbax.cli", "run", str(projects / project), "train", *flags],
            cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True, timeout=12,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.splitlines()[0].removeprefix("Submission ")

    labelled_id = cli("--label", "explicit actor", env_label="environment actor")
    env_id = cli(env_label="environment actor")
    unlabelled_id = cli()
    jobs = JobService(host, new_application_registry_bundle())
    labelled = jobs.submission(labelled_id)
    (origin,) = labelled.items
    assert origin.disposition == "admitted"
    assert labelled.actor == StudioSubmissionActor(client="cli", label="explicit actor")
    assert jobs.submission(env_id).actor == StudioSubmissionActor(client="cli", label="environment actor")
    assert jobs.submission(unlabelled_id).actor == StudioSubmissionActor(client="cli")
    assert all(
        item.job_id == origin.job_id and item.disposition == "attached"
        for submission_id in (env_id, unlabelled_id) for item in jobs.submission(submission_id).items
    )
    assert jobs.admit_plan(folders, reviewed, run_keys=(key,), token=labelled.token) == labelled
    raw = host.submissions(id=labelled_id)[0]["record"]
    assert "actor" not in raw["command"] and "actor" not in raw["runs"][origin.run_key]
    unlabelled = host.submissions(id=unlabelled_id)[0]["record"]
    assert unlabelled["command"] == raw["command"]
    assert unlabelled["authored_train_plan"] == raw["authored_train_plan"]
    before = host.jobs(job_id=origin.job_id, include_launch=True)[0]
    assert "actor" not in before["launch"]["payload"]
    assert before["request"] == planned.request_id

    legacy = run_job("actor-legacy", "pass", project="legacy")
    legacy_record = {"project": "legacy", "continues": None, "runs": {}}
    legacy_data = {
        "run_key": "old", "stage_id": "old", "scenario_id": "default",
        "label": "Old attempt", "coordinates": {},
    }
    old_submission = host.admit(
        "actor-legacy", [AdmissionItem("old", replace(legacy, data=legacy_data))], record=legacy_record,
    )
    restart(host)
    with TestClient(create_app(projects_dir=projects)) as client:
        assert "actor" not in client.get(f"/api/submissions/{old_submission['id']}").json()
        assert "actor" not in client.get("/api/jobs", params={"project": "legacy"}).json()["jobs"][0]
        stored = client.get(f"/api/submissions/{labelled_id}").json()
        assert stored["actor"] == {"client": "cli", "label": "explicit actor"}
        attached = client.post(f"/api/projects/{project}/train-runs", json={
            "revision": reviewed.revision, "run_keys": [asdict(key)], "token": "studio-attached",
        })
        assert attached.status_code == 202, attached.text
        assert attached.json()["actor"] == {"client": "studio"}
        assert attached.json()["items"][0]["disposition"] == "attached"
        activity = client.get("/api/jobs", params={"project": project}).json()["jobs"]
        assert [(job["job_id"], job["actor"]) for job in activity] == [(origin.job_id, stored["actor"])]
        _, events = jobs.events(after=0, projects=(project,), wait=0, machine=False)
        assert any(event.activity and event.activity.actor == labelled.actor for event in events.events)

        # A queued legacy attempt can receive an attachment from a current client.
        legacy_attachment = host.admit(
            "actor-legacy-attach",
            [AdmissionItem("old", replace(legacy, id="actor-later", data=legacy_data))],
            record=legacy_record | {"actor": {"client": "cli", "label": "Later actor"}},
        )
        assert legacy_attachment["items"][0]["disposition"] == "attached"
        assert jobs.submission(legacy_attachment["id"]).actor.label == "Later actor"
        assert jobs.jobs(project="legacy").jobs[0].actor is None

        release.touch()
        finished = wait_until(lambda: job_row(host, origin.job_id, "succeeded"), timeout=24)
        receipt = finished["outcome"]["receipt"]
        manifest = canonical_manifest_path(
            "TrainingRunManifest", receipt["manifest_id"], root=reviewed.store_path,
        ).read_bytes()
        assert b'"actor"' not in manifest and b'explicit actor' not in manifest
        new = client.post(f"/api/projects/{project}/train-runs", json={
            "revision": reviewed.revision, "run_keys": [asdict(key)], "token": "studio-new",
        })
        assert new.status_code == 202, new.text
        studio = new.json()
        assert studio["actor"] == {"client": "studio"}
        assert studio["items"][0]["disposition"] == "admitted"
        activity = client.get("/api/jobs", params={"project": project}).json()["jobs"]
        assert [(job["request_id"], job["actor"]) for job in activity] == [
            (planned.request_id, stored["actor"]), (planned.request_id, studio["actor"]),
        ]
        wait_until(lambda: job_row(host, studio["items"][0]["job_id"], "succeeded"), timeout=24)
