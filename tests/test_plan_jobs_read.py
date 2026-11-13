"""Qualify the saved Plan join against the real local Host and training worker."""

from hashlib import sha256
import json
import os
from pathlib import Path
import signal
from statistics import median
import sys
from time import monotonic, perf_counter_ns, sleep

from fastapi.testclient import TestClient
import pytest

from feedbax import jobs
from feedbax.contracts.base import DEFAULT_MANIFEST_ROOT_ENV
from feedbax.contracts.jobs import JobReceipt, JobStatus
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request, encode_request
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.host import AdmissionItem, HostClient, HostError, Job, Launch
from feedbax.job_completion import verify_bound_training_completion
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectRevisionConflict
from tests.support.managed_train_fixture import managed_train_payload, managed_train_project
from tests.test_project_plan import _project_data


def _until(read, ready=bool, *, timeout=34):
    deadline = monotonic() + timeout
    result = None
    while monotonic() < deadline:
        result = read()
        if ready(result):
            return result
        sleep(0.02)
    pytest.fail(f"Host did not reach the expected state. Last observation was {result!r}")


def _write_project(root: Path, spec: DocumentTrainingRequestSpec) -> None:
    assert spec.seed is not None
    project = Project.model_validate({
        "steps": {"fit": {"kind": "train", "base": {"inline": spec.model_dump(mode="json")}}},
        "sweep": {"rows": {
            "original": {},
            "same": {},
            "different": {"patches": {"steps.fit.base.inline.seed": {"value": spec.seed + 1}}},
        }},
    })
    for relative, data in dump_text_project(
        project, ProjectConfiguration(project="reach", store="results")
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _measure(service, revision: str, selection: tuple[RunKey, ...], expected, monkeypatch):
    samples = {name: [] for name in ("baseline", "joined", "compiler", "jobs", "snapshot", "receipt")}

    def timed(function, sample):
        def measure(*args, **kwargs):
            start = perf_counter_ns()
            result = function(*args, **kwargs)
            samples[sample].append((perf_counter_ns() - start) / 1e6)
            return result
        return measure

    base = service.folders.read_plan
    with monkeypatch.context() as patch:
        patch.setattr(service.folders, "read_plan", timed(base, "compiler"))
        patch.setattr(service.jobs, "read_plan_attempts", timed(service.jobs.read_plan_attempts, "jobs"))
        patch.setattr(service.jobs.host, "jobs_snapshot", timed(service.jobs.host.jobs_snapshot, "snapshot"))
        patch.setattr(jobs, "_verified_training_receipt", timed(jobs._verified_training_receipt, "receipt"))
        baseline = timed(base, "baseline")
        joined = timed(service.read, "joined")
        for _ in range(12):
            assert baseline("reach", revision=revision, source_runs=selection) == expected.settled
            assert joined("reach", revision=revision, source_runs=selection) == expected
    return {
        name: {"median_ms": median(values), "samples_ms": values}
        for name, values in samples.items()
    }


def test_plan_join_does_not_start_an_absent_host(tmp_path, monkeypatch):
    host_root = tmp_path / "host"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host_root))
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    projects = tmp_path / "projects"
    (projects / "empty").mkdir(parents=True)
    (projects / "empty" / "empty.feedbax").write_text(
        '{"schema_version":"feedbax.project.v1","store":"results"}'
    )
    with TestClient(create_app(projects_dir=projects)) as client:
        snapshot = client.get("/api/projects/empty").json()
        with pytest.raises(HostError) as error:
            client.app.state.project_plan.read("empty", revision=snapshot["revision"])
        assert error.value.code == "disconnected"
    assert not host_root.exists()
    assert not (projects / "empty" / "results").exists()


def test_saved_train_read_can_include_the_settled_window_map(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    root = projects / "reach"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    data, _ = _project_data()
    managed = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    data["documents"] = managed["documents"]
    data["steps"]["fit"] = managed["steps"]["train"]
    data["steps"]["fit"]["base"]["inline"]["seed"] = 0
    data["steps"]["fit"]["sweep"] = {"axes": {"seed": {"values": {
        "a": {"label": "11", "patches": {"seed": {"value": 11}}},
        "b": {"label": "12", "patches": {"seed": {"value": 12}}},
    }}}}
    data["steps"]["evaluate"]["inputs"]["subject"]["where"] = {"rows": ["seed=a"]}
    data["steps"]["report"] = {
        "kind": "report", "base": {"inline": {}}, "inputs": {"source": {"step": "evaluate"}},
    }
    data["steps"]["broken"] = {"kind": "train", "base": {"document": "missing"}}
    data["steps"]["invalid"] = {"kind": "train", "base": {"inline": {}}}
    for name, content in dump_text_project(
        Project.model_validate(data), ProjectConfiguration(project="reach", store="results")
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    pid = HostClient().ping()["pid"]
    try:
        with TestClient(create_app(projects_dir=projects)) as client:
            revision = client.get("/api/projects/reach").json()["revision"]
            route = "/api/projects/reach/train-runs"
            old = client.get(route, params={"revision": revision})
            projected = client.get(route, params={"revision": revision, "window_map": "true"})
    finally:
        os.kill(pid, signal.SIGTERM)

    assert old.status_code == projected.status_code == 200
    response = projected.json()
    window = response.pop("window_map")
    assert response == old.json()
    assert set(response) == {"revision", "rows"}
    assert window["variants"] == ["default"]
    assert all("evaluation_conditions" not in row for row in window["rows"])
    steps = {step["step_id"]: step for step in window["steps"]}
    assert steps["fit"]["document_ids"] == {
        "payload.trial_documents.model": "point-mass",
        "payload.trial_documents.task": "simple-reaches-task",
        "payload.trial_documents.objective": "simple-reaches-objective",
    }
    assert steps["fit"]["status_counts"] == {"not_run": 2}
    assert steps["broken"]["status_counts"] == {}
    assert steps["invalid"]["status_counts"] == {"cannot_run": 1}
    assert sum(problem["step"] == "invalid" for problem in window["problems"]) > 1
    [subject] = steps["evaluate"]["inputs"]
    assert subject["source_step_id"] == "fit"
    assert subject["subset"]["terms"]["rows"] == ["seed=a"]
    assert (subject["subset"]["count"], subject["subset"]["total"]) == (1, 2)
    assert steps["evaluate"]["status_counts"] == {"cannot_run": 1}
    [evaluation] = window["evaluations"]
    assert evaluation["status"] == "cannot_run" and not evaluation["can_run"]
    assert evaluation["result"] is None
    assert any(
        (problem["step"], problem["path"], problem["code"]) == ("evaluate", "/evaluation_type", "cannot_run")
        for problem in window["problems"]
    )
    assert steps["report"]["inputs"][0]["source_step_id"] == "evaluate"
    fit_a = {"variant": "default", "step": "fit", "row": "seed=a"}
    [edge] = window["edges"]
    assert edge == {
        "from_key": fit_a,
        "to_key": {"variant": "default", "step": "evaluate", "row": "default"},
    }
    row = next(item for item in window["rows"] if item["key"] == fit_a)
    assert row["coordinates"]["seed"]["label"] == "11"
    assert all(item["key"]["step"] == "fit" for item in response["rows"])
    assert any(
        (problem["step"], problem["document"], problem["code"]) == ("broken", "missing", "missing_document")
        for problem in window["problems"]
    )
    assert not (root / "results").exists()


def test_plan_joins_exact_train_attempts_and_receipts_across_reload(tmp_path, monkeypatch, record_property):
    projects = tmp_path / "projects"
    root = projects / "reach"
    store = root / "results"
    host_root = tmp_path / "host"
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host_root))
    monkeypatch.setenv(DEFAULT_MANIFEST_ROOT_ENV, str(store))
    monkeypatch.setenv("FEEDBAX_ORCHESTRATION_ROOT", str(tmp_path / "orchestration"))
    monkeypatch.setenv("FEEDBAX_CONTROLLER_EVENT_LOG", str(tmp_path / "controller.jsonl"))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    canonical = DocumentTrainingRequestSpec.model_validate(managed_train_payload(batches=160) | {"seed": 0})
    _write_project(root, canonical)
    same = RunKey("same", "fit", "default")
    original = RunKey("original", "fit", "default")
    different = RunKey("different", "fit", "default")
    selection = (same,)
    host = HostClient()
    pid = host.ping()["pid"]
    host.policy(compute=1)
    release = tmp_path / "release"
    seen = []

    try:
        with TestClient(create_app(projects_dir=projects)) as client:
            service = client.app.state.project_plan
            opened = client.get("/api/projects/reach").json()
            revision = opened["revision"]

            def read():
                view = service.read("reach", revision=revision, source_runs=selection)
                assert view.settled.revision == revision
                assert view.settled.semantic_root == opened["semantic_root"]
                return view

            before = read()
            plan = before.settled.plan
            ids = {run.key: run.request_id for run in plan.runs}
            request_id = ids[original]
            assert ids[same] == request_id and ids[different] != request_id
            assert set(plan.sources) == {same}
            encoded = encode_request(canonical, {})
            assert (request_id, plan.requests[request_id]) == (encoded.id, encoded.data)
            decoded = decode_request(plan.requests[request_id], plan.documents.__getitem__)
            assert decoded.spec == canonical and decoded.inputs == {}
            assert not before.attempts_by_run and not store.exists()
            assert "post" not in client.app.openapi()["paths"].get("/api/projects/{project_id}/plan", {})

            wrong_store = Job(
                "different-store", "reach", request_id, Launch((sys.executable, "-c", "pass")),
                data={"manifest_root": str(tmp_path / "other-results")},
            )
            host.admit("wrong-store", [AdmissionItem(wrong_store.id, wrong_store)], record={})
            _until(lambda: host.jobs(job_id=wrong_store.id)[0]["outcome"])
            gate = Job(
                "different-project", "other-project", request_id,
                Launch((sys.executable, "-c", (
                    "from pathlib import Path; from time import sleep\n"
                    f"while not Path({str(release)!r}).exists(): sleep(.02)"
                ))),
                data={"manifest_root": str(store)},
            )
            host.admit("slot-gate", [AdmissionItem(gate.id, gate)], record={})
            _until(lambda: host.jobs(job_id=gate.id)[0]["state"] == "running")

            renamed = client.post("/api/projects/reach/rename", json={"revision": revision, "name": "renamed"})
            assert renamed.status_code == 200, renamed.text
            revision = renamed.json()["snapshot"]["revision"]
            renamed_plan = read()
            assert renamed_plan.settled.project_id == before.settled.project_id == "reach"
            assert renamed_plan.settled.store_path == before.settled.store_path
            assert renamed_plan.settled.plan == before.settled.plan
            admission = service.jobs.admit_plan(
                service.folders, before.settled, run_keys=(same,), token="canonical-train"
            )
            (admitted,) = admission.items
            assert admitted.disposition == "admitted"
            attempt_id = admitted.job_id

            def current_attempt():
                view = read()
                assert view.settled == renamed_plan.settled
                assert set(view.attempts_by_run) == {original, same}
                assert view.attempts_by_run[original] == view.attempts_by_run[same]
                attempt, = view.jobs.by_request[request_id]
                assert attempt.job_id == attempt_id
                if not seen or seen[-1] != attempt.status.value:
                    seen.append(attempt.status.value)
                return attempt

            assert current_attempt().status is JobStatus.QUEUED
            release.touch()
            finished = _until(current_attempt, lambda row: row.status in {
                JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CORRUPT,
            })
            canonical_job, = host.jobs(job_id=attempt_id)
            assert finished.status is JobStatus.SUCCEEDED, (finished, canonical_job)
            assert JobStatus.STARTING.value in seen and finished.progress is not None
            assert finished.progress.completed_updates == 160
            canonical_outcome = canonical_job["outcome"]
            completion = canonical_outcome["completion"]
            assert completion["request_id"] == request_id
            assert completion["producer_job_id"] == attempt_id
            verify_bound_training_completion(completion)
            events = host.events(projects=("reach",), limit=1000)["events"]
            kinds = {event["kind"] for event in events if event["job_id"] == attempt_id}
            assert {"queued", "accepted", "binding", "progress", "terminal"} <= kinds

            receipt = JobReceipt.model_validate(canonical_outcome["receipt"])
            assert finished.successful_receipt == receipt
            path = canonical_manifest_path("TrainingRunManifest", receipt.manifest_id, root=store)
            raw = path.read_bytes()
            assert sha256(raw).hexdigest() == receipt.manifest_sha256
            worker_binding = BoundStudioTrainingJobInput.model_validate_json(
                (store / "checkpoints" / attempt_id / "binding.json").read_bytes()
            )
            assert (worker_binding.project_id, worker_binding.manifest_root) == ("reach", store)
            assert worker_binding.request_id == request_id
            assert worker_binding.canonical_request is not None

        with TestClient(create_app(projects_dir=projects)) as client:
            service = client.app.state.project_plan
            start = perf_counter_ns()
            reopened = read()
            cold_ms = (perf_counter_ns() - start) / 1e6
            assert reopened.attempts_by_run[original] == (finished,)
            assert reopened.attempts_by_run[same] == (finished,)
            fingerprint = {file.relative_to(root): file.stat().st_mtime_ns for file in root.rglob("*")}
            timing = _measure(service, revision, selection, reopened, monkeypatch)
            assert fingerprint == {file.relative_to(root): file.stat().st_mtime_ns for file in root.rglob("*")}
            assert not (store / "store.json").exists() and not (store / "refs").exists()

            path.write_bytes(raw + b"\n")
            corrupt, = read().attempts_by_run[same]
            assert corrupt.status is JobStatus.CORRUPT and corrupt.successful_receipt is None
            path.write_bytes(raw)

            failure = Job(
                "later-failure", "reach", request_id, Launch((sys.executable, "-c", "raise SystemExit(7)")),
                data={"manifest_root": str(store)},
            )
            host.admit("later-failure", [AdmissionItem(failure.id, failure)], record={})
            _until(lambda: host.jobs(job_id=failure.id)[0]["state"] == "failed")
            after_failure = read()
            retained, failed = after_failure.attempts_by_run[same]
            assert retained == finished and failed.status is JobStatus.FAILED
            assert failed.successful_receipt is None
            saved = after_failure.history[same]
            assert saved.status is JobStatus.SUCCEEDED and saved.reason
            assert saved.result is not None and saved.result.receipt == receipt
            projected = next(row for row in after_failure.status_rows() if row.key == same)
            assert (projected.status, projected.reason) == (saved.status, saved.reason)

            edit = client.patch("/api/projects/reach", json={
                "revision": revision, "address": {"kind": "step", "id": "fit"},
                "patches": [{"op": "replace", "path": "/base/inline/seed", "value": canonical.seed + 2}],
            })
            assert edit.status_code == 200, edit.text
            with pytest.raises(ProjectRevisionConflict):
                read()
            updated = service.read("reach", revision=edit.json()["revision"], source_runs=selection)
            assert updated.settled.revision == edit.json()["revision"]
            assert not updated.attempts_by_run and request_id not in updated.settled.plan.requests
            record_property("plan_jobs_evidence", json.dumps({
                "revision": revision,
                "edited_revision": updated.settled.revision,
                "canonical_request": request_id,
                "canonical_job": attempt_id,
                "observed_statuses": seen,
                "receipt": receipt.model_dump(mode="json"),
                "receipt_bytes": len(raw),
                "canonical_completion_update": completion["checkpoint"]["completed_updates"],
                "host_cursor": reopened.jobs.cursor,
                "cold_joined_ms": cold_ms,
                "warm_timings": timing,
            }))
    finally:
        release.touch()
        live = [job["id"] for job in host.jobs() if job["outcome"] is None]
        if live:
            host.stop(live, "force")
            _until(lambda: all(job["outcome"] is not None for job in host.jobs()), timeout=8)
        os.kill(pid, signal.SIGTERM)
        _until(lambda: not host.address.exists(), timeout=8)
