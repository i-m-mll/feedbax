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
from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    StudioGraphMethodPayload,
    StudioTrainingJobInput,
)
from feedbax.contracts.training import TrainingRunSpec
from feedbax.host import AdmissionItem, HostClient, HostError, Job, Launch
from feedbax.job_completion import verify_bound_training_completion
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.training_requests import lower_studio_training_runs
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectRevisionConflict
from web.e2e.studio.fixtures.reaching_project import reaching_project


def _until(read, ready=bool, *, timeout=34):
    deadline = monotonic() + timeout
    result = None
    while monotonic() < deadline:
        result = read()
        if ready(result):
            return result
        sleep(0.02)
    pytest.fail(f"Host did not reach the expected state. Last observation was {result!r}")


def _write_project(root: Path, spec: TrainingRequestSpec) -> None:
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


def _fixture_request(spec: TrainingRunSpec) -> TrainingRequestSpec:
    assert spec.graph.inline is not None and spec.graph.ref is None
    assert spec.method_ref == STUDIO_GRAPH_METHOD
    assert spec.worker_execution.mapping_levels is None and spec.worker_execution.resume is None
    assert spec.checkpoint_progress.resume_from is None
    assert spec.checkpoint_progress.continuation is None
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    method.require_supported_launch(resume=False)
    return TrainingRequestSpec.model_validate({
        "graph": spec.graph.inline,
        "task": spec.task,
        "task_binding_spec": spec.task_binding_spec,
        "training_config": spec.training_config,
        "objective": spec.objective,
        "method_ref": spec.method_ref,
        "method_payload": spec.method_payload,
        "risk_aggregation": spec.risk_aggregation,
        "on_nan": spec.on_nan,
        "seed": method.seed,
    })


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
    (projects / "empty" / "feedbax.project.json").write_text(
        '{"schema_version":"feedbax.project.v1","project":"empty","store":"results"}'
    )
    with TestClient(create_app(projects_dir=projects)) as client:
        snapshot = client.get("/api/projects/empty").json()
        with pytest.raises(HostError) as error:
            client.app.state.project_plan.read("empty", revision=snapshot["revision"])
        assert error.value.code == "disconnected"
    assert not host_root.exists()
    assert not (projects / "empty" / "results").exists()


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
    authored = reaching_project(batches=160)
    lowered, = lower_studio_training_runs(
        authored.workspace, authored.graph, stage_id="stage:train",
        registry_bundle=new_application_registry_bundle(),
    )
    canonical = _fixture_request(lowered.spec)
    assert canonical.seed is not None
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

            response = client.post("/api/submissions", json={
                "project_id": "reach",
                "workspace": authored.workspace.model_dump(mode="json", exclude_none=True),
                "graph": authored.graph.model_dump(mode="json", exclude_none=True),
                "selections": [{"stage_id": "stage:train"}],
                "token": "normal-studio-train",
                "policy": "all",
            })
            assert response.status_code == 202, response.text
            legacy_item, = response.json()["items"]
            assert legacy_item["disposition"] == "admitted"
            legacy_id = legacy_item["job_id"]
            legacy_job, = host.jobs(job_id=legacy_id, include_launch=True)
            assert legacy_job["request"] == lowered.request_id != request_id
            assert legacy_job["state"] == "queued"
            assert not read().attempts_by_run

            launch = Launch(**legacy_job["launch"])
            bound = StudioTrainingJobInput.model_validate(launch.payload)
            assert bound.spec == lowered.spec and bound.resume is None
            assert (bound.project_id, bound.manifest_root) == ("reach", store)
            assert encode_request(_fixture_request(bound.spec), {}).data == encoded.data

            admission = service.jobs.admit_plan(
                service.folders, before.settled, run_keys=(same,), token="canonical-train"
            )
            (admitted,) = admission.items
            assert admitted.disposition == "admitted"
            attempt_id = admitted.job_id

            def current_attempt():
                view = read()
                assert view.settled == before.settled
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
            legacy_finished, = host.jobs(job_id=legacy_id)
            assert legacy_finished["state"] == "succeeded"
            legacy_outcome = legacy_finished["outcome"]
            legacy_receipt = JobReceipt.model_validate(legacy_outcome["receipt"])
            legacy_completion = legacy_outcome["completion"]
            assert legacy_completion["request_id"] == lowered.request_id
            assert legacy_completion["receipt"] == legacy_outcome["receipt"]
            assert legacy_completion["checkpoint"]["completed_updates"] == 160

            canonical_outcome = canonical_job["outcome"]
            completion = canonical_outcome["completion"]
            assert completion["request_id"] == request_id
            assert completion["producer_job_id"] == attempt_id
            verify_bound_training_completion(completion)
            legacy_path = canonical_manifest_path(
                "TrainingRunManifest", legacy_receipt.manifest_id, root=store
            )
            assert sha256(legacy_path.read_bytes()).hexdigest() == legacy_receipt.manifest_sha256
            events = host.events(projects=("reach",), limit=1000)["events"]
            for job_id in (legacy_id, attempt_id):
                kinds = {event["kind"] for event in events if event["job_id"] == job_id}
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
            retained, failed = read().attempts_by_run[same]
            assert retained == finished and failed.status is JobStatus.FAILED
            assert failed.successful_receipt is None

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
                "legacy_request": lowered.request_id,
                "legacy_job": legacy_id,
                "canonical_job": attempt_id,
                "observed_statuses": seen,
                "receipt": receipt.model_dump(mode="json"),
                "receipt_bytes": len(raw),
                "legacy_receipt": legacy_receipt.model_dump(mode="json"),
                "legacy_completion_update": legacy_completion["checkpoint"]["completed_updates"],
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
