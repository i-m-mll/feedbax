"""Detached M4 rehearsal with the real saved-plan admission, workers and Store."""

from __future__ import annotations

import os
import secrets
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.execution_target import SessionTargetProfile
from feedbax.contracts.project import Project
from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project_io import ProjectConfiguration, project_declaration_name
from feedbax.planner.project_plan import RunKey
from feedbax.host import HostError
from feedbax.jobs import JobService
from feedbax.orchestration.session import (
    KeepOpen, ProviderResource, SessionAuthorization, SessionDriver, SessionError, SessionRequest,
)
from feedbax.orchestration.session_jobs import CollectedAttempt, SessionJobs
from feedbax.orchestration.targets import TargetRegistry
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.services.remote_runs import StudioRemoteRuns, StudioVenue, StudioVenueQuote
from tests.support.session_venue import ENVIRONMENT, LocalVenueProvider, evidence, verify_authorization
from tests.test_canonical_evaluate_admission import managed_evaluation_definition
from tests.support.host import HELD_UNTIL_STOPPED, admit, run_job, until_exists, wait_until
from tests.support.hang import HANG_SECONDS


PROJECT = "reaching"
TRAIN = RunKey("default", "train", "default")
EVALUATE = RunKey("default", "baseline", "default")
TOKEN = "save-train-and-evaluate"
ISSUER_KEY = b"fixture issuing key for the local rehearsal"


class _LostAcquisitionReply(LocalVenueProvider):
    def acquire(self, launch):
        super().acquire(launch)
        raise ConnectionError("provider reply lost")


def _driver(root, provider):
    return SessionDriver(root, provider, verify_authorization=partial(verify_authorization, key=ISSUER_KEY))


def _request(provider, *, maximum=120_000):
    return SessionRequest(
        id=uuid4(), target=provider.target, resources={"compute": 1}, environment=ENVIRONMENT,
        lifecycle=KeepOpen(), maximum_microusd=maximum,
    )


def _authorization(request, **changes):
    fields = {
        "operator": "local-operator", "method": "test-issuer", "request_digest": request.digest,
        "expires_at_ms": time.time_ns() // 1_000_000 + 150_000,
        "ceiling_microusd": request.maximum_microusd,
    } | changes
    return SessionAuthorization(**fields, evidence=evidence(fields, ISSUER_KEY))


def _jobs(client):
    return JobService(client, new_application_registry_bundle())


def _controller(root: Path, provider_root: Path, projects: Path, gate: Path, session_id):
    """Run one real delegation, lose the reply, then exit the controller process."""
    def make_jobs(client):
        call = client._call

        def lose_reply(op, **request):
            result = call(op, **request)
            if op == "admit_with_completion":
                raise ConnectionError("admission reply lost")
            return result

        client._call = lose_reply
        service = _jobs(client)
        launch = service._training_worker
        service._training_worker = replace(
            launch,
            env=launch.env | {"PYTHONPATH": f"{gate}:{Path.cwd()}", "SESSION_TEST_RELEASE": str(gate / "release")},
            preload=(*launch.preload, "session_test_gate"),
        )
        return service

    session = SessionJobs(_driver(root, LocalVenueProvider(provider_root)), session_id, make_jobs)
    folders = ProjectFolderService(projects)
    reviewed = folders.read_plan(PROJECT, revision=None)
    try:
        session.delegate(folders, reviewed, run_keys=(TRAIN, EVALUATE), token=TOKEN)
    except ConnectionError as error:
        assert str(error) == "admission reply lost"
    else:
        raise AssertionError("Didn't interrupt the accepted delegation.")


def _terminal_without_rpc(database: Path, *, count: int = 2):
    if not database.exists():
        return None
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT id, state, outcome FROM jobs ORDER BY seq").fetchall()
    return rows if len(rows) == count and all(row[2] is not None for row in rows) else None


def test_remote_stop_accepts_before_contacting_the_venue(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    provider = LocalVenueProvider(tmp_path / "provider")
    driver = _driver(tmp_path / "controller", provider)
    profile = SessionTargetProfile(
        kind="session", id="cpu-rehearsal", label="CPU rehearsal", driver=provider.target,
        resources={"compute": 1}, lifecycle=KeepOpen(), maximum_microusd=120_000,
    )
    request = TargetRegistry((profile,), (provider,)).session_request(
        profile.id, session_id=uuid4(), environment=ENVIRONMENT,
    )
    accepting = False
    deliveries = []

    def jobs(driver, session_id):
        assert not accepting, "The stop endpoint contacted the venue before acknowledging the request."
        session = SessionJobs(driver, session_id, _jobs)
        send = session.host.stop

        def deliver(job_ids, mode):
            deliveries.append((tuple(job_ids), mode))
            return send(job_ids, mode)

        session.host.stop = deliver
        return session

    remote = StudioRemoteRuns(
        (profile,), (StudioVenue(driver, ENVIRONMENT, jobs, 90, lambda req: StudioVenueQuote(provider.rate)),),
        attest=lambda _, req: _authorization(req),
    )
    venue = None
    ids = []
    try:
        driver.open(request, _authorization(request), adoption_token=secrets.token_urlsafe())
        venue = driver.attach(request.id)
        held = []
        for index in range(3):
            name = f"remote-stop-{index}"
            job = run_job(name, HELD_UNTIL_STOPPED, project=PROJECT, request=f"train:{name}")
            held.append(replace(job, data={
                "run_key": name, "stage_id": "train", "scenario_id": "default", "label": name, "coordinates": {},
            }))
        ids = [job.id for job in held]
        admit(venue, "remote-stop", *held)

        # Advance the same observer the app runs in the background, at deterministic boundaries.
        with TestClient(create_app(projects_dir=tmp_path / "projects")) as client:
            app = client.app
            app.state.remote_runs = remote
            remote.observe(app.state.project_folders, app.state.job_service)
            rows = {row["job_id"]: row for row in client.get("/api/jobs").json()["jobs"] if row["job_id"] in ids}
            assert rows.keys() == set(ids)
            assert all(row["ended_at_ms"] is None for row in rows.values())
            submitted = {id_: row["submitted_at_ms"] for id_, row in rows.items()}
            accepting = True
            body = {"job_ids": ids, "mode": "force"}
            for _ in range(2):
                response = client.post("/api/jobs/stop", json=body)
                assert response.status_code == 202, response.text
                assert [(item["job_id"], item["error"]) for item in response.json()] == [(id_, None) for id_ in ids]
            assert not deliveries

            accepting = False
            remote.observe(app.state.project_folders, app.state.job_service)
            assert deliveries == [(tuple(ids), "force")]
            wait_until(lambda: all(row["state"] == "stopped" for row in venue.jobs()))
            remote.observe(app.state.project_folders, app.state.job_service)
            assert deliveries == [(tuple(ids), "force")]
            rows = {row["job_id"]: row for row in client.get("/api/jobs").json()["jobs"] if row["job_id"] in ids}
            assert {id_: row["status"] for id_, row in rows.items()} == dict.fromkeys(ids, "stopped")
            venue_times = {
                (event["job_id"], event["kind"]): int(event["time"] * 1000)
                for event in venue.events()["events"] if event["job_id"] in ids
            }
            for id_, row in rows.items():
                assert row["submitted_at_ms"] == submitted[id_] == venue_times[id_, "queued"]
                assert row["ended_at_ms"] == venue_times[id_, "terminal"]
    finally:
        remote.stop_collection()
        if venue is not None:
            venue.stop(ids, "force")
            wait_until(lambda: not venue.session_status()["live_jobs"])
            driver.close(request.id)
        for resource in provider.inventory():
            provider.remove(resource)


@pytest.mark.round(
    "A controller process exits after delegating to a venue Host of its own, whose real Train "
    "and Evaluate are collected exactly."
)
def test_session_rehearses_saved_train_and_evaluate_after_controller_exit(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    provider_root = tmp_path / "provider"
    controller_root = tmp_path / "controller"
    provider = _LostAcquisitionReply(provider_root)
    driver = _driver(controller_root, provider)
    profile = SessionTargetProfile(
        kind="session", id="cpu-rehearsal", label="CPU rehearsal", driver=provider.target,
        resources={"compute": 1}, lifecycle=KeepOpen(), maximum_microusd=120_000,
    )
    registry = TargetRegistry((profile,), (provider,))
    request = registry.session_request(profile.id, session_id=uuid4(), environment=ENVIRONMENT)
    authorization = _authorization(request)
    adoption = secrets.token_urlsafe()

    with pytest.raises(SessionError, match="different session"):
        driver.open(request.model_copy(update={"profile": "another-profile"}), authorization, adoption_token=adoption)
    with pytest.raises(SessionError, match="cost ceiling"):
        driver.open(request, _authorization(request, ceiling_microusd=1), adoption_token=adoption)
    with pytest.raises(SessionError, match="issuer authentication"):
        driver.open(request, authorization.model_copy(update={"evidence": "forged"}), adoption_token=adoption)
    assert driver.inventory() == ()

    try:
        with pytest.raises(ConnectionError, match="provider reply lost"):
            driver.open(request, authorization, adoption_token=adoption)
        assert driver.record(request.id).phase == "opening"
        state = driver.open(request, authorization, adoption_token=adoption)
        assert state.phase == "open"
        assert driver.open(request, authorization, adoption_token=adoption) == state
        assert state.venue is not None
        venue = provider.directory(state.venue)
        secret = ProviderResource(id="session-secret", intent=state.launch.intent, kind="secret")
        provider._write_resource(secret)
        foreign = ProviderResource(id="foreign", intent=None, kind="volume")
        provider._write_resource(foreign)
        other = ProviderResource(id="other-session", intent="feedbax-session-" + "f" * 64, kind="venue")
        provider._write_resource(other)

        folders = ProjectFolderService(tmp_path / "projects")
        project_root = folders.root / PROJECT
        authored = Project.model_validate(managed_evaluation_definition())
        configuration = ProjectConfiguration(project=PROJECT, store=str(venue / "results"))
        for name, data in dump_text_project(authored, configuration).items():
            path = project_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        local_plan = folders.read_plan(PROJECT, revision=None)
        project_id = local_plan.project_id
        destination_store = tmp_path / "collected-results"
        source = ImmutableArtifactBlobProvider(local_plan.store_path)

        gate = tmp_path / "gate"
        gate.mkdir()
        (gate / "session_test_gate.py").write_text(
            f"""
import os

from feedbax.training.method_runtime import TrainingLoop

original = TrainingLoop.run_block

def execute(*args, **kwargs):
{until_exists('os.environ["SESSION_TEST_RELEASE"]', indent="    ")}    return original(*args, **kwargs)

TrainingLoop.run_block = execute
"""
        )
        process = subprocess.run(
            [sys.executable, "-m", __name__, str(controller_root), str(provider_root),
             str(folders.root), str(gate), str(request.id)],
            capture_output=True, text=True, timeout=HANG_SECONDS,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        (gate / "release").touch()
        rows = wait_until(lambda: _terminal_without_rpc(venue / "host" / "host.sqlite"))
        assert [row[1] for row in rows] == ["succeeded", "succeeded"], rows
        train_id, evaluation_id = (row[0] for row in rows)

        session = SessionJobs(driver, request.id, _jobs)
        session.replay(folders)
        first = session.poll(limit=1)
        assert len(first["events"]) == 1
        reconnected = SessionJobs(_driver(controller_root, provider), request.id, _jobs)
        second = reconnected.poll(limit=1)
        assert second["events"][0]["cursor"] > first["cursor"]
        def submit_session(selected, app_folders, reviewed, *, run_keys, token, reuse_completed):
            assert selected == profile and not reuse_completed
            return reconnected.delegate(app_folders, reviewed, run_keys=run_keys, token=token)

        with TestClient(create_app(projects_dir=folders.root, targets=registry, submit_session=submit_session)) as client:
            machines = client.get("/api/orchestration/targets").json()
            assert machines["targets"][1]["availability"]["state"] == "off"
            response = client.post(f"/api/projects/{project_id}/train-runs", json={
                "revision": local_plan.revision, "run_keys": [asdict(TRAIN), asdict(EVALUATE)],
                "token": TOKEN, "target": profile.id,
            })
            assert response.status_code == 409, response.text
        submission = JobService(
            reconnected.host, new_application_registry_bundle(), targets=registry, submit_session=submit_session,
        ).submit_plan(folders, local_plan, run_keys=(TRAIN, EVALUATE), token=TOKEN, target=profile.id)
        assert {item.job_id for item in submission.items} == {train_id, evaluation_id}
        assert submission.project_id == project_id
        assert submission.target == state.placement
        assert state.placement.realization.placement == {"device": "cpu"}
        assert len(reconnected.host.jobs()) == 2
        with pytest.raises(HostError) as busy:
            driver.close(request.id)
        assert busy.value.code == "session_busy"

        adopted = _driver(tmp_path / "new-controller", provider)
        assert adopted.adopt(request, authorization, adoption_token=adoption).epoch == state.epoch + 1
        session = SessionJobs(adopted, request.id, _jobs)
        with pytest.raises(HostError) as stale:
            reconnected.host.stop((train_id,), "force")
        assert stale.value.code == "session_fenced"
        with pytest.raises(HostError) as stale:
            driver.close(request.id)
        assert stale.value.code == "session_fenced"
        assert set(provider.inventory()) == {state.venue, secret, other, foreign}

        attempts = session.host.jobs_snapshot(include_launch=True)["jobs"]
        receipts = {
            row["id"]: ContentObjectRef.model_validate(row["outcome"]["store_success"]) for row in attempts
        }
        original = {key: source.get_object(ref) for key, ref in receipts.items()}
        (edge,) = attempts[1]["edges"]
        assert edge["job_id"] == train_id and edge["authenticated_completion"]
        assert edge["binding"]["terminal"]["receipt"] == receipts[train_id].model_dump(mode="json")
        request_ids = {run.key: run.request_id for run in local_plan.plan.runs}
        assert {row["request"] for row in attempts} == {request_ids[TRAIN], request_ids[EVALUATE]}
        assert all(row["project"] == project_id for row in attempts)
        assert all(row["data"]["target"] == state.placement.model_dump(mode="json") for row in attempts)
        assert all(SuccessView(source).read_receipt(ref).target == state.placement for ref in receipts.values())
        local_jobs = _jobs(host)

        def collect(attempt):
            result = local_jobs.collect_session_attempt(
                source, attempt, project=project_id, store=destination_store, project_root=local_plan.project_root,
            )
            assert result.activity.target == state.placement
            assert result.activity.submitted_at_ms == attempt["submitted_at_ms"]
            assert result.activity.ended_at_ms == attempt["ended_at_ms"]
            return CollectedAttempt(result.activity.job_id, result.evidence_sha256)

        ack = session.host.acknowledge_collection

        def lose_ack_reply(job_id, digest):
            ack(job_id, digest)
            raise ConnectionError("acknowledgement reply lost")

        session.host.acknowledge_collection = lose_ack_reply
        with pytest.raises(ConnectionError, match="acknowledgement reply lost"):
            session.collect(collect)
        assert session.host.session_status()["collections"] == {train_id: receipts[train_id].sha256}
        resumed = SessionJobs(_driver(tmp_path / "new-controller", provider), request.id, _jobs)
        assert {item.attempt_id: item.sha256 for item in resumed.collect(collect)} == {
            key: ref.sha256 for key, ref in receipts.items()
        }
        assert not resumed.host.session_status()["pending"]

        make_client = provider.client

        def client_with_lost_close_reply(resource, access):
            client = make_client(resource, access)
            claim = client.claim_session_close

            def lose_close_reply():
                claim()
                raise ConnectionError("close claim reply lost")

            client.claim_session_close = lose_close_reply
            return client

        with monkeypatch.context() as faults:
            faults.setattr(provider, "client", client_with_lost_close_reply)
            with pytest.raises(ConnectionError, match="close claim reply lost"):
                adopted.close(request.id)
        assert adopted.record(request.id).phase == "open"
        assert set(provider.inventory()) == {state.venue, secret, other, foreign}

        recovered = _driver(tmp_path / "recovered-controller", provider)
        with pytest.raises(SessionError, match="adoption capability"):
            recovered.recover_close(request, authorization, adoption_token=secrets.token_urlsafe())
        remove = provider.remove
        with monkeypatch.context() as faults:
            faults.setattr(provider, "remove", lambda item: None if item == secret else remove(item))
            with pytest.raises(SessionError, match="Owned resources remain"):
                recovered.recover_close(request, authorization, adoption_token=adoption)
        assert recovered.record(request.id).close_evidence.remaining == (secret,)
        evidence = recovered.close(request.id)
        assert set(evidence.owned) == {state.venue, secret}
        assert not evidence.remaining
        assert recovered.close(request.id) == evidence
        assert set(provider.inventory()) == {other, foreign}
        assert not venue.exists()

        destination = ImmutableArtifactBlobProvider(destination_store)
        assert {key: destination.get_object(ref) for key, ref in receipts.items()} == original
        declaration = project_declaration_name(PROJECT)
        local_configuration = configuration.model_copy(update={"store": str(destination_store)})
        (project_root / declaration).write_bytes(dump_text_project(authored, local_configuration)[declaration])
        collected_plan = folders.read_plan(project_id, revision=None)
        assert {run.key: run.request_id for run in collected_plan.plan.runs} == request_ids
        plan = ProjectPlanService(folders, local_jobs).read(project_id, revision=collected_plan.revision)
        (observed,) = plan.attempts_by_run[EVALUATE]
        assert observed.job_id == evaluation_id
        assert observed.checkpoint_source.job_id == train_id
        result = local_jobs.read_evaluation_result(project=project_id, store=destination_store, job_id=evaluation_id)
        with result.open_states(verify=True) as recorded:
            assert recorded.descriptor.dimensions["checkpoint"] == 1
            assert recorded.descriptor.dimensions["trial"] > 0
    finally:
        for resource in provider.inventory():
            provider.remove(resource)


@pytest.mark.round(
    "The venue's Host is killed, and the watchdog closes the venue at a real cost ceiling some seconds later."
)
def test_watchdog_closes_venue_at_cost_ceiling_after_host_dies(tmp_path):
    provider = LocalVenueProvider(tmp_path / "provider")
    driver = _driver(tmp_path / "controller", provider)
    # Fixed when the venue is acquired, the ceiling must cover the venue Host's start, up to about
    # 2.5 s on a loaded machine.
    request = _request(provider, maximum=6_000)
    state = driver.open(request, _authorization(request), adoption_token=secrets.token_urlsafe())
    assert state.launch.ends_at_ms == state.launch.acquired_at_ms + 6_000
    venue = driver.attach(request.id)
    os.kill(venue.ping()["pid"], signal.SIGKILL)
    wait_until(lambda: not driver.inventory())
    assert driver.close(request.id).remaining == ()


if __name__ == "__main__":
    from uuid import UUID

    _controller(*(Path(argument) for argument in sys.argv[1:5]), UUID(sys.argv[5]))
