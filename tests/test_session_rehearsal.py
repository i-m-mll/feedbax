"""Detached M4 rehearsal with the real saved-plan admission, workers and Store."""

from __future__ import annotations

import os
import secrets
import shutil
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
from feedbax.contracts.project_plan import RunKey
from feedbax.host import HostError
from feedbax.jobs import JobService
from feedbax.orchestration.session import (
    KeepOpen, ProviderResource, SessionAuthorization, SessionDriver, SessionError, SessionRequest,
)
from feedbax.orchestration.session_jobs import CollectedAttempt, SessionJobs
from feedbax.orchestration.targets import TargetRegistry
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.session_venue import ENVIRONMENT, LocalVenueProvider, evidence, verify_authorization
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests import test_local_job_host


host_fixture = test_local_job_host.host
wait_until = test_local_job_host.wait_until

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


def _controller(root: Path, provider_root: Path, venue: Path, gate: Path, session_id):
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
    folders = ProjectFolderService(venue / "projects")
    reviewed = folders.read_plan(PROJECT, revision=None)
    try:
        session.delegate(folders, reviewed, run_keys=(TRAIN, EVALUATE), token=TOKEN)
    except ConnectionError as error:
        assert str(error) == "admission reply lost"
    else:
        raise AssertionError("Didn't interrupt the accepted delegation.")


def _terminal_without_rpc(database: Path):
    if not database.exists():
        return None
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT id, state, outcome FROM jobs ORDER BY seq").fetchall()
    return rows if len(rows) == 2 and all(row[2] is not None for row in rows) else None


def test_session_rehearses_saved_train_and_evaluate_after_controller_exit(host_fixture, tmp_path, monkeypatch):
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

        folders = ProjectFolderService(venue / "projects")
        _save_project(folders, PROJECT, Project.model_validate(managed_evaluation_definition()))
        local_folders = ProjectFolderService(tmp_path / "projects")
        shutil.copytree(folders.root / PROJECT, local_folders.root / PROJECT)
        local_plan = local_folders.read_plan(PROJECT, revision=None)
        source = ImmutableArtifactBlobProvider(folders.read_plan(PROJECT, revision=None).store_path)

        gate = tmp_path / "gate"
        gate.mkdir()
        (gate / "session_test_gate.py").write_text(
            """
import os
import time
from pathlib import Path

from feedbax.training import executor

original = executor.execute_managed_training_run

def execute(*args, **kwargs):
    release = Path(os.environ["SESSION_TEST_RELEASE"])
    deadline = time.monotonic() + 15
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Controller did not detach")
        time.sleep(.02)
    return original(*args, **kwargs)

executor.execute_managed_training_run = execute
"""
        )
        process = subprocess.run(
            [sys.executable, "-m", __name__, str(controller_root), str(provider_root),
             str(venue), str(gate), str(request.id)],
            capture_output=True, text=True, timeout=22,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        (gate / "release").touch()
        rows = wait_until(lambda: _terminal_without_rpc(venue / "host" / "host.sqlite"), timeout=31)
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
            response = client.post(f"/api/projects/{PROJECT}/train-runs", json={
                "revision": local_plan.revision, "run_keys": [asdict(TRAIN), asdict(EVALUATE)],
                "token": TOKEN, "target": profile.id,
            })
            assert response.status_code == 409, response.text
        submission = JobService(
            reconnected.host, new_application_registry_bundle(), targets=registry, submit_session=submit_session,
        ).submit_plan(folders, local_plan, run_keys=(TRAIN, EVALUATE), token=TOKEN, target=profile.id)
        assert {item.job_id for item in submission.items} == {train_id, evaluation_id}
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
        assert all(row["data"]["target"] == state.placement.model_dump(mode="json") for row in attempts)
        assert all(SuccessView(source).read_receipt(ref).target == state.placement for ref in receipts.values())
        local_jobs = _jobs(host_fixture)

        def collect(attempt):
            result = local_jobs.collect_session_attempt(source, attempt, project=PROJECT, store=local_plan.store_path)
            assert result.activity.target == state.placement
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

        destination = ImmutableArtifactBlobProvider(local_plan.store_path)
        assert {key: destination.get_object(ref) for key, ref in receipts.items()} == original
        plan = ProjectPlanService(local_folders, local_jobs).read(PROJECT, revision=local_plan.revision)
        (observed,) = plan.attempts_by_run[EVALUATE]
        assert observed.job_id == evaluation_id
        assert observed.checkpoint_source.job_id == train_id
        result = local_jobs.read_evaluation_result(project=PROJECT, store=local_plan.store_path, job_id=evaluation_id)
        with result.open_states(verify=True) as recorded:
            assert recorded.descriptor.dimensions["checkpoint"] == 1
            assert recorded.descriptor.dimensions["trial"] > 0
    finally:
        for resource in provider.inventory():
            provider.remove(resource)


def test_watchdog_closes_venue_at_cost_ceiling_after_host_dies(tmp_path):
    provider = LocalVenueProvider(tmp_path / "provider")
    driver = _driver(tmp_path / "controller", provider)
    request = _request(provider, maximum=2_000)
    state = driver.open(request, _authorization(request), adoption_token=secrets.token_urlsafe())
    assert state.launch.ends_at_ms == state.launch.acquired_at_ms + 2_000
    venue = driver.attach(request.id)
    os.kill(venue.ping()["pid"], signal.SIGKILL)
    wait_until(lambda: not driver.inventory(), timeout=6)
    assert driver.close(request.id).remaining == ()


if __name__ == "__main__":
    from uuid import UUID

    _controller(*(Path(argument) for argument in sys.argv[1:5]), UUID(sys.argv[5]))
