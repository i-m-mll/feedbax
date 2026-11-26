"""Restart the controller across the durable Jobs approval boundary."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import perf_counter_ns
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from feedbax.contracts.execution_target import KeepOpen, SessionTargetProfile
from feedbax.contracts.jobs import StudioJobSubmission
from feedbax.contracts.project import Project
from feedbax.desktop.attestation import DesktopApproval
from feedbax.desktop.control import write_private_json
from feedbax.desktop.release import AppPaths, Release
from feedbax.desktop import remote_runs as installed_remote
from feedbax.orchestration.repo_snapshot import seal_repo_snapshot
from feedbax.orchestration.runpod import SubprocessRunPodTransport
from feedbax.orchestration.runpod_session import RunPodEnvironment
from feedbax.orchestration.session import ProviderResource, SessionDriver, SessionError, SessionOpeningFailed
from feedbax.orchestration.session_jobs import SessionJobs
from feedbax.orchestration.venue_jobs import VenueJobs
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.remote_runs import StudioRemoteRuns, StudioVenue, StudioVenueQuote
from tests.support.git_fixtures import run_fixture_git
from tests.support.runpod_session import FakeRunPod, runpod_resources
from tests.support.session_venue import ENVIRONMENT, LocalVenueProvider
from tests.support.store_transport import loopback_store
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project
from tests.support.host import wait_until
from tests.test_session_rehearsal import EVALUATE, PROJECT, TRAIN
from tests.support.hang import HANG_SECONDS


PROFILE = "recovery"
RUNS = (TRAIN, EVALUATE)
API = f"/api/projects/{PROJECT}/train-runs"
REFUSED = "The original machine refused initialization."


def _park(root: Path) -> None:
    (root / "boundary").touch()
    signal.pause()
    raise AssertionError("The controller must be killed by the test.")


class _Provider(LocalVenueProvider):
    def __init__(self, root: Path, action: str):
        super().__init__(root / "venues")
        self.action = action

    def acquire(self, launch):
        resource = super().acquire(launch)
        if self.action == "after-acquisition":
            _park(self.root.parent)
        return resource

    def realize(self, *args, **kwargs):
        if self.action == "refuse":
            raise SessionOpeningFailed(REFUSED)
        return super().realize(*args, **kwargs)


def _app(root: Path, action: str, native_proof=None):
    provider = _Provider(root, action)

    def attest(_launcher, challenge, _reason):
        assert action in ("after-acquisition", "before-delegation"), "Recovery must use the original approval."
        return challenge

    approval = DesktopApproval(root / "approval.json", root / "launcher", native_proof=native_proof or attest)
    driver = SessionDriver(root / "controller", provider, verify_authorization=approval.verify)
    transport = loopback_store(root / "transfer", root / f"ssh-{action}").endpoint.connection
    profile = SessionTargetProfile(
        kind="session", id=PROFILE, label="Recovery machine", driver=provider.target,
        resources={"compute": 1}, lifecycle=KeepOpen(), maximum_microusd=90_000,
    )

    def jobs(driver, session_id):
        if action == "before-delegation":
            _park(root)
        venue = driver.record(session_id).venue
        assert venue is not None
        directory = provider.directory(venue)
        return SessionJobs(driver, session_id, venue_jobs=VenueJobs(
            transport, python=sys.executable, cwd=Path.cwd(), host_root=directory / "host",
            projects_root=directory / "projects", stores_root=directory / "stores",
        ))

    venue = StudioVenue(driver, ENVIRONMENT, jobs, 90, lambda request: StudioVenueQuote(
        driver.provider.hourly_microusd(request),
    ))
    remote = StudioRemoteRuns((profile,), (venue,), attest=approval)
    return create_app(projects_dir=root / "projects", remote_runs=remote), driver


def _first_controller(root: Path, action: str) -> None:
    app, driver = _app(root, action)
    with TestClient(app) as client:
        reviewed = app.state.project_folders.read_plan(PROJECT, revision=None)
        quote = client.post(f"{API}/quote", json={
            "revision": reviewed.revision, "target": PROFILE, "run_keys": [asdict(key) for key in RUNS],
        })
        quote.raise_for_status()
        assert all(resource.intent is None for resource in driver.inventory())
        quote_id = quote.json()["id"]
        (root / "quote-id").write_text(quote_id)
        accepted = client.post(f"{API}/approve", json={"quote": quote_id})
        assert accepted.status_code == 202 and accepted.json() == {"session_id": quote_id}
        (root / "acknowledged").touch()
        signal.pause()
    raise AssertionError("The first controller must be killed during recovery.")


def _kill_controller(root: Path, action: str) -> UUID:
    output = root / "first-controller.log"
    with output.open("wb") as log, subprocess.Popen(
        [sys.executable, "-m", "tests.test_approved_session_recovery", str(root), action],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
    ) as child:
        def reached_boundary():
            assert child.poll() is None, output.read_text()
            return (root / "boundary").is_file() and (root / "acknowledged").is_file()

        try:
            wait_until(reached_boundary)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=HANG_SECONDS)
        assert child.returncode == -signal.SIGKILL, output.read_text()
    return UUID((root / "quote-id").read_text())


def _local_project(tmp_path: Path, monkeypatch, host):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host.root))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    folders = ProjectFolderService(tmp_path / "projects")
    raw = document_project(authored_documents(), evaluation=True)
    _save_project(folders, PROJECT, Project.model_validate(raw))
    foreign = ProviderResource(id="foreign", kind="venue", intent=None)
    LocalVenueProvider(tmp_path / "venues")._write_resource(foreign)
    return folders, foreign


def _approval_row(driver, quote_id):
    with driver.connection() as db:
        return db.execute(
            """SELECT d.submission, a.failure FROM session_openings a
               JOIN session_delegations d ON d.session=a.session AND d.token=a.session WHERE a.session=?""",
            (str(quote_id),),
        ).fetchone()


def test_native_refusal_and_edit_during_approval_leave_no_admission(tmp_path, monkeypatch, host):
    folders, foreign = _local_project(tmp_path, monkeypatch, host)
    reviewed = folders.read_plan(PROJECT, revision=None)
    challenges = []

    def native_proof(launcher, challenge, reason):
        assert launcher == tmp_path / "launcher" and "Approve 2 runs" in reason
        challenges.append(challenge)
        if len(challenges) == 1:
            raise SessionError("Touch ID was cancelled.")
        if len(challenges) == 2:
            return challenges[0]
        changed_path = folders.root / PROJECT / "steps/train.json"
        document = json.loads(changed_path.read_bytes())
        document["base"]["inline"]["settings"]["optimizer"] = {
            "type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.002},
        }
        changed_path.write_text(json.dumps(document))
        return challenge

    app, driver = _app(tmp_path, "recover", native_proof)
    with TestClient(app) as client:
        quote = client.post(f"{API}/quote", json={
            "revision": reviewed.revision, "target": PROFILE, "run_keys": [asdict(key) for key in RUNS],
        })
        quote.raise_for_status()
        quote_id = quote.json()["id"]
        assert challenges == [] and _approval_row(driver, quote_id) is None
        assert driver.inventory() == (foreign,)

        untrusted = client.post(f"{API}/approve", json={"quote": quote_id, "evidence": "browser-proof"})
        assert untrusted.status_code == 422 and challenges == []
        for _ in range(3):
            assert client.post(f"{API}/approve", json={"quote": quote_id}).status_code == 409
            assert _approval_row(driver, quote_id) is None
            assert driver.inventory() == (foreign,)
        assert len(set(challenges)) == 3
        assert client.post(f"{API}/approve", json={"quote": quote_id}).status_code == 409
        assert len(challenges) == 3


def test_staging_one_hundred_saved_runs_replies_before_provider_acquisition(
    tmp_path, monkeypatch, host, record_property,
):
    folders, foreign = _local_project(tmp_path, monkeypatch, host)
    raw = document_project(authored_documents())
    base = raw["steps"].pop("train")
    for index in range(100):
        step = deepcopy(base)
        step["base"]["inline"]["settings"]["optimizer"]["learning_rate"]["value"] += index / 1e6
        raw["steps"][f"train-{index}"] = step
    _save_project(folders, PROJECT, Project.model_validate(raw))
    reviewed = folders.read_plan(PROJECT, revision=None)
    selected = [run for run in reviewed.plan.runs if run.key.step.startswith("train-")]
    assert len(selected) == 100
    attested_ns = None
    staged_ns = None
    quote_id = None

    def native_proof(_launcher, challenge, _reason):
        nonlocal attested_ns
        attested_ns = perf_counter_ns()
        return challenge

    app, driver = _app(tmp_path, "refuse", native_proof)

    async def check_response(scope, receive, send):
        async def observe(message):
            nonlocal staged_ns
            if scope.get("path") == f"{API}/approve" and message["type"] == "http.response.body":
                staged_ns = perf_counter_ns()
                assert _approval_row(driver, quote_id) == (None, None)
                assert not await asyncio.to_thread(
                    app.state.remote_runs.approve, app.state.project_folders, PROJECT, UUID(quote_id),
                )
                await asyncio.to_thread(app.state.remote_runs.advance_openings, app.state.project_folders)
                assert driver.records() == () and driver.inventory() == (foreign,)
            await send(message)
        await app(scope, receive, observe)

    with TestClient(check_response) as client:
        quote = client.post(f"{API}/quote", json={
            "revision": reviewed.revision, "target": PROFILE,
            "run_keys": [asdict(run.key) for run in selected],
        })
        quote.raise_for_status()
        quote_id = quote.json()["id"]
        reply = client.post(f"{API}/approve", json={"quote": quote_id})
        assert reply.status_code == 202 and reply.json() == {"session_id": quote_id}
        assert staged_ns is not None and attested_ns is not None
        elapsed_ms = (staged_ns - attested_ns) / 1e6
        record_property("approval_stage_ms", round(elapsed_ms, 3))
        with driver.connection() as db:
            (stored,) = db.execute("SELECT request FROM session_delegations WHERE session=?", (quote_id,)).fetchone()
        delegation = json.loads(stored)
        assert delegation["run_keys"] == [asdict(run.key) for run in selected]
        assert delegation["request_ids"] == [run.request_id for run in selected]
        assert wait_until(lambda: _approval_row(driver, quote_id)[1]) == REFUSED


@pytest.fixture
def installed_runpod(tmp_path, monkeypatch, host):
    folders, _ = _local_project(tmp_path, monkeypatch, host)
    source = tmp_path / "source"
    package = source / "feedbax"
    package.mkdir(parents=True)
    run_fixture_git(source, "init")
    (source / "pyproject.toml").write_text('[project]\nname = "feedbax"\nversion = "0.0.0"\n')
    (package / "__init__.py").touch()
    run_fixture_git(source, "add", "pyproject.toml", "feedbax/__init__.py")
    run_fixture_git(source, "commit", "-m", "Fixture source")
    snapshot = seal_repo_snapshot("feedbax", source, snapshot_parent=tmp_path / "snapshots")
    toolchain = b"isolated Linux uv fixture"
    environment = RunPodEnvironment(
        workspace="/workspace", bootstrap_python="/usr/bin/python3", uv_sha256=hashlib.sha256(toolchain).hexdigest(),
        uv_version="0.11.0", snapshot=snapshot.record, backend="gpu",
    )
    resources = runpod_resources()
    fake = FakeRunPod(resources)

    def provider_response(request):
        if request.url.host == "api.runpod.io":
            assert request.headers["Authorization"] == "Bearer fixture-key"
        return fake.handle(request)

    monkeypatch.setattr(
        installed_remote, "SubprocessRunPodTransport",
        lambda: SubprocessRunPodTransport(http_transport=httpx.MockTransport(provider_response)),
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "unselected-fixture-key")
    paths = AppPaths.resolve(instance_root=tmp_path / "installed")
    paths.control.mkdir(mode=0o700, parents=True)
    launcher = tmp_path / "Contents" / "MacOS" / "feedbax-studio"
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    venue = launcher.parent.parent / "Resources" / "runpod"
    venue.mkdir(parents=True)
    shutil.copytree(snapshot.staging_root, venue / "source")
    (venue / "environment.json").write_text(environment.model_dump_json())
    (venue / "uv").write_bytes(toolchain)
    release = Release(launcher.parent.parent, {
        "source_commit": snapshot.record.commit, "resources": {
            "launcher": "MacOS/feedbax-studio", "runpod_environment": "Resources/runpod/environment.json",
            "runpod_uv": "Resources/runpod/uv", "runpod_source": "Resources/runpod/source",
        },
    }, "fixture")
    assert installed_remote.load_remote_runs(paths, release) is None
    credentials = tmp_path / "account.toml"
    credentials.write_text('apikey = "fixture-key"\n')
    identity = tmp_path / "ssh-identity"
    identity.touch()
    profile = SessionTargetProfile(
        kind="session", id=PROFILE, label="RunPod test", driver="runpod",
        resources=resources.model_dump(mode="json") | {"datacenter": "US-TEST-1"},
        lifecycle=KeepOpen(), maximum_microusd=90_000,
    )
    write_private_json(paths.control / "runpod.json", {
        "schema_id": "feedbax.desktop.runpod.v1", "profiles": [profile.model_dump(mode="json")],
        "authorization_seconds": 120, "runpod_config": str(credentials),
        "ssh_identity": str(identity),
    })
    return folders, paths, release, fake, credentials


def test_installed_runpod_composes_the_real_read_only_quote(installed_runpod):
    folders, paths, release, fake, credentials = installed_runpod
    resources = fake.resources
    reviewed = folders.read_plan(PROJECT, revision=None)
    other_release = Release(release.contents, release.document | {"source_commit": "0" * 40}, "other")
    with pytest.raises(ValueError, match="must match the installed Studio source commit"):
        installed_remote.load_remote_runs(paths, other_release)
    remote = installed_remote.load_remote_runs(paths, release)
    assert remote is not None
    with TestClient(create_app(projects_dir=folders.root, remote_runs=remote)) as client:
        quote = client.post(f"{API}/quote", json={
            "revision": reviewed.revision, "target": PROFILE, "run_keys": [asdict(key) for key in RUNS],
        })
        quote.raise_for_status()
        assert quote.json()["hourly_microusd"] == resources.hourly_microusd
        assert "datacenter" not in quote.json()["resources"]
        assert {key: quote.json()["resources"][key] for key in ("region", "stock", "cloud")} == {
            "region": "Any", "stock": "Low", "cloud": "Secure",
        }
        assert fake.requests == [("GET", f"/v2/catalog/gpus/{resources.gpu_id}")]
        fake.quoted_rate = 0.76
        refused = client.post(f"{API}/quote", json={
            "revision": reviewed.revision, "target": PROFILE, "run_keys": [asdict(key) for key in RUNS],
        })
        assert refused.status_code == 409
        assert all(method == "GET" for method, _ in fake.requests)
        assert not fake.pods and not fake.secrets
    credentials.unlink()
    unavailable = installed_remote.load_remote_runs(paths, release)
    assert unavailable is not None
    configured = next(choice for choice in unavailable.catalog(("train",)).targets if choice.profile.id == PROFILE)
    assert configured.availability.state != "ready"
    assert len(fake.requests) == 2


def test_plain_text_runpod_rejection_becomes_a_failed_saved_run(installed_runpod, monkeypatch):
    folders, paths, release, fake, _ = installed_runpod
    reviewed = folders.read_plan(PROJECT, revision=None)
    handle = fake.handle
    expected = "RunPod HTTP 400: The container entrypoint exceeds the allowed length for [redacted]."

    def reject_pod(request):
        if (request.method, request.url.path) != ("POST", "/v2/pods"):
            return handle(request)
        fake.requests.append((request.method, request.url.path))
        assert "dataCenterIds" not in json.loads(request.content)
        return httpx.Response(
            400, text="The container entrypoint exceeds the allowed length for fixture-key.",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

    monkeypatch.setattr(fake, "handle", reject_pod)
    monkeypatch.setattr(installed_remote, "DesktopApproval", lambda key_path, launcher: DesktopApproval(
        key_path, launcher, native_proof=lambda _launcher, challenge, _reason: challenge,
    ))
    remote = installed_remote.load_remote_runs(paths, release)
    assert remote is not None
    with TestClient(create_app(projects_dir=folders.root, remote_runs=remote)) as client:
        quote = client.post(f"{API}/quote", json={
            "revision": reviewed.revision, "target": PROFILE, "run_keys": [asdict(TRAIN)],
        })
        quote.raise_for_status()
        session_id = quote.json()["id"]
        assert client.post(f"{API}/approve", json={"quote": session_id}).status_code == 202

        def failed_row():
            plan = client.get(API, params={"revision": reviewed.revision, "window_map": "true"})
            plan.raise_for_status()
            row = next(row for row in plan.json()["window_map"]["rows"] if row["key"] == asdict(TRAIN))
            return row if row["status"] == "failed" else None

        row = wait_until(failed_row)
        assert row["reason"] == expected
        assert row["opening_step"] == "acquiring"
        sessions = client.get("/api/sessions")
        sessions.raise_for_status()
        machine = next(machine for machine in sessions.json() if machine["id"] == session_id)
        assert machine["state"] == "off"
        assert fake.requests.count(("POST", "/v2/pods")) == 1
        assert not fake.pods and not fake.secrets


@pytest.mark.round(
    "A controller process is killed after approval, and its restart delivers the selection to a"
    " venue Host with real workers."
)
def test_controller_restart_delivers_approved_selection_on_the_same_machine(tmp_path, monkeypatch, host):
    from feedbax.contracts.studio_project import StudioProjectRename

    folders, foreign = _local_project(tmp_path, monkeypatch, host)
    authored = folders.read(PROJECT)
    renamed = folders.rename(PROJECT, StudioProjectRename(revision=authored.revision, name="renamed_experiment"))
    assert renamed.snapshot.project_id == authored.project_id
    reviewed = folders.read_plan(PROJECT, revision=None)
    files = folders.read_authored_snapshot(PROJECT, revision=reviewed.revision).files
    requests = {run.key: str(run.request_id) for run in reviewed.plan.runs}

    quote_id = _kill_controller(tmp_path, "before-delegation")
    app, restarted = _app(tmp_path, "recover")
    opened = restarted.record(quote_id)
    assert opened.phase == "open" and opened.venue is not None
    venue = restarted.provider.directory(opened.venue)

    try:
        assert _approval_row(restarted, quote_id) == (None, None)
        assert restarted.attach(quote_id).jobs() == []
        changed = "steps/train.json"
        changed_path = folders.root / PROJECT / changed
        document = json.loads(files[changed])
        document["base"]["inline"]["seed"] = 17
        changed_path.write_text(json.dumps(document))
        edited = folders.read_plan(PROJECT, revision=None)
        assert edited.revision != reviewed.revision
        assert str(next(run.request_id for run in edited.plan.runs if run.key == TRAIN)) != requests[TRAIN]

        with TestClient(app) as client:
            accepted = wait_until(lambda: _approval_row(restarted, quote_id)[0])
            submission = StudioJobSubmission.model_validate_json(accepted)
            reply = client.post(f"{API}/approve", json={"quote": str(quote_id)})
            assert reply.status_code == 202 and reply.json() == {"session_id": str(quote_id)}
            assert submission.target == opened.placement
            assert submission.project_id == reviewed.project_id
            assert [item.disposition for item in submission.items] == ["admitted", "admitted"]
            venue_host = restarted.attach(quote_id)
            assert len(venue_host.submissions(token=str(quote_id))) == 1
            assert [row["request"] for row in venue_host.jobs()] == [requests[key] for key in RUNS]
            installed = venue / "projects" / reviewed.revision / reviewed.project_id
            assert {name: (installed / name).read_bytes() for name in files} == files
            assert (venue / "stores" / reviewed.project_id).is_dir()
            assert set(restarted.inventory()) == {opened.venue, foreign}
            assert list((tmp_path / "venues").glob(".used-*")) == [tmp_path / "venues" / f".used-{opened.venue.id}"]

            def acknowledged_both():
                collections = venue_host.session_status()["collections"]
                return collections if len(collections) == 2 else None

            acknowledged = wait_until(acknowledged_both)
            local = SuccessView(ImmutableArtifactBlobProvider(reviewed.store_path))
            for item in submission.items:
                ref = local.attempt_ref(item.job_id)
                assert ref is not None and ref.sha256 == acknowledged[item.job_id]
            assert restarted.close(quote_id).remaining == ()
    finally:
        restarted.provider.remove(opened.venue)
    assert restarted.inventory() == (foreign,)


def test_recovered_opening_failure_closes_the_original_machine(tmp_path, monkeypatch, host):
    folders, foreign = _local_project(tmp_path, monkeypatch, host)
    quote_id = _kill_controller(tmp_path, "after-acquisition")
    challenges = []

    def attest(_launcher, challenge, _reason):
        challenges.append(challenge)
        return challenge

    app, restarted = _app(tmp_path, "refuse", attest)
    original = restarted.record(quote_id)
    assert original.phase == "opening" and original.venue is None
    assert _approval_row(restarted, quote_id) == (None, None)
    (owned,) = [item for item in restarted.inventory() if item.intent == original.launch.intent]

    with TestClient(app) as client:
        failure = wait_until(lambda: _approval_row(restarted, quote_id)[1])
        reply = client.post(f"{API}/approve", json={"quote": str(quote_id)})
        assert reply.status_code == 202 and reply.json() == {"session_id": str(quote_id)} and failure == REFUSED
        closed = restarted.record(quote_id)
        assert closed.phase == "closed" and closed.venue == owned
        assert closed.close_evidence is not None
        assert closed.close_evidence.owned == (owned,) and closed.close_evidence.remaining == ()
        assert restarted.inventory() == (foreign,)
        assert list((tmp_path / "venues").glob(".used-*")) == [tmp_path / "venues" / f".used-{owned.id}"]
        assert challenges == []

        revised = folders.read_plan(PROJECT, revision=None)
        new_quote = client.post(f"{API}/quote", json={
            "revision": revised.revision, "target": PROFILE, "run_keys": [asdict(key) for key in RUNS],
        })
        new_quote.raise_for_status()
        next_id = new_quote.json()["id"]
        assert next_id != str(quote_id)
        assert client.post(f"{API}/approve", json={"quote": next_id}).status_code == 202
        assert len(challenges) == 1
        assert wait_until(lambda: _approval_row(restarted, next_id)[1]) == REFUSED
        assert restarted.inventory() == (foreign,)


if __name__ == "__main__":
    _first_controller(Path(sys.argv[1]), sys.argv[2])
