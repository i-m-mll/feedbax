"""Restart the controller across the durable Jobs approval boundary."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import sys
from uuid import UUID

from fastapi.testclient import TestClient

from feedbax.contracts.execution_target import KeepOpen, SessionTargetProfile
from feedbax.contracts.jobs import StudioJobSubmission
from feedbax.contracts.project import Project
from feedbax.orchestration.session import ProviderResource, SessionOpeningFailed
from feedbax.orchestration.session_jobs import SessionJobs
from feedbax.orchestration.venue_jobs import VenueJobs
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.web.app import create_app
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.remote_runs import StudioRemoteRuns, StudioVenue
from tests.support.session_venue import ENVIRONMENT, LocalVenueProvider
from tests.support.store_transport import loopback_store
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import authored_documents, document_project
from tests import test_local_job_host
from tests.test_local_job_host import wait_until
from tests.test_session_rehearsal import EVALUATE, PROJECT, TRAIN, _authorization, _driver


PROFILE = "recovery"
RUNS = (TRAIN, EVALUATE)
API = f"/api/projects/{PROJECT}/train-runs"
REFUSED = "The original machine refused initialization."
host_fixture = test_local_job_host.host


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


def _app(root: Path, action: str):
    provider = _Provider(root, action)
    driver = _driver(root / "controller", provider)
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

    def attest(quote, request):
        assert action in ("after-acquisition", "before-delegation"), "Recovery must use the original approval."
        return _authorization(request, expires_at_ms=quote.expires_at_ms)

    remote = StudioRemoteRuns((profile,), (StudioVenue(driver, ENVIRONMENT, jobs, 90),), attest=attest)
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
        client.post(f"{API}/approve", json={"quote": quote_id}).raise_for_status()
    raise AssertionError("The first controller must exit before replying.")


def _kill_controller(root: Path, action: str) -> UUID:
    output = root / "first-controller.log"
    with output.open("wb") as log, subprocess.Popen(
        [sys.executable, "-m", "tests.test_approved_session_recovery", str(root), action],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
    ) as child:
        def reached_boundary():
            assert child.poll() is None, output.read_text()
            return (root / "boundary").is_file()

        try:
            wait_until(reached_boundary, timeout=15)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        assert child.returncode == -signal.SIGKILL, output.read_text()
    return UUID((root / "quote-id").read_text())


def _local_project(tmp_path: Path, monkeypatch, host_fixture):
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(host_fixture.root))
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


def test_controller_restart_delivers_approved_selection_on_the_same_machine(tmp_path, monkeypatch, host_fixture):
    folders, foreign = _local_project(tmp_path, monkeypatch, host_fixture)
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
        document["base"]["inline"]["method_payload"]["payload"]["optimizer"]["params"]["learning_rate"] = 0.002
        changed_path.write_text(json.dumps(document))
        edited = folders.read_plan(PROJECT, revision=None)
        assert edited.revision != reviewed.revision
        assert str(next(run.request_id for run in edited.plan.runs if run.key == TRAIN)) != requests[TRAIN]

        with TestClient(app) as client:
            accepted = wait_until(lambda: _approval_row(restarted, quote_id)[0], timeout=16)
            submission = StudioJobSubmission.model_validate_json(accepted)
            reply = client.post(f"{API}/approve", json={"quote": str(quote_id)})
            assert reply.status_code == 202 and StudioJobSubmission.model_validate(reply.json()) == submission
            assert submission.target == opened.placement
            assert [item.disposition for item in submission.items] == ["admitted", "admitted"]
            host = restarted.attach(quote_id)
            assert len(host.submissions(token=str(quote_id))) == 1
            assert [row["request"] for row in host.jobs()] == [requests[key] for key in RUNS]
            installed = venue / "projects" / reviewed.revision / PROJECT
            assert {name: (installed / name).read_bytes() for name in files} == files
            assert set(restarted.inventory()) == {opened.venue, foreign}
            assert list((tmp_path / "venues").glob(".used-*")) == [tmp_path / "venues" / f".used-{opened.venue.id}"]

            acknowledged = wait_until(
                lambda: host.session_status()["collections"] if len(host.session_status()["collections"]) == 2 else None,
                timeout=28,
            )
            local = SuccessView(ImmutableArtifactBlobProvider(reviewed.store_path))
            for item in submission.items:
                ref = local.attempt_ref(item.job_id)
                assert ref is not None and ref.sha256 == acknowledged[item.job_id]
            assert restarted.close(quote_id).remaining == ()
    finally:
        restarted.provider.remove(opened.venue)
    assert restarted.inventory() == (foreign,)


def test_recovered_opening_failure_closes_the_original_machine(tmp_path, monkeypatch, host_fixture):
    _, foreign = _local_project(tmp_path, monkeypatch, host_fixture)
    quote_id = _kill_controller(tmp_path, "after-acquisition")
    app, restarted = _app(tmp_path, "refuse")
    original = restarted.record(quote_id)
    assert original.phase == "opening" and original.venue is None
    assert _approval_row(restarted, quote_id) == (None, None)
    (owned,) = [item for item in restarted.inventory() if item.intent == original.launch.intent]

    with TestClient(app) as client:
        failure = wait_until(lambda: _approval_row(restarted, quote_id)[1])
        reply = client.post(f"{API}/approve", json={"quote": str(quote_id)})
        assert reply.status_code == 409 and reply.json() == {"detail": REFUSED} and failure == REFUSED
        closed = restarted.record(quote_id)
        assert closed.phase == "closed" and closed.venue == owned
        assert closed.close_evidence is not None
        assert closed.close_evidence.owned == (owned,) and closed.close_evidence.remaining == ()
        assert restarted.inventory() == (foreign,)
        assert list((tmp_path / "venues").glob(".used-*")) == [tmp_path / "venues" / f".used-{owned.id}"]


if __name__ == "__main__":
    _first_controller(Path(sys.argv[1]), sys.argv[2])
