"""RunPod controller path over injected HTTP and a local subprocess venue."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.execution_target import SessionTargetProfile, attempt_target
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.host.protocol import HostError
from feedbax.jobs import JobService
from feedbax.orchestration.repo_snapshot import seal_repo_snapshot
from feedbax.orchestration.runpod import SubprocessRunPodTransport
from feedbax.orchestration.runpod_cutoff import NativeRunPodCutoff
from feedbax.orchestration.runpod_session import RunPodEnvironment, RunPodSessionProvider
from feedbax.orchestration.runpod_session_provider import RunPodAcquisition
from feedbax.orchestration.session import KeepOpen, SessionDriver, SessionError, SessionRequest
from feedbax.orchestration.targets import TargetRegistry
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService
from tests import test_local_job_host
from tests.support.runpod_native import ACCOUNT_KEY, NativeLoopbackRunPod
from tests.support.runpod_session import runpod_resources
from tests.support.session_venue import verify_authorization
from tests.test_document_train_pair import authored_documents, document_project
from tests.test_session_rehearsal import EVALUATE, ISSUER_KEY, PROJECT, TRAIN, _authorization, _jobs, _terminal_without_rpc


host_fixture = test_local_job_host.host
wait_until = test_local_job_host.wait_until


def _loopback_ssh(path, *, gate, lost_reply, trace, resources, login):
    path.write_text(f"""#!{sys.executable}
import os
from pathlib import Path
from types import SimpleNamespace
import shlex
import subprocess
import sys

position = sys.argv.index("root@127.0.0.1")
options = sys.argv[1:position]
assert "StrictHostKeyChecking=yes" in options and "IdentitiesOnly=yes" in options and "2222" in options
forwarded = shlex.split(" ".join(sys.argv[position + 1:]))
with open({str(trace)!r}, "a") as log:
    log.write(("rsync" if "--server" in forwarded else " ".join(forwarded[:3])) + "\\n")
if forwarded[0] == "nvidia-smi":
    print({f'{resources.gpu_id}, {resources.min_driver_major}.86.10, {resources.compute_capability}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'!r})
    sys.exit(0)
if "-m" in forwarded and "feedbax.host" in forwarded and "attach" in forwarded:
    assert "-n" not in options
    if Path({str(gate)!r}).is_file():
        sys.exit(74)
os.chdir({str(login)!r})
os.environ["UV_OFFLINE"] = "true"
os.environ.pop("PYTHONPATH", None)
if "feedbax.orchestration.venue_jobs" in forwarded and Path({str(lost_reply)!r}).exists():
    result = subprocess.run(forwarded, input=sys.stdin.buffer.read(), capture_output=True, check=True)
    assert b'"value"' in result.stdout
    Path({str(lost_reply)!r}).unlink()
    sys.exit(74)
os.execvp(forwarded[0], forwarded)
""")
    path.chmod(0o700)


def _terminate_local_venue(venue):
    if not venue.exists():
        return
    (venue / "ready").unlink(missing_ok=True)
    rows = subprocess.check_output(["ps", "-axo", "pid=,command="], timeout=4, text=True)
    for line in rows.splitlines():
        pid, _, command = line.strip().partition(" ")
        watchdog = f"watchdog.py --config {venue / 'watchdog.json'}"
        host = f"-m feedbax.host --root {venue / 'host'} serve --require-session"
        if watchdog in command or host in command:
            with suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGTERM)
    for directory, _, _ in os.walk(venue):
        Path(directory).chmod(0o700)
    shutil.rmtree(venue)


@pytest.fixture
def native_execution(tmp_path):
    controller = tmp_path / "controller"
    workspace = tmp_path / "venue-workspace"
    login = tmp_path / "ssh-login"
    login.mkdir()
    resources = runpod_resources()
    venue = workspace / "feedbax-sessions" / NativeLoopbackRunPod.pod_id
    fake = NativeLoopbackRunPod(
        resources, workspace=workspace, scratch=tmp_path,
        on_delete=lambda pod_id: _terminate_local_venue(venue) if pod_id == fake.pod_id else None,
    )
    gate, trace = tmp_path / "sshd-link-down", tmp_path / "ssh.trace"
    ssh, lost_reply = tmp_path / "injected-ssh", tmp_path / "lost-admission-reply"
    _loopback_ssh(ssh, gate=gate, lost_reply=lost_reply, trace=trace, resources=resources, login=login)
    uv = shutil.which("uv")
    assert uv is not None
    uv_version = subprocess.check_output([uv, "--version"], text=True, timeout=4).split()[1]
    src = Path(__file__).resolve().parents[1]
    snapshot = seal_repo_snapshot(
        "feedbax", src, snapshot_parent=tmp_path / "snapshots", exclude_gitlinks=("vendor/window-shell",),
    )
    environment = RunPodEnvironment(
        workspace=str(workspace), bootstrap_python=sys._base_executable,
        uv_sha256=hashlib.sha256(Path(uv).read_bytes()).hexdigest(), uv_version=uv_version,
        snapshot=snapshot.record, backend="cpu", extras=("web",),
    )
    with fake.serve_scoped() as origin:
        acquisition = RunPodAcquisition(
            controller / "provider", SubprocessRunPodTransport(http_transport=httpx.MockTransport(fake.handle)),
            api_key=ACCOUNT_KEY, ssh_key_path=tmp_path / "unused-client-key",
            running_cutoff=NativeRunPodCutoff(str(workspace), sys._base_executable, _test_origin=origin),
            ssh_executable=str(ssh),
        )
        provider = RunPodSessionProvider(acquisition, environment, snapshot, uv_binary=Path(uv))
        verifier = partial(verify_authorization, key=ISSUER_KEY)
        driver = SessionDriver(controller / "sessions", provider, verify_authorization=verifier)
        try:
            yield SimpleNamespace(controller=controller, venue=venue, resources=resources, fake=fake,
                                  gate=gate, trace=trace, lost_reply=lost_reply, environment=environment,
                                  provider=provider, verifier=verifier, driver=driver)
        finally:
            _terminate_local_venue(venue)


def test_runpod_saved_plan_uses_real_venue_paths_and_collects_over_ssh(host_fixture, native_execution):
    run = native_execution
    controller, venue, fake = run.controller, run.venue, run.fake
    gate, trace, lost_reply = run.gate, run.trace, run.lost_reply
    provider, driver, verifier = run.provider, run.driver, run.verifier
    profile = SessionTargetProfile(
        kind="session", id="runpod-witness", label="RunPod witness", driver=provider.target,
        resources=run.resources.model_dump(mode="json"), lifecycle=KeepOpen(), maximum_microusd=750_000,
    )
    targets = TargetRegistry((profile,), (provider,))
    request = targets.session_request(profile.id, session_id=uuid4(), environment=run.environment.digest)
    quote = provider.acquisition.review_quote(request)
    assert quote["gpu_hourly_usd"] == 0.75 and quote["ephemeral_disk_gb"] == 40
    assert quote["authorized_total_hourly_microusd"] == run.resources.hourly_microusd
    assert not fake.pods and not fake.secrets
    authorization = _authorization(request)
    adoption = secrets.token_urlsafe()
    with pytest.raises(SessionError):
        driver.open(request.model_copy(update={"profile": "runpod-other"}), authorization, adoption_token=adoption)
    assert all(method != "POST" for method, _ in fake.requests)
    local_folders = ProjectFolderService(controller / "projects")
    project_root = local_folders.root / PROJECT
    local_store = project_root / "local-results"
    for name, data in dump_text_project(
        Project.model_validate(document_project(authored_documents(), evaluation=True)),
        ProjectConfiguration(project=PROJECT, store=str(local_store)),
    ).items():
        target = project_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    approved = local_folders.read_plan(PROJECT, revision=None)
    assert approved.store_path == local_store
    approved_files = local_folders.read_authored_snapshot(PROJECT, revision=approved.revision).files

    opened = driver.open(request, authorization, adoption_token=adoption)
    assert fake.boot_armed.is_file()
    assert opened.realization.probe["running_cutoff"] == {
        "version": 1, "pod_id": fake.pod_id, "intent": opened.launch.intent, "ends_at_ms": opened.launch.ends_at_ms,
    }
    assert opened.realization.probe["probe_result"] == 5
    assert opened.realization.probe["python"] == str(venue / "venv" / "bin" / "python")
    assert not provider.supports_suspension
    session = provider.jobs(driver, request.id)

    def submit_to_venue(selected, folders, reviewed, *, run_keys, token, reuse_completed):
        assert selected == profile and not reuse_completed
        return session.delegate(folders, reviewed, run_keys=run_keys, token=token)

    typed_jobs = JobService(
        host_fixture, new_application_registry_bundle(), targets=targets, submit_session=submit_to_venue,
    )
    lost_reply.touch()
    with pytest.raises(OSError, match="exited 74"):
        typed_jobs.submit_plan(local_folders, approved, target=profile.id,
                               run_keys=(TRAIN, EVALUATE), token="saved-chain")
    assert not lost_reply.exists()
    changed_file = next(name for name in approved_files if name.endswith(".json"))
    (project_root / changed_file).write_bytes(approved_files[changed_file] + b"\n")
    assert local_folders.read_plan(PROJECT, revision=None).revision != approved.revision
    gate.touch()
    rows = wait_until(lambda: _terminal_without_rpc(venue / "host" / "host.sqlite"), timeout=33)
    assert [row[1] for row in rows] == ["succeeded", "succeeded"], rows
    gate.unlink()

    reconnected = provider.jobs(driver, request.id)
    reconnected.replay(local_folders)
    submitted = reconnected.delegate(
        local_folders, approved, run_keys=(TRAIN, EVALUATE), token="saved-chain",
    )
    assert all(item.disposition == "admitted" for item in submitted.items), submitted.items
    assert submitted.target == opened.placement and submitted.target.profile == profile.id
    saved = venue / "projects" / approved.revision / PROJECT
    assert {name: (saved / name).read_bytes() for name in approved_files} == approved_files
    first = session.poll(limit=1)
    second = reconnected.poll(limit=1)
    assert second["events"][0]["cursor"] > first["cursor"]
    assert reconnected.delegate(
        local_folders, approved, run_keys=(TRAIN, EVALUATE), token="saved-chain",
    ) == submitted
    train_id, evaluate_id = (item.job_id for item in submitted.items)
    venue_rows = reconnected.host.jobs_snapshot(include_launch=True)["jobs"]
    assert len(venue_rows) == 2
    assert all(attempt_target(row["data"]["target"]) == opened.placement for row in venue_rows)
    assert all(row["launch"]["argv"] == [opened.realization.probe["python"]] for row in venue_rows)
    assert all(row["data"]["manifest_root"] == str(venue / "stores" / PROJECT) for row in venue_rows)
    train_row, evaluate_row = venue_rows
    (edge,) = evaluate_row["edges"]
    assert edge["job_id"] == train_id and edge["authenticated_completion"]
    assert edge["binding"]["terminal"]["receipt"] == train_row["outcome"]["store_success"]

    original = []
    venue_store = ImmutableArtifactBlobProvider(venue / "stores" / PROJECT)
    for row in venue_rows:
        ref = ContentObjectRef.model_validate(row["outcome"]["store_success"])
        original.append((ref, venue_store.get_object(ref)))
    assert len(original) == 2
    assert reconnected.host.session_status()["pending"]

    gate.touch()
    local_jobs = _jobs(host_fixture)
    copied = reconnected.collect_project(local_jobs, approved, observed_only=True)
    assert {item.attempt_id for item in copied} == {train_id, evaluate_id}
    gate.unlink()
    assert set(reconnected.host.session_status()["pending"]) == {train_id, evaluate_id}
    assert reconnected.collect_project(local_jobs, approved) == copied
    assert not reconnected.host.session_status()["pending"]
    assert "rsync" in trace.read_text().splitlines()

    adopted = SessionDriver(controller / "adopted", provider, verify_authorization=verifier)
    assert adopted.adopt(request, authorization, adoption_token=adoption).epoch == 1
    with pytest.raises(HostError) as stale:
        driver.close(request.id)
    assert stale.value.code == "session_fenced"
    foreign = {"id": "foreign-pod", "name": "another user's job"}
    fake.pods["foreign-pod"] = foreign
    close = adopted.close(request.id)
    assert {item.kind for item in close.owned} == {"venue", "secret"}
    assert not close.remaining
    assert fake.pods == {"foreign-pod": foreign}
    assert not venue.exists()

    destination = ImmutableArtifactBlobProvider(local_store)
    assert [(ref, destination.get_object(ref)) for ref, _ in original] == original
    view = SuccessView(destination)
    assert all(view.read_owned_success(ref).receipt.target == opened.placement for ref, _ in original)
    recorded = view.read_owned_success(view.attempt_ref(evaluate_id)).receipt
    subject = recorded.inputs["subject"][0].evaluation_subject
    assert subject.terminal_receipt == original[0][0]
    assert subject.terminal_checkpoint.run_id == train_id
    result = local_jobs.read_evaluation_result(project=PROJECT, store=local_store, job_id=evaluate_id)
    with result.open_states(verify=True) as states:
        assert states.descriptor.dimensions["checkpoint"] == 1



def test_runpod_native_cutoff_removes_only_its_pod_after_host_and_controller_detach(native_execution):
    run = native_execution
    driver, fake = run.driver, run.fake
    request = SessionRequest(
        id=uuid4(), target=run.provider.target, resources=run.resources.model_dump(mode="json"),
        environment=run.environment.digest, lifecycle=KeepOpen(), maximum_microusd=3_750,
    )
    opened = driver.open(request, _authorization(request), adoption_token=secrets.token_urlsafe())
    assert opened.launch.ends_at_ms - opened.launch.acquired_at_ms == 18_000
    assert run.fake.boot_armed.is_file()
    assert opened.realization.probe["running_cutoff"]["ends_at_ms"] == opened.launch.ends_at_ms
    foreign = {"id": "other-pod", "name": "unrelated run"}
    fake.pods[foreign["id"]] = foreign
    fake.refuse_delete_once = True
    pid = driver.attach(request.id).ping()["pid"]
    # Disable this fixture's Host readiness so the watcher cannot replace the killed daemon.
    (run.venue / "ready").unlink()
    os.kill(pid, signal.SIGKILL)
    fake.controller_attached = False
    wait_until(lambda: any(method == "GET" and status == 404 for method, _, status, _ in fake.scoped_requests),
               timeout=22)
    assert fake.pods == {foreign["id"]: foreign}
    requests = [(status, when) for method, path, status, when in fake.scoped_requests
                if method == "DELETE" and path == f"/v2/pods/{fake.pod_id}"]
    assert [status for status, _ in requests] == [503, 204]
    assert all(when >= opened.launch.ends_at_ms for _, when in requests)
    assert not run.venue.exists()

    # The detached venue uses only its pod key, leaving its account-level boot secret.
    assert fake.secret_id in fake.secrets
    fake.controller_attached = True
    close = driver.close(request.id)
    assert {item.kind for item in close.owned} == {"venue", "secret"}
    assert not close.remaining and not fake.secrets
    assert fake.pods == {foreign["id"]: foreign}
