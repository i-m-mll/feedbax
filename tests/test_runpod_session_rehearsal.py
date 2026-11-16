"""RunPod controller path over injected HTTP and a local subprocess venue."""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.host.protocol import HostError
from feedbax.orchestration.repo_snapshot import seal_repo_snapshot
from feedbax.orchestration.runpod import SubprocessRunPodTransport
from feedbax.orchestration.runpod_session import RunPodEnvironment, RunPodSessionProvider
from feedbax.orchestration.runpod_session_provider import RunPodAcquisition
from feedbax.orchestration.session import KeepOpen, SessionDriver, SessionRequest
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.web.services.project_folder import ProjectFolderService
from tests import test_local_job_host
from tests.support.runpod_session import FakeRunPod, FakeRunPodCutoff, runpod_resources
from tests.support.session_venue import verify_authorization
from tests.test_canonical_evaluate_admission import managed_evaluation_definition
from tests.test_session_rehearsal import EVALUATE, ISSUER_KEY, PROJECT, TRAIN, _authorization, _jobs, _terminal_without_rpc


host_fixture = test_local_job_host.host
wait_until = test_local_job_host.wait_until


def _loopback_ssh(path, *, gate, lost_reply, trace, resources, login):
    path.write_text(f"""#!{sys.executable}
import os
from pathlib import Path
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


def test_runpod_saved_plan_uses_real_venue_paths_and_collects_over_ssh(host_fixture, tmp_path):
    controller = tmp_path / "controller"
    venue_workspace = tmp_path / "venue-workspace"
    login = tmp_path / "ssh-login"
    login.mkdir()
    venue = venue_workspace / "feedbax-sessions" / "pod-1"
    resources = runpod_resources()
    fake = FakeRunPod(resources, on_delete=lambda pod_id: _terminate_local_venue(venue) if pod_id == "pod-1" else None)
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
        workspace=str(venue_workspace), bootstrap_python=sys._base_executable, uv=uv, uv_version=uv_version,
        snapshot=snapshot.record, backend="cpu", extras=("web",),
    )
    acquisition = RunPodAcquisition(
        controller / "provider", SubprocessRunPodTransport(http_transport=httpx.MockTransport(fake.handle)),
        api_key="fixture-issued-key", ssh_key_path=tmp_path / "unused-client-key",
        running_cutoff=FakeRunPodCutoff(("/usr/bin/true",)), ssh_executable=str(ssh),
    )
    provider = RunPodSessionProvider(acquisition, environment, snapshot)
    verifier = partial(verify_authorization, key=ISSUER_KEY)
    driver = SessionDriver(controller / "sessions", provider, verify_authorization=verifier)
    request = SessionRequest(
        id=uuid4(), target=provider.target, resources=resources.model_dump(mode="json"),
        environment=environment.digest, lifecycle=KeepOpen(), maximum_microusd=1_000_000,
    )
    authorization = _authorization(request)
    adoption = secrets.token_urlsafe()
    local_folders = ProjectFolderService(controller / "projects")
    project_root = local_folders.root / PROJECT
    local_store = project_root / "local-results"
    for name, data in dump_text_project(
        Project.model_validate(managed_evaluation_definition()),
        ProjectConfiguration(project=PROJECT, store=str(local_store)),
    ).items():
        target = project_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    approved = local_folders.read_plan(PROJECT, revision=None)
    assert approved.store_path == local_store
    approved_files = local_folders.read_authored_snapshot(PROJECT, revision=approved.revision).files

    try:
        opened = driver.open(request, authorization, adoption_token=adoption)
        assert opened.realization.probe["probe_result"] == 5
        assert opened.realization.probe["python"] == str(venue / "venv" / "bin" / "python")
        assert not provider.supports_suspension
        session = provider.jobs(driver, request.id)
        lost_reply.touch()
        with pytest.raises(OSError, match="exited 74"):
            session.delegate(local_folders, approved, run_keys=(TRAIN, EVALUATE), token="saved-chain")
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
        assert all(row["launch"]["argv"] == [opened.realization.probe["python"]] for row in venue_rows)
        assert all(row["data"]["manifest_root"] == str(venue / "stores" / PROJECT) for row in venue_rows)
        assert venue_rows[1]["edges"][0]["binding"]["completion"]["producer_job_id"] == train_id

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
        result = local_jobs.read_evaluation_result(project=PROJECT, store=local_store, job_id=evaluate_id)
        with result.open_states(verify=True) as states:
            assert states.descriptor.dimensions["checkpoint"] == 1
    finally:
        _terminate_local_venue(venue)
