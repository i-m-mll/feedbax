from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from feedbax.orchestration import runpod
from feedbax.orchestration.runpod_session_provider import (
    RunPodAcquisition, RunPodResources, _BOOT_COMMAND, _BOOT_KEY, _fresh_boot_host_key,
)
from feedbax.orchestration.session import (
    KeepOpen, ProviderResource, SessionAuthorization, SessionError, SessionLaunch, SessionRequest,
)
from tests.support.runpod_session import FakeRunPod, FakeRunPodCutoff, runpod_resources


def _launch(resources: RunPodResources) -> SessionLaunch:
    now = time.time_ns() // 1_000_000
    request = SessionRequest(
        id=uuid4(), target="runpod", resources=resources.model_dump(mode="json"),
        environment="e" * 64, lifecycle=KeepOpen(kind="keep-open"), maximum_microusd=100_000,
    )
    authorization = SessionAuthorization(
        request_digest=request.digest, operator="local", method="fixture", evidence="fixture",
        expires_at_ms=now + 120_000, ceiling_microusd=request.maximum_microusd,
    )
    return SessionLaunch(
        request=request, authorization=authorization, adoption_sha256="b" * 64, acquired_at_ms=now,
        hourly_microusd=resources.hourly_microusd,
    )


def _provider(tmp_path: Path, handle, cutoff: FakeRunPodCutoff | None) -> RunPodAcquisition:
    return RunPodAcquisition(
        tmp_path / "acquisition records", runpod.SubprocessRunPodTransport(http_transport=httpx.MockTransport(handle)),
        api_key="fixture-api-key", ssh_key_path=tmp_path / "operator-key", running_cutoff=cutoff,
    )


def test_runpod_gate_guards_all_mutations(tmp_path):
    resources = runpod_resources()
    fake = FakeRunPod(resources)
    launch = _launch(resources)
    with pytest.raises(SessionError, match="preboot cutoff"):
        _provider(tmp_path, fake.handle, None).acquire(launch)
    assert fake.requests == []

    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    fake.quoted_rate *= 2
    with pytest.raises(SessionError, match="authorized GPU, rate, CUDA and region"):
        _provider(tmp_path, fake.handle, cutoff).acquire(launch)
    assert not any(method == "POST" for method, _ in fake.requests)
    assert cutoff.launches == []


def test_runpod_boot_installs_pinned_key_and_arms_guard_before_start(tmp_path):
    private, public = _fresh_boot_host_key()
    start = tmp_path / "start"
    start.write_text("#!/bin/bash\nprintf 'start\n'\n")
    start.chmod(0o700)
    command = ["/bin/bash", "-c", _BOOT_COMMAND, "feedbax-host-key", public, str(tmp_path), str(start)]
    environment = {**os.environ, _BOOT_KEY: private}
    unarmed = subprocess.run(
        [*command, "/usr/bin/false"], env=environment, timeout=5, capture_output=True, text=True,
    )
    assert unarmed.returncode != 0 and unarmed.stdout == ""
    armed = subprocess.run([
        *command, "/bin/bash", "-c",
        'test -z "${FEEDBAX_BOOT_SSH_KEY:-}" && test -s "$1" && printf "armed\\n"',
        "guard", str(tmp_path / "ssh_host_ed25519_key"),
    ], env=environment, timeout=5, capture_output=True, text=True, check=True)
    assert armed.stdout.splitlines() == ["armed", "start"]
    assert (tmp_path / "ssh_host_ed25519_key.pub").read_text() == public + "\n"


@pytest.mark.parametrize("path, commit", [
    ("/v2/account/secrets", True),
    ("/v2/pods", True),
    ("/v2/pods", False),
])
def test_runpod_recovers_exactly_one_create_or_leaves_it_unresolved(tmp_path, path, commit):
    resources = runpod_resources()
    fake = FakeRunPod(resources)
    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    launch = _launch(resources)
    attempted = False

    def lose_reply(request: httpx.Request) -> httpx.Response:
        nonlocal attempted
        if request.method == "POST" and request.url.path == path:
            assert not attempted
            attempted = True
            journal = tmp_path / "acquisition records" / f"{launch.intent}.json"
            expected = "secret-requested" if path.endswith("secrets") else "pod-requested"
            text = journal.read_text()
            assert expected in text and "fixture-api-key" not in text
            if commit:
                fake.handle(request)
            raise httpx.ReadTimeout("lost create response", request=request)
        return fake.handle(request)

    with pytest.raises(SessionError, match="transport failed"):
        _provider(tmp_path, lose_reply, cutoff).acquire(launch)
    assert attempted
    resumed = _provider(tmp_path, lose_reply, cutoff)
    if not commit:
        with pytest.raises(SessionError, match="pod creation remains unresolved"):
            resumed.acquire(launch)
        assert not fake.pods
        return
    venue = resumed.acquire(launch)
    assert venue.id == fake.pod_id
    assert venue.intent == launch.intent
    assert cutoff.launches == [launch]
    assert fake.requests.count(("POST", "/v2/account/secrets")) == 1
    assert fake.requests.count(("POST", "/v2/pods")) == 1


def test_runpod_pins_endpoint_and_removes_only_the_exact_owned_resources(tmp_path, monkeypatch):
    resources = runpod_resources()
    fake = FakeRunPod(resources)
    launch = _launch(resources)
    fake.pods["other"] = {"id": "other", "name": "unrelated"}
    fake.secrets["other"] = {"id": "other", "name": "unrelated", "description": "unrelated"}
    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    provider = _provider(tmp_path, fake.handle, cutoff)
    venue = provider.acquire(launch)
    created = fake.created_pod
    assert created["fixtureRunningCutoffMs"] == launch.ends_at_ms
    assert created["entrypoint"][-2:] == ["/start.sh", "/usr/bin/true"]
    probe_calls: list[list[str]] = []

    def probe(argv: list[str], *, timeout_seconds: float) -> runpod.CommandResult:
        probe_calls.append(argv)
        assert timeout_seconds <= 8
        assert argv[-2] == "root@127.0.0.1" and argv[-1].startswith("nvidia-smi ")
        known_hosts = provider.root / f"{launch.intent}.known-hosts"
        assert known_hosts.read_text() == f"[127.0.0.1]:2222 {created['entrypoint'][4]}\n"
        output = f"{resources.gpu_id}, 570.86.10, 8.9, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n"
        return runpod.CommandResult(0, output, "")

    monkeypatch.setattr(runpod, "_run_command", probe)
    endpoint = provider.endpoint(venue)
    assert endpoint.host == "root@127.0.0.1"
    assert "StrictHostKeyChecking=yes" in endpoint.ssh_argv
    assert "HostKeyAlgorithms=ssh-ed25519" in endpoint.ssh_argv
    assert "-n" not in endpoint.ssh_argv
    assert len(probe_calls) == 1
    assert endpoint.placement["device"] == resources.gpu_id

    secret = ProviderResource(id=fake.secret_id, intent=launch.intent, kind="secret")
    with pytest.raises(SessionError, match="owned pod remains"):
        provider.remove(secret)
    fake.pods[fake.pod_id]["name"] = "unrelated"
    with pytest.raises(SessionError, match="intent changed"):
        provider.remove(venue)
    assert not any(method == "DELETE" for method, _ in fake.requests)
    fake.pods[fake.pod_id]["name"] = launch.intent
    provider.remove(venue)
    provider.remove(venue)
    provider.remove(secret)
    assert [(item.id, item.intent) for item in provider.inventory()] == [("other", None), ("other", None)]
    assert fake.requests.count(("DELETE", f"/v2/pods/{fake.pod_id}")) == 1
    with pytest.raises(SessionError, match="pod creation remains unresolved"):
        provider.acquire(launch)
