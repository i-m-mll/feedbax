from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
import pytest

from feedbax.orchestration import runpod, runpod_session_provider
from feedbax.orchestration.repo_snapshot import RepoSnapshotRecord, SealedRepoSnapshot
from feedbax.orchestration.runpod_session import RunPodEnvironment, RunPodSessionProvider
from feedbax.orchestration.runpod_session_provider import (
    RunPodAcquisition, RunPodResources, _BOOT_COMMAND, _BOOT_KEY, _fresh_boot_host_key,
)
from feedbax.orchestration.session import (
    KeepOpen, ProviderResource, ProviderUnreachable, SessionAuthorization, SessionDriver, SessionError, SessionLaunch,
    SessionRequest,
)
from tests.support.runpod_session import FakeRunPod, FakeRunPodCutoff, runpod_resources


def _launch(resources: RunPodResources, *, environment: str = "e" * 64) -> SessionLaunch:
    now = time.time_ns() // 1_000_000
    request = SessionRequest(
        id=uuid4(), target="runpod", resources=resources.model_dump(mode="json"),
        environment=environment, lifecycle=KeepOpen(kind="keep-open"), maximum_microusd=100_000,
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
    quote = _provider(tmp_path, fake.handle, None).review_quote(launch.request)
    assert quote["gpu_hourly_usd"] == fake.quoted_rate
    assert quote["ephemeral_disk_gb"] == resources.ephemeral_disk_gb
    assert quote["authorized_total_hourly_microusd"] == resources.hourly_microusd

    def unavailable(request):
        assert request.method == "GET"
        raise httpx.ConnectError("fixture outage", request=request)

    with pytest.raises(ProviderUnreachable, match="transport failed"):
        _provider(tmp_path, unavailable, None).review_quote(launch.request)

    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    fake.quoted_rate *= 2
    with pytest.raises(SessionError, match="authorized GPU, rate, CUDA and region"):
        _provider(tmp_path, fake.handle, cutoff).acquire(launch)
    assert not any(method == "POST" for method, _ in fake.requests)
    assert cutoff.launches == []


def test_machine_status_uses_owned_provider_state_without_treating_price_as_billed_cost(tmp_path):
    launch = _launch(runpod_resources())
    resource = ProviderResource(id="pod", intent=launch.intent, kind="venue")
    body = {
        "id": resource.id, "name": resource.intent, "status": "RUNNING", "desiredStatus": "EXITED",
        "costPerHr": 0.75, "createdAt": "2026-01-01T00:00:00Z",
    }

    def read(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/v2/pods/pod"
        return httpx.Response(200, json=body)

    provider = _provider(tmp_path, read, None)
    running = provider.machine_observation(resource)
    assert running.state == "on"
    assert running.billing_started_at_ms == 1767225600000
    assert running.billed_microusd is None
    body["status"] = "EXITED"
    assert provider.machine_observation(resource).state == "off"
    body["name"] = "foreign"
    with pytest.raises(SessionError, match="intent changed"):
        provider.machine_observation(resource)


def test_runpod_boot_installs_pinned_key_and_arms_guard_before_start(tmp_path):
    private, public = _fresh_boot_host_key()
    start = tmp_path / "start"
    start.write_text('#!/bin/bash\ntest -z "${RUNPOD_API_KEY:-}${LD_LIBRARY_PATH:-}" && printf "start\\n"\n')
    start.chmod(0o700)
    command = ["/bin/bash", "-c", _BOOT_COMMAND, "feedbax-host-key", public, str(tmp_path), str(start)]
    environment = {
        **os.environ, _BOOT_KEY: private, "RUNPOD_API_KEY": "fixture-pod-token", "LD_LIBRARY_PATH": "/image/cuda",
    }
    unarmed = subprocess.run(
        [*command, "/usr/bin/false"], env=environment, timeout=5, capture_output=True, text=True,
    )
    assert unarmed.returncode != 0 and unarmed.stdout == ""
    armed = subprocess.run([
        *command, "/bin/bash", "-c",
        'test -z "${FEEDBAX_BOOT_SSH_KEY:-}${LD_LIBRARY_PATH:-}" && test "$RUNPOD_API_KEY" = fixture-pod-token'
        ' && test ! -e "$1" && printf "armed\\n"',
        "guard", str(tmp_path / "ssh_host_ed25519_key"),
    ], env=environment, timeout=5, capture_output=True, text=True, check=True)
    assert armed.stdout.splitlines() == ["armed", "start"]
    assert armed.stderr.splitlines() == ["RunPod venue cutoff guard armed.", "RunPod pinned SSH host key installed."]
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


@pytest.mark.parametrize("failure", ["deadline", "read-deadline", 400, 402, 404, 422, 429, "ambiguous"])
def test_runpod_open_closes_only_definitive_uninitialized_openings(tmp_path, monkeypatch, failure):
    resources = runpod_resources()
    fake = FakeRunPod(resources)
    fake.pods["other"] = {"id": "other", "name": "unrelated"}
    record = RepoSnapshotRecord(
        commit="a" * 40, dirty=False, content_sha256="b" * 64, source_state_sha256="c" * 64, file_count=0,
    )
    snapshot = SealedRepoSnapshot("fixture", tmp_path, tmp_path, record)
    environment = RunPodEnvironment(
        workspace=str(tmp_path), bootstrap_python="/usr/bin/python3", uv="/usr/bin/uv",
        uv_version="1.2.3", snapshot=record, backend="gpu",
    )
    launch = _launch(resources, environment=environment.digest)
    lost_read = False

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal lost_read
        if failure == "read-deadline" and not lost_read and (
            request.method, request.url.path
        ) == ("GET", f"/v2/pods/{fake.pod_id}"):
            lost_read = True
            expired = launch.acquired_at_ms + (resources.startup_seconds + 1) * 1000
            monkeypatch.setattr(runpod_session_provider, "_epoch_ms", lambda: expired)
            raise httpx.ReadTimeout("SSH readiness poll timed out", request=request)
        if (request.method, request.url.path) == ("POST", "/v2/pods"):
            if isinstance(failure, int):
                return httpx.Response(failure, json={
                    "status": failure,
                    "title": "Request rejected",
                    "detail": "entrypoint exceeded its allowed length for fixture-api-key",
                    "errors": ["Schema error: custom error text", *[f"field {i}" for i in range(9)]],
                }, headers={"Content-Type": "application/problem+json"})
            if failure == "ambiguous":
                raise httpx.ReadTimeout("lost create response", request=request)
            response = fake.handle(request)
            if failure == "deadline":
                expired = launch.acquired_at_ms + (resources.startup_seconds + 1) * 1000
                monkeypatch.setattr(runpod_session_provider, "_epoch_ms", lambda: expired)
            return response
        return fake.handle(request)

    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    acquisition = _provider(tmp_path, fail, cutoff)
    provider = RunPodSessionProvider(acquisition, environment, snapshot)
    driver = SessionDriver(tmp_path / "driver", provider, verify_authorization=lambda _: None)
    expected = f"RunPod HTTP {failure}" if isinstance(failure, int) else {
        "deadline": "SSH readiness exceeded", "read-deadline": "SSH readiness exceeded",
        "ambiguous": "transport failed",
    }[failure]
    with pytest.raises(SessionError, match=expected):
        driver.open(launch.request, launch.authorization, adoption_token="x" * 32)
    session = driver.record(launch.request.id)
    assert set(fake.pods) == {"other"}
    if isinstance(failure, int):
        record_path = tmp_path / "acquisition records" / f"{launch.intent}.json"
        assert json.loads(record_path.read_text())["rejection"] == {
            "call": "pod", "status": failure, "title": "Request rejected",
            "detail": "entrypoint exceeded its allowed length for [redacted]",
            "errors": ["Schema error: custom error text", *[f"field {i}" for i in range(7)]],
        }
        assert session.opening_failure is not None
        assert session.opening_failure.splitlines()[1] == "Schema error: custom error text"
        assert "fixture-api-key" not in session.opening_failure
    if failure == "ambiguous":
        assert session.phase == "opening" and set(fake.secrets) == {fake.secret_id}
        with pytest.raises(SessionError, match="no reconciled venue identity"):
            driver.close(launch.request.id)
        assert not any(method == "DELETE" for method, _ in fake.requests)
    else:
        assert session.phase == "closed" and not fake.secrets
        assert session.close_evidence is not None and not session.close_evidence.remaining


def test_runpod_redacts_actual_boot_secret_and_keeps_unfamiliar_provider_vocabulary(tmp_path, monkeypatch):
    resources = runpod_resources()
    fake = FakeRunPod(resources)
    launch = _launch(resources)
    private_key = "private/input=="
    monkeypatch.setattr(runpod_session_provider, "_fresh_boot_host_key", lambda: (private_key, "ssh-ed25519 YWJj"))

    def reject(request):
        if (request.method, request.url.path) == ("POST", "/v2/account/secrets"):
            assert json.loads(request.content)["value"] == private_key
            return httpx.Response(422, json={
                "status": 422, "title": "Unexpected credential field",
                "detail": f"Unknown FieldName\u202e\n{private_key} {quote(private_key, safe='')} fixture-api-key",
                "errors": ["arbitrary vocabulary" + "x" * 700],
            }, headers={"Content-Type": "application/problem+json"})
        return fake.handle(request)

    provider = _provider(tmp_path, reject, FakeRunPodCutoff(("/usr/bin/true",)))
    with pytest.raises(SessionError, match=r"Unknown FieldName\s+\[redacted\]") as error:
        provider.acquire(launch)
    journal = (tmp_path / "acquisition records" / f"{launch.intent}.json").read_text()
    rejection = json.loads(journal)["rejection"]
    assert rejection["call"] == "boot-secret"
    assert "Unexpected credential field" in rejection["title"]
    assert len(rejection["errors"][0]) == 500
    for record in (journal, str(error.value)):
        for secret in (private_key, quote(private_key, safe=""), "fixture-api-key", "\u202e"):
            assert secret not in record
    assert ("POST", "/v2/pods") not in fake.requests


@pytest.mark.parametrize("late", ["never", "before-end", "after-end", "inventory-unavailable"])
def test_unconfirmed_create_fails_at_startup_but_preserves_cleanup_until_end(tmp_path, monkeypatch, late):
    resources = runpod_resources()
    fake = FakeRunPod(resources)
    fake.pods["other"] = {"id": "other", "name": "foreign"}
    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    snapshot = RepoSnapshotRecord(
        commit="a" * 40, dirty=False, content_sha256="b" * 64, source_state_sha256="c" * 64, file_count=0,
    )
    environment = RunPodEnvironment(
        workspace=str(tmp_path), bootstrap_python="/usr/bin/python3", uv="/usr/bin/uv",
        uv_version="1.2.3", snapshot=snapshot, backend="gpu",
    )
    launch = _launch(resources, environment=environment.digest)
    now = launch.acquired_at_ms
    monkeypatch.setattr(runpod_session_provider, "_epoch_ms", lambda: now)
    captured = []
    unavailable = False

    def handle(request):
        if (request.method, request.url.path) == ("POST", "/v2/pods"):
            assert not captured
            captured.append(request.content)
            # A 400 outside the documented error envelope is still unresolved.
            return httpx.Response(400, json={"error": "bad gateway"})
        if unavailable and (request.method, request.url.path) == ("GET", "/v2/pods"):
            raise httpx.ConnectError("inventory outage", request=request)
        return fake.handle(request)

    def driver():
        api = _provider(tmp_path, handle, cutoff)
        provider = RunPodSessionProvider(api, environment, SealedRepoSnapshot("fixture", tmp_path, tmp_path, snapshot))
        return SessionDriver(tmp_path / "driver", provider, verify_authorization=lambda _: None, clock_ms=lambda: now)

    def open_session(service):
        return service.open(launch.request, launch.authorization, adoption_token="x" * 32)

    def publish_late():
        request = httpx.Request("POST", "https://api.runpod.io/v2/pods", content=captured[0])
        fake.handle(request)

    with pytest.raises(SessionError, match="remains unresolved"):
        open_session(driver())
    with pytest.raises(SessionError, match="remains unresolved"):
        open_session(driver())
    assert set(fake.secrets) == {fake.secret_id}
    now += resources.startup_seconds * 1000
    unavailable = late == "inventory-unavailable"
    with pytest.raises(SessionError, match="did not confirm creation"):
        open_session(driver())
    record = driver().record(launch.request.id)
    assert record.phase == "closing" and record.unconfirmed_create
    assert record.opening_failure == "RunPod did not confirm creation."
    assert set(fake.secrets) == {fake.secret_id}
    assert set(fake.pods) == {"other"}
    with pytest.raises(SessionError, match="can only reconcile and close"):
        driver().adopt(launch.request, launch.authorization, adoption_token="x" * 32)
    if late == "before-end":
        publish_late()
        with pytest.raises(SessionError, match="approved end time"):
            driver().close(launch.request.id)
        assert set(fake.pods) == {"other"} and set(fake.secrets) == {fake.secret_id}

    now = record.launch.ends_at_ms
    if unavailable:
        with pytest.raises(ProviderUnreachable):
            driver().close(launch.request.id)
        assert driver().record(launch.request.id).phase == "closing"
        assert set(fake.secrets) == {fake.secret_id}
        unavailable = False
    closed = driver().close(launch.request.id)
    assert driver().record(launch.request.id).phase == "closed"
    assert bool(closed.note) == (late != "before-end")
    assert set(fake.pods) == {"other"} and not fake.secrets
    if late == "after-end":
        publish_late()
        driver().reconcile_unconfirmed_creates()
        assert set(fake.pods) == {"other"}
        assert driver().record(launch.request.id).close_evidence.note is None
    assert len(captured) == 1


def test_runpod_waits_for_direct_ssh_on_the_same_running_pod(tmp_path, monkeypatch):
    resources = RunPodResources.model_validate(runpod_resources().model_dump() | {"startup_seconds": 240})
    fake = FakeRunPod(resources)
    launch = _launch(resources)
    cutoff = FakeRunPodCutoff(("/usr/bin/true",))
    provider = _provider(tmp_path, fake.handle, cutoff)
    venue = provider.acquire(launch)
    pod = fake.pods[venue.id]
    ready = pod["ssh"]
    pod["ssh"] = {"direct": None}
    sleeps = []

    def publish_direct_endpoint(seconds: float) -> None:
        sleeps.append(seconds)
        after_cold_pull = launch.acquired_at_ms + 91_000
        monkeypatch.setattr(runpod_session_provider, "_epoch_ms", lambda: after_cold_pull)
        pod["ssh"] = ready

    monkeypatch.setattr(runpod_session_provider.time, "sleep", publish_direct_endpoint)
    probe = f"{resources.gpu_id}, 570.86.10, 8.9, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n"
    monkeypatch.setattr(runpod, "_run_command", lambda *_args, **_kwargs: runpod.CommandResult(0, probe, ""))

    endpoint = provider.endpoint(venue, opening=True)
    assert endpoint.host == "root@127.0.0.1" and endpoint.opening_deadline is not None
    assert len(sleeps) == 1
    assert fake.requests.count(("POST", "/v2/pods")) == 1
    assert not any(method == "DELETE" for method, _ in fake.requests)


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
    expired = launch.acquired_at_ms + (resources.startup_seconds + 1) * 1000
    monkeypatch.setattr(runpod_session_provider, "_epoch_ms", lambda: expired)
    with pytest.raises(SessionError, match="SSH readiness exceeded"):
        provider.endpoint(venue, opening=True)
    assert len(probe_calls) == 1
    assert provider.endpoint(venue).opening_deadline is None
    provider.begin_completion(venue, launch)
    assert provider.claim_uninitialized(venue, launch) is None

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
