"""Independent RunPod transport and response classification tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from feedbax.orchestration.runpod import (
    CommandResult,
    RunPodDriverError,
    SubprocessRunPodTransport,
    classify_create_failure,
    classify_pod_state,
    endpoint_classification,
    project_runpod_provision_facts,
    rank_datacenters_for_gpu,
)


def test_classifies_secure_endpoint_shapes_and_dead_states() -> None:
    ready = {
        "desiredStatus": "RUNNING",
        "ssh": {"ip": "203.0.113.10", "port": 2222, "ssh_command": "ssh root@203.0.113.10 -p 2222"},
    }
    assert classify_pod_state(ready).status == "ready"
    assert endpoint_classification(ready).kind == "ssh_command"

    command_only = {"ssh": {"ssh_command": "ssh -i key -p 39125 root@213.1.2.3"}}
    endpoint = endpoint_classification(command_only)
    assert endpoint.kind == "ssh_command"
    assert endpoint.ip == "213.1.2.3"
    assert endpoint.port == 39125

    assert (
        classify_pod_state(
            {"desiredStatus": "EXITED", "lastStatusChange": "Exited by Runpod"}
        ).status
        == "dead"
    )


def test_subprocess_transport_detaches_ssh_stdin() -> None:
    transport = SubprocessRunPodTransport(ssh_host="198.51.100.10", ssh_port=2222)

    assert transport._ssh_base(detach_stdin=True)[:2] == ["ssh", "-n"]
    assert transport._ssh_base()[0:2] != ["ssh", "-n"]
    assert (
        classify_pod_state({"desiredStatus": "RUNNING", "ssh": {"error": "pod not ready"}}).status
        == "not_ready"
    )


def test_ranks_datacenters_for_gpu_by_stock() -> None:
    datacenters: list[dict[str, Any]] = [
        {"id": "low", "gpuAvailability": [{"gpuId": "RTX_5090", "stockStatus": "Low"}]},
        {"id": "none", "gpuAvailability": [{"gpuId": "RTX_5090", "stockStatus": ""}]},
        {"id": "high", "gpuAvailability": [{"gpuId": "RTX_5090", "stockStatus": "High"}]},
        {"id": "other", "gpuAvailability": [{"gpuId": "A100", "stockStatus": "High"}]},
        {"id": "medium", "gpuAvailability": [{"gpuId": "RTX_5090", "stockStatus": "Medium"}]},
    ]

    assert rank_datacenters_for_gpu(datacenters, "RTX_5090") == [
        "high",
        "medium",
        "low",
        "none",
    ]


def test_projects_provision_facts_from_existing_pod_response() -> None:
    facts = project_runpod_provision_facts(
        {
            "dataCenter": {"id": "CA-MTL-1"},
            "machine": {"costPerHr": "0.74"},
            "template": {"imageName": "runpod/pytorch@sha256:" + "a" * 64},
            "createdAt": "2026-07-19 18:05:00.898 +0000 UTC",
        }
    )

    assert facts == {
        "provider": "runpod",
        "region": "CA-MTL-1",
        "immutable_image_id": "runpod/pytorch@sha256:" + "a" * 64,
        "hourly_rate": 0.74,
        "hourly_rate_raw": "0.74",
        "billing_started_at": "2026-07-19T18:05:00.898000+00:00",
        "billing_started_at_raw": "2026-07-19 18:05:00.898 +0000 UTC",
        "currency": "USD",
        "provider_observation_basis": "runpodctl pod get response",
    }


@pytest.mark.parametrize("raw_rate", ["invalid", "nan", "inf"])
def test_projects_raw_malformed_or_non_finite_provision_rate(raw_rate: str) -> None:
    facts = project_runpod_provision_facts(
        {
            "imageName": "runpod/pytorch@sha256:" + "b" * 64,
            "costPerHr": raw_rate,
        }
    )

    assert facts["hourly_rate_raw"] == raw_rate
    assert facts["hourly_rate"] is None
    assert facts["immutable_image_id"] == "runpod/pytorch@sha256:" + "b" * 64


@pytest.mark.parametrize(
    ("raw_timestamp", "canonical"),
    [
        ("2026-07-19T18:05:00.898Z", "2026-07-19T18:05:00.898000+00:00"),
        ("2026-07-19T14:05:00.898-04:00", "2026-07-19T18:05:00.898000+00:00"),
        ("2026-07-19 18:05:00 +0000 UTC", "2026-07-19T18:05:00+00:00"),
        ("2026-07-19 18:05:00.898 +0000 UTC", "2026-07-19T18:05:00.898000+00:00"),
        ("2026-07-19 18:05:00.898 UTC", None),
        ("not-a-timestamp", None),
        (None, None),
    ],
)
def test_projects_canonical_and_raw_runpod_billing_timestamp(
    raw_timestamp: str | None,
    canonical: str | None,
) -> None:
    facts = project_runpod_provision_facts({"createdAt": raw_timestamp})

    assert facts["billing_started_at"] == canonical
    assert facts["billing_started_at_raw"] == raw_timestamp


@pytest.mark.parametrize("digest_matches", [True, False])
def test_image_probe_verifies_the_immutable_manifest(digest_matches: bool) -> None:
    digest = "sha256:" + "a" * 64
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "auth.docker.io":
            assert request.method == "GET"
            assert request.url.params["scope"] == "repository:runpod/base:pull"
            return httpx.Response(200, json={"token": "public-read-token"})
        assert request.method == "HEAD"
        assert request.url.host == "registry-1.docker.io"
        assert request.url.path == f"/v2/runpod/base/manifests/{digest}"
        assert request.headers["Authorization"] == "Bearer public-read-token"
        return httpx.Response(
            200, headers={"Docker-Content-Digest": digest if digest_matches else "other"}
        )

    transport = SubprocessRunPodTransport(http_transport=httpx.MockTransport(handle))
    assert not transport.image_exists("runpod/base:latest")
    assert not requests
    assert transport.image_exists(f"runpod/base@{digest}") is digest_matches
    assert len(requests) == 2


def test_provision_projection_does_not_invent_declared_image() -> None:
    facts = project_runpod_provision_facts({"costPerHr": 0.5})

    assert facts["immutable_image_id"] is None


def test_subprocess_rsync_protects_remote_upload_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run_command(args: list[str], *, timeout_seconds: float | None = None) -> CommandResult:
        assert timeout_seconds is None
        calls.append(args)
        if args[-1] == "--version":
            return CommandResult(0, "rsync 3.4.1")
        return CommandResult(0, "")

    monkeypatch.setattr("feedbax.orchestration.runpod._run_command", run_command)
    monkeypatch.setattr(
        "feedbax.orchestration.runpod.shutil.which",
        lambda executable: "/opt/homebrew/bin/rsync",
    )
    source = tmp_path / "checkpoint [draft]; $HOME"
    source.mkdir()
    transport = SubprocessRunPodTransport(
        ssh_host="198.51.100.10",
        ssh_port=2222,
        ssh_key_path="/keys/runpod key",
    )

    transport.rsync(
        str(source) + "/",
        "/workspace/checkout [draft]; $HOME/",
        delete=True,
        excludes=("*.pyc",),
    )

    assert calls == [
        ["/opt/homebrew/bin/rsync", "--version"],
        ["/opt/homebrew/bin/rsync", "--secluded-args", "--version"],
        [
            "/opt/homebrew/bin/rsync",
            "-az",
            "--no-owner",
            "--no-group",
            "--secluded-args",
            "--progress",
            "--stats",
            "--delete",
            "--exclude",
            "*.pyc",
            "-e",
            "ssh -i '/keys/runpod key' -p 2222 -o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null",
            str(source) + "/",
            "root@198.51.100.10:/workspace/checkout [draft]; $HOME/",
        ],
    ]


def test_subprocess_rsync_protects_remote_download_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run_command(args: list[str], *, timeout_seconds: float | None = None) -> CommandResult:
        calls.append(args)
        if args[-1] == "--version":
            assert timeout_seconds is None
            if "--secluded-args" in args:
                return CommandResult(1, stderr="rsync: unrecognized option `--secluded-args'")
            return CommandResult(0, "openrsync: protocol version 29")
        assert timeout_seconds == 17
        return CommandResult(0, "")

    monkeypatch.setattr("feedbax.orchestration.runpod._run_command", run_command)
    monkeypatch.setattr(
        "feedbax.orchestration.runpod.shutil.which",
        lambda executable: "/usr/bin/rsync",
    )
    target = tmp_path / "collected outputs"
    key_path = tmp_path / "runpod key"
    transport = SubprocessRunPodTransport(
        ssh_host="198.51.100.10",
        ssh_port=2222,
        ssh_key_path=key_path,
    )

    transport.rsync(
        "/workspace/run [r5]; $(touch nope)/",
        str(target) + "/",
        timeout_seconds=17,
    )

    assert calls == [
        ["/usr/bin/rsync", "--version"],
        ["/usr/bin/rsync", "--secluded-args", "--version"],
        [
            "/usr/bin/rsync",
            "-az",
            "--no-owner",
            "--no-group",
            "--progress",
            "--stats",
            "-e",
            f"ssh -i '{key_path}' -p 2222 -o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null",
            "root@198.51.100.10:'/workspace/run [r5]; $(touch nope)/'",
            str(target) + "/",
        ],
    ]


def test_subprocess_rsync_preserves_normal_path_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run_command(args: list[str], *, timeout_seconds: float | None = None) -> CommandResult:
        calls.append(args)
        if args[-1] == "--version":
            if "--secluded-args" in args:
                return CommandResult(1, stderr="rsync: unrecognized option `--secluded-args'")
            return CommandResult(0, "openrsync: protocol version 29")
        return CommandResult(0, "")

    monkeypatch.setattr("feedbax.orchestration.runpod._run_command", run_command)
    monkeypatch.setattr(
        "feedbax.orchestration.runpod.shutil.which",
        lambda executable: "/usr/bin/rsync",
    )
    source = tmp_path / "checkpoint"
    source.mkdir()
    transport = SubprocessRunPodTransport(ssh_host="198.51.100.10", ssh_port=2222)

    transport.rsync(str(source) + "/", "/workspace/checkpoint/", delete=True)

    assert calls[-1][-2:] == [
        str(source) + "/",
        "root@198.51.100.10:/workspace/checkpoint/",
    ]
    assert all(not arg.startswith("--info=") for arg in calls[-1])


@pytest.mark.parametrize("direction", ["upload", "download"])
@pytest.mark.parametrize(
    "rsync_executable",
    [
        pytest.param("/usr/bin/rsync", id="apple-openrsync"),
        pytest.param("/opt/homebrew/bin/rsync", id="modern-rsync"),
    ],
)
def test_subprocess_rsync_real_protocol_preserves_exact_remote_path(
    direction: str,
    rsync_executable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path(rsync_executable).is_file():
        pytest.skip(f"rsync implementation is not installed: {rsync_executable}")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        "    -i|-p|-o|-l) shift 2 ;;\n"
        "    -n) shift ;;\n"
        "    *) shift; break ;;\n"
        "  esac\n"
        "done\n"
        '[ "$1" = rsync ] || exit 92\n'
        'if [ -n "$FEEDBAX_FAKE_REMOTE_SOURCE" ]; then\n'
        '  mkdir -p "$FEEDBAX_FAKE_REMOTE_SOURCE" || exit 93\n'
        '  printf download > "$FEEDBAX_FAKE_REMOTE_SOURCE/payload.txt" || exit 94\n'
        "fi\n"
        "shift\n"
        'exec /bin/sh -c "$FEEDBAX_REMOTE_RSYNC $*"\n'
    )
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("FEEDBAX_REMOTE_RSYNC", rsync_executable)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    local_path = tmp_path / "local payload [r5]; $HOME"
    remote_path = tmp_path / "remote payload [r5]; $(printf injected)"
    marker = "payload.txt"
    if direction == "upload":
        local_path.mkdir()
        (local_path / marker).write_text("upload")
        source, target = str(local_path) + "/", str(remote_path) + "/"
    else:
        monkeypatch.setenv("FEEDBAX_FAKE_REMOTE_SOURCE", str(remote_path))
        local_path.mkdir()
        source, target = str(remote_path) + "/", str(local_path) + "/"
    transport = SubprocessRunPodTransport(
        ssh_host="198.51.100.10",
        ssh_port=2222,
        rsync_executable=rsync_executable,
    )

    result = transport.rsync(source, target)

    assert result.returncode == 0, result.stderr
    destination = remote_path if direction == "upload" else local_path
    assert (destination / marker).read_text() == direction
    assert not any("'" in path.name for path in tmp_path.rglob("*"))


def test_subprocess_rsync_fails_before_transfer_when_executable_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "feedbax.orchestration.runpod.shutil.which",
        lambda executable: None,
    )
    monkeypatch.setattr(
        "feedbax.orchestration.runpod._run_command",
        lambda args, **kwargs: calls.append(args),
    )
    transport = SubprocessRunPodTransport(ssh_host="198.51.100.10", ssh_port=2222)

    with pytest.raises(RunPodDriverError, match="rsync executable is unavailable"):
        transport.rsync(str(source) + "/", "/workspace/target/")

    assert calls == []


@pytest.mark.parametrize(
    ("probe_result", "error"),
    [
        (CommandResult(72, stderr="loader failure"), "rsync executable is unusable"),
        (
            CommandResult(0, "rsync 3.4.1"),
            "could not determine rsync secluded-argument support",
        ),
    ],
    ids=["unusable", "ambiguous-capability"],
)
def test_subprocess_rsync_fails_before_transfer_on_invalid_capability_probe(
    probe_result: CommandResult,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "feedbax.orchestration.runpod.shutil.which",
        lambda executable: "/resolved/rsync",
    )

    def run_command(args: list[str], **kwargs: Any) -> CommandResult:
        calls.append(args)
        if len(calls) == 1:
            return probe_result
        return CommandResult(74, stderr="unexpected secluded probe failure")

    monkeypatch.setattr("feedbax.orchestration.runpod._run_command", run_command)
    transport = SubprocessRunPodTransport(ssh_host="198.51.100.10", ssh_port=2222)

    with pytest.raises(RunPodDriverError, match=error):
        transport.rsync(str(source) + "/", "/workspace/target/")

    expected_calls = 1 if probe_result.returncode else 2
    assert len(calls) == expected_calls


def test_subprocess_runpodctl_applies_endpoint_poll_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[list[str], float | None]] = []

    def run_command(
        args: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        observed.append((args, timeout_seconds))
        return CommandResult(0, "{}")

    monkeypatch.setattr("feedbax.orchestration.runpod._run_command", run_command)

    SubprocessRunPodTransport().runpodctl(
        "pod",
        "get",
        "pod-1",
        "--output",
        "json",
        timeout_seconds=12.5,
    )

    assert observed == [(["runpodctl", "pod", "get", "pod-1", "--output", "json"], 12.5)]


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(1, "This machine does not have the resources to deploy your pod"),
        CommandResult(
            1,
            "There are no longer any instances available with the requested specifications.",
        ),
        CommandResult(1, '{"error":"pod limit exceeded for this account"}'),
        CommandResult(1, "", "transport connection lost"),
    ],
)
def test_unstructured_or_lost_create_response_remains_ambiguous(result: CommandResult) -> None:
    classification, _detail = classify_create_failure(result, None)

    assert classification == "ambiguous"


CREATE_FAILURE_CORPUS_PATH = (
    Path(__file__).parent / "fixtures" / "runpod_provider_errors" / "create_failure_corpus.json"
)


def _create_failure_corpus() -> list[dict[str, Any]]:
    """Regression corpus for feedbax/32d1d73; see the fixture README for curation policy."""
    return json.loads(CREATE_FAILURE_CORPUS_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", _create_failure_corpus(), ids=lambda entry: entry["name"])
def test_create_failure_corpus_classification(entry: dict[str, Any]) -> None:
    """Every curated payload must keep its human-adjudicated classification."""
    for stdout, stderr in ((entry["stdout"], entry["stderr"]), (entry["stderr"], entry["stdout"])):
        result = CommandResult(entry["returncode"], stdout, stderr)
        classification, _detail = classify_create_failure(result, None)
        assert classification == entry["expected_classification"], entry["name"]
