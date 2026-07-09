from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

from feedbax.orchestration.bundle import (
    BudgetPolicy,
    EnvironmentDeclaration,
    RunBundle,
    RunRowSpec,
)
from feedbax.orchestration.drivers.runpod import (
    CommandResult,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
    build_literal_path_patch_command,
    classify_pod_state,
    endpoint_classification,
    rank_datacenters_for_gpu,
)
from feedbax.orchestration.state import RowState, RunSetState


class FakeRunPodTransport:
    def __init__(self) -> None:
        self.runpodctl_calls: list[tuple[str, ...]] = []
        self.ssh_commands: list[str] = []
        self.rsync_calls: list[tuple[str, str, bool, tuple[str, ...]]] = []
        self.runpodctl_results: dict[tuple[str, ...], list[CommandResult]] = {}
        self.ssh_results: list[CommandResult] = []

    def queue_runpodctl(self, args: tuple[str, ...], result: CommandResult) -> None:
        self.runpodctl_results.setdefault(args, []).append(result)

    def queue_ssh(self, result: CommandResult) -> None:
        self.ssh_results.append(result)

    def runpodctl(self, *args: str) -> CommandResult:
        self.runpodctl_calls.append(args)
        queued = self.runpodctl_results.get(args)
        if queued:
            return queued.pop(0)
        return CommandResult(0, "{}")

    def ssh(self, command: str) -> CommandResult:
        self.ssh_commands.append(command)
        if command.startswith("cat ") and not self.ssh_results:
            return CommandResult(0, "4321\n")
        if self.ssh_results:
            return self.ssh_results.pop(0)
        return CommandResult(0, "")

    def rsync(
        self,
        source: str,
        target: str,
        *,
        delete: bool = False,
        excludes: tuple[str, ...] = (),
    ) -> CommandResult:
        self.rsync_calls.append((source, target, delete, tuple(excludes)))
        return CommandResult(0, "")


def _bundle(tmp_path: Path, *, keep_alive: bool = False) -> RunBundle:
    return RunBundle(
        run_set_id="2026-01-02-deadbeef",
        driver="runpod",
        rows=[
            RunRowSpec(
                row_id="warm",
                command=[sys.executable, "-m", "feedbax.train", "--row", "warm"],
                collect=["events/warm.events.jsonl"],
                run_spec={
                    "resume": {
                        "baseline_checkpoint_path": "_artifacts/run-a/checkpoint_100",
                        "baseline_completed_batch": 100,
                    }
                },
            )
        ],
        environment=EnvironmentDeclaration(
            python_version="3.12", overlay_steps=["uv pip install extra"]
        ),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path),
        keep_alive=keep_alive,
    )


def _state(bundle: RunBundle) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState() for row in bundle.rows},
        environment_fingerprint="fingerprint-123",
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

    assert rank_datacenters_for_gpu(datacenters, "RTX_5090") == ["high", "medium", "low"]


def test_provision_reuses_provided_endpoint_and_disables_teardown(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    record = driver.provision(bundle, _state(bundle))
    teardown = driver.teardown(bundle, _state(bundle))

    assert record["provided_endpoint"] is True
    assert "nvidia-smi >/dev/null" in transport.ssh_commands
    assert teardown["teardown"] == "skipped"
    assert transport.runpodctl_calls == []


def test_realize_env_rsyncs_repos_literal_patches_and_bootstrap(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))  # mkdir
    transport.queue_ssh(CommandResult(1, ""))  # fingerprint probe mismatch
    local_repo = tmp_path / "feedbax"
    local_repo.mkdir()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            local_repos={"feedbax": local_repo},
            remote_repos={"feedbax": "/workspace/feedbax"},
            path_patches=(("/Users/mll/local feedbax", "/workspace/feedbax"),),
        ),
        transport=transport,
    )

    fingerprint = driver.realize_env(bundle, _state(bundle))

    assert fingerprint
    assert transport.rsync_calls == [
        (
            str(local_repo) + "/",
            "/workspace/feedbax/",
            True,
            (
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "_artifacts",
                "web/node_modules",
            ),
        )
    ]
    joined = "\n".join(transport.ssh_commands)
    assert "perl -0pi" in joined
    assert "\\Q$ENV{PATCH_FROM}\\E" in joined
    assert "PATCH_FROM='/Users/mll/local feedbax'" in joined
    assert "uv sync --frozen" in joined
    assert "uv pip install extra" in joined
    assert "uv pip install -U" in joined
    assert "jax[cuda12]" in joined


def test_literal_patch_command_uses_perl_quotemeta_not_regex_globs() -> None:
    command = build_literal_path_patch_command(
        "/a/path+[x]", "/remote/path", ["/workspace/feedbax"]
    )

    assert "perl -0pi" in command
    assert "\\Q$ENV{PATCH_FROM}\\E" in command
    assert "/a/path+[x]" in command


def test_stage_inputs_stages_and_verifies_declared_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    baseline = repo / "_artifacts" / "run-a" / "checkpoint_100"
    baseline.mkdir(parents=True)
    (baseline / "latest.json").write_text(
        json.dumps({"completed_training_batches": 100}), encoding="utf-8"
    )
    monkeypatch.chdir(repo)
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    outputs = driver.stage_inputs(bundle, _state(bundle))

    assert outputs["baseline_count"] == 1
    assert transport.rsync_calls == [
        (str(baseline) + "/", "/workspace/_artifacts/run-a/checkpoint_100/", True, ())
    ]
    assert "completed_training_batches" in transport.ssh_commands[-1]


def test_launch_row_exports_contract_env_and_starts_deadman(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            ssh_host="198.51.100.10",
            ssh_port=2222,
            deadman_enabled=True,
            deadman_silence_seconds=60,
        ),
        transport=transport,
    )

    outputs = driver.launch_row(bundle, bundle.rows[0], _state(bundle))

    assert outputs["pid"] == 4321
    launch_command = transport.ssh_commands[0]
    assert "nohup bash -lc" in launch_command
    assert "FEEDBAX_RUN_SET_ID=2026-01-02-deadbeef" in launch_command
    assert "FEEDBAX_ROW_ID=warm" in launch_command
    assert (
        "FEEDBAX_RUN_EVENTS_DIR=/workspace/feedbax_runs/2026-01-02-deadbeef/events"
        in launch_command
    )
    assert "FEEDBAX_ENV_FINGERPRINT=fingerprint-123" in launch_command
    assert "JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache" in launch_command
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launch_command
    assert "kill -0 \"$pid\"" in launch_command
    assert "orphaned launch: started sentinel present, process dead, no terminal sentinel" in launch_command
    assert "rm -f" not in launch_command
    assert 'runpodctl remove pod "$pod_id"' in transport.ssh_commands[1]


def test_deadman_disabled_when_keep_alive(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, "4321\n"))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            ssh_host="198.51.100.10",
            ssh_port=2222,
            deadman_enabled=True,
        ),
        transport=transport,
    )

    driver.launch_row(bundle, bundle.rows[0], _state(bundle))

    assert len(transport.ssh_commands) == 2  # launch + pid read, no watchdog
    assert all("deadman" not in command for command in transport.ssh_commands)


def test_probe_parses_one_round_trip_report(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_ssh(
        CommandResult(
            0,
            json.dumps(
                {
                    "gpu": "RTX 5090, 12, 1024, 32000",
                    "rows": {"warm": {"status": "running", "pid": 111, "detail": None}},
                }
            ),
        )
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    probe = driver.probe(bundle, bundle.rows[0], _state(bundle))

    assert probe.status == "running"
    assert probe.pid == 111
    assert probe.metadata and probe.metadata["gpu"].startswith("RTX 5090")


def test_collect_rsyncs_requested_outputs_and_verifies_payload(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    payload = bundle.run_set_dir / "collected" / "warm" / "warm.events.jsonl"
    payload.parent.mkdir(parents=True)
    payload.write_text("payload\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    row = bundle.rows[0].model_copy(update={"payload_sha256": digest})
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    collected = driver.collect(bundle, row, _state(bundle))

    assert collected == {"warm.events.jsonl": str(payload)}
    assert transport.rsync_calls == [
        (
            "/workspace/feedbax_runs/2026-01-02-deadbeef/events/warm.events.jsonl",
            str(payload),
            False,
            (),
        )
    ]


def test_teardown_removes_acquired_pod_and_falls_back_to_stop(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "busy"))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    result = driver.teardown(bundle, _state(bundle))

    assert result["teardown"] == "stopped"
    assert ("remove", "pod", "pod-123") in transport.runpodctl_calls
    assert ("stop", "pod", "pod-123") in transport.runpodctl_calls


def test_local_baseline_mismatch_raises_before_rsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    baseline = repo / "_artifacts" / "run-a" / "checkpoint_100"
    baseline.mkdir(parents=True)
    (baseline / "latest.json").write_text(
        json.dumps({"completed_training_batches": 99}), encoding="utf-8"
    )
    monkeypatch.chdir(repo)
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="completed_batch mismatch"):
        driver.stage_inputs(bundle, _state(bundle))

    assert transport.rsync_calls == []
