from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowProvenance
from feedbax.contracts.spec_storage import training_spec_canonical_bytes
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingAssemblySpec,
    StudioTrainingIdentityAdapter,
)
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    EnvironmentDeclaration,
    RunBundle,
    RowLaunchSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.drivers.runpod import (
    CommandResult,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
    SubprocessRunPodTransport,
    build_literal_path_patch_command,
    classify_pod_state,
    compute_runpod_environment_fingerprint,
    declared_baselines,
    endpoint_classification,
    rank_datacenters_for_gpu,
)
from feedbax.orchestration.stages import StageEngine
from feedbax.orchestration.state import RowState, RunSetState, RunSetStateStore


class FakeRunPodTransport:
    def __init__(self) -> None:
        self.runpodctl_calls: list[tuple[str, ...]] = []
        self.ssh_commands: list[str] = []
        self.rsync_calls: list[tuple[str, str, bool, tuple[str, ...]]] = []
        self.runpodctl_results: dict[tuple[str, ...], list[CommandResult]] = {}
        self.ssh_results: list[CommandResult] = []
        self.sentinel_results: list[CommandResult] = []
        self.log_tail_result = CommandResult(0, "")
        self.operations: list[str] = []
        self.rsync_timeouts: list[float | None] = []
        self.rsync_result = CommandResult(0, "")

    def queue_runpodctl(self, args: tuple[str, ...], result: CommandResult) -> None:
        self.runpodctl_results.setdefault(args, []).append(result)

    def queue_ssh(self, result: CommandResult) -> None:
        self.ssh_results.append(result)

    def runpodctl(self, *args: str) -> CommandResult:
        self.runpodctl_calls.append(args)
        self.operations.append(f"runpodctl:{' '.join(args)}")
        queued = self.runpodctl_results.get(args)
        if queued:
            return queued.pop(0)
        return CommandResult(0, "{}")

    def image_exists(self, image: str) -> bool:
        return image == "runpod/pytorch:1.0.3"

    def ssh(self, command: str) -> CommandResult:
        self.ssh_commands.append(command)
        self.operations.append(f"ssh:{command}")
        if command.startswith("if [ -f ") and "printf pending" in command:
            if self.sentinel_results:
                return self.sentinel_results.pop(0)
            return CommandResult(0, "done")
        if command.startswith("tail -n 50 -- "):
            return self.log_tail_result
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
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        self.rsync_calls.append((source, target, delete, tuple(excludes)))
        self.rsync_timeouts.append(timeout_seconds)
        self.operations.append(f"rsync:{source}->{target}")
        return self.rsync_result


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _RunPodFixtureCompiler:
    """Compile a governed Studio document into one RunPod-focused v3 row."""

    def compile(
        self,
        request: RunAssemblyRequest,
        *,
        authored: dict[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del request, run_set_id, context
        payload = StudioTrainingAssemblySpec.model_validate(authored).worker_payload()
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id="warm",
                    payload=payload,
                    resolved_semantics={"payload": payload},
                    immutable_inputs=[],
                    launch=RowLaunchSpec(
                        command=["python", "-m", "feedbax.train", "--row", "warm"],
                        collect=["events/warm.events.jsonl"],
                        payload_routing={"kind": "registered-execution-payload"},
                    ),
                )
            ]
        )


def _bundle(
    tmp_path: Path,
    *,
    keep_alive: bool = False,
    deadman_enabled: bool = False,
    baseline: bool = True,
) -> RunBundle:
    training_config = None
    if baseline:
        # The baseline extension lives inside a schema-valid authored field and
        # is carried through to the registered executable payload by ASSEMBLE.
        training_config = {
            "resume": {
                "baseline_checkpoint_path": "_artifacts/run-a/checkpoint_100",
                "baseline_completed_batch": 100,
            }
        }
    authored = StudioTrainingAssemblySpec(total_batches=1, training_config=training_config)
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = tmp_path / "authored-studio-training.json"
    authored_path.write_bytes(authored_bytes)
    authored_sha = hashlib.sha256(authored_bytes).hexdigest()
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id=f"artifact://sha256/{authored_sha}",
            sha256=authored_sha,
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id="feedbax.tests.runpod-fixture",
            compiler_version="feedbax.tests.runpod-fixture.v1",
        ),
        driver="runpod",
        environment=EnvironmentDeclaration(
            python_version="3.12",
            image_id="runpod/pytorch:1.0.3",
            overlay_steps=["uv pip install extra"],
        ),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path),
        keep_alive=keep_alive,
        deadman_enabled=deadman_enabled,
        deadman_silence_seconds=60,
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=request.compiler.compiler_id,
        compiler_version=request.compiler.compiler_version,
        compiler=_RunPodFixtureCompiler(),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    return assemble_run_bundle(
        request,
        run_set_id="2026-01-02-deadbeef",
        context=AssemblyContext(custody_root=tmp_path / "custody"),
        registry=registry,
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


def test_subprocess_rsync_uses_portable_progress_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run_command(args: list[str], *, timeout_seconds: float | None = None) -> CommandResult:
        assert timeout_seconds is None
        calls.append(args)
        return CommandResult(0, "")

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)
    source = tmp_path / "checkpoint"
    source.mkdir()
    transport = SubprocessRunPodTransport(ssh_host="198.51.100.10", ssh_port=2222)

    transport.rsync(str(source) + "/", "/workspace/checkpoint/", delete=True)

    assert len(calls) == 1
    assert "--progress" in calls[0]
    assert "--stats" in calls[0]
    assert all(not arg.startswith("--info=") for arg in calls[0])


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


def test_create_pod_uses_current_runpodctl_pod_create_surface(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    base_call = (
        "pod",
        "create",
        "--name",
        "feedbax-orchestration-2026-01-02-deadbeef",
        "--image",
        "runpod/pytorch:1.0.3",
        "--ports",
        "22/tcp,8080/http",
        "--gpu-id",
        "NVIDIA GeForce RTX 4090",
    )
    first_call = (
        *base_call,
        "--data-center-ids",
        "CA-MTL-1",
        "--env",
        '{"FEEDBAX_RUNPOD_API_KEY": "dummy-key"}',
    )
    expected_call = (*base_call, "--data-center-ids", "EU-CZ-1", *first_call[-2:])
    transport.queue_runpodctl(first_call, CommandResult(1, "", "no capacity"))
    transport.queue_runpodctl(expected_call, CommandResult(0, json.dumps({"id": "pod-123"})))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=("CA-MTL-1", "EU-CZ-1"),
            api_key="dummy-key",
            image="runpod/pytorch:1.0.3",
        ),
        transport=transport,
    )

    assert driver._create_pod(bundle) == "pod-123"
    assert transport.runpodctl_calls == [first_call, expected_call]
    assert "dummy-key" not in repr(driver.config)
    assert "--gpuType" not in expected_call
    assert "--dataCenterId" not in expected_call


def test_provision_timeout_removes_pod_and_reprovisions(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    clock = FakeClock()
    create_call = (
        "pod",
        "create",
        "--name",
        "feedbax-orchestration-2026-01-02-deadbeef",
        "--image",
        "runpod/pytorch:1.0.3",
        "--ports",
        "22/tcp,8080/http",
        "--gpu-id",
        "NVIDIA RTX 2000 Ada Generation",
    )
    transport.queue_runpodctl(create_call, CommandResult(0, json.dumps({"id": "pod-1"})))
    transport.queue_runpodctl(create_call, CommandResult(0, json.dumps({"id": "pod-2"})))
    transport.queue_runpodctl(
        ("pod", "get", "pod-2", "--output", "json"),
        CommandResult(0, '{"createdAt":"now","ssh":{"ip":"203.0.113.2","port":22}}'),
    )
    transport.queue_runpodctl(("user", "--output", "json"), CommandResult(0, '{"clientBalance": 10}'))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA RTX 2000 Ada Generation",
            image="runpod/pytorch:1.0.3",
            max_acquire_seconds=0,
            max_provision_attempts=2,
            poll_seconds=1,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert all(check.status == "pass" for check in driver.preflight_checks(bundle))

    record = driver.provision(bundle, _state(bundle))

    assert ("remove", "pod", "pod-1") in transport.runpodctl_calls
    assert record["pod_id"] == "pod-2"


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
            path_patches=(
                (
                    "/workspace/feedbax/pyproject.toml",
                    "/Users/mll/local feedbax",
                    "/workspace/feedbax",
                ),
            ),
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
    assert "jax.__version__" in joined
    assert "jax[cuda12]" in joined


def test_realize_env_waits_for_delayed_done_sentinel(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={"environment": bundle.environment.model_copy(update={"overlay_steps": []})}
    )
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(1, ""))
    transport.sentinel_results = [
        CommandResult(0, "pending"),
        CommandResult(0, "pending"),
        CommandResult(0, "done"),
    ]
    clock = FakeClock()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            overlay_steps=(),
            poll_seconds=2,
            env_step_timeout_seconds=10,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    driver.realize_env(bundle, _state(bundle))

    assert clock.sleeps == [2, 2]
    fingerprint_write = next(
        index
        for index, command in enumerate(transport.ssh_commands)
        if "env-fingerprint.txt" in command and "printf %s" in command
    )
    sentinel_probe = max(
        index
        for index, command in enumerate(transport.ssh_commands)
        if "uv-sync.done" in command and "printf pending" in command
    )
    assert sentinel_probe < fingerprint_write


def test_realize_env_failed_sentinel_raises_with_remote_log_tail(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={"environment": bundle.environment.model_copy(update={"overlay_steps": []})}
    )
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(1, ""))
    transport.sentinel_results = [CommandResult(0, "failed")]
    transport.log_tail_result = CommandResult(0, "line 49\nimportant failure detail\n")
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(overlay_steps=()),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="important failure detail"):
        driver.realize_env(bundle, _state(bundle))


def test_realize_env_sentinel_timeout_raises(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={"environment": bundle.environment.model_copy(update={"overlay_steps": []})}
    )
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(1, ""))
    transport.sentinel_results = [CommandResult(0, "pending")] * 4
    clock = FakeClock()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            overlay_steps=(),
            poll_seconds=1,
            env_step_timeout_seconds=3,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(RunPodDriverError, match="uv sync timed out after 3s"):
        driver.realize_env(bundle, _state(bundle))

    assert clock.now == 3


def test_realize_env_fingerprint_match_skips_environment_steps(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fingerprint = compute_runpod_environment_fingerprint(bundle)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(0, fingerprint))
    driver = RunPodOrchestrationDriver(transport=transport)

    assert driver.realize_env(bundle, _state(bundle)) == fingerprint
    assert transport.rsync_calls == []
    assert all("uv sync --frozen" not in command for command in transport.ssh_commands)
    assert all("overlay-" not in command for command in transport.ssh_commands)


def test_literal_patch_command_uses_perl_quotemeta_not_regex_globs() -> None:
    command = build_literal_path_patch_command(
        "/workspace/feedbax/pyproject.toml", "/a/path+[x]", "/remote/path"
    )

    assert "perl -0pi" in command
    assert "\\Q$ENV{PATCH_FROM}\\E" in command
    assert "/a/path+[x]" in command
    assert "/workspace/feedbax/pyproject.toml" in command
    assert "find " not in command


def test_literal_patch_command_cannot_rewrite_out_of_scope_file() -> None:
    command = build_literal_path_patch_command(
        "/workspace/feedbax/pyproject.toml", "/local/feedbax", "/workspace/feedbax"
    )

    assert "/workspace/feedbax/other.py" not in command
    assert command.count("/workspace/feedbax/pyproject.toml") == 1


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
    assert outputs["payload_count"] == 1
    payload = bundle.rows[0].execution.payload
    assert transport.rsync_calls == [
        (str(baseline) + "/", "/workspace/_artifacts/run-a/checkpoint_100/", True, ()),
        (
            str(payload.uri),
            "/workspace/feedbax_runs/2026-01-02-deadbeef/inputs/warm.json",
            False,
            (),
        ),
    ]
    assert all((any("completed_training_batches" in command for command in transport.ssh_commands), any(payload.sha256 in command for command in transport.ssh_commands)))


def test_declared_baselines_accepts_bundle_metadata(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(
        update={"metadata": {"runpod_baselines": [{"checkpoint_path": "/custody", "completed_batches": 12000}]}}
    )

    assert declared_baselines(bundle)[-1].completed_batch == "12000"


def test_launch_row_exports_contract_env_without_per_row_deadman(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            ssh_host="198.51.100.10",
            ssh_port=2222,
        ),
        transport=transport,
    )

    outputs = driver.launch_row(bundle, bundle.rows[0], _state(bundle))

    assert outputs["pid"] == 4321
    launch_command = transport.ssh_commands[0]
    assert "nohup bash -lc" in launch_command
    assert "</dev/null" in launch_command
    assert "FEEDBAX_RUN_SET_ID=2026-01-02-deadbeef" in launch_command
    assert "FEEDBAX_ROW_ID=warm" in launch_command
    assert (
        "FEEDBAX_RUN_EVENTS_DIR=/workspace/feedbax_runs/2026-01-02-deadbeef/events"
        in launch_command
    )
    assert "FEEDBAX_ENV_FINGERPRINT=fingerprint-123" in launch_command
    assert "JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache" in launch_command
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launch_command
    assert 'kill -0 "$pid"' in launch_command
    assert (
        "orphaned launch: started sentinel present, process dead, no terminal sentinel"
        in launch_command
    )
    assert "rm -f" not in launch_command
    assert all("deadman" not in command for command in transport.ssh_commands)


def test_launch_row_injects_native_execution_context_from_bundle_row(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.rows[0]
    planned_run_id = "feedbax-training-run:planned-warm"
    row = original.model_copy(
        update={
            "launch": RowLaunchSpec(
                command=[
                    "python",
                    "-m",
                    "feedbax",
                    "execute-training-run-spec",
                    "specs/warm.json",
                ]
            ),
            "execution": original.execution.model_copy(
                update={
                    "row_provenance": TrainingRowProvenance(
                        row_id=original.row_id,
                        row_index=0,
                        planned_run_id=planned_run_id,
                        authored_payload_hash="a" * 64,
                        lowered_execution_payload_hash=original.execution.payload.sha256,
                        axis_coordinates={"temperature": 1.0},
                        lowerer_identities=[
                            RowLowererIdentity(
                                lowerer_id="feedbax.tests.runpod",
                                lowerer_version="v1",
                            )
                        ],
                    )
                }
            ),
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            ssh_host="198.51.100.10",
            ssh_port=2222,
        ),
        transport=transport,
    )

    driver.launch_row(bundle, row, _state(bundle))

    launch_command = transport.ssh_commands[0]
    assert "--execution-context-json" in launch_command
    assert planned_run_id in launch_command
    assert '"environment_fingerprint":"fingerprint-123"' in launch_command
    assert '"row_id":"warm"' in launch_command
    assert '"lowerer_id":"feedbax.tests.runpod"' in launch_command
    assert "feedbax-training-run:feedbax-training-run:" not in launch_command


def test_launch_row_routes_staged_payload_to_native_executor(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    original = bundle.rows[0]
    row = original.model_copy(
        update={
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "execute-training-run-spec"],
                payload_routing={"kind": "registered-execution-payload"},
            ),
            "execution": original.execution.model_copy(
                update={
                    "row_provenance": TrainingRowProvenance(
                        row_id=original.row_id,
                        row_index=0,
                        planned_run_id="feedbax-training-run:smoke",
                        authored_payload_hash="a" * 64,
                        lowered_execution_payload_hash=original.execution.payload.sha256,
                        axis_coordinates={},
                        lowerer_identities=[
                            RowLowererIdentity(
                                lowerer_id="feedbax.tests.runpod",
                                lowerer_version="v1",
                            )
                        ],
                    )
                }
            ),
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(config=RunPodDriverConfig(), transport=transport)

    driver.launch_row(bundle, row, _state(bundle))

    command = transport.ssh_commands[0]
    remote = "/workspace/feedbax_runs/2026-01-02-deadbeef"
    assert f"{remote}/inputs/warm.json" in command
    assert str(original.execution.payload.uri) not in command
    assert f"{remote}/rows/warm/checkpoints" in command
    assert "--manifest-root" in command
    assert "--checkpoint-root" in command
    assert "--run-id" in command
    assert "feedbax-training-run:smoke" in command


def test_deadman_disabled_when_keep_alive(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, deadman_enabled=True)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, "4321\n"))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            ssh_host="198.51.100.10",
            ssh_port=2222,
        ),
        transport=transport,
    )

    driver.launch_row(bundle, bundle.rows[0], _state(bundle))

    assert len(transport.ssh_commands) == 2  # launch + pid read, no watchdog
    assert all("deadman" not in command for command in transport.ssh_commands)


def test_deadman_is_verified_and_started_once_during_environment_realization(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, deadman_enabled=True)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            ssh_host="198.51.100.10",
            ssh_port=2222,
        ),
        transport=transport,
    )

    driver.realize_env(bundle, _state(bundle))

    joined = "\n".join(transport.ssh_commands)
    assert "command -v runpodctl" in joined
    assert all(text in joined for text in ("RUNPOD_API_KEY=$(tr", "runpodctl get pod pod-123"))
    watchdog = next(command for command in transport.ssh_commands if "deadman.pid" in command)
    assert 'kill -0 "$(cat "$pid_file")"' in watchdog
    assert 'runpodctl remove pod "$pod_id"' in watchdog
    assert ">>" in watchdog
    assert "/logs/deadman.log" in watchdog


def test_named_runpod_preflight_checks_happen_before_provision(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(0, json.dumps({"clientBalance": 12.5})),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(gpu_id="NVIDIA GeForce RTX 4090"),
        transport=transport,
    )

    checks = driver.preflight_checks(bundle)

    assert [check.name for check in checks] == [
        "runpod-image-tag-exists",
        "runpod-gpu-policy-declared",
        "runpod-credentials",
        "runpod-balance-floor",
        "runpod-deadman-credentials",
    ]
    assert all(check.status == "pass" for check in checks)
    assert transport.runpodctl_calls == [("user", "--output", "json")]


def test_stage_engine_records_named_runpod_checks_before_provision(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )
    store = RunSetStateStore(bundle.run_set_dir / "state.json")

    state = StageEngine(bundle=bundle, driver=driver, store=store).run(stop_after_stage="PREFLIGHT")

    names = [check.name for check in state.stage("PREFLIGHT").checks]
    assert "runpod-image-tag-exists" in names
    assert "runpod-balance-floor" in names
    assert state.stage("PROVISION").status == "pending"
    assert transport.ssh_commands == []
    assert transport.runpodctl_calls == []


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


def test_probe_rows_batches_all_rows_into_one_ssh_round_trip(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    second = bundle.rows[0].model_copy(update={"row_id": "second"})
    bundle = bundle.model_copy(update={"rows": [*bundle.rows, second]})
    transport = FakeRunPodTransport()
    transport.queue_ssh(
        CommandResult(
            0,
            json.dumps(
                {
                    "gpu": "RTX 5090",
                    "rows": {
                        "warm": {"status": "running", "pid": 111, "detail": None},
                        "second": {"status": "completed", "pid": 222, "detail": None},
                    },
                }
            ),
        )
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    probes = driver.probe_rows(bundle, bundle.rows, _state(bundle))

    assert probes["warm"].status == "running"
    assert probes["second"].status == "completed"
    assert len(transport.ssh_commands) == 1
    assert "'warm', 'second'" in transport.ssh_commands[0]


def test_collect_rsyncs_requested_outputs_and_verifies_payload(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    payload = bundle.run_set_dir / "collected" / "warm" / "warm.events.jsonl"
    payload.parent.mkdir(parents=True)
    payload.write_text("payload\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    row = bundle.rows[0].model_copy(
        update={
            "launch": bundle.rows[0].launch.model_copy(
                update={"metadata": {"payload_sha256": digest}}
            )
        }
    )
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


def test_collect_native_outputs_uses_row_dir_and_canonical_events(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    row = bundle.rows[0].model_copy(
        update={
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "execute-training-run-spec", "spec.json"],
                collect=["manifest.json", "training-diagnostics.json"],
            )
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(config=RunPodDriverConfig(), transport=transport)

    collected = driver.collect(bundle, row, _state(bundle))

    remote = "/workspace/feedbax_runs/2026-01-02-deadbeef"
    assert [call[0] for call in transport.rsync_calls] == [
        f"{remote}/rows/warm/manifest.json",
        f"{remote}/rows/warm/training-diagnostics.json",
        f"{remote}/events/warm.events.jsonl",
    ]
    assert collected["warm.events.jsonl"] == str(
        bundle.run_set_dir / "events/warm.events.jsonl"
    )


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


def test_abort_teardown_pulls_failure_logs_before_pod_removal(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.rsync_result = CommandResult(1, "", "diagnostic pull failed")
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            failure_log_pull_timeout_seconds=17,
        ),
        transport=transport,
    )
    engine = StageEngine(bundle=bundle, driver=driver)

    state = engine._run_teardown(_state(bundle), abort=True)

    assert state.stage("TEARDOWN").status == "completed"
    pull_index = next(
        index
        for index, operation in enumerate(transport.operations)
        if operation.startswith("rsync:")
    )
    remove_index = transport.operations.index("runpodctl:remove pod pod-123")
    assert pull_index < remove_index
    assert transport.rsync_calls[-1] == (
        "/workspace/feedbax_runs/2026-01-02-deadbeef/logs/",
        str(bundle.run_set_dir / "failure-logs") + "/",
        False,
        (),
    )
    assert transport.rsync_timeouts[-1] == 17


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
