from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shlex
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
    DeploymentPolicy,
    DeploymentResourceRequest,
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
    build_launch_row_command,
    build_literal_path_patch_command,
    build_remote_nohup_sentinel_command,
    classify_pod_state,
    compute_runpod_environment_fingerprint,
    endpoint_classification,
    project_runpod_provision_facts,
    rank_datacenters_for_gpu,
)
from feedbax.orchestration.drivers.base import ProvisioningAttemptError
from feedbax.training.interruption import CancellationDecision
from feedbax.orchestration.conformance import CheckEntry, CheckRegistry
from feedbax.orchestration.stages import (
    STAGE_PROVISION,
    STAGE_REALIZE_ENV,
    STAGE_STAGE_INPUTS,
    OrchestrationStageError,
    StageEngine,
)
from feedbax.orchestration.state import RowState, RunSetState, RunSetStateStore


class FakeRunPodTransport:
    def __init__(self) -> None:
        self.runpodctl_calls: list[tuple[str, ...]] = []
        self.runpodctl_timeouts: list[float | None] = []
        self.ssh_commands: list[str] = []
        self.rsync_calls: list[tuple[str, str, bool, tuple[str, ...]]] = []
        self.runpodctl_results: dict[tuple[str, ...], list[CommandResult]] = {}
        self.ssh_results: list[CommandResult] = []
        self.sentinel_results: list[CommandResult] = []
        self.log_tail_result = CommandResult(0, "")
        self.operations: list[str] = []
        self.rsync_timeouts: list[float | None] = []
        self.rsync_result = CommandResult(0, "")
        self.environment_probe_result: CommandResult | None = None

    def queue_runpodctl(self, args: tuple[str, ...], result: CommandResult) -> None:
        self.runpodctl_results.setdefault(args, []).append(result)

    def queue_empty_global_inventory(self, payload: str = "[]") -> None:
        self.queue_runpodctl(
            ("pod", "list", "--output", "json"),
            CommandResult(0, payload),
        )

    def queue_ssh(self, result: CommandResult) -> None:
        self.ssh_results.append(result)

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        self.runpodctl_calls.append(args)
        self.runpodctl_timeouts.append(timeout_seconds)
        self.operations.append(f"runpodctl:{' '.join(args)}")
        queued = self.runpodctl_results.get(args)
        if queued:
            return queued.pop(0)
        return CommandResult(0, "{}")

    def image_exists(self, image: str) -> bool:
        return image == "runpod/pytorch:1.0.3@sha256:" + "a" * 64

    def ssh(self, command: str) -> CommandResult:
        self.ssh_commands.append(command)
        self.operations.append(f"ssh:{command}")
        if command.startswith("if [ -f ") and "printf pending" in command:
            if self.sentinel_results:
                return self.sentinel_results.pop(0)
            return CommandResult(0, "done")
        if command.startswith("tail -n 50 -- "):
            return self.log_tail_result
        if (
            "uv run --no-sync python -c" in command
            and "feedbax.runpod_environment_fingerprint.v1" in command
        ):
            if self.environment_probe_result is not None:
                return self.environment_probe_result
            declaration = json.loads(shlex.split(command)[-1])
            return CommandResult(0, _realized_fingerprint(declaration))
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
        *,
        authored: dict[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del run_set_id, context
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
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    lockfile_sha256 = hashlib.sha256(lockfile.read_bytes()).hexdigest()
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
        deployment_policy=DeploymentPolicy(
            driver="runpod",
            venue="remote",
            cloud_authorized=True,
            review_required=False,
            review_authorized=False,
            resources=DeploymentResourceRequest(
                gpu_id="NVIDIA GeForce RTX 4090",
                regions=["CA-MTL-1", "US-OR-1"],
            ),
        ),
        environment=EnvironmentDeclaration(
            python_version="3.12",
            image_id="runpod/pytorch:1.0.3@sha256:" + "a" * 64,
            lockfile_hashes={"uv.lock": lockfile_sha256},
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


def _realized_fingerprint(declaration: dict[str, Any]) -> str:
    return json.dumps(
        {
            "schema_version": "feedbax.runpod_environment_fingerprint.v1",
            "declaration_sha256": declaration["declaration_sha256"],
            "image_id": declaration["image_id"],
            "lockfile_hashes": declaration["lockfile_hashes"],
            "runtime": {
                "device_count": 1,
                "device_kind": "NVIDIA GeForce RTX 5090",
                "equinox": "0.13.2",
                "feedbax": "0.1.0",
                "jax": "0.7.2",
                "jax_platform": "gpu",
                "jax_platform_version": "CUDA 12.8",
                "jaxlib": "0.7.2",
                "python": "3.12.8",
                "python_implementation": "CPython",
            },
            "feedbax_plugins": [
                {
                    "distribution": "rlrmp2",
                    "distribution_version": "0.1.0",
                    "name": "rlrmp2",
                    "value": "rlrmp2.feedbax_plugin",
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _state(bundle: RunBundle) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState() for row in bundle.rows},
        environment_fingerprint="fingerprint-123",
    )


class GovernedProvisionDriver:
    """Fake one-attempt RunPod driver for stage retry policy tests."""

    govern_provisioning_retries = True
    provision_retry_delay_seconds = 1.0

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
        del bundle, state
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return dict(outcome)


def _failed_attempt(
    *,
    acquired: bool = False,
    retryable: bool = True,
    stop_reason: str | None = None,
    billing: bool = False,
) -> ProvisioningAttemptError:
    record: dict[str, Any] = {"driver": "runpod", "acquired": acquired}
    if acquired:
        record["cleanup"] = {"pod_absence": {"verified": True}}
    if billing:
        record.update(
            {
                "billing_started_at": "1969-12-31T23:59:59+00:00",
                "hourly_rate": 7200.0,
                "currency": "USD",
            }
        )
    return ProvisioningAttemptError(
        "transient provisioning failure",
        retryable=retryable,
        attempt_record=record,
        stop_reason=stop_reason,
    )


def _governed_engine(tmp_path: Path, outcomes: list[object], **kwargs: Any) -> tuple[StageEngine, GovernedProvisionDriver, RunSetStateStore]:
    bundle = _bundle(tmp_path)
    clock = FakeClock()
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = GovernedProvisionDriver(outcomes)
    return StageEngine(bundle=bundle, driver=driver, store=store, sleep=clock.sleep, wall_time=clock.monotonic, **kwargs), driver, store


def test_stage_engine_governs_runpod_provisioning(tmp_path: Path) -> None:
    engine, driver, store = _governed_engine(
        tmp_path, [*[_failed_attempt() for _ in range(10)], {"pod_id": "ok"}]
    )
    state = engine.run(stop_after_stage=STAGE_PROVISION)
    assert driver.calls == 11 and len(state.provisioning_attempts) == 10

    deadline_root = tmp_path / "deadline"
    deadline_root.mkdir()
    engine, driver, store = _governed_engine(
        deadline_root, [_failed_attempt(), {"pod_id": "must-not-run"}]
    )
    engine.bundle = engine.bundle.model_copy(
        update={"budget": BudgetPolicy(max_wall_clock_seconds=1)}
    )
    with pytest.raises(OrchestrationStageError, match="wall-clock"):
        engine.run(stop_after_stage=STAGE_PROVISION)
    assert driver.calls == 1 and store.load().provisioning_stop_reason == "wall-clock-exceeded"


@pytest.mark.parametrize(
    ("failure", "reason", "probe"),
    [
        (_failed_attempt(retryable=False), "non-retryable-error", None),
        (_failed_attempt(acquired=True, stop_reason="teardown-failure"), "teardown-failure", None),
        (None, "cancelled", lambda: CancellationDecision("stop", "test", 0.0)),
    ],
)
def test_stage_engine_does_not_retry_governed_terminal_stops(
    tmp_path: Path, failure: ProvisioningAttemptError | None, reason: str, probe: Any
) -> None:
    engine, driver, store = _governed_engine(tmp_path, [failure] if failure else [{"pod_id": "no"}], interruption_probe=probe)
    with pytest.raises(OrchestrationStageError, match=reason):
        engine.run(stop_after_stage=STAGE_PROVISION)
    assert driver.calls == (0 if failure is None else 1)
    assert store.load().provisioning_stop_reason == reason


def test_provisioning_resume_reuses_deadline_and_failed_cost(tmp_path: Path) -> None:
    engine, driver, store = _governed_engine(
        tmp_path,
        [_failed_attempt(acquired=True, billing=True), _failed_attempt(acquired=True, billing=True)],
    )
    engine.bundle = engine.bundle.model_copy(
        update={"budget": BudgetPolicy(max_wall_clock_seconds=30, max_spend_usd=3.0)}
    )
    engine._sleep = lambda _seconds: (_ for _ in ()).throw(RuntimeError("restart"))
    with pytest.raises(RuntimeError, match="restart"):
        engine.run(stop_after_stage=STAGE_PROVISION)
    interrupted = store.load()
    deadline = interrupted.budget_counters["provisioning_deadline_at"]
    cost = interrupted.budget_counters["failed_provision_cost_usd"]
    resumed_engine = StageEngine(
        bundle=engine.bundle, driver=driver, store=store, wall_time=engine._wall_time
    )
    with pytest.raises(OrchestrationStageError, match="spend-exceeded"):
        resumed_engine.run(stop_after_stage=STAGE_PROVISION)
    resumed = store.load()
    assert resumed.budget_counters["provisioning_deadline_at"] == deadline
    assert resumed.budget_counters["failed_provision_cost_usd"] > cost


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
        }
    )

    assert facts == {
        "provider": "runpod",
        "region": "CA-MTL-1",
        "immutable_image_id": "runpod/pytorch@sha256:" + "a" * 64,
        "hourly_rate": 0.74,
        "hourly_rate_raw": "0.74",
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


def test_provision_projection_does_not_invent_declared_image() -> None:
    facts = project_runpod_provision_facts({"costPerHr": 0.5})

    assert facts["immutable_image_id"] is None


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

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)

    SubprocessRunPodTransport().runpodctl(
        "pod",
        "get",
        "pod-1",
        "--output",
        "json",
        timeout_seconds=12.5,
    )

    assert observed == [
        (["runpodctl", "pod", "get", "pod-1", "--output", "json"], 12.5)
    ]


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


def test_provided_endpoint_certify_fails_without_provider_realization_facts(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )
    provision_record = dict(driver.provision(bundle, _state(bundle)))
    declaration_fingerprint = compute_runpod_environment_fingerprint(bundle)
    fingerprint = _realized_fingerprint(
        {
            "declaration_sha256": declaration_fingerprint,
            "image_id": bundle.environment.image_id,
            "lockfile_hashes": bundle.environment.lockfile_hashes,
            "python_version": bundle.environment.python_version,
        }
    )
    completed_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)
    started_at = completed_at - timedelta(seconds=1)
    row_id = bundle.rows[0].row_id
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        provision_record=provision_record,
        environment_fingerprint=fingerprint,
        rows={
            row_id: RowState(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
            )
        },
    )
    state = state.with_stage(
        STAGE_PROVISION,
        state.stage(STAGE_PROVISION).model_copy(
            update={
                "status": "completed",
                "completed_at": started_at - timedelta(seconds=1),
                "outputs": provision_record,
            }
        ),
    )
    state = state.with_stage(
        STAGE_REALIZE_ENV,
        state.stage(STAGE_REALIZE_ENV).model_copy(
            update={
                "status": "completed",
                "completed_at": started_at,
                "outputs": {"environment_fingerprint": fingerprint},
            }
        ),
    )
    state = state.with_stage(
        STAGE_STAGE_INPUTS,
        state.stage(STAGE_STAGE_INPUTS).model_copy(
            update={"status": "completed", "outputs": {"inputs": [], "payloads": []}}
        ),
    )
    engine = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=CheckRegistry(
            {"fixture": lambda row: CheckEntry(check_id="fixture", status="pass")}
        ),
    )

    _state_after, outputs = engine._stage_certify(state)
    certificate = json.loads((bundle.run_set_dir / "conformance.json").read_text())
    row = certificate["rows"][row_id]
    realized_check = next(
        check for check in row["checks"] if check["check_id"] == "realized_deployment"
    )

    assert outputs["overall"] == "fail"
    assert realized_check["status"] == "fail"
    assert row["realized_deployment_evidence"]["provider"] is None
    assert row["realized_deployment_evidence"]["immutable_image_id"] is None
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


def test_provider_authorization_failure_stops_stage_once(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    call = (
        "pod", "create", "--name", "feedbax-orchestration-2026-01-02-deadbeef",
        "--image", "runpod/pytorch:1.0.3@sha256:" + "a" * 64,
        "--ports", "22/tcp,8080/http", "--gpu-id", "NVIDIA GeForce RTX 4090",
        "--data-center-ids", "CA-MTL-1",
    )
    transport.queue_runpodctl(("user", "--output", "json"), CommandResult(0, '{"clientBalance":10}'))
    transport.queue_runpodctl(call, CommandResult(401, '{"statusCode":401,"code":"unauthorized"}'))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=("CA-MTL-1",),
            image=bundle.environment.image_id,
        ),
        transport=transport,
    )
    store = RunSetStateStore(bundle.run_set_dir / "authorization.json")
    with pytest.raises(OrchestrationStageError, match="non-retryable-error"):
        StageEngine(bundle=bundle, driver=driver, store=store).run(stop_after_stage=STAGE_PROVISION)
    assert transport.runpodctl_calls.count(call) == 1
    assert store.load().provisioning_stop_reason == "non-retryable-error"


def test_provision_timeout_removes_pod_and_reprovisions(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(
        update={"budget": BudgetPolicy(max_wall_clock_seconds=30, max_spend_usd=3.0)}
    )
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
        ("pod", "get", "pod-1", "--output", "json"),
        CommandResult(
            0,
            '{"createdAt":"1969-12-31T23:59:59+00:00","costPerHr":3600}',
        ),
    )
    transport.queue_runpodctl(
        ("pod", "get", "pod-1", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_runpodctl(
        ("pod", "get", "pod-2", "--output", "json"),
        CommandResult(
            0,
            '{"createdAt":"1970-01-01T00:00:01+00:00","costPerHr":1,"ssh":{"ip":"203.0.113.2","port":22}}',
        ),
    )
    transport.queue_runpodctl(("user", "--output", "json"), CommandResult(0, '{"clientBalance": 10}'))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA RTX 2000 Ada Generation",
            image="runpod/pytorch:1.0.3",
            max_acquire_seconds=1,
            poll_seconds=1,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    state = StageEngine(
        bundle=bundle,
        driver=driver,
        store=RunSetStateStore(bundle.run_set_dir / "state.json"),
        sleep=clock.sleep,
        wall_time=clock.monotonic,
    ).run(stop_after_stage=STAGE_PROVISION)

    assert ("remove", "pod", "pod-1") in transport.runpodctl_calls
    assert state.provision_record["pod_id"] == "pod-2"
    assert state.provisioning_attempts[0]["cleanup"]["pod_absence"]["verified"] is True
    assert state.budget_counters["failed_provision_cost_usd"] == 2.0
    pod_get_timeouts = [
        timeout
        for call, timeout in zip(
            transport.runpodctl_calls,
            transport.runpodctl_timeouts,
            strict=True,
        )
        if call[:2] == ("pod", "get")
    ]
    assert pod_get_timeouts == [1, 60, 1]


def test_endpoint_ready_after_deadline_is_rejected_and_torn_down(tmp_path: Path) -> None:
    clock = FakeClock()

    class LateReadyTransport(FakeRunPodTransport):
        def runpodctl(
            self,
            *args: str,
            timeout_seconds: float | None = None,
        ) -> CommandResult:
            result = super().runpodctl(*args, timeout_seconds=timeout_seconds)
            if args[:2] == ("pod", "get"):
                assert timeout_seconds == 2
                clock.sleep(timeout_seconds + 1)
                return CommandResult(
                    0,
                    '{"createdAt":"now","ssh":{"ip":"203.0.113.1","port":22}}',
                )
            return result

    bundle = _bundle(tmp_path)
    transport = LateReadyTransport()
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
    transport.queue_runpodctl(create_call, CommandResult(0, '{"id":"pod-late"}'))
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(0, '{"clientBalance":10}'),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA RTX 2000 Ada Generation",
            image="runpod/pytorch:1.0.3",
            max_acquire_seconds=2,
            poll_seconds=1,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert all(check.status == "pass" for check in driver.preflight_checks(bundle))

    with pytest.raises(ProvisioningAttemptError, match="timed out waiting.*after 2s"):
        driver.provision(bundle, _state(bundle))

    assert ("remove", "pod", "pod-late") in transport.runpodctl_calls
    assert all("nvidia-smi" not in command for command in transport.ssh_commands)


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

    fingerprint_payload = json.loads(fingerprint)
    assert fingerprint_payload["schema_version"] == (
        "feedbax.runpod_environment_fingerprint.v1"
    )
    assert fingerprint_payload["runtime"]["jax_platform"] == "gpu"
    assert fingerprint_payload["feedbax_plugins"][0]["name"] == "rlrmp2"
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
    assert "entry_point.load()" in joined
    assert "lockfile digest mismatch" in joined


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
        if "env-fingerprint.json" in command and "printf %s" in command
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
    declaration_fingerprint = compute_runpod_environment_fingerprint(bundle)
    declaration = {
        "declaration_sha256": declaration_fingerprint,
        "image_id": bundle.environment.image_id,
        "lockfile_hashes": bundle.environment.lockfile_hashes,
        "python_version": bundle.environment.python_version,
    }
    realized_fingerprint = _realized_fingerprint(declaration)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(0, declaration_fingerprint))
    transport.queue_ssh(CommandResult(0, realized_fingerprint))
    driver = RunPodOrchestrationDriver(transport=transport)

    assert driver.realize_env(bundle, _state(bundle)) == realized_fingerprint
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


def test_stage_inputs_ignores_checkpoint_shaped_payload_keys(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    outputs = driver.stage_inputs(bundle, _state(bundle))

    assert outputs["input_count"] == 0
    assert outputs["payload_count"] == 1
    payload = bundle.rows[0].execution.payload
    assert transport.rsync_calls == [
        (
            str(
                bundle.run_set_dir
                / ".stage-attempts/stage-inputs-0/inputs"
            )
            + "/",
            "/workspace/feedbax_runs/2026-01-02-deadbeef/"
            ".stage-attempts/stage-inputs-0/inputs/",
            True,
            (),
        ),
    ]
    assert any(payload.sha256 in command for command in transport.ssh_commands)
    publish = next(command for command in transport.ssh_commands if "renameat2" in command)
    assert "stage-inputs-0/inputs" in publish
    assert "/feedbax_runs/2026-01-02-deadbeef/inputs" in publish
    assert transport.operations.index(
        next(operation for operation in transport.operations if operation.startswith("rsync:"))
    ) < transport.operations.index(f"ssh:{publish}")
    assert not any("checkpoint_100" in command for command in transport.ssh_commands)


def test_stage_inputs_ignores_legacy_runpod_baseline_metadata(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(
        update={"metadata": {"runpod_baselines": [{"checkpoint_path": "/custody", "completed_batches": 12000}]}}
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    outputs = driver.stage_inputs(bundle, _state(bundle))

    assert outputs["input_count"] == 0
    assert all(call[0] != "/custody" for call in transport.rsync_calls)
    assert not any("/_artifacts/" in call[1] for call in transport.rsync_calls)


def test_stage_inputs_fails_closed_when_final_input_tree_exists(tmp_path: Path) -> None:
    class ExistingFinalInputsTransport(FakeRunPodTransport):
        def ssh(self, command: str) -> CommandResult:
            result = super().ssh(command)
            if "renameat2" in command:
                return CommandResult(
                    1,
                    stderr="input publication target already exists",
                )
            return result

    bundle = _bundle(tmp_path)
    transport = ExistingFinalInputsTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="publication target already exists"):
        driver.stage_inputs(bundle, _state(bundle))

    assert len(transport.rsync_calls) == 1
    assert ".stage-attempts/stage-inputs-0/inputs/" in transport.rsync_calls[0][1]
    assert transport.rsync_calls[0][1] != (
        "/workspace/feedbax_runs/2026-01-02-deadbeef/inputs/"
    )


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
    assert "setsid -f bash -lc" in launch_command
    assert "</dev/null" in launch_command
    assert "while [ ! -s" in launch_command
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
    assert "rm -f" in launch_command
    assert all("deadman" not in command for command in transport.ssh_commands)


def test_remote_sentinel_uses_session_detacher_instead_of_background_shell() -> None:
    command = build_remote_nohup_sentinel_command(
        workdir="/workspace/feedbax",
        command="python -c 'import time; time.sleep(30)'",
        done_file="/workspace/run/sentinels/bootstrap.done",
        failed_file="/workspace/run/sentinels/bootstrap.failed",
        log_file="/workspace/run/logs/bootstrap.log",
    )

    assert "setsid -f bash -lc" in command
    assert "nohup" not in command
    assert "</dev/null" in command
    assert command.endswith("2>&1")


def test_launch_row_command_returns_before_buffered_child_and_replaces_stale_pid(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setsid = fake_bin / "setsid"
    setsid.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "assert sys.argv[1] == '-f'\n"
        "if os.fork():\n"
        "    raise SystemExit(0)\n"
        "os.setsid()\n"
        "os.execvp(sys.argv[2], sys.argv[2:])\n",
        encoding="utf-8",
    )
    setsid.chmod(0o755)
    bundle = _bundle(tmp_path)
    original = bundle.rows[0]
    row = original.model_copy(
        update={
            "launch": RowLaunchSpec(
                command=[
                    sys.executable,
                    "-c",
                    "import time; print('buffered', end=''); time.sleep(10)",
                ]
            )
        }
    )
    remote_run_dir = tmp_path / "run"
    sentinel_dir = remote_run_dir / "sentinels"
    sentinel_dir.mkdir(parents=True)
    pid_path = sentinel_dir / "warm.pid"
    pid_path.write_text("999999\n", encoding="utf-8")
    command = build_launch_row_command(
        bundle=bundle,
        row=row,
        remote_run_dir=str(remote_run_dir),
        remote_sentinel_dir=str(sentinel_dir),
        workdir=str(tmp_path),
        jax_cache_dir=str(tmp_path / "jax-cache"),
        env_fingerprint="fingerprint-123",
    )

    started = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        timeout=3,
        check=False,
    )
    elapsed = time.monotonic() - started
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    try:
        assert result.returncode == 0, result.stderr
        assert elapsed < 2
        assert pid != 999999
        os.kill(pid, 0)
    finally:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


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
    assert "command -v setsid >/dev/null" in transport.ssh_commands[0]
    assert "command -v runpodctl" in joined
    assert all(text in joined for text in ("RUNPOD_API_KEY=$(tr", "runpodctl get pod pod-123"))
    watchdog = next(command for command in transport.ssh_commands if "deadman.pid" in command)
    assert 'kill -0 "$(cat "$pid_file")"' in watchdog
    assert "setsid -f bash -lc" in watchdog
    assert "echo $$ >" in watchdog
    assert 'rm -f "$pid_file"' in watchdog
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
        "input-provider-bindings",
        "runpod-image-immutable",
        "runpod-image-tag-exists",
        "runpod-lockfiles-declared",
        "runpod-python-version-declared",
        "runpod-gpu-policy-declared",
        "runpod-credentials",
        "runpod-balance-floor",
        "runpod-deadman-credentials",
    ]
    assert all(check.status == "pass" for check in checks)
    assert transport.runpodctl_calls == [("user", "--output", "json")]


@pytest.mark.parametrize(
    ("environment_update", "failed_check"),
    [
        ({"image_id": "runpod/pytorch:mutable"}, "runpod-image-immutable"),
        ({"image_id": None}, "runpod-image-immutable"),
        ({"lockfile_hashes": {}}, "runpod-lockfiles-declared"),
        (
            {"lockfile_hashes": {"uv.lock": "not-a-sha256"}},
            "runpod-lockfiles-declared",
        ),
        ({"python_version": None}, "runpod-python-version-declared"),
    ],
)
def test_preflight_rejects_non_deterministic_environment_declarations(
    tmp_path: Path,
    environment_update: dict[str, Any],
    failed_check: str,
) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={
            "environment": bundle.environment.model_copy(update=environment_update)
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(gpu_id="NVIDIA GeForce RTX 5090"),
        transport=transport,
    )

    checks = {check.name: check for check in driver.preflight_checks(bundle)}

    assert checks[failed_check].status == "fail"
    assert driver._preflight_passed is False


def test_realize_env_rejects_mutable_image_before_remote_access(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={
            "environment": bundle.environment.model_copy(
                update={"image_id": "runpod/pytorch:mutable"}
            )
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(transport=transport)

    with pytest.raises(RunPodDriverError, match="pinned by @sha256"):
        driver.realize_env(bundle, _state(bundle))

    assert transport.ssh_commands == []


@pytest.mark.parametrize("mismatch", ["lockfile", "cuda"])
def test_realize_env_rejects_runtime_provenance_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(1, ""))
    declaration_sha256 = compute_runpod_environment_fingerprint(bundle)
    mismatched = json.loads(
        _realized_fingerprint(
            {
                "declaration_sha256": declaration_sha256,
                "image_id": bundle.environment.image_id,
                "lockfile_hashes": bundle.environment.lockfile_hashes,
                "python_version": bundle.environment.python_version,
            }
        )
    )
    if mismatch == "lockfile":
        mismatched["lockfile_hashes"]["uv.lock"] = "b" * 64
        expected_error = "lockfile_hashes"
    else:
        mismatched["runtime"]["jax_platform"] = "cpu"
        expected_error = "JAX CUDA backend"
    transport.environment_probe_result = CommandResult(0, json.dumps(mismatched))
    driver = RunPodOrchestrationDriver(transport=transport)

    with pytest.raises(RunPodDriverError, match=expected_error):
        driver.realize_env(bundle, _state(bundle))


def test_rlrmp2_is_the_primary_environment_workdir() -> None:
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            remote_repos={
                "feedbax": "/workspace/feedbax",
                "rlrmp2": "/workspace/rlrmp2",
            }
        ),
        transport=FakeRunPodTransport(),
    )

    assert driver._primary_workdir() == "/workspace/rlrmp2"


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
    assert "runpod-image-immutable" in names
    assert "runpod-lockfiles-declared" in names
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


def test_teardown_remove_failure_stops_then_removes_owned_pod(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "busy"))
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    result = driver.teardown(bundle, _state(bundle))

    assert result["teardown"] == "stopped-then-removed"
    assert result["pod_absence"] == {
        "verified": True,
        "pod_id": "pod-123",
        "polls": 1,
        "terminal_observation": "not-found",
    }
    assert result["final_pod_inventory"] | {"observed_at": "<time>"} == {
        "scope": "provider-account",
        "verified": True,
        "observed_at": "<time>",
        "observation_basis": "runpodctl pod list --output json",
        "outcome": "empty",
        "pod_count": 0,
        "pod_ids": [],
    }
    assert transport.runpodctl_calls.count(("remove", "pod", "pod-123")) == 2
    assert ("stop", "pod", "pod-123") in transport.runpodctl_calls
    second_remove_index = transport.runpodctl_calls.index(
        ("remove", "pod", "pod-123"), 1
    )
    assert transport.runpodctl_timeouts[second_remove_index] == 60


@pytest.mark.parametrize(
    "inventory_payload",
    [
        "[]",
        '{"pods":[]}',
        '{"data":[]}',
        '{"data":{"pods":[]}}',
    ],
)
def test_teardown_accepts_supported_empty_provider_inventory_shapes(
    tmp_path: Path,
    inventory_payload: str,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory(inventory_payload)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    result = driver.teardown(bundle, _state(bundle))

    assert result["pod_absence"]["verified"] is True
    assert result["final_pod_inventory"]["verified"] is True
    assert result["final_pod_inventory"]["pod_ids"] == []
    assert transport.runpodctl_calls[-1] == ("pod", "list", "--output", "json")


@pytest.mark.parametrize(
    ("inventory_result", "expected_outcome", "expected_count", "expected_ids"),
    [
        (
            CommandResult(0, '[{"id":"pod-other","name":"top-secret-name"}]'),
            "non-empty",
            1,
            ["pod-other"],
        ),
        (CommandResult(0, "{}"), "invalid", None, []),
        (CommandResult(0, '{"pods":[],"data":[]}'), "invalid", None, []),
        (CommandResult(0, '[{"id":"unsafe secret"}]'), "invalid", None, []),
        (
            CommandResult(1, "", "provider failed with top-secret-token"),
            "unavailable",
            None,
            [],
        ),
    ],
)
def test_teardown_records_sanitized_unverified_provider_inventory(
    tmp_path: Path,
    inventory_result: CommandResult,
    expected_outcome: str,
    expected_count: int | None,
    expected_ids: list[str],
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_runpodctl(
        ("pod", "list", "--output", "json"),
        inventory_result,
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    result = driver.teardown(bundle, _state(bundle))

    inventory = result["final_pod_inventory"]
    assert inventory["verified"] is False
    assert inventory["outcome"] == expected_outcome
    assert inventory["pod_count"] == expected_count
    assert inventory["pod_ids"] == expected_ids
    serialized = json.dumps(result, sort_keys=True)
    assert "top-secret" not in serialized
    assert "unsafe secret" not in serialized
    assert transport.runpodctl_calls[-1] == ("pod", "list", "--output", "json")


@pytest.mark.parametrize(
    ("inventory_result", "expected_outcome"),
    [
        (CommandResult(0, '[{"id":"pod-other"}]'), "non-empty"),
        (CommandResult(0, "{}"), "invalid"),
        (CommandResult(1, "", "provider unavailable"), "unavailable"),
    ],
)
def test_unverified_global_inventory_survives_teardown_and_blocks_register(
    tmp_path: Path,
    inventory_result: CommandResult,
    expected_outcome: str,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_runpodctl(
        ("pod", "list", "--output", "json"),
        inventory_result,
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )
    engine = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=CheckRegistry(
            {"fixture": lambda _row: CheckEntry(check_id="fixture", status="pass")}
        ),
    )

    state = engine._run_teardown(_state(bundle), abort=False)

    assert state.stage("TEARDOWN").status == "completed"
    inventory = state.stage("TEARDOWN").outputs["final_pod_inventory"]
    assert inventory["verified"] is False
    assert inventory["outcome"] == expected_outcome
    with pytest.raises(
        OrchestrationStageError,
        match="globally empty RunPod provider inventory",
    ):
        engine._stage_register(state)


def test_teardown_fails_closed_when_remove_after_stop_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("remove", "pod", "pod-123"), CommandResult(1, "", "busy")
    )
    transport.queue_runpodctl(
        ("remove", "pod", "pod-123"), CommandResult(1, "", "still busy")
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="remove pod after stop failed"):
        driver.teardown(bundle, _state(bundle))

    assert transport.runpodctl_calls == [
        ("remove", "pod", "pod-123"),
        ("stop", "pod", "pod-123"),
        ("remove", "pod", "pod-123"),
    ]


def test_teardown_polls_until_exact_owned_pod_is_absent(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    query = ("pod", "get", "pod-123", "--output", "json")
    transport.queue_runpodctl(query, CommandResult(0, '{"id":"pod-123"}'))
    transport.queue_runpodctl(query, CommandResult(1, "", "pod does not exist"))
    transport.queue_empty_global_inventory()
    clock = FakeClock()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            poll_seconds=2,
            teardown_absence_timeout_seconds=10,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    result = driver.teardown(bundle, _state(bundle))

    assert result["teardown"] == "removed"
    assert result["pod_absence"]["polls"] == 2
    assert clock.sleeps == [2]


def test_teardown_fails_when_exact_owned_pod_remains_present(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    query = ("pod", "get", "pod-123", "--output", "json")
    for _ in range(2):
        transport.queue_runpodctl(query, CommandResult(0, '{"id":"pod-123"}'))
    clock = FakeClock()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-123",
            poll_seconds=2,
            teardown_absence_timeout_seconds=4,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(RunPodDriverError, match="remained present for 4s"):
        driver.teardown(bundle, _state(bundle))


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(0, "{}"),
        CommandResult(0, '{"id":"other-pod"}'),
        CommandResult(1, "", "provider query unavailable"),
    ],
)
def test_teardown_fails_closed_on_ambiguous_absence_query(
    tmp_path: Path,
    result: CommandResult,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        result,
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="ambiguous absence query"):
        driver.teardown(bundle, _state(bundle))


def test_teardown_keep_alive_skips_owned_pod_query(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-123"),
        transport=transport,
    )

    result = driver.teardown(bundle, _state(bundle))

    assert result["teardown"] == "skipped"
    assert transport.runpodctl_calls == []


def test_abort_teardown_pulls_failure_logs_before_pod_removal(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.rsync_result = CommandResult(1, "", "diagnostic pull failed")
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory()
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
