from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shlex
from types import SimpleNamespace
from typing import Any

import pytest

import feedbax.orchestration.collection_recovery as collection_recovery
import feedbax.orchestration.drivers.runpod as runpod_module
from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.remote_smoke import RemoteSmokeEvidence
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowProvenance
from feedbax.contracts.spec_storage import training_spec_canonical_bytes
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingAssemblySpec,
    StudioTrainingIdentityAdapter,
)
from feedbax.contracts.training import default_training_program_registry
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
from feedbax.orchestration.collection_recovery import (
    CollectionRecoveryBinding,
    CollectionRecoveryError,
)
from feedbax.orchestration.drivers.runpod import (
    _run_command,
    CommandResult,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
    RunPodRemoteSmokeError,
    SubprocessRunPodTransport,
    build_launch_row_command,
    build_runpod_execution_namespace,
    build_literal_path_patch_command,
    build_remote_nohup_sentinel_command,
    classify_pod_state,
    compute_runpod_environment_fingerprint,
    endpoint_classification,
    project_runpod_provision_facts,
    rank_datacenters_for_gpu,
    runpod_row_workdir,
    validate_runpod_repo_realization_plan,
)
from feedbax.orchestration.drivers.capabilities import (
    AcquisitionSemantics,
    DriverCapabilityEnvelope,
    DriverHook,
)
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.drivers.base import (
    AcquisitionCreateError,
    AcquisitionResult,
    ProvisioningAttemptError,
)
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.repo_realization import (
    EditableSourceResolution,
    RepoRealizationError,
    RepoRealizationPlan,
)
from feedbax.training.interruption import CancellationDecision
from feedbax.orchestration.conformance import CheckEntry, CheckRegistry
from feedbax.orchestration.stages import (
    STAGE_PREFLIGHT,
    STAGE_PROVISION,
    STAGE_REALIZE_ENV,
    STAGE_SMOKE,
    STAGE_STAGE_INPUTS,
    OrchestrationStageError,
    PreflightFailed,
    StageEngine,
    _DeferredOperatorSignal,
    _ScopedSignalSupervisor,
)
from feedbax.orchestration.state import (
    ProcessIdentity,
    RowState,
    RunSetState,
    RunSetStateStore,
    StageState,
)
from feedbax.orchestration.revision import resolve_feedbax_revision


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
        self.image_exists_calls: list[str] = []

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
        self.image_exists_calls.append(image)
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
        if "identity_status='missing'" in command:
            if self.ssh_results:
                return self.ssh_results.pop(0)
            identity = {
                "schema_id": "feedbax.orchestration.process_identity",
                "schema_version": "feedbax.orchestration.process_identity.v1",
                "mechanism": "environment-token-v1",
                "run_set_id": "2026-01-02-deadbeef",
                "row_id": "warm",
                "pid": 4321,
                "process_group_id": 4321,
                "launch_token": "a" * 64,
            }
            return CommandResult(
                0,
                json.dumps(
                    {
                        "gpu": "",
                        "rows": {
                            "warm": {
                                "status": "running",
                                "pid": 4321,
                                "process_identity": identity,
                                "identity_status": "owned",
                                "detail": None,
                            }
                        },
                    }
                ),
            )
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


class AcquisitionLeaseTransport(FakeRunPodTransport):
    """State-aware fake for the engine-owned acquisition WAL protocol."""

    def __init__(self, store: RunSetStateStore) -> None:
        super().__init__()
        self.store = store
        self.create_results: list[CommandResult] = []
        self.inventory: dict[str, str] = {}
        self.create_names: list[str] = []
        self.register_on_create: list[tuple[str, ...]] = []

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        if args[:2] == ("pod", "create"):
            self.runpodctl_calls.append(args)
            self.runpodctl_timeouts.append(timeout_seconds)
            name = args[args.index("--name") + 1]
            self.create_names.append(name)
            persisted = self.store.load()
            assert persisted.acquisition_intents[-1].state == "intended"
            assert name.endswith(persisted.acquisition_intents[-1].intent_id)
            for pod_id in self.register_on_create.pop(0) if self.register_on_create else ():
                self.inventory[pod_id] = name
            return self.create_results.pop(0)
        if args == ("pod", "list", "--output", "json"):
            self.runpodctl_calls.append(args)
            self.runpodctl_timeouts.append(timeout_seconds)
            return CommandResult(
                0,
                json.dumps(
                    [
                        {"id": pod_id, "name": name}
                        for pod_id, name in sorted(self.inventory.items())
                    ]
                ),
            )
        if args[:2] == ("remove", "pod"):
            self.runpodctl_calls.append(args)
            self.runpodctl_timeouts.append(timeout_seconds)
            self.inventory.pop(args[2], None)
            return CommandResult(0, "")
        if args[:2] == ("pod", "get"):
            self.runpodctl_calls.append(args)
            self.runpodctl_timeouts.append(timeout_seconds)
            pod_id = args[2]
            if pod_id not in self.inventory:
                return CommandResult(1, "", "pod not found")
            return CommandResult(
                0,
                json.dumps(
                    {
                        "id": pod_id,
                        "name": self.inventory[pod_id],
                        "ssh": {"ip": "203.0.113.10", "port": 22},
                        "imageName": "runpod/pytorch:1.0.3",
                    }
                ),
            )
        return super().runpodctl(*args, timeout_seconds=timeout_seconds)


def _acquisition_engine(
    bundle: RunBundle,
    store: RunSetStateStore,
    transport: AcquisitionLeaseTransport,
    *,
    datacenters: tuple[str, ...] = ("CA-MTL-1", "EU-CZ-1"),
) -> tuple[StageEngine, RunPodOrchestrationDriver, RunSetState]:
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            image=bundle.environment.image_id or "",
            datacenters=datacenters,
            poll_seconds=1,
        ),
        transport=transport,
    )
    driver._preflight_passed = True
    state = _state(bundle)
    store.save(state)
    engine = StageEngine(bundle=bundle, driver=driver, store=store)
    engine._signal_supervisor = _ScopedSignalSupervisor()
    return engine, driver, state


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


def _git_seal_ready(root: Path, *tracked: str) -> None:
    """Make an existing directory a sealable Git top-level.

    Governed repo snapshots (lane E-ii) require every configured local repo to
    be a Git working tree, while the layout-vs-lock check (lane E-i) requires it
    to hold the declared lockfile. This commits the requested tracked files (or a
    placeholder when none is supplied) so both checks are satisfied by the same
    fixture root.
    """
    root.mkdir(parents=True, exist_ok=True)
    if (root / ".git").exists():
        return
    to_add = [name for name in tracked if (root / name).exists()]
    if not to_add:
        (root / ".sealed").write_text("sealed\n", encoding="utf-8")
        to_add = [".sealed"]
    # Every fixture repo in this module is built this way and forking `git` is
    # the file's dominant cost, so this is deliberately three processes rather
    # than one per file plus two for identity: the adds are batched and the
    # committer identity rides on the commit as `-c` overrides.
    git = ["git", "-C", str(root), "--no-optional-locks"]
    subprocess.run([*git, "init", "--quiet"], check=True, capture_output=True)
    subprocess.run([*git, "add", "--", *to_add], check=True, capture_output=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.email=runpod@example.invalid",
            "-c",
            "user.name=RunPod Test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )


def _bundle(
    tmp_path: Path,
    *,
    keep_alive: bool = False,
    deadman_enabled: bool = False,
    baseline: bool = True,
    smoke_enabled: bool = False,
) -> RunBundle:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_bytes((Path.cwd() / "uv.lock").read_bytes())
    lockfile_sha256 = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    _git_seal_ready(tmp_path, "uv.lock")
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
        feedbax_revision=resolve_feedbax_revision(),
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
    bundle = assemble_run_bundle(
        request,
        run_set_id="2026-01-02-deadbeef",
        context=AssemblyContext(custody_root=tmp_path / "custody"),
        registry=registry,
    )
    return bundle.model_copy(update={"smoke_enabled": smoke_enabled})


def _layout_case(
    tmp_path: Path,
    lock_text: str,
    *,
    local_repos: dict[str, Path] | None = None,
    remote_repos: dict[str, str] | None = None,
    primary_repo: str = "consumer",
    path_patches: tuple[tuple[str, str, str], ...] = (),
    write_lock: bool = True,
) -> tuple[RunBundle, RunPodDriverConfig]:
    bundle = _bundle(tmp_path)
    configured_local = local_repos or {primary_repo: tmp_path / primary_repo}
    configured_remote = remote_repos or {primary_repo: f"/workspace/{primary_repo}"}
    lock_path = configured_local[primary_repo] / "uv.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_bytes = lock_text.encode("utf-8")
    if write_lock:
        lock_path.write_bytes(lock_bytes)
    for repo_root in configured_local.values():
        _git_seal_ready(Path(repo_root), "uv.lock")
    environment = bundle.environment.model_copy(
        update={"lockfile_hashes": {"uv.lock": hashlib.sha256(lock_bytes).hexdigest()}}
    )
    return (
        bundle.model_copy(update={"environment": environment}),
        RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos=configured_local,
            remote_repos=configured_remote,
            primary_repo=primary_repo,
            path_patches=path_patches,
        ),
    )


def _realization_layout_error(
    bundle: RunBundle,
    config: RunPodDriverConfig,
) -> tuple[str | None, Mapping[str, Any]]:
    driver = RunPodOrchestrationDriver(config=config, transport=FakeRunPodTransport())
    try:
        plan = driver.seal_repo_realization_plan(bundle)
    except (RepoRealizationError, RunPodDriverError) as exc:
        return str(exc), {}
    return validate_runpod_repo_realization_plan(bundle, config, plan, driver._repo_snapshots)


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


def _sealed_state(
    driver: RunPodOrchestrationDriver,
    bundle: RunBundle,
) -> RunSetState:
    assert driver.config.local_repos, "snapshot tests must configure explicit fixture repos"
    plan = driver.seal_repo_realization_plan(bundle)
    return _state(bundle).model_copy(
        update={
            "repo_realization_plan": plan,
            "stages": {
                "PREFLIGHT": StageState(
                    status="completed",
                    outputs={
                        "driver_evidence": {
                            "repo_realization_plan_digest": plan.plan_digest,
                        }
                    },
                )
            },
        }
    )


def _init_snapshot_repo(root: Path) -> None:
    root.mkdir(parents=True)
    subprocess.run(["git", "-C", str(root), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "runpod@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "RunPod Test"],
        check=True,
    )
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (root / "uv.lock").write_bytes((Path.cwd() / "uv.lock").read_bytes())
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt", "uv.lock"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )


def _owned_state(bundle: RunBundle, pod_id: str = "pod-123") -> RunSetState:
    return _state(bundle).model_copy(
        update={
            "provision_record": {
                "driver": "runpod",
                "pod_id": pod_id,
                "provided_pod": False,
                "provided_endpoint": False,
                "teardown_allowed": True,
                "status": "RUNNING",
            }
        }
    )


def _owned_driver(
    *,
    transport: FakeRunPodTransport,
    pod_id: str = "pod-123",
    config: RunPodDriverConfig | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> RunPodOrchestrationDriver:
    """Build the engine-acquired fixture variant and adopt its exact pod."""
    driver = RunPodOrchestrationDriver(
        config=config or RunPodDriverConfig(),
        transport=transport,
        sleep=sleep,
        monotonic=monotonic,
    )
    driver.adopt_owned_pod(pod_id)
    return driver


class GovernedProvisionDriver:
    """Fake one-attempt RunPod driver for stage retry policy tests."""

    realized_capabilities = DriverCapabilityEnvelope.single(
        "runpod",
        replace(
            RunPodOrchestrationDriver.capability_envelope.variants["engine-acquired"],
            variant_id="governed-fixture",
            acquisition=AcquisitionSemantics.EXTERNALLY_PROVIDED,
            optional_hooks=frozenset(
                {
                    DriverHook.GOVERN_PROVISIONING_RETRIES,
                    DriverHook.PROVISION_RETRY_DELAY,
                }
            ),
        ),
    ).realize("governed-fixture")
    poll_interval_seconds = 0.05

    def govern_provisioning_retries(self) -> bool:
        return True

    def provision_retry_delay(self) -> float:
        return 1.0

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


def _governed_engine(
    tmp_path: Path, outcomes: list[object], **kwargs: Any
) -> tuple[StageEngine, GovernedProvisionDriver, RunSetStateStore]:
    bundle = _bundle(tmp_path)
    clock = FakeClock()
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    driver = GovernedProvisionDriver(outcomes)
    return (
        StageEngine(
            bundle=bundle,
            driver=driver,
            store=store,
            sleep=clock.sleep,
            wall_time=clock.monotonic,
            **kwargs,
        ),
        driver,
        store,
    )


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


def test_engine_persists_one_intent_before_each_single_create(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "lease-state.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [
        CommandResult(400, '{"statusCode":400,"code":"invalid_request"}'),
        CommandResult(0, '{"id":"pod-second"}'),
    ]
    transport.register_on_create = [(), ("pod-second",)]
    engine, _driver, state = _acquisition_engine(bundle, store, transport)

    state, outputs = engine._engine_owned_provision(state, attempt_ordinal=1)

    assert outputs["pod_id"] == "pod-second"
    persisted = store.load()
    assert [intent.state for intent in persisted.acquisition_intents] == [
        "failed-unacquired",
        "acquired",
    ]
    assert [intent.datacenter_candidate for intent in persisted.acquisition_intents] == [
        "CA-MTL-1",
        "EU-CZ-1",
    ]
    assert len(set(transport.create_names)) == 2


_RUNPOD_RESOURCE_UNAVAILABLE = (
    '{"error":"This machine does not have the resources to deploy your pod. '
    'Please try a different machine"}\n'
    "Usage:\n  runpodctl pod create [flags]\n\n"
    '{"error":"failed to create pod: This machine does not have the resources to deploy '
    'your pod. Please try a different machine"}\n'
)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_resource_unavailable_create_response_is_definitive(stream: str) -> None:
    result = CommandResult(1, **{stream: _RUNPOD_RESOURCE_UNAVAILABLE})

    classification, _detail = runpod_module._classify_create_failure(result, None)

    assert classification == "non-retryable"


_RUNPOD_NO_INSTANCES_AVAILABLE = (
    '{"error":"There are no longer any instances available with the requested '
    'specifications. Please refresh and try again."}\n'
)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_no_instances_available_create_response_is_definitive(stream: str) -> None:
    """EU-RO-1 variant surfaced from rlrmp2/5ea2a98: a distinct RunPod no-capacity message."""
    result = CommandResult(1, **{stream: _RUNPOD_NO_INSTANCES_AVAILABLE})

    classification, _detail = runpod_module._classify_create_failure(result, None)

    assert classification == "non-retryable"


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
    classification, _detail = runpod_module._classify_create_failure(result, None)

    assert classification == "retryable"


CREATE_FAILURE_CORPUS_PATH = (
    Path(__file__).parent / "fixtures" / "runpod_provider_errors" / "create_failure_corpus.json"
)


def _create_failure_corpus() -> list[dict[str, Any]]:
    """Regression corpus for feedbax/32d1d73; see the fixture README for curation policy."""
    return json.loads(CREATE_FAILURE_CORPUS_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", _create_failure_corpus(), ids=lambda entry: entry["name"])
def test_create_failure_corpus_classification(entry: dict[str, Any]) -> None:
    """Every curated payload must keep its human-adjudicated classification."""
    result = CommandResult(entry["returncode"], entry["stdout"], entry["stderr"])

    classification, _detail = runpod_module._classify_create_failure(result, None)

    assert classification == entry["expected_classification"], entry["name"]


@pytest.mark.parametrize(
    "entry",
    [
        entry
        for entry in _create_failure_corpus()
        if entry["expected_behavior"] == "region-rejected-continues"
    ],
    ids=lambda entry: entry["name"],
)
def test_create_failure_corpus_definitive_rejects_region_and_continues(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    """A definitive corpus entry rejects its candidate region and moves to the next one."""
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / f"corpus-definitive-{entry['name']}.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [
        CommandResult(entry["returncode"], entry["stdout"], entry["stderr"]),
        CommandResult(0, '{"id":"pod-second"}'),
    ]
    transport.register_on_create = [(), ("pod-second",)]
    engine, _driver, state = _acquisition_engine(bundle, store, transport)

    _state_after, outputs = engine._engine_owned_provision(state, attempt_ordinal=1)

    assert outputs["pod_id"] == "pod-second"
    persisted = store.load()
    assert [intent.state for intent in persisted.acquisition_intents] == [
        "failed-unacquired",
        "acquired",
    ]
    assert len(transport.create_names) == 2


@pytest.mark.parametrize(
    "entry",
    [entry for entry in _create_failure_corpus() if entry["expected_behavior"] == "halt-ambiguous"],
    ids=lambda entry: entry["name"],
)
def test_create_failure_corpus_ambiguous_halts_acquisition(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    """An ambiguous corpus entry stops provisioning rather than guessing at a classification."""
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / f"corpus-ambiguous-{entry['name']}.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [
        CommandResult(entry["returncode"], entry["stdout"], entry["stderr"])
    ]
    engine, _driver, state = _acquisition_engine(bundle, store, transport)

    with pytest.raises(OrchestrationStageError, match="ambiguous-acquisition-unresolved"):
        engine._engine_owned_provision(state, attempt_ordinal=1)

    persisted = store.load()
    assert len(transport.create_names) == 1
    assert persisted.acquisition_intents[0].state == "ambiguous-unresolved"
    assert persisted.provisioning_stop_reason == "ambiguous-acquisition-unresolved"


def test_resource_unavailable_create_advances_to_next_candidate(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "resource-unavailable-state.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [
        CommandResult(1, "", _RUNPOD_RESOURCE_UNAVAILABLE),
        CommandResult(0, '{"id":"pod-second"}'),
    ]
    transport.register_on_create = [(), ("pod-second",)]
    engine, _driver, state = _acquisition_engine(bundle, store, transport)

    _state_after, outputs = engine._engine_owned_provision(state, attempt_ordinal=1)

    assert outputs["pod_id"] == "pod-second"
    persisted = store.load()
    assert [intent.state for intent in persisted.acquisition_intents] == [
        "failed-unacquired",
        "acquired",
    ]
    assert [intent.datacenter_candidate for intent in persisted.acquisition_intents] == [
        "CA-MTL-1",
        "EU-CZ-1",
    ]
    assert len(transport.create_names) == 2


def test_ambiguous_create_zero_match_stops_without_next_candidate(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "ambiguous-state.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [CommandResult(0, "garbage")]
    engine, _driver, state = _acquisition_engine(bundle, store, transport)

    with pytest.raises(OrchestrationStageError, match="ambiguous-acquisition-unresolved"):
        engine._engine_owned_provision(state, attempt_ordinal=1)

    persisted = store.load()
    assert len(transport.create_names) == 1
    assert persisted.acquisition_intents[0].state == "ambiguous-unresolved"
    assert persisted.provisioning_stop_reason == "ambiguous-acquisition-unresolved"
    assert "unresolved_owned_pod" in persisted.acquisition_intents[0].evidence


@pytest.mark.parametrize("pod_ids", [("pod-lost",), ("pod-duplicate-a", "pod-duplicate-b")])
def test_ambiguous_create_adopts_all_name_matches_before_retry(
    tmp_path: Path,
    pod_ids: tuple[str, ...],
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "lost-response-state.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [CommandResult(0, "not-json")]
    transport.register_on_create = [pod_ids]
    engine, _driver, state = _acquisition_engine(bundle, store, transport)

    with pytest.raises(ProvisioningAttemptError) as raised:
        engine._engine_owned_provision(state, attempt_ordinal=1)

    persisted = store.load()
    intent = persisted.acquisition_intents[0]
    assert raised.value.retryable is True
    assert len(transport.create_names) == 1
    assert intent.state == "resolved-torn-down"
    assert intent.pod_ids == list(pod_ids)
    assert len(intent.teardown_evidence) == len(pod_ids)
    assert transport.inventory == {}
    assert all(("remove", "pod", pod_id) in transport.runpodctl_calls for pod_id in pod_ids)


def test_restart_reconciles_intended_create_before_new_acquisition(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "restart-intent-state.json")
    first_transport = AcquisitionLeaseTransport(store)
    first_engine, first_driver, state = _acquisition_engine(
        bundle, store, first_transport, datacenters=("CA-MTL-1",)
    )
    state, intent = first_engine._new_acquisition_intent(
        state,
        attempt_ordinal=1,
        candidate_ordinal=1,
        candidate="CA-MTL-1",
    )
    pod_name = first_driver.acquisition_pod_name(intent.intent_id)

    second_transport = AcquisitionLeaseTransport(store)
    second_transport.inventory["pod-after-kill"] = pod_name
    second_driver = RunPodOrchestrationDriver(
        config=first_driver.config,
        transport=second_transport,
    )
    second_engine = StageEngine(bundle=bundle, driver=second_driver, store=store)
    reconciled = second_engine._reconcile_acquisition_intents(store.load())

    assert reconciled.acquisition_intents[0].state == "resolved-torn-down"
    assert second_transport.inventory == {}


def test_reconciliation_defers_signal_through_bounded_verified_teardown(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "signal-reconcile-state.json")
    transport = AcquisitionLeaseTransport(store)
    engine, driver, state = _acquisition_engine(bundle, store, transport)
    state, intent = engine._new_acquisition_intent(
        state,
        attempt_ordinal=1,
        candidate_ordinal=1,
        candidate="CA-MTL-1",
    )
    transport.inventory["pod-signal"] = driver.acquisition_pod_name(intent.intent_id)
    supervisor = _ScopedSignalSupervisor()
    original_observe = driver.observe_pod_inventory

    def signalled_observe(**kwargs: Any) -> Any:
        supervisor._handle(signal.SIGTERM, None)
        return original_observe(**kwargs)

    driver.observe_pod_inventory = signalled_observe
    with pytest.raises(_DeferredOperatorSignal):
        with supervisor.defer_signals():
            engine._reconcile_acquisition_intents(state)

    assert transport.inventory == {}
    assert store.load().acquisition_intents[0].state == "resolved-torn-down"
    assert all(
        timeout is None or timeout <= driver.reconciliation_timeout_seconds
        for timeout in transport.runpodctl_timeouts
    )


def test_reconciliation_primary_failure_is_not_masked_by_deferred_signal(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "signal-primary-state.json")
    transport = AcquisitionLeaseTransport(store)
    engine, driver, state = _acquisition_engine(bundle, store, transport)
    state, _intent = engine._new_acquisition_intent(
        state,
        attempt_ordinal=1,
        candidate_ordinal=1,
        candidate="CA-MTL-1",
    )
    supervisor = _ScopedSignalSupervisor()

    def failing_observe(**_kwargs: Any) -> Any:
        supervisor._handle(signal.SIGTERM, None)
        raise RunPodDriverError("primary inventory failure")

    driver.observe_pod_inventory = failing_observe
    with pytest.raises(RunPodDriverError, match="primary inventory failure"):
        with supervisor.defer_signals():
            engine._reconcile_acquisition_intents(state)


def test_fresh_driver_restores_active_lease_endpoint_and_tears_down_real_pod(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "active-lease-state.json")
    transport = AcquisitionLeaseTransport(store)
    engine, driver, state = _acquisition_engine(bundle, store, transport)
    state, intent = engine._new_acquisition_intent(
        state,
        attempt_ordinal=1,
        candidate_ordinal=1,
        candidate="CA-MTL-1",
    )
    state = engine._replace_acquisition_intent(
        state, intent.intent_id, state="acquired", pod_ids=["pod-active"]
    ).model_copy(
        update={
            "provision_record": {
                "driver": "runpod",
                "pod_id": "pod-active",
                "ssh_host": "203.0.113.9",
                "ssh_port": 2222,
                "provided_pod": False,
                "provided_endpoint": False,
                "teardown_allowed": True,
                "intent_id": intent.intent_id,
            }
        }
    )
    store.save(state)

    fresh_transport = AcquisitionLeaseTransport(store)
    fresh_transport.inventory["pod-active"] = driver.acquisition_pod_name(intent.intent_id)
    fresh_driver = RunPodOrchestrationDriver(config=driver.config, transport=fresh_transport)
    fresh_engine = StageEngine(bundle=bundle, driver=fresh_driver, store=store)
    fresh_engine._restore_driver_from_provision_record(state)
    assert fresh_driver._pod_id == "pod-active"
    assert fresh_driver._endpoint is not None
    assert fresh_driver._endpoint.ip == "203.0.113.9"
    assert fresh_driver._endpoint.port == 2222
    teardown = fresh_driver.teardown(bundle, state)

    assert teardown["pod_id"] == "pod-active"
    assert teardown["pod_absence"]["verified"] is True


def test_acquired_intent_without_verified_teardown_blocks_retry(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "unverified-acquired-state.json")
    transport = AcquisitionLeaseTransport(store)
    engine, _driver, state = _acquisition_engine(bundle, store, transport)
    state, intent = engine._new_acquisition_intent(
        state,
        attempt_ordinal=1,
        candidate_ordinal=1,
        candidate="CA-MTL-1",
    )
    state = engine._replace_acquisition_intent(
        state, intent.intent_id, state="acquired", pod_ids=["pod-missing"]
    )
    store.save(state)

    with pytest.raises(OrchestrationStageError, match="ambiguous-acquisition-unresolved"):
        engine._reconcile_acquisition_intents(state)

    assert store.load().acquisition_intents[0].state == "ambiguous-unresolved"


def test_post_provision_early_failure_retains_identity_and_verifies_teardown(
    tmp_path: Path,
) -> None:
    """Regression for feedbax/9e44e27.

    A failure in the provision-to-SMOKE window (here, REALIZE_ENV) must not
    orphan the pod that PROVISION already acquired: ``run`` must retain the
    acquired pod's identity, execute teardown through the driver, and prove
    provider-side absence before the acquisition intent is considered
    resolved.
    """
    bundle = _bundle(tmp_path)
    store = RunSetStateStore(bundle.run_set_dir / "post-provision-failure-state.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [CommandResult(0, '{"id":"pod-early-failure"}')]
    transport.register_on_create = [("pod-early-failure",)]
    transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, '{"clientBalance": 10}')
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            datacenters=("CA-MTL-1",),
            poll_seconds=1,
        ),
        transport=transport,
    )
    engine = StageEngine(bundle=bundle, driver=driver, store=store)

    def _fail_realize_env(_state: RunSetState) -> tuple[RunSetState, Mapping[str, Any]]:
        raise RuntimeError("simulated post-provision failure before SMOKE")

    engine._stage_realize_env = _fail_realize_env

    with pytest.raises(RuntimeError, match="simulated post-provision failure before SMOKE"):
        engine.run()

    final = store.load()
    teardown = final.stage("TEARDOWN")
    assert teardown.status == "completed"
    assert teardown.outputs["abort_path"] is True
    assert teardown.outputs["pod_id"] == "pod-early-failure"
    assert teardown.outputs["pod_absence"]["verified"] is True
    assert transport.inventory == {}
    intent = final.acquisition_intents[0]
    assert intent.state == "resolved-torn-down"
    assert intent.pod_ids == ["pod-early-failure"]
    assert any(("remove", "pod", "pod-early-failure") == call for call in transport.runpodctl_calls)


def test_deadman_is_installed_after_endpoint_and_before_gpu_readiness(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, deadman_enabled=True)
    store = RunSetStateStore(bundle.run_set_dir / "early-deadman-state.json")
    transport = AcquisitionLeaseTransport(store)
    transport.create_results = [CommandResult(0, '{"id":"pod-deadman"}')]
    transport.register_on_create = [("pod-deadman",)]
    engine, _driver, state = _acquisition_engine(
        bundle, store, transport, datacenters=("CA-MTL-1",)
    )

    engine._engine_owned_provision(state, attempt_ordinal=1)

    watchdog_index = next(
        index for index, command in enumerate(transport.ssh_commands) if "deadman.pid" in command
    )
    gpu_index = transport.ssh_commands.index("nvidia-smi >/dev/null")
    assert watchdog_index < gpu_index


def test_provided_endpoint_never_creates_or_adopts_acquisition_intents(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="203.0.113.5", ssh_port=2222),
        transport=FakeRunPodTransport(),
    )
    engine = StageEngine(bundle=bundle, driver=driver)

    state, outputs = engine._stage_provision(_state(bundle))
    fresh = RunPodOrchestrationDriver(transport=FakeRunPodTransport())
    fresh.restore_from_provision_record(outputs)

    assert outputs["provided_endpoint"] is True
    assert state.acquisition_intents == []
    assert fresh.engine_acquisition_required() is False


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
    engine, driver, store = _governed_engine(
        tmp_path, [failure] if failure else [{"pod_id": "no"}], interruption_probe=probe
    )
    with pytest.raises(OrchestrationStageError, match=reason):
        engine.run(stop_after_stage=STAGE_PROVISION)
    assert driver.calls == (0 if failure is None else 1)
    assert store.load().provisioning_stop_reason == reason


def test_provisioning_resume_reuses_deadline_and_failed_cost(tmp_path: Path) -> None:
    engine, driver, store = _governed_engine(
        tmp_path,
        [
            _failed_attempt(acquired=True, billing=True),
            _failed_attempt(acquired=True, billing=True),
        ],
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

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)
    monkeypatch.setattr(
        "feedbax.orchestration.drivers.runpod.shutil.which",
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

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)
    monkeypatch.setattr(
        "feedbax.orchestration.drivers.runpod.shutil.which",
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

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)
    monkeypatch.setattr(
        "feedbax.orchestration.drivers.runpod.shutil.which",
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
        "feedbax.orchestration.drivers.runpod.shutil.which",
        lambda executable: None,
    )
    monkeypatch.setattr(
        "feedbax.orchestration.drivers.runpod._run_command",
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
        "feedbax.orchestration.drivers.runpod.shutil.which",
        lambda executable: "/resolved/rsync",
    )

    def run_command(args: list[str], **kwargs: Any) -> CommandResult:
        calls.append(args)
        if len(calls) == 1:
            return probe_result
        return CommandResult(74, stderr="unexpected secluded probe failure")

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)
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

    monkeypatch.setattr("feedbax.orchestration.drivers.runpod._run_command", run_command)

    SubprocessRunPodTransport().runpodctl(
        "pod",
        "get",
        "pod-1",
        "--output",
        "json",
        timeout_seconds=12.5,
    )

    assert observed == [(["runpodctl", "pod", "get", "pod-1", "--output", "json"], 12.5)]


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
    bundle = _bundle(tmp_path).model_copy(update={"smoke_enabled": False})
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )
    provision_record = dict(driver.provision(bundle, _state(bundle)))
    declaration_fingerprint = compute_runpod_environment_fingerprint(
        bundle, driver.seal_repo_realization_plan(bundle)
    )
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
    state, smoke_outputs = engine._stage_smoke(state)
    state = state.with_stage(
        STAGE_SMOKE,
        state.stage(STAGE_SMOKE).model_copy(
            update={"status": "completed", "outputs": dict(smoke_outputs)}
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
        "feedbax-orchestration-intent-123",
        "--image",
        bundle.environment.image_id,
        "--ports",
        "22/tcp,8080/http",
        "--gpu-id",
        "NVIDIA GeForce RTX 4090",
    )
    expected_call = (
        *base_call,
        "--data-center-ids",
        "CA-MTL-1",
        "--env",
        '{"FEEDBAX_RUNPOD_API_KEY": "dummy-key"}',
    )
    transport.queue_runpodctl(expected_call, CommandResult(0, json.dumps({"id": "pod-123"})))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=("CA-MTL-1", "EU-CZ-1"),
            api_key="dummy-key",
            image=bundle.environment.image_id or "",
        ),
        transport=transport,
    )

    driver._preflight_passed = True
    assert driver.create_pod_once(bundle, "CA-MTL-1", "intent-123") == AcquisitionResult(
        pod_id="pod-123",
        accepted_datacenter="CA-MTL-1",
    )
    assert transport.runpodctl_calls == [expected_call]
    assert "dummy-key" not in repr(driver.config)
    assert "--gpuType" not in expected_call
    assert "--dataCenterId" not in expected_call


def test_provision_record_preserves_accepted_datacenter_when_pod_get_omits_it(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    base_call = (
        "pod",
        "create",
        "--name",
        "feedbax-orchestration-2026-01-02-deadbeef",
        "--image",
        bundle.environment.image_id,
        "--ports",
        "22/tcp,8080/http",
        "--gpu-id",
        "NVIDIA GeForce RTX 4090",
    )
    transport.queue_runpodctl(
        (*base_call, "--data-center-ids", "CA-MTL-1"),
        CommandResult(1, "", "no capacity"),
    )
    transport.queue_runpodctl(
        (*base_call, "--data-center-ids", "EU-CZ-1"),
        CommandResult(0, '{"id":"pod-123"}'),
    )
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(
            0,
            json.dumps(
                {
                    "id": "pod-123",
                    "ssh": {"ip": "203.0.113.10", "port": 22},
                    "imageName": bundle.environment.image_id,
                    "costPerHr": 0.74,
                    "createdAt": "2026-07-19T18:05:00Z",
                }
            ),
        ),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=("CA-MTL-1", "EU-CZ-1"),
            image=bundle.environment.image_id or "",
        ),
        transport=transport,
    )
    driver._preflight_passed = True

    record = driver.finish_acquired_pod(
        bundle,
        AcquisitionResult("pod-123", "EU-CZ-1"),
        "intent-accepted-datacenter",
    )
    store = RunSetStateStore(tmp_path / "state.json")
    store.save(_state(bundle).model_copy(update={"provision_record": record}))

    assert store.load().provision_record["region"] == "EU-CZ-1"
    assert record["intent_id"] == "intent-accepted-datacenter"
    assert record["provider_observation_basis"] == (
        "accepted singleton runpodctl pod create datacenter; "
        "runpodctl pod get response omitted datacenter"
    )


def test_accepted_datacenter_persists_into_realized_deployment_evidence(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    driver = RunPodOrchestrationDriver(transport=FakeRunPodTransport())
    provision = driver._provision_record(
        {
            "id": "pod-123",
            "ssh": {"ip": "203.0.113.10", "port": 22},
            "imageName": bundle.environment.image_id,
            "costPerHr": 0.74,
            "createdAt": "2026-07-19T18:05:00Z",
        },
        provided_pod=False,
        accepted_datacenter="CA-MTL-1",
    )
    declaration = {
        "declaration_sha256": compute_runpod_environment_fingerprint(
            bundle, driver.seal_repo_realization_plan(bundle)
        ),
        "image_id": bundle.environment.image_id,
        "lockfile_hashes": bundle.environment.lockfile_hashes,
        "python_version": bundle.environment.python_version,
    }
    fingerprint_payload = json.loads(_realized_fingerprint(declaration))
    fingerprint_payload["runtime"]["device_kind"] = "NVIDIA GeForce RTX 4090"
    fingerprint = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    started_at = datetime(2026, 7, 19, 18, 6, tzinfo=timezone.utc)
    completed_at = started_at + timedelta(seconds=1)
    row_id = bundle.rows[0].row_id
    state = RunSetState(
        run_set_id=bundle.run_set_id,
        provision_record=provision,
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
                "outputs": provision,
            }
        ),
    ).with_stage(
        STAGE_REALIZE_ENV,
        state.stage(STAGE_REALIZE_ENV).model_copy(
            update={
                "status": "completed",
                "completed_at": started_at,
                "outputs": {"environment_fingerprint": fingerprint},
            }
        ),
    )
    store = RunSetStateStore(tmp_path / "certification-state.json")
    store.save(state)

    evidence = StageEngine(bundle=bundle, driver=driver)._realized_deployment_evidence(
        bundle.rows[0],
        store.load(),
        observed_at=completed_at + timedelta(seconds=1),
    )

    assert evidence["region"] == "CA-MTL-1"
    assert evidence["provider_observations"]["region_raw"] == "CA-MTL-1"
    assert evidence["unavailable"].get("region") is None


def test_provision_record_rejects_conflicting_accepted_datacenter() -> None:
    driver = RunPodOrchestrationDriver(transport=FakeRunPodTransport())

    with pytest.raises(RunPodDriverError, match="datacenter conflicts"):
        driver._provision_record(
            {
                "id": "pod-123",
                "dataCenterId": "US-OR-1",
                "ssh": {"ip": "203.0.113.10", "port": 22},
            },
            provided_pod=False,
            accepted_datacenter="EU-CZ-1",
        )


def test_provision_conflict_tears_down_and_is_not_retried(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    create_call = (
        "pod",
        "create",
        "--name",
        "feedbax-orchestration-2026-01-02-deadbeef",
        "--image",
        bundle.environment.image_id,
        "--ports",
        "22/tcp,8080/http",
        "--gpu-id",
        "NVIDIA GeForce RTX 4090",
        "--data-center-ids",
        "EU-CZ-1",
    )
    transport.queue_runpodctl(create_call, CommandResult(0, '{"id":"pod-conflict"}'))
    transport.queue_runpodctl(
        ("pod", "get", "pod-conflict", "--output", "json"),
        CommandResult(
            0,
            json.dumps(
                {
                    "id": "pod-conflict",
                    "dataCenterId": "US-OR-1",
                    "ssh": {"ip": "203.0.113.10", "port": 22},
                }
            ),
        ),
    )
    transport.queue_runpodctl(
        ("pod", "get", "pod-conflict", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=("EU-CZ-1",),
            image=bundle.environment.image_id or "",
        ),
        transport=transport,
    )
    driver._preflight_passed = True

    with pytest.raises(RunPodDriverError, match="datacenter conflicts"):
        driver.finish_acquired_pod(
            bundle,
            AcquisitionResult("pod-conflict", "EU-CZ-1"),
            "intent-conflict",
        )
    cleanup = driver.teardown(bundle, _state(bundle))

    assert cleanup["pod_absence"]["verified"] is True
    assert ("remove", "pod", "pod-conflict") in transport.runpodctl_calls


def test_provided_pod_does_not_inherit_configured_datacenter_authority() -> None:
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-provided",
            datacenters=("EU-CZ-1",),
        ),
        transport=FakeRunPodTransport(),
    )

    record = driver._provision_record(
        {
            "id": "pod-provided",
            "ssh": {"ip": "203.0.113.10", "port": 22},
        },
        provided_pod=True,
    )

    assert record["region"] is None
    assert record["provider_observation_basis"] == "runpodctl pod get response"


def test_keyboard_interrupt_during_provision_self_heals_owned_pod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("pod", "get", "pod-interrupt", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(image=bundle.environment.image_id or ""),
        transport=transport,
    )
    driver._preflight_passed = True
    monkeypatch.setattr(
        driver,
        "create_pod_once",
        lambda _bundle, _candidate, _intent_id: AcquisitionResult("pod-interrupt", None),
    )

    def interrupted_endpoint(_pod_id: str) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(driver, "_wait_for_endpoint", interrupted_endpoint)

    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    state = _state(bundle)
    store.save(state)
    engine = StageEngine(bundle=bundle, driver=driver, store=store)
    with _ScopedSignalSupervisor() as supervisor:
        engine._signal_supervisor = supervisor
        with pytest.raises(KeyboardInterrupt):
            engine._engine_owned_provision(state, attempt_ordinal=1)

    assert ("remove", "pod", "pod-interrupt") in transport.runpodctl_calls
    assert driver.has_pending_owned_resource() is False


def test_provision_cleanup_failure_does_not_mask_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-interrupt"), CommandResult(1, "", "busy"))
    transport.queue_runpodctl(("stop", "pod", "pod-interrupt"), CommandResult(1, "", "still busy"))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(image=bundle.environment.image_id or ""),
        transport=transport,
    )
    driver._preflight_passed = True
    monkeypatch.setattr(
        driver,
        "create_pod_once",
        lambda _bundle, _candidate, _intent_id: AcquisitionResult("pod-interrupt", None),
    )

    def interrupted_endpoint(_pod_id: str) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(driver, "_wait_for_endpoint", interrupted_endpoint)

    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    state = _state(bundle)
    store.save(state)
    engine = StageEngine(bundle=bundle, driver=driver, store=store)
    with _ScopedSignalSupervisor() as supervisor:
        engine._signal_supervisor = supervisor
        with pytest.raises(KeyboardInterrupt):
            engine._engine_owned_provision(state, attempt_ordinal=1)

    assert driver.has_pending_owned_resource() is True


def test_provider_authorization_failure_stops_stage_once(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    call = (
        "pod",
        "create",
        "--name",
        "feedbax-orchestration-2026-01-02-deadbeef",
        "--image",
        "runpod/pytorch:1.0.3@sha256:" + "a" * 64,
        "--ports",
        "22/tcp,8080/http",
        "--gpu-id",
        "NVIDIA GeForce RTX 4090",
        "--data-center-ids",
        "CA-MTL-1",
    )
    transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, '{"clientBalance":10}')
    )
    transport.queue_runpodctl(call, CommandResult(401, '{"statusCode":401,"code":"unauthorized"}'))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=("CA-MTL-1",),
            image=bundle.environment.image_id,
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )
    store = RunSetStateStore(bundle.run_set_dir / "authorization.json")
    state = _state(bundle)
    store.save(state)
    driver._preflight_passed = True
    driver.create_pod_once = lambda *_args: (_ for _ in ()).throw(
        AcquisitionCreateError(
            "unauthorized",
            clean_rejection=True,
            evidence={"returncode": 401, "classification": "clean"},
        )
    )
    engine = StageEngine(bundle=bundle, driver=driver, store=store)
    with pytest.raises(ProvisioningAttemptError) as raised:
        engine._engine_owned_provision(state, attempt_ordinal=1)
    assert raised.value.retryable is True
    assert store.load().acquisition_intents[0].state == "failed-unacquired"


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
        bundle.environment.image_id,
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
    transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, '{"clientBalance": 10}')
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA RTX 2000 Ada Generation",
            image=bundle.environment.image_id or "",
            max_acquire_seconds=1,
            poll_seconds=1,
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    acquisitions = iter([AcquisitionResult("pod-1", None), AcquisitionResult("pod-2", None)])
    driver.create_pod_once = lambda *_args: next(acquisitions)
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
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert all(check.status == "pass" for check in driver.preflight_checks(bundle))
    driver.create_pod_once = lambda *_args: AcquisitionResult("pod-late", None)
    store = RunSetStateStore(bundle.run_set_dir / "late-state.json")
    state = _state(bundle)
    store.save(state)
    engine = StageEngine(bundle=bundle, driver=driver, store=store)
    with _ScopedSignalSupervisor() as supervisor:
        engine._signal_supervisor = supervisor
        with pytest.raises(ProvisioningAttemptError, match="timed out waiting.*after 2s"):
            engine._engine_owned_provision(state, attempt_ordinal=1)

    assert ("remove", "pod", "pod-late") in transport.runpodctl_calls
    assert all("nvidia-smi" not in command for command in transport.ssh_commands)


def test_realize_env_rsyncs_repos_literal_patches_and_bootstrap(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))  # mkdir
    transport.queue_ssh(CommandResult(1, ""))  # fingerprint probe mismatch
    local_repo = tmp_path / "local repos" / "feedbax [dev]"
    local_repo.parent.mkdir()
    _init_snapshot_repo(local_repo)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            local_repos={"feedbax": local_repo},
            remote_repos={"feedbax": "/workspace/dev repos/feedbax [dev]"},
            path_patches=(
                (
                    "/workspace/feedbax/runtime.cfg",
                    "/Users/mll/local feedbax",
                    "/workspace/feedbax",
                ),
            ),
        ),
        transport=transport,
    )

    fingerprint = driver.realize_env(bundle, _sealed_state(driver, bundle))

    fingerprint_payload = json.loads(fingerprint)
    assert fingerprint_payload["schema_version"] == ("feedbax.runpod_environment_fingerprint.v1")
    assert fingerprint_payload["runtime"]["jax_platform"] == "gpu"
    assert fingerprint_payload["feedbax_plugins"][0]["name"] == "rlrmp2"
    assert len(transport.rsync_calls) == 1
    source, target, delete, excludes = transport.rsync_calls[0]
    assert source.endswith("/")
    assert source != str(local_repo) + "/"
    assert (
        Path(source).name
        == driver.repo_realization_plan().snapshot_manifest.repos["feedbax"].content_sha256
    )
    assert target == "/workspace/dev repos/feedbax [dev]/"
    assert delete is True
    assert excludes == ()
    joined = "\n".join(transport.ssh_commands)
    assert "perl -0pi" in joined
    assert "\\Q$ENV{PATCH_FROM}\\E" in joined
    assert "PATCH_FROM='/Users/mll/local feedbax'" in joined
    assert "uv sync --frozen" in joined
    assert "uv pip install extra" in joined
    assert "jax.__version__" in joined
    assert "jax[cuda12]" in joined
    assert "compose_application()" in joined
    assert "lockfile digest mismatch" in joined


def test_runpod_snapshot_digest_changes_environment_reuse_key(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    local_repo = tmp_path / "feedbax"
    _init_snapshot_repo(local_repo)
    config = RunPodDriverConfig(local_repos={"feedbax": local_repo})
    (local_repo / "tracked.txt").write_text("dirty one\n", encoding="utf-8")
    first_driver = RunPodOrchestrationDriver(config=config, transport=FakeRunPodTransport())
    first_plan = first_driver.seal_repo_realization_plan(bundle)
    first_fingerprint = compute_runpod_environment_fingerprint(bundle, first_plan)
    (local_repo / "tracked.txt").write_text("dirty two\n", encoding="utf-8")
    second_driver = RunPodOrchestrationDriver(config=config, transport=FakeRunPodTransport())
    second_plan = second_driver.seal_repo_realization_plan(bundle)
    second_fingerprint = compute_runpod_environment_fingerprint(bundle, second_plan)

    first = first_plan.repos["feedbax"].snapshot
    second = second_plan.repos["feedbax"].snapshot
    assert first.commit == second.commit
    assert first.dirty is second.dirty is True
    assert first.content_sha256 != second.content_sha256
    assert first_fingerprint != second_fingerprint


def test_runpod_plan_root_and_resolution_changes_invalidate_reuse_key(
    tmp_path: Path,
) -> None:
    lock_text = (
        'version = 1\n[[package]]\nname = "consumer"\nsource = { editable = "../provider" }\n'
    )
    bundle, config = _layout_case(
        tmp_path,
        lock_text,
        local_repos={
            "consumer": tmp_path / "consumer",
            "provider": tmp_path / "provider",
        },
        remote_repos={
            "consumer": "/workspace/consumer",
            "provider": "/workspace/provider",
        },
    )
    driver = RunPodOrchestrationDriver(config=config, transport=FakeRunPodTransport())
    plan = driver.seal_repo_realization_plan(bundle)
    changed_entries = dict(plan.repos)
    changed_entries["provider"] = changed_entries["provider"].model_copy(
        update={"remote_root": "/workspace/provider-changed"}
    )
    changed_root = RepoRealizationPlan.create(
        primary_repo=plan.primary_repo,
        repos=changed_entries,
        editable_source_resolutions=plan.editable_source_resolutions,
        snapshot_manifest=plan.snapshot_manifest,
    )
    original_resolution = plan.editable_source_resolutions[0]
    changed_resolution = RepoRealizationPlan.create(
        primary_repo=plan.primary_repo,
        repos=plan.repos,
        editable_source_resolutions=[
            original_resolution.model_copy(update={"target_subpath": "variant"})
        ],
        snapshot_manifest=plan.snapshot_manifest,
    )

    original = compute_runpod_environment_fingerprint(bundle, plan)
    assert compute_runpod_environment_fingerprint(bundle, changed_root) != original
    assert compute_runpod_environment_fingerprint(bundle, changed_resolution) != original


def test_plan_records_deduplicated_lock_sources_with_complete_keys(tmp_path: Path) -> None:
    repeated = (
        'version = 1\n[[package]]\nname = "one"\n'
        'source = { editable = "../provider" }\n'
        '[[package]]\nname = "two"\nsource = { editable = "../provider" }\n'
    )
    bundle, config = _layout_case(
        tmp_path,
        repeated,
        local_repos={
            "consumer": tmp_path / "consumer",
            "provider": tmp_path / "provider",
        },
        remote_repos={
            "consumer": "/workspace/consumer",
            "provider": "/workspace/provider",
        },
    )
    plan = RunPodOrchestrationDriver(
        config=config, transport=FakeRunPodTransport()
    ).seal_repo_realization_plan(bundle)

    assert plan.editable_source_resolutions == [
        EditableSourceResolution(
            consumer_repo="consumer",
            lock_relative_path="uv.lock",
            source_form="editable",
            spelling="../provider",
            target_repo="provider",
            target_subpath=".",
        )
    ]


def test_realize_env_rejects_preflight_plan_digest_mismatch(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(local_repos={"feedbax": tmp_path}),
        transport=transport,
    )
    state = _sealed_state(driver, bundle)
    preflight = state.stage(STAGE_PREFLIGHT).model_copy(
        update={"outputs": {"driver_evidence": {"repo_realization_plan_digest": "0" * 64}}}
    )
    state = state.model_copy(update={"stages": {STAGE_PREFLIGHT: preflight}})

    with pytest.raises(RunPodDriverError, match="between PREFLIGHT and REALIZE_ENV"):
        driver.realize_env(bundle, state)

    assert transport.ssh_commands == []


def test_runpod_wholesale_snapshot_sync_deletes_stale_secret(tmp_path: Path) -> None:
    rsync = shutil.which("rsync")
    assert rsync is not None, "rsync is required for the governed-transfer contract test"

    class LocalRsyncTransport(FakeRunPodTransport):
        def rsync(
            self,
            source: str,
            target: str,
            *,
            delete: bool = False,
            excludes: tuple[str, ...] = (),
            timeout_seconds: float | None = None,
        ) -> CommandResult:
            super().rsync(
                source,
                target,
                delete=delete,
                excludes=excludes,
                timeout_seconds=timeout_seconds,
            )
            args = [rsync, "-a"]
            if delete:
                args.append("--delete")
            args.extend([source, target])
            completed = subprocess.run(args, capture_output=True, text=True)
            return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    bundle = _bundle(tmp_path)
    local_repo = tmp_path / "feedbax"
    _init_snapshot_repo(local_repo)
    (local_repo / ".gitignore").write_text("*.secret\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(local_repo), "add", ".gitignore"], check=True)
    subprocess.run(
        ["git", "-C", str(local_repo), "commit", "-m", "ignore secrets"],
        check=True,
        capture_output=True,
    )
    (local_repo / "local.secret").write_text("never ship\n", encoding="utf-8")
    remote = tmp_path / "remote repo"
    remote.mkdir()
    (remote / "stale.secret").write_text("remove me\n", encoding="utf-8")
    transport = LocalRsyncTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(local_repos={"feedbax": local_repo}),
        transport=transport,
    )
    manifest = driver.seal_repo_realization_plan(bundle).snapshot_manifest
    snapshot_root = driver._repo_snapshots.snapshots["feedbax"].staging_root

    driver._rsync_repo(str(snapshot_root), str(remote))

    assert not (remote / "stale.secret").exists()
    assert not (remote / "local.secret").exists()
    assert (remote / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"
    assert manifest.repos["feedbax"].file_count == 3
    assert transport.rsync_calls == [(f"{snapshot_root}/", f"{remote}/", True, ())]


@pytest.mark.parametrize("repo_order", [("outer", "inner"), ("inner", "outer")])
def test_nested_remote_destinations_fail_before_wholesale_sync_in_both_orders(
    tmp_path: Path,
    repo_order: tuple[str, str],
) -> None:
    local_repos = {
        "outer": tmp_path / "local-outer",
        "inner": tmp_path / "local-inner",
    }
    bundle, _config = _layout_case(
        tmp_path,
        "version = 1\n",
        local_repos=local_repos,
        remote_repos={"outer": "/unused/outer", "inner": "/unused/inner"},
        primary_repo="outer",
    )
    destination = tmp_path / "remote" / "a"
    nested = destination / "b"
    nested.mkdir(parents=True)
    marker = nested / "must-survive.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    remote_roots = {name: str(destination if name == "outer" else nested) for name in repo_order}
    config = RunPodDriverConfig(
        local_repos=local_repos,
        remote_repos=remote_roots,
        primary_repo="outer",
    )
    transport = FakeRunPodTransport()

    with pytest.raises(RepoRealizationError, match="overlapping remote repo roots"):
        RunPodOrchestrationDriver(config=config, transport=transport).seal_repo_realization_plan(
            bundle
        )

    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert transport.rsync_calls == []


def test_realize_env_fails_closed_when_repo_rsync_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, ""))  # mkdir
    transport.rsync_result = CommandResult(1, "", "remote path rejected")
    local_repo = tmp_path / "feedbax"
    _init_snapshot_repo(local_repo)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            local_repos={"feedbax": local_repo},
            remote_repos={"feedbax": "/workspace/feedbax"},
        ),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="rsync repo .*remote path rejected"):
        driver.realize_env(bundle, _sealed_state(driver, bundle))


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
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    driver.realize_env(bundle, _sealed_state(driver, bundle))

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
        config=RunPodDriverConfig(
            overlay_steps=(),
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match="important failure detail"):
        driver.realize_env(bundle, _sealed_state(driver, bundle))


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
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(RunPodDriverError, match="uv sync timed out after 3s"):
        driver.realize_env(bundle, _sealed_state(driver, bundle))

    assert clock.now == 3


def test_realize_env_fingerprint_match_skips_environment_steps(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(local_repos={"feedbax": tmp_path}),
        transport=transport,
    )
    state = _sealed_state(driver, bundle)
    declaration_fingerprint = compute_runpod_environment_fingerprint(
        bundle, driver.repo_realization_plan()
    )
    declaration = {
        "declaration_sha256": declaration_fingerprint,
        "image_id": bundle.environment.image_id,
        "lockfile_hashes": bundle.environment.lockfile_hashes,
        "python_version": bundle.environment.python_version,
    }
    realized_fingerprint = _realized_fingerprint(declaration)
    transport.queue_ssh(CommandResult(0, ""))
    transport.queue_ssh(CommandResult(0, declaration_fingerprint))
    transport.queue_ssh(CommandResult(0, realized_fingerprint))
    assert driver.realize_env(bundle, state) == realized_fingerprint
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
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            image=bundle.environment.image_id or "",
        ),
        transport=transport,
    )

    outputs = driver.stage_inputs(bundle, _state(bundle))

    assert outputs["input_count"] == 0
    assert outputs["payload_count"] == 1
    payload = bundle.rows[0].execution.payload
    assert transport.rsync_calls == [
        (
            str(bundle.run_set_dir / ".stage-attempts/stage-inputs-0/inputs") + "/",
            "/workspace/feedbax_runs/2026-01-02-deadbeef/.stage-attempts/stage-inputs-0/inputs/",
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


def test_stage_inputs_heartbeats_while_blocking_transfer(tmp_path: Path) -> None:
    class BlockingTransferTransport(FakeRunPodTransport):
        def rsync(
            self,
            source: str,
            target: str,
            *,
            delete: bool = False,
            excludes: tuple[str, ...] = (),
            timeout_seconds: float | None = None,
        ) -> CommandResult:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                heartbeats = [
                    command
                    for command in self.ssh_commands
                    if command.startswith("touch -- ") and "/.host-active" in command
                ]
                if len(heartbeats) >= 2:
                    return super().rsync(
                        source,
                        target,
                        delete=delete,
                        excludes=excludes,
                        timeout_seconds=timeout_seconds,
                    )
                time.sleep(0.005)
            return CommandResult(1, stderr="host heartbeat did not recur")

    bundle = _bundle(tmp_path, deadman_enabled=True)
    transport = BlockingTransferTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            image=bundle.environment.image_id or "",
            poll_seconds=0.01,
        ),
        transport=transport,
    )

    driver.stage_inputs(bundle, _state(bundle))

    heartbeats = [
        command
        for command in transport.ssh_commands
        if command.startswith("touch -- ") and "/.host-active" in command
    ]
    assert len(heartbeats) >= 2
    assert all(".stage-attempts/stage-inputs-0" in command for command in heartbeats)


def test_stage_inputs_ignores_legacy_runpod_baseline_metadata(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path).model_copy(
        update={
            "metadata": {
                "runpod_baselines": [{"checkpoint_path": "/custody", "completed_batches": 12000}]
            }
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            image=bundle.environment.image_id or "",
        ),
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
    assert transport.rsync_calls[0][1] != ("/workspace/feedbax_runs/2026-01-02-deadbeef/inputs/")


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
    assert "while { [ ! -s" in launch_command
    assert "FEEDBAX_RUN_SET_ID=2026-01-02-deadbeef" in launch_command
    assert "FEEDBAX_ROW_ID=warm" in launch_command
    assert (
        "FEEDBAX_RUN_EVENTS_DIR=/workspace/feedbax_runs/2026-01-02-deadbeef/events"
        in launch_command
    )
    assert "FEEDBAX_ENV_FINGERPRINT=fingerprint-123" in launch_command
    assert "JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache" in launch_command
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launch_command
    assert "FEEDBAX_PROCESS_IDENTITY=" in launch_command
    assert "feedbax.orchestration.process_identity.v1" in launch_command
    assert "refusing to adopt started row without verified process identity" in launch_command
    assert outputs["process_identity"]["run_set_id"] == bundle.run_set_id
    assert "rm -f" in launch_command
    assert all("deadman" not in command for command in transport.ssh_commands)


def test_runpod_launch_refuses_unverified_recovered_pid(tmp_path: Path) -> None:
    class UnverifiedRecoveryTransport(FakeRunPodTransport):
        def ssh(self, command: str) -> CommandResult:
            if "identity_status='missing'" in command:
                self.ssh_commands.append(command)
                return CommandResult(
                    0,
                    json.dumps(
                        {
                            "gpu": "",
                            "rows": {
                                "warm": {
                                    "status": "failed",
                                    "pid": 4321,
                                    "process_identity": None,
                                    "identity_status": "missing",
                                    "detail": "process identity record is missing",
                                }
                            },
                        }
                    ),
                )
            return super().ssh(command)

    bundle = _bundle(tmp_path)
    transport = UnverifiedRecoveryTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    outputs = driver.launch_row(bundle, bundle.rows[0], _state(bundle))

    assert outputs["status"] == "failed"
    assert outputs["pid"] == 4321
    assert outputs["event_discrepancies"][0]["code"] == "unverified_process_identity"


def test_runpod_stop_requires_and_checks_durable_process_identity(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    row = bundle.rows[0]
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )
    pid_only = _state(bundle).model_copy(
        update={"rows": {row.row_id: RowState(status="running", pid=4321)}}
    )

    with pytest.raises(RunPodDriverError, match="durable process identity is absent"):
        driver.stop_row(bundle, row, pid_only)

    assert transport.ssh_commands == []

    identity = ProcessIdentity(
        run_set_id=bundle.run_set_id,
        row_id=row.row_id,
        pid=4321,
        process_group_id=4321,
        launch_token="b" * 64,
    )
    owned = pid_only.model_copy(
        update={
            "rows": {
                row.row_id: RowState(status="running", pid=identity.pid, process_identity=identity)
            }
        }
    )
    transport.queue_ssh(CommandResult(0, json.dumps({"status": "stopped", "detail": None})))

    stopped = driver.stop_row(bundle, row, owned)

    assert stopped["status"] == "stopped"
    command = transport.ssh_commands[-1]
    assert "os.killpg(process_group_id, signal.SIGTERM)" in command
    assert "process group member" in command
    assert identity.launch_token in command


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
        workdir=str(tmp_path),
        jax_cache_dir=str(tmp_path / "jax-cache"),
        env_fingerprint="fingerprint-123",
        execution_namespace=build_runpod_execution_namespace(
            bundle=bundle,
            row=row,
            remote_run_dir=str(remote_run_dir),
            remote_sentinel_dir=str(sentinel_dir),
            env_fingerprint="fingerprint-123",
        ),
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
    assert launch_command.count("uv run --no-sync") == 1
    assert "--execution-context-json" in launch_command
    assert planned_run_id in launch_command
    assert '"environment_fingerprint":"fingerprint-123"' in launch_command
    assert '"row_id":"warm"' in launch_command
    assert '"lowerer_id":"feedbax.tests.runpod"' in launch_command
    assert "feedbax-training-run:feedbax-training-run:" not in launch_command


def test_launch_row_does_not_double_wrap_normalized_native_command(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.rows[0]
    row = original.model_copy(
        update={
            "launch": RowLaunchSpec(
                command=[
                    "uv",
                    "run",
                    "--no-sync",
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
                        planned_run_id="feedbax-training-run:normalized-warm",
                        authored_payload_hash="a" * 64,
                        lowered_execution_payload_hash=original.execution.payload.sha256,
                        axis_coordinates={},
                        lowerer_identities=[],
                    )
                }
            ),
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(config=RunPodDriverConfig(), transport=transport)

    driver.launch_row(bundle, row, _state(bundle))

    launch_command = transport.ssh_commands[0]
    assert launch_command.count("uv run --no-sync") == 1
    assert "--execution-context-json" in launch_command
    assert "specs/warm.json" in launch_command


def test_launch_row_runs_evaluation_matrix_in_realized_uv_environment(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.rows[0]
    row = original.model_copy(
        update={
            "execution_family": "evaluation-matrix",
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "matrix-harness"],
                payload_routing={"kind": "registered-execution-payload"},
            ),
        }
    )
    bundle = bundle.model_copy(
        update={
            "execution_family": "evaluation-matrix",
            "rows": [row],
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(config=RunPodDriverConfig(), transport=transport)

    driver.launch_row(bundle, row, _state(bundle))

    launch_command = transport.ssh_commands[0]
    assert launch_command.count("uv run --no-sync") == 1
    assert "python -m feedbax matrix-harness" in launch_command


def test_launch_row_entry_fallback_keeps_single_uv_environment_prefix(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.rows[0]
    row = original.model_copy(
        update={
            "launch": RowLaunchSpec(
                entry="scripts/run worker.py",
                collect=original.launch.collect,
            )
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(config=RunPodDriverConfig(), transport=transport)

    driver.launch_row(bundle, row, _state(bundle))

    launch_command = transport.ssh_commands[0]
    assert launch_command.count("uv run --no-sync") == 1
    assert shlex.quote("scripts/run worker.py") in launch_command


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
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    driver.realize_env(bundle, _sealed_state(driver, bundle))

    joined = "\n".join(transport.ssh_commands)
    assert "command -v setsid >/dev/null" in transport.ssh_commands[0]
    assert "command -v runpodctl" in joined
    assert all(text in joined for text in ("RUNPOD_API_KEY=$(tr", "runpodctl get pod pod-123"))
    watchdog = next(command for command in transport.ssh_commands if "deadman.pid" in command)
    assert "feedbax.orchestration.process_identity.v1" in watchdog
    assert "FEEDBAX_PROCESS_IDENTITY=" in watchdog
    assert "setsid -f bash -lc" in watchdog
    assert "echo $$ >" in watchdog
    assert "deadman.installed" in watchdog
    assert 'newest=$(stat -c %Y "$installed"' in watchdog
    assert 'find "$run_dir/.stage-attempts" -type f -print' in watchdog
    assert 'rm -f "$pid_file"' in watchdog
    assert "deadman.process.json" in watchdog
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
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    checks = driver.preflight_checks(bundle)

    assert [check.name for check in checks] == [
        "runpod-repo-snapshots",
        "input-provider-bindings",
        "runpod-remote-smoke-applicability",
        "continuation-schedule-consistency",
        "runpod-lockfiles-declared",
        "runpod-remote-layout-vs-lock",
        "runpod-image-immutable",
        "runpod-image-tag-exists",
        "runpod-python-version-declared",
        "runpod-gpu-policy-declared",
        "runpod-credentials",
        "runpod-balance-floor",
        "runpod-deadman-credentials",
    ]
    assert all(check.status == "pass" for check in checks)
    assert transport.runpodctl_calls == [("user", "--output", "json")]


def test_runpod_preflight_reports_independent_static_failures_in_canonical_order(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={
            "environment": bundle.environment.model_copy(
                update={
                    "image_id": "runpod/pytorch:mutable",
                    "lockfile_hashes": {},
                    "python_version": None,
                }
            )
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(local_repos={"feedbax": tmp_path}),
        transport=transport,
    )

    checks = driver.preflight_checks(bundle)

    assert [check.name for check in checks] == [
        "runpod-repo-snapshots",
        "input-provider-bindings",
        "runpod-remote-smoke-applicability",
        "continuation-schedule-consistency",
        "runpod-lockfiles-declared",
        "runpod-remote-layout-vs-lock",
        "runpod-image-immutable",
        "runpod-image-tag-exists",
        "runpod-python-version-declared",
        "runpod-gpu-policy-declared",
        "runpod-credentials",
        "runpod-balance-floor",
        "runpod-deadman-credentials",
    ]
    assert [check.name for check in checks if check.status == "fail"] == [
        "runpod-lockfiles-declared",
        "runpod-image-immutable",
        "runpod-python-version-declared",
        "runpod-gpu-policy-declared",
    ]
    image_check = next(check for check in checks if check.name == "runpod-image-tag-exists")
    assert image_check.observed == {
        "outcome": "skipped-due-to-dependency",
        "dependencies": ["runpod-lockfiles-declared", "runpod-image-immutable"],
    }
    credential_check = next(check for check in checks if check.name == "runpod-credentials")
    assert credential_check.observed == {
        "outcome": "skipped-due-to-dependency",
        "dependencies": ["runpod-lockfiles-declared"],
    }
    assert transport.operations == []
    assert driver._preflight_passed is False


def test_stage_preflight_composes_core_and_runpod_static_failures_before_provider_queries(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(
        update={
            "environment": bundle.environment.model_copy(
                update={
                    "image_id": "runpod/pytorch:mutable",
                    "python_version": None,
                }
            )
        }
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    with pytest.raises(PreflightFailed) as raised:
        StageEngine(bundle=bundle, driver=driver).run()

    state = RunSetStateStore(bundle.run_set_dir / "state.json").load()
    checks = state.stage(STAGE_PREFLIGHT).checks
    assert [check.name for check in checks if check.status == "fail"] == [
        "environment-declaration",
        "runpod-image-immutable",
        "runpod-python-version-declared",
    ]
    message = str(raised.value)
    assert (
        message.index("environment-declaration")
        < message.index("runpod-image-immutable")
        < message.index("runpod-python-version-declared")
    )
    provider_check = next(check for check in checks if check.name == "runpod-credentials")
    assert provider_check.observed["outcome"] == "skipped-due-to-dependency"
    assert provider_check.observed["dependencies"] == [
        "environment-declaration",
    ]
    assert transport.operations == []


def test_runpod_preflight_queries_credentials_despite_independent_declaration_failures(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, deadman_enabled=True)
    bundle = bundle.model_copy(
        update={"environment": bundle.environment.model_copy(update={"python_version": None})}
    )
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(0, json.dumps({"clientBalance": 12.5})),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    checks = driver.preflight_checks(bundle)
    named = {check.name: check for check in checks}

    assert [check.name for check in checks if check.status == "fail"] == [
        "runpod-python-version-declared",
        "runpod-gpu-policy-declared",
        "runpod-deadman-credentials",
    ]
    assert named["runpod-credentials"].status == "pass"
    assert named["runpod-balance-floor"].status == "pass"
    assert transport.runpodctl_calls == [("user", "--output", "json")]


def test_runpod_preflight_collects_image_probe_exception_and_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    monkeypatch.setattr(
        transport,
        "image_exists",
        lambda _image: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(0, json.dumps({"clientBalance": 12.5})),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    checks = driver.preflight_checks(bundle)
    named = {check.name: check for check in checks}

    assert named["runpod-image-tag-exists"].status == "fail"
    assert named["runpod-image-tag-exists"].detail == (
        "image existence query raised RuntimeError: registry unavailable"
    )
    assert named["runpod-credentials"].status == "pass"
    assert named["runpod-balance-floor"].status == "pass"
    assert [check.name for check in checks].index("runpod-image-tag-exists") < [
        check.name for check in checks
    ].index("runpod-credentials")


def test_completed_preflight_validation_rejects_dependency_skip_sentinel(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(0, json.dumps({"clientBalance": 12.5})),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )
    checks = driver.preflight_checks(bundle)
    balance_index = next(
        index for index, check in enumerate(checks) if check.name == "runpod-balance-floor"
    )
    checks[balance_index] = checks[balance_index].model_copy(
        update={
            "detail": "skipped-due-to-dependency: runpod-credentials",
            "observed": {
                "outcome": "skipped-due-to-dependency",
                "dependencies": ["runpod-credentials"],
            },
        }
    )

    with pytest.raises(RunPodDriverError, match="includes a failing check"):
        driver._validate_preflight_checks(bundle, checks)


def test_runpod_preflight_records_balance_skip_after_credential_failure(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(1, "", "credential rejected"),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    named = {check.name: check for check in driver.preflight_checks(bundle)}

    assert named["runpod-credentials"].status == "fail"
    assert named["runpod-credentials"].detail == "credential rejected"
    assert named["runpod-balance-floor"].status == "pass"
    assert named["runpod-balance-floor"].observed == {
        "outcome": "skipped-due-to-dependency",
        "dependencies": ["runpod-credentials"],
    }
    assert driver._preflight_passed is False


class RemoteSmokeTransport(FakeRunPodTransport):
    def __init__(self, *, probe_status: str) -> None:
        super().__init__()
        self.probe_status = probe_status
        self.identity_status = "owned"
        self.terminal_status = probe_status if probe_status in {"completed", "failed"} else None

    def ssh(self, command: str) -> CommandResult:
        self.ssh_commands.append(command)
        if "paths=json.loads(sys.argv[1])" in command:
            digest = "d" * 64
            return CommandResult(
                0,
                json.dumps(
                    {
                        "events": digest,
                        "row_roots": digest,
                        "sentinels": digest,
                        "staged_inputs": digest,
                    }
                ),
            )
        if "reports={}" in command:

            def report(row_id: str) -> dict[str, Any]:
                identity = {
                    "schema_id": "feedbax.orchestration.process_identity",
                    "schema_version": "feedbax.orchestration.process_identity.v1",
                    "mechanism": "environment-token-v1",
                    "run_set_id": "2026-01-02-deadbeef",
                    "row_id": f"smoke-{row_id}",
                    "pid": 5000,
                    "process_group_id": 5000,
                    "launch_token": "c" * 64,
                }
                return {
                    "status": self.probe_status,
                    "terminal_status": self.terminal_status,
                    "pid": identity["pid"],
                    "process_identity": identity,
                    "identity_status": self.identity_status,
                }

            return CommandResult(
                0,
                json.dumps(
                    {
                        "rows": {
                            f"smoke-{row_id}": report(row_id) for row_id in ("warm", "cool", "hot")
                        }
                    }
                ),
            )
        if "os.killpg(process_group_id, signal.SIGTERM)" in command:
            return CommandResult(0, json.dumps({"status": "stopped", "detail": None}))
        if "smoke executor log lacks a typed result" in command:
            return CommandResult(
                0,
                json.dumps(
                    {
                        "start_completed_batches": 0,
                        "end_completed_batches": 2,
                        "payload_binding_status": "verified",
                        "executor_result_sha256": "e" * 64,
                    }
                ),
            )
        return CommandResult(0, "")


def _native_smoke_bundle(tmp_path: Path, *, deadline_seconds: int = 1800) -> RunBundle:
    bundle = _bundle(tmp_path, smoke_enabled=True)
    original = bundle.rows[0]
    provenance = TrainingRowProvenance(
        row_id=original.row_id,
        row_index=0,
        planned_run_id="feedbax-training-run:remote-smoke-test",
        authored_payload_hash="a" * 64,
        lowered_execution_payload_hash=original.execution.payload.sha256,
        axis_coordinates={},
        lowerer_identities=[
            RowLowererIdentity(lowerer_id="feedbax.tests.remote-smoke", lowerer_version="v1")
        ],
    )
    row = original.model_copy(
        update={
            "execution": original.execution.model_copy(update={"row_provenance": provenance}),
            "launch": original.launch.model_copy(
                update={"command": ["python", "-m", "feedbax", "execute-training-run-spec"]}
            ),
        }
    )
    return bundle.model_copy(update={"rows": [row], "smoke_deadline_seconds": deadline_seconds})


def test_runpod_remote_smoke_records_derived_bounded_evidence(tmp_path: Path) -> None:
    bundle = _native_smoke_bundle(tmp_path)
    transport = RemoteSmokeTransport(probe_status="completed")
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-smoke",
            remote_run_root="/remote/runs",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    evidence = driver.smoke_row(bundle, bundle.rows[0], _state(bundle))

    assert evidence["status"] == "passed"
    assert evidence["start_completed_batches"] == 0
    assert evidence["end_completed_batches"] == 2
    assert evidence["payload_binding_status"] == "verified"
    assert evidence["cleanup_status"] == "removed"
    assert evidence["protected_paths_before"] == evidence["protected_paths_after"]
    assert evidence["derived_run_id"] != evidence["planned_run_id"]
    provenance = evidence["derived_producer_context"]["execution"]["row_provenance"]
    assert provenance["planned_run_id"] == evidence["derived_run_id"]
    launch = next(command for command in transport.ssh_commands if "--update-budget" in command)
    assert "--update-budget 2" in launch
    assert "/smoke/warm" in launch
    assert "/rows/warm/checkpoints" not in launch
    assert "/sentinels/smoke-warm.started" in launch


def test_smoke_policy_does_not_change_real_launch_identity(tmp_path: Path) -> None:
    smoke_bundle = _native_smoke_bundle(tmp_path)
    no_smoke_bundle = smoke_bundle.model_copy(update={"smoke_enabled": False})
    row = smoke_bundle.rows[0]
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-smoke"),
        transport=FakeRunPodTransport(),
    )

    def launch_command(bundle: RunBundle) -> str:
        namespace = build_runpod_execution_namespace(
            bundle=bundle,
            row=row,
            remote_run_dir=driver._remote_run_dir(bundle),
            remote_sentinel_dir=driver._remote_sentinel_dir(bundle),
            env_fingerprint="fingerprint-123",
        )
        return build_launch_row_command(
            bundle=bundle,
            row=row,
            workdir=driver._row_workdir(row),
            env_fingerprint="fingerprint-123",
            jax_cache_dir="/workspace/jax_cache",
            execution_namespace=namespace,
        )

    def normalize(command: str) -> str:
        return re.sub(r"\b[0-9a-f]{64}\b", "<digest>", command)

    assert normalize(launch_command(smoke_bundle)) == normalize(launch_command(no_smoke_bundle))
    assert row.execution.row_provenance is not None
    assert row.execution.row_provenance.planned_run_id in launch_command(smoke_bundle)


def test_smoke_launch_copies_declared_continuation_seed_into_scratch_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _native_smoke_bundle(tmp_path)
    row = bundle.rows[0]
    source = SimpleNamespace(custody=SimpleNamespace(target_role="checkpoint"))
    monkeypatch.setattr(runpod_module, "native_resume_checkpoint_source", lambda *_: source)
    monkeypatch.setattr(
        runpod_module,
        "build_native_resume_seed_command",
        lambda source_path, attempt, target, resolved: (
            f"SEED {source_path} {attempt} {target} {resolved is source}"
        ),
    )
    namespace = build_runpod_execution_namespace(
        bundle=bundle,
        row=row,
        remote_run_dir="/remote/runs",
        remote_sentinel_dir="/remote/runs/sentinels",
        env_fingerprint="fingerprint-123",
        scratch_root="/remote/runs/smoke/warm",
        run_identity="feedbax-training-run:remote-smoke-test--smoke",
        sentinel_stem="smoke-warm",
    )

    command = build_launch_row_command(
        bundle=bundle,
        row=row,
        workdir="/workspace/feedbax",
        env_fingerprint="fingerprint-123",
        jax_cache_dir="/workspace/jax_cache",
        execution_namespace=namespace,
        update_budget=2,
    )

    assert "SEED /remote/runs/inputs/" in command
    assert "/remote/runs/smoke/warm/.checkpoint-seed-attempt" in command
    assert "/remote/runs/smoke/warm/checkpoints True" in command
    assert command.index("SEED ") < command.index("touch /remote/runs/sentinels/smoke-warm.started")


def test_stage_smoke_internal_exception_records_valid_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _native_smoke_bundle(tmp_path)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-smoke"),
        transport=FakeRunPodTransport(),
    )
    monkeypatch.setattr(
        driver,
        "smoke_row",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("internal smoke defect")),
    )

    with pytest.raises(OrchestrationStageError, match="internal smoke defect") as raised:
        StageEngine(bundle=bundle, driver=driver)._stage_smoke(_state(bundle))

    evidence = RemoteSmokeEvidence.model_validate(raised.value.stage_outputs)
    assert evidence.rows[0].status == "failed"
    assert evidence.rows[0].cleanup_status == "failed"
    assert evidence.rows[0].payload_binding_status == "not-run"


def test_stage_smoke_records_evidence_for_every_non_opted_out_row(tmp_path: Path) -> None:
    bundle = _native_smoke_bundle(tmp_path)
    first = bundle.rows[0]
    assert first.execution.row_provenance is not None
    rows = [
        first.model_copy(
            update={
                "row_id": row_id,
                "execution": first.execution.model_copy(
                    update={
                        "row_provenance": first.execution.row_provenance.model_copy(
                            update={"row_id": row_id, "row_index": index}
                        )
                    }
                ),
            }
        )
        for index, row_id in enumerate(("warm", "cool", "hot"))
    ]
    bundle = bundle.model_copy(update={"rows": rows})
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-smoke"),
        transport=RemoteSmokeTransport(probe_status="completed"),
    )

    _state_after, outputs = StageEngine(bundle=bundle, driver=driver)._stage_smoke(_state(bundle))
    evidence = RemoteSmokeEvidence.model_validate(outputs)

    assert [row.row_id for row in evidence.rows] == ["warm", "cool", "hot"]
    assert all(row.status == "passed" for row in evidence.rows)


def test_runpod_remote_smoke_deadline_escalates_and_records_failure(tmp_path: Path) -> None:
    bundle = _native_smoke_bundle(tmp_path, deadline_seconds=60)
    transport = RemoteSmokeTransport(probe_status="running")
    clock = FakeClock()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-smoke",
            remote_run_root="/remote/runs",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(RunPodRemoteSmokeError, match="wall-clock deadline") as raised:
        driver.smoke_row(bundle, bundle.rows[0], _state(bundle))

    assert raised.value.evidence["status"] == "failed"
    assert raised.value.evidence["cleanup_status"] == "removed"
    termination = next(
        command
        for command in transport.ssh_commands
        if "os.killpg(process_group_id, signal.SIGTERM)" in command
    )
    assert "os.killpg(process_group_id, signal.SIGKILL)" in termination
    assert "/sentinels/smoke-warm.failed" in termination


def test_runpod_remote_smoke_failed_probe_records_failure(tmp_path: Path) -> None:
    bundle = _native_smoke_bundle(tmp_path)
    transport = RemoteSmokeTransport(probe_status="failed")
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-smoke",
            remote_run_root="/remote/runs",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    with pytest.raises(RunPodRemoteSmokeError, match="remote smoke row 'warm' failed") as raised:
        driver.smoke_row(bundle, bundle.rows[0], _state(bundle))

    assert raised.value.evidence["status"] == "failed"
    assert raised.value.evidence["cleanup_status"] == "removed"
    assert not any("os.killpg" in command for command in transport.ssh_commands)


def test_runpod_remote_smoke_preserves_unverified_live_process(tmp_path: Path) -> None:
    bundle = _native_smoke_bundle(tmp_path)
    transport = RemoteSmokeTransport(probe_status="failed")
    transport.identity_status = "mismatch"
    transport.terminal_status = None
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-smoke",
            remote_run_root="/remote/runs",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
    )

    with pytest.raises(RunPodRemoteSmokeError) as raised:
        driver.smoke_row(bundle, bundle.rows[0], _state(bundle))

    assert raised.value.evidence["cleanup_status"] == "failed"
    assert not any("os.killpg" in command for command in transport.ssh_commands)
    assert not any("rm -rf" in command for command in transport.ssh_commands)


def test_runpod_preflight_rejects_non_native_smoke_row_by_name(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, smoke_enabled=True)
    row = bundle.rows[0].model_copy(
        update={"launch": bundle.rows[0].launch.model_copy(update={"command": ["echo", "full"]})}
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=FakeRunPodTransport(),
    )

    checks = driver.preflight_checks(bundle)

    applicability = next(
        check for check in checks if check.name == "runpod-remote-smoke-applicability"
    )
    assert applicability.status == "fail"
    assert "warm" in (applicability.detail or "")


def test_declared_continuation_without_source_custody_fails_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    continuation = CheckpointContinuationRequest(
        source_completed_batches=10,
        additional_batches=4,
    )
    monkeypatch.setattr(
        runpod_module,
        "_authenticated_row_training_spec",
        lambda _row: SimpleNamespace(
            checkpoint_progress=SimpleNamespace(continuation=continuation)
        ),
    )
    transport = FakeRunPodTransport()
    monkeypatch.setattr(
        transport,
        "image_exists",
        lambda _image: pytest.fail("provider transport must not run after schedule failure"),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=transport,
        training_method_registry=default_training_program_registry(),
    )

    checks = driver.preflight_checks(bundle)

    assert len(checks) == 13
    schedule_check = next(
        check for check in checks if check.name == "continuation-schedule-consistency"
    )
    assert schedule_check.status == "fail"
    assert "no exact authenticated resume checkpoint source" in (schedule_check.detail or "")
    provider_check = next(check for check in checks if check.name == "runpod-credentials")
    assert provider_check.observed == {
        "outcome": "skipped-due-to-dependency",
        "dependencies": ["continuation-schedule-consistency"],
    }
    assert transport.operations == []


def test_remote_layout_lock_without_path_sources_passes_after_digest_verification(
    tmp_path: Path,
) -> None:
    bundle, config = _layout_case(tmp_path, "version = 1\n")

    error, observed = _realization_layout_error(bundle, config)

    assert error is None
    assert observed["path_sources"] == []


@pytest.mark.parametrize(
    ("lock_text", "expected"),
    [
        ("version = [\n", "malformed TOML"),
        ('version = 1\n[[package]]\nname = "bad"\nsource = { virtual = "." }\n', "virtual"),
        ('version = 1\n[[package]]\nname = "bad"\nsource = { editable = "/tmp/x" }\n', "absolute"),
        (
            'version = 1\n[[package]]\nname = "bad"\nsource = { editable = "../../escape" }\n',
            "unmatched local target",
        ),
        ("version = 999\n", "unsupported lock version"),
        (
            'version = 1\n[[package]]\nname = "bad"\nsource = { local = "../repo" }\n',
            "local",
        ),
        (
            'version = 1\n[[package]]\nname = "bad"\n'
            'source = { editable = ".", registry = "https://example.invalid" }\n',
            "mixed source forms",
        ),
    ],
)
def test_remote_layout_rejects_invalid_lock_content(
    tmp_path: Path,
    lock_text: str,
    expected: str,
) -> None:
    bundle, config = _layout_case(tmp_path, lock_text)

    error, _ = _realization_layout_error(bundle, config)

    assert error is not None
    assert expected in error


def test_remote_layout_rejects_missing_or_hash_mismatched_lock(tmp_path: Path) -> None:
    missing_bundle, config = _layout_case(tmp_path, "version = 1\n", write_lock=False)
    missing_error, _ = _realization_layout_error(missing_bundle, config)
    assert missing_error is not None
    assert "cannot inspect sealed lock" in missing_error

    lock_path = Path(config.local_repos["consumer"]) / "uv.lock"
    lock_path.write_text("version = 2\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(lock_path.parent), "add", "uv.lock"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(lock_path.parent), "commit", "-m", "add mismatched lock"],
        check=True,
        capture_output=True,
    )
    mismatch_error, _ = _realization_layout_error(missing_bundle, config)
    assert mismatch_error is not None
    assert "hash mismatch" in mismatch_error


@pytest.mark.parametrize("form", ["editable", "path"])
def test_remote_layout_handles_self_local_source(tmp_path: Path, form: str) -> None:
    bundle, config = _layout_case(
        tmp_path,
        f'version = 1\n[[package]]\nname = "consumer"\nsource = {{ {form} = "." }}\n',
    )

    error, observed = _realization_layout_error(bundle, config)

    assert error is None
    assert observed["path_sources"][0]["spelling"] == "."
    assert observed["path_sources"][0]["planned_remote_target"] == "/workspace/consumer"
    assert str(tmp_path) not in json.dumps(observed)


def test_remote_layout_uses_sealed_bytes_after_live_mutation(tmp_path: Path) -> None:
    live_root = tmp_path / "live" / "consumer"
    bundle, config = _layout_case(
        tmp_path,
        'version = 1\n[[package]]\nname = "consumer"\nsource = { editable = "." }\n',
        local_repos={"consumer": live_root},
    )
    driver = RunPodOrchestrationDriver(config=config, transport=FakeRunPodTransport())
    plan = driver.seal_repo_realization_plan(bundle)
    (live_root / "uv.lock").write_text("version = 999\n", encoding="utf-8")
    sealed_error, _ = validate_runpod_repo_realization_plan(
        bundle, config, plan, driver._repo_snapshots
    )

    assert sealed_error is None


def test_remote_layout_preserves_exact_spaced_sibling_spelling(tmp_path: Path) -> None:
    consumer = tmp_path / "10 Projects" / "10 PhD" / "rlrmp2"
    feedbax = tmp_path / "10 Projects" / "10 PhD" / "20 Feedbax" / "feedbax"
    local = {"rlrmp2": consumer, "feedbax": feedbax}
    remote = {
        "rlrmp2": "/workspace/10 Projects/10 PhD/rlrmp2",
        "feedbax": "/workspace/10 Projects/10 PhD/20 Feedbax/feedbax",
    }
    lock_text = (
        'version = 1\n[[package]]\nname = "feedbax"\n'
        'source = { editable = "../20 Feedbax/feedbax" }\n'
    )
    bundle, config = _layout_case(
        tmp_path,
        lock_text,
        local_repos=local,
        remote_repos=remote,
        primary_repo="rlrmp2",
    )

    error, observed = _realization_layout_error(bundle, config)

    assert error is None
    assert observed["path_sources"][0]["spelling"] == "../20 Feedbax/feedbax"


def test_remote_layout_mismatch_names_spelling_and_both_planned_targets(
    tmp_path: Path,
) -> None:
    consumer = tmp_path / "rlrmp2"
    target = tmp_path / "feedbax"
    bundle, config = _layout_case(
        tmp_path,
        'version = 1\n[[package]]\nname = "feedbax"\nsource = { editable = "../feedbax" }\n',
        local_repos={"rlrmp2": consumer, "feedbax": target},
        remote_repos={"rlrmp2": "/workspace/rlrmp2", "feedbax": "/workspace/wrong-name"},
        primary_repo="rlrmp2",
    )

    error, _ = _realization_layout_error(bundle, config)

    assert error is not None
    assert "../feedbax" in error
    assert "/workspace/feedbax" in error
    assert "/workspace/wrong-name" in error


def test_remote_layout_rejects_ambiguous_repo_containment(tmp_path: Path) -> None:
    consumer = tmp_path / "outer" / "consumer"
    bundle, config = _layout_case(
        tmp_path,
        'version = 1\n[[package]]\nname = "consumer"\nsource = { editable = "." }\n',
        local_repos={"outer": tmp_path / "outer", "consumer": consumer},
        remote_repos={"outer": "/workspace/outer", "consumer": "/workspace/consumer"},
    )

    error, _ = _realization_layout_error(bundle, config)

    assert error is not None
    assert "ambiguous local target" in error


def test_remote_layout_rejects_mapping_mismatch_duplicate_roots_and_lock_patch(
    tmp_path: Path,
) -> None:
    bundle, config = _layout_case(tmp_path, "version = 1\n")
    mismatch = config.__class__(
        **{**config.__dict__, "remote_repos": {"other": "/workspace/other"}}
    )
    mismatch_error, _ = _realization_layout_error(bundle, mismatch)
    assert mismatch_error is not None
    assert "repo key mismatch" in mismatch_error

    duplicate = config.__class__(
        **{
            **config.__dict__,
            "local_repos": {"consumer": tmp_path / "consumer", "other": tmp_path / "other"},
            "remote_repos": {"consumer": "/workspace/same", "other": "/workspace/same"},
        }
    )
    duplicate_error, _ = _realization_layout_error(bundle, duplicate)
    assert duplicate_error is not None
    assert "overlapping remote repo roots" in duplicate_error

    patched = config.__class__(
        **{
            **config.__dict__,
            "path_patches": (("/workspace/consumer/uv.lock", "old", "new"),),
        }
    )
    patch_error, _ = _realization_layout_error(bundle, patched)
    assert patch_error is not None
    assert "path_patches" in patch_error
    assert "uv.lock" in patch_error


def test_layout_failure_short_circuits_before_all_provider_queries(tmp_path: Path) -> None:
    bundle, config = _layout_case(tmp_path, "version = [\n")
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(config=config, transport=transport)

    checks = driver.preflight_checks(bundle)

    assert len(checks) == 13
    named = {check.name: check for check in checks}
    assert named["runpod-remote-layout-vs-lock"].status == "fail"
    assert named["runpod-repo-snapshots"].observed == {
        "outcome": "skipped-due-to-dependency",
        "dependencies": ["repo-realization-plan-sealing"],
    }
    assert named["runpod-image-tag-exists"].observed["outcome"] == ("skipped-due-to-dependency")
    assert named["runpod-credentials"].observed["dependencies"] == ["runpod-remote-layout-vs-lock"]
    assert transport.image_exists_calls == []
    assert transport.runpodctl_calls == []
    assert transport.operations == []


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
        update={"environment": bundle.environment.model_copy(update=environment_update)}
    )
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 5090",
            local_repos={"feedbax": tmp_path},
        ),
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
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(local_repos={"feedbax": tmp_path}),
        transport=transport,
    )
    state = _sealed_state(driver, bundle)
    declaration_sha256 = compute_runpod_environment_fingerprint(
        bundle, driver.repo_realization_plan()
    )
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
    with pytest.raises(RunPodDriverError, match=expected_error):
        driver.realize_env(bundle, state)


def test_explicit_primary_repo_controls_environment_and_row_workdirs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            remote_repos={
                "feedbax": "/workspace/feedbax",
                "rlrmp2": "/workspace/rlrmp2",
            },
            primary_repo="feedbax",
        ),
        transport=FakeRunPodTransport(),
    )

    assert driver._primary_workdir() == "/workspace/feedbax"
    assert runpod_row_workdir(driver.config, bundle.rows[0]) == "/workspace/feedbax"


def test_stage_engine_records_named_runpod_checks_before_provision(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
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


def test_fresh_runpod_driver_restores_completed_preflight_before_provision(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, baseline=False)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    first_transport = FakeRunPodTransport()
    first_transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )
    first_driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            pod_id="pod-restored",
            gpu_id="NVIDIA GeForce RTX 4090",
            datacenters=tuple(bundle.deployment_policy.resources.regions),
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=first_transport,
    )
    StageEngine(bundle=bundle, driver=first_driver, store=store).run(stop_after_stage="PREFLIGHT")

    class RestoredDriver(RunPodOrchestrationDriver):
        provision_calls = 0

        def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            del bundle, state
            assert self._preflight_passed is True
            type(self).provision_calls += 1
            return {"driver": "fake-restored-runpod"}

    resume_transport = FakeRunPodTransport()
    resumed = StageEngine(
        bundle=bundle,
        driver=RestoredDriver(
            config=RunPodDriverConfig(
                pod_id="pod-restored",
                gpu_id="NVIDIA GeForce RTX 4090",
                datacenters=tuple(bundle.deployment_policy.resources.regions),
                image=bundle.environment.image_id or "",
                local_repos={"feedbax": tmp_path},
            ),
            transport=resume_transport,
            realized_capabilities=RunPodOrchestrationDriver.capability_envelope.realize(
                "externally-managed"
            ),
        ),
        store=store,
    ).run(stop_after_stage="PROVISION")

    assert resumed.stage(STAGE_PROVISION).status == "completed"
    assert RestoredDriver.provision_calls == 1
    assert resume_transport.operations == []


def test_completed_preflight_without_schedule_check_is_rerun(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, baseline=False)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    first_transport = FakeRunPodTransport()
    first_transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )
    config = RunPodDriverConfig(
        gpu_id="NVIDIA GeForce RTX 4090",
        image=bundle.environment.image_id or "",
        local_repos={"feedbax": tmp_path},
    )
    state = StageEngine(
        bundle=bundle,
        driver=RunPodOrchestrationDriver(config=config, transport=first_transport),
        store=store,
    ).run(stop_after_stage=STAGE_PREFLIGHT)
    preflight = state.stage(STAGE_PREFLIGHT)
    old_checks = [
        check for check in preflight.checks if check.name != "continuation-schedule-consistency"
    ]
    old_outputs = {
        **preflight.outputs,
        "checks": [check.model_dump(mode="json") for check in old_checks],
    }
    store.save(
        state.with_stage(
            STAGE_PREFLIGHT,
            preflight.model_copy(update={"checks": old_checks, "outputs": old_outputs}),
        )
    )
    rerun_transport = FakeRunPodTransport()
    rerun_transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )

    rerun = StageEngine(
        bundle=bundle,
        driver=RunPodOrchestrationDriver(config=config, transport=rerun_transport),
        store=store,
    ).run(stop_after_stage=STAGE_PREFLIGHT)

    assert any(
        check.name == "continuation-schedule-consistency"
        for check in rerun.stage(STAGE_PREFLIGHT).checks
    )
    assert rerun_transport.runpodctl_calls == [("user", "--output", "json")]


@pytest.mark.parametrize(
    "tamper", ["evidence-shape", "checks", "bundle", "timestamps", "driver", "teardown"]
)
def test_fresh_runpod_driver_rejects_invalid_completed_preflight_before_provision(
    tmp_path: Path, tamper: str
) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, baseline=False)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    first_transport = FakeRunPodTransport()
    first_transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )
    first_driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            gpu_id="NVIDIA GeForce RTX 4090",
            image=bundle.environment.image_id or "",
            local_repos={"feedbax": tmp_path},
        ),
        transport=first_transport,
    )
    state = StageEngine(bundle=bundle, driver=first_driver, store=store).run(
        stop_after_stage="PREFLIGHT"
    )
    preflight = state.stage(STAGE_PREFLIGHT)
    if tamper == "evidence-shape":
        preflight = preflight.model_copy(
            update={"outputs": {**preflight.outputs, "driver_evidence": []}}
        )
    elif tamper == "checks":
        checks = list(preflight.checks)
        checks[0] = checks[0].model_copy(update={"status": "fail"})
        preflight = preflight.model_copy(update={"checks": checks})
    elif tamper == "bundle":
        assemble = state.stage("ASSEMBLE").model_copy(
            update={"outputs": {**state.stage("ASSEMBLE").outputs, "bundle_sha256": "0" * 64}}
        )
        state = state.with_stage("ASSEMBLE", assemble)
    elif tamper == "timestamps":
        preflight = preflight.model_copy(update={"completed_at": None})
    state = state.with_stage(STAGE_PREFLIGHT, preflight)
    store.save(state)

    class MustNotProvision(RunPodOrchestrationDriver):
        def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            del bundle, state
            pytest.fail("invalid completed PREFLIGHT must fail before provision")

    resume_transport = FakeRunPodTransport()
    resume_config = RunPodDriverConfig(
        gpu_id="NVIDIA GeForce RTX 4090",
        min_balance_usd=6.0 if tamper == "driver" else 5.0,
        image=bundle.environment.image_id or "",
        local_repos={"feedbax": tmp_path},
        auto_teardown=tamper != "teardown",
    )
    with pytest.raises(PreflightFailed, match="persisted driver PREFLIGHT evidence is invalid"):
        StageEngine(
            bundle=bundle,
            driver=MustNotProvision(
                config=resume_config,
                transport=resume_transport,
            ),
            store=store,
        ).run(stop_after_stage="PROVISION")

    assert resume_transport.operations == []


def test_fresh_runpod_driver_rejects_unbound_completed_preflight_offline(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, baseline=False)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    first_transport = FakeRunPodTransport()
    first_transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )
    state = StageEngine(
        bundle=bundle,
        driver=RunPodOrchestrationDriver(
            config=RunPodDriverConfig(
                gpu_id="NVIDIA GeForce RTX 4090",
                image=bundle.environment.image_id or "",
                local_repos={"feedbax": tmp_path},
            ),
            transport=first_transport,
        ),
        store=store,
    ).run(stop_after_stage="PREFLIGHT")
    preflight = state.stage(STAGE_PREFLIGHT).model_copy(
        update={
            "outputs": {
                key: value
                for key, value in state.stage(STAGE_PREFLIGHT).outputs.items()
                if key != "driver_evidence"
            }
        }
    )
    failed_provision = state.stage(STAGE_PROVISION).model_copy(
        update={"status": "failed", "attempts": 1, "error": "non-retryable-error"}
    )
    state = state.with_stage(STAGE_PREFLIGHT, preflight).with_stage(
        STAGE_PROVISION, failed_provision
    )
    state = state.model_copy(
        update={
            "abort_reason": "non-retryable-error",
            "provisioning_stop_reason": "non-retryable-error",
            "budget_counters": {"provisioning_stop_reason": "non-retryable-error"},
        }
    )
    store.save(state)

    class LegacyRestoredDriver(RunPodOrchestrationDriver):
        provision_calls = 0

        def provision(self, bundle: RunBundle, state: RunSetState) -> dict[str, Any]:
            del bundle, state
            assert self._preflight_passed is True
            type(self).provision_calls += 1
            return {"driver": "fake-legacy-restored-runpod"}

    resume_transport = FakeRunPodTransport()
    with pytest.raises(PreflightFailed, match="lacks repo realization plan binding"):
        StageEngine(
            bundle=bundle,
            driver=LegacyRestoredDriver(
                config=RunPodDriverConfig(
                    gpu_id="NVIDIA GeForce RTX 4090",
                    image=bundle.environment.image_id or "",
                    local_repos={"feedbax": tmp_path},
                ),
                transport=resume_transport,
            ),
            store=store,
        ).run(stop_after_stage="PROVISION")

    assert LegacyRestoredDriver.provision_calls == 0
    assert resume_transport.operations == []


def test_completed_preflight_rejects_evidence_without_repo_plan_binding(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    config = RunPodDriverConfig(
        gpu_id="NVIDIA GeForce RTX 4090",
        image=bundle.environment.image_id or "",
        local_repos={"feedbax": tmp_path},
    )
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )
    state = StageEngine(
        bundle=bundle,
        driver=RunPodOrchestrationDriver(config=config, transport=transport),
        store=RunSetStateStore(bundle.run_set_dir / "state.json"),
    ).run(stop_after_stage=STAGE_PREFLIGHT)
    preflight = state.stage(STAGE_PREFLIGHT)
    old_checks = [
        check for check in preflight.checks if check.name != "runpod-remote-layout-vs-lock"
    ]
    old_outputs = {
        **preflight.outputs,
        "checks": [check.model_dump(mode="json") for check in old_checks],
    }
    old_outputs.pop("driver_evidence", None)
    old_state = state.with_stage(
        STAGE_PREFLIGHT,
        preflight.model_copy(update={"checks": old_checks, "outputs": old_outputs}),
    )

    with pytest.raises(RunPodDriverError, match="lacks repo realization plan binding"):
        RunPodOrchestrationDriver(
            config=config, transport=FakeRunPodTransport()
        ).restore_completed_preflight(bundle, old_state)


def test_separate_process_existing_run_rejects_legacy_preflight(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, baseline=False)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("user", "--output", "json"),
        CommandResult(0, json.dumps({"clientBalance": 12.5})),
    )
    state = StageEngine(
        bundle=bundle,
        driver=RunPodOrchestrationDriver(
            config=RunPodDriverConfig(
                gpu_id="NVIDIA GeForce RTX 4090",
                datacenters=tuple(bundle.deployment_policy.resources.regions),
                image=bundle.environment.image_id or "",
                local_repos={"feedbax": tmp_path},
            ),
            transport=transport,
        ),
        store=store,
    ).run(stop_after_stage=STAGE_PREFLIGHT)
    assert len(state.stage(STAGE_PREFLIGHT).checks) == 25
    preflight = state.stage(STAGE_PREFLIGHT)
    preflight = preflight.model_copy(
        update={
            "outputs": {
                key: value for key, value in preflight.outputs.items() if key != "driver_evidence"
            }
        }
    )
    failed_provision = state.stage(STAGE_PROVISION).model_copy(
        update={"status": "failed", "attempts": 1, "error": "non-retryable-error"}
    )
    state = state.with_stage(STAGE_PREFLIGHT, preflight).with_stage(
        STAGE_PROVISION, failed_provision
    )
    state = state.model_copy(
        update={
            "abort_reason": "non-retryable-error",
            "provisioning_stop_reason": "non-retryable-error",
            "budget_counters": {"provisioning_stop_reason": "non-retryable-error"},
        }
    )
    store.save(state)

    script = r"""
import asyncio
import sys

from feedbax.bin import orchestrate
from feedbax.plugins.composition import compose_application

bootstrap = asyncio.run(compose_application(local_component_source=None))
orchestrate._run_existing(
    sys.argv[1],
    stop_after_stage="PROVISION",
    conformance_registry=bootstrap.bundle.conformance_checks,
    training_method_registry=bootstrap.bundle.training_programs,
    driver_registry=bootstrap.bundle.drivers,
    plugin_provenance=bootstrap.provenance,
)
"""
    env = {**os.environ, "FEEDBAX_ORCHESTRATION_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", script, bundle.run_set_id],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "lacks repo realization plan binding" in result.stderr


@pytest.mark.parametrize(
    "tamper",
    ["checks-both", "balance", "timestamps", "bindings", "config", "image", "run-set", "bundle"],
)
def test_legacy_completed_preflight_rejects_semantic_tampering_offline(
    tmp_path: Path, tamper: str
) -> None:
    bundle = _bundle(tmp_path, keep_alive=True, baseline=False)
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    first_transport = FakeRunPodTransport()
    first_transport.queue_runpodctl(
        ("user", "--output", "json"), CommandResult(0, json.dumps({"clientBalance": 12.5}))
    )
    state = StageEngine(
        bundle=bundle,
        driver=RunPodOrchestrationDriver(
            config=RunPodDriverConfig(
                gpu_id="NVIDIA GeForce RTX 4090",
                image=bundle.environment.image_id or "",
                local_repos={"feedbax": tmp_path},
            ),
            transport=first_transport,
        ),
        store=store,
    ).run(stop_after_stage="PREFLIGHT")
    preflight = state.stage(STAGE_PREFLIGHT)
    copied_checks = list(preflight.checks)
    if tamper == "checks-both":
        index = next(
            i for i, check in enumerate(copied_checks) if check.name == "runpod-image-immutable"
        )
        copied_checks[index] = copied_checks[index].model_copy(update={"observed": "tampered"})
    elif tamper == "balance":
        index = next(
            i for i, check in enumerate(copied_checks) if check.name == "runpod-balance-floor"
        )
        copied_checks[index] = copied_checks[index].model_copy(update={"observed": "not-a-number"})
    elif tamper == "timestamps":
        preflight = preflight.model_copy(update={"completed_at": None})
    outputs = {
        "checks": [check.model_dump(mode="json") for check in copied_checks],
    }
    preflight = preflight.model_copy(update={"checks": copied_checks, "outputs": outputs})
    state = state.with_stage(STAGE_PREFLIGHT, preflight)
    if tamper == "run-set":
        state = state.model_copy(update={"run_set_id": "different-run-set"})
    store.save(state)

    bindings = (
        [InputProviderRootBinding("unexpected.binding", tmp_path)] if tamper == "bindings" else []
    )
    resume_config = RunPodDriverConfig(
        gpu_id="NVIDIA GeForce RTX 5090" if tamper == "config" else "NVIDIA GeForce RTX 4090",
        image=(
            "runpod/pytorch:different" if tamper == "image" else bundle.environment.image_id or ""
        ),
        local_repos={"feedbax": tmp_path},
    )
    resume_bundle = (
        bundle.model_copy(update={"keep_alive": False}) if tamper == "bundle" else bundle
    )
    resume_transport = FakeRunPodTransport()
    with pytest.raises(PreflightFailed, match="persisted driver PREFLIGHT evidence is invalid"):
        StageEngine(
            bundle=resume_bundle,
            driver=RunPodOrchestrationDriver(
                config=resume_config,
                transport=resume_transport,
                input_provider_bindings=bindings,
            ),
            store=store,
        ).run(stop_after_stage="PROVISION")

    assert resume_transport.operations == []


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
    transport.queue_ssh(CommandResult(0, "file"))
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


@pytest.mark.parametrize(
    "source_kind",
    ["row-relative", "row-relative-dot", "run-relative", "absolute"],
)
def test_collect_skips_legacy_evaluation_raw_store(
    tmp_path: Path,
    source_kind: str,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )
    row_id = bundle.rows[0].row_id
    source = {
        "row-relative": "evaluation",
        "row-relative-dot": "./evaluation",
        "run-relative": f"rows/{row_id}/evaluation",
        "absolute": f"{driver._remote_run_dir(bundle)}/rows/{row_id}/evaluation",
    }[source_kind]
    row = bundle.rows[0].model_copy(
        update={
            "execution_family": "evaluation-matrix",
            "launch": bundle.rows[0].launch.model_copy(update={"collect": [source]}),
        }
    )
    bundle = bundle.model_copy(
        update={
            "execution_family": "evaluation-matrix",
            "rows": [row],
        }
    )

    collected = driver.collect(bundle, row, _state(bundle))

    event_name = f"{row.row_id}.events.jsonl"
    assert collected == {
        event_name: str(bundle.run_set_dir / "events" / event_name),
    }
    assert transport.ssh_commands == []
    assert len(transport.rsync_calls) == 1
    assert transport.rsync_calls[0][0].endswith(f"/events/{event_name}")


def test_collect_native_outputs_uses_row_dir_and_canonical_events(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    row = bundle.rows[0].model_copy(
        update={
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "execute-training-run-spec", "spec.json"],
                collect=[
                    "manifest.json",
                    "training-diagnostics.json",
                    "checkpoints",
                    "manifests",
                ],
            )
        }
    )
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, "file"))
    transport.queue_ssh(CommandResult(0, "file"))
    transport.queue_ssh(CommandResult(0, "directory"))
    transport.queue_ssh(CommandResult(0, "directory"))
    driver = RunPodOrchestrationDriver(config=RunPodDriverConfig(), transport=transport)

    collected = driver.collect(bundle, row, _state(bundle))

    remote = "/workspace/feedbax_runs/2026-01-02-deadbeef"
    assert [call[0] for call in transport.rsync_calls] == [
        f"{remote}/rows/warm/manifest.json",
        f"{remote}/rows/warm/training-diagnostics.json",
        f"{remote}/rows/warm/checkpoints/",
        f"{remote}/rows/warm/manifests/",
        f"{remote}/events/warm.events.jsonl",
    ]
    assert transport.rsync_calls[2][1].endswith("/collected/warm/checkpoints/")
    assert transport.rsync_calls[2][2] is True
    assert transport.rsync_calls[3][1].endswith("/collected/warm/manifests/")
    assert transport.rsync_calls[3][2] is True
    assert collected["warm.events.jsonl"] == str(bundle.run_set_dir / "events/warm.events.jsonl")


@pytest.mark.parametrize(
    ("source_kind", "error"),
    [
        ("missing", "declared collection output is absent"),
        ("symlink", "declared collection output is a symlink"),
        ("unsupported", "not a regular file or directory"),
    ],
)
def test_collect_rejects_unsafe_or_absent_remote_outputs(
    source_kind: str,
    error: str,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_ssh(CommandResult(0, source_kind))
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(ssh_host="198.51.100.10", ssh_port=2222),
        transport=transport,
    )

    with pytest.raises(RunPodDriverError, match=error):
        driver.collect(bundle, bundle.rows[0], _state(bundle))

    assert transport.rsync_calls == []


@pytest.mark.parametrize(
    ("directory_name", "payload_paths"),
    [
        ("checkpoints", ("latest.json", "transactions/tx-terminal/manifest.json")),
        ("manifests", ("artifacts/sha256/ab/abcdef/training-history.json",)),
    ],
)
def test_local_and_runpod_collect_directory_contents_at_declared_target(
    directory_name: str,
    payload_paths: tuple[str, ...],
    tmp_path: Path,
) -> None:
    rsync_executable = next(
        (path for path in ("/usr/bin/rsync", "/opt/homebrew/bin/rsync") if Path(path).is_file()),
        None,
    )
    if rsync_executable is None:
        pytest.skip("rsync is not installed")
    source = tmp_path / "remote" / directory_name
    source.mkdir(parents=True)
    for relative_path in payload_paths:
        payload = source / relative_path
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text('{"source":"terminal"}\n')

    local_root = tmp_path / "local"
    local_root.mkdir()
    local_bundle = _bundle(local_root)
    local_row = local_bundle.rows[0].model_copy(
        update={
            "launch": local_bundle.rows[0].launch.model_copy(
                update={"collect": [str(source)], "metadata": {}}
            )
        }
    )
    local_collected = LocalOrchestrationDriver(cwd=tmp_path, freeze_lines=[]).collect(
        local_bundle, local_row, _state(local_bundle)
    )

    class RealRsyncTransport(FakeRunPodTransport):
        def ssh(self, command: str) -> CommandResult:
            self.ssh_commands.append(command)
            return CommandResult(0, "directory")

        def rsync(
            self,
            source_path: str,
            target_path: str,
            *,
            delete: bool = False,
            excludes: tuple[str, ...] = (),
            timeout_seconds: float | None = None,
        ) -> CommandResult:
            self.rsync_calls.append((source_path, target_path, delete, tuple(excludes)))
            args = [rsync_executable, "-a"]
            if delete:
                args.append("--delete")
            args.extend([source_path, target_path])
            completed = subprocess.run(args, check=False, capture_output=True, text=True)
            return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    runpod_root = tmp_path / "runpod"
    runpod_root.mkdir()
    runpod_bundle = _bundle(runpod_root)
    runpod_row = runpod_bundle.rows[0].model_copy(
        update={
            "launch": runpod_bundle.rows[0].launch.model_copy(
                update={"collect": [str(source)], "metadata": {}}
            )
        }
    )
    stale_nested = (
        runpod_bundle.run_set_dir / "collected" / "warm" / directory_name / directory_name
    )
    stale_nested.mkdir(parents=True)
    (stale_nested / "stale.json").write_text('{"source":"stale"}\n')
    transport = RealRsyncTransport()
    runpod_collected = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(), transport=transport
    ).collect(runpod_bundle, runpod_row, _state(runpod_bundle))

    for collected in (local_collected, runpod_collected):
        collected_root = Path(collected[directory_name])
        for relative_path in payload_paths:
            assert (collected_root / relative_path).read_text(encoding="utf-8") == (
                '{"source":"terminal"}\n'
            )
        assert not (collected_root / directory_name).exists()
    assert transport.rsync_calls == [
        (
            str(source) + "/",
            str(Path(runpod_collected[directory_name])) + "/",
            True,
            (),
        )
    ]


def _completed_recovery_state(bundle: RunBundle) -> RunSetState:
    provision = {"driver": "runpod", "pod_id": "pod-gone"}
    return RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState(status="completed") for row in bundle.rows},
        provision_record=provision,
        stages={
            "COLLECT": StageState(status="running", attempts=6),
            "PROVISION": StageState(status="completed", outputs=provision),
            "TEARDOWN": StageState(
                status="completed",
                outputs={
                    "driver": "runpod",
                    "teardown": "removed",
                    "pod_id": "pod-gone",
                    "pod_absence": {
                        "verified": True,
                        "pod_id": "pod-gone",
                        "terminal_observation": "not-found",
                    },
                    "final_pod_inventory": {
                        "verified": True,
                        "outcome": "empty",
                        "pod_count": 0,
                        "pod_ids": [],
                        "scope": "provider-account",
                        "observation_basis": "runpodctl pod list --output json",
                    },
                },
            ),
        },
    )


def test_collect_recovers_preserved_nested_directory_without_provider_calls(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    row = bundle.rows[0].model_copy(
        update={
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "execute-training-run-spec", "spec.json"],
                collect=["manifest.json", "training-diagnostics.json", "checkpoints"],
            )
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    preserved = bundle.run_set_dir / "collected" / row.row_id
    nested = preserved / "checkpoints" / "checkpoints"
    nested.mkdir(parents=True)
    (preserved / "manifest.json").write_text('{"kind":"TrainingRunManifest"}\n')
    (preserved / "training-diagnostics.json").write_text("{}\n")
    (nested / "latest.json").write_text('{"transaction_id":"tx-terminal"}\n')
    transaction = nested / "transactions" / "tx-terminal"
    transaction.mkdir(parents=True)
    (transaction / "manifest.json").write_text("{}\n")
    state = _completed_recovery_state(bundle)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=transport,
        collection_recovery_bindings=[CollectionRecoveryBinding(row.row_id, preserved)],
    )

    collected = driver.collect(bundle, row, state)

    recovered = Path(collected["checkpoints"])
    assert recovered != preserved / "checkpoints"
    assert (recovered / "latest.json").is_file()
    assert (recovered / "transactions" / "tx-terminal" / "manifest.json").is_file()
    assert not (recovered / "checkpoints").exists()
    assert (nested / "latest.json").is_file()
    assert transport.runpodctl_calls == []
    assert transport.ssh_commands == []
    assert transport.rsync_calls == []
    evidence = driver.collection_recovery_evidence(row.row_id)
    assert evidence is not None
    assert evidence["provider_calls"] == 0
    assert evidence["original_evidence_untouched"] is True


@pytest.mark.parametrize("tamper", ["missing", "extra", "symlink", "ambiguous"])
def test_collect_recovery_rejects_invalid_preserved_output_map(
    tamper: str,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, baseline=False)
    row = bundle.rows[0].model_copy(
        update={
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "execute-training-run-spec", "spec.json"],
                collect=["manifest.json", "training-diagnostics.json", "checkpoints"],
            )
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    preserved = bundle.run_set_dir / "collected" / row.row_id
    preserved.mkdir(parents=True)
    (preserved / "manifest.json").write_text("{}\n")
    (preserved / "training-diagnostics.json").write_text("{}\n")
    checkpoints = preserved / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "latest.json").write_text("{}\n")
    if tamper == "missing":
        (preserved / "manifest.json").unlink()
    elif tamper == "extra":
        (preserved / "unexpected.json").write_text("{}\n")
    elif tamper == "symlink":
        (preserved / "manifest.json").unlink()
        (preserved / "manifest.json").symlink_to(preserved / "training-diagnostics.json")
    else:
        nested = checkpoints / "checkpoints"
        nested.mkdir()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(row.row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError):
        driver.collect(bundle, row, _completed_recovery_state(bundle))


def test_collect_recovery_requires_verified_empty_final_inventory(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    preserved = bundle.run_set_dir / "collected" / bundle.rows[0].row_id
    preserved.mkdir(parents=True)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(bundle.rows[0].row_id, preserved)],
    )
    state = _completed_recovery_state(bundle).model_copy(
        update={
            "stages": {
                "COLLECT": StageState(status="running", attempts=6),
                "TEARDOWN": StageState(),
            }
        }
    )

    with pytest.raises(CollectionRecoveryError, match="exact pod absence"):
        driver.collect(bundle, bundle.rows[0], state)


def test_collect_recovery_refuses_configured_provider_target(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    preserved = bundle.run_set_dir / "collected" / bundle.rows[0].row_id
    preserved.mkdir(parents=True)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-still-configured"),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(bundle.rows[0].row_id, preserved)],
    )

    with pytest.raises(RunPodDriverError, match="refuses a configured live"):
        driver.collect(bundle, bundle.rows[0], _completed_recovery_state(bundle))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("driver", "other"),
        ("teardown", "skipped"),
        ("pod_id", "other-pod"),
    ],
)
def test_collect_recovery_binds_exact_runpod_teardown(
    field: str,
    value: str,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    preserved = bundle.run_set_dir / "collected" / bundle.rows[0].row_id
    preserved.mkdir(parents=True)
    state = _completed_recovery_state(bundle)
    teardown = state.stage("TEARDOWN")
    state = state.with_stage(
        "TEARDOWN",
        teardown.model_copy(update={"outputs": {**teardown.outputs, field: value}}),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(bundle.rows[0].row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError, match="exact pod absence"):
        driver.collect(bundle, bundle.rows[0], state)


def test_collect_recovery_accepts_stopped_then_removed_teardown(tmp_path: Path) -> None:
    bundle, row, preserved = _recovery_fixture(tmp_path)
    state = _completed_recovery_state(bundle)
    teardown = state.stage("TEARDOWN")
    state = state.with_stage(
        "TEARDOWN",
        teardown.model_copy(
            update={"outputs": {**teardown.outputs, "teardown": "stopped-then-removed"}}
        ),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(row.row_id, preserved)],
    )

    collected = driver.collect(bundle, row, state)

    assert Path(collected["checkpoints"], "latest.json").is_file()


def test_collect_recovery_binds_teardown_to_provision_record(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    preserved = bundle.run_set_dir / "collected" / bundle.rows[0].row_id
    preserved.mkdir(parents=True)
    state = _completed_recovery_state(bundle).model_copy(
        update={"provision_record": {"driver": "runpod", "pod_id": "other-pod"}}
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(bundle.rows[0].row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError, match="exact pod absence"):
        driver.collect(bundle, bundle.rows[0], state)


def test_collect_recovery_requires_canonical_inventory_basis(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    preserved = bundle.run_set_dir / "collected" / bundle.rows[0].row_id
    preserved.mkdir(parents=True)
    state = _completed_recovery_state(bundle)
    teardown = state.stage("TEARDOWN")
    inventory = {
        **teardown.outputs["final_pod_inventory"],
        "observation_basis": "untrusted-cache",
    }
    state = state.with_stage(
        "TEARDOWN",
        teardown.model_copy(
            update={
                "outputs": {
                    **teardown.outputs,
                    "final_pod_inventory": inventory,
                }
            }
        ),
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(bundle.rows[0].row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError, match="provider-account inventory"):
        driver.collect(bundle, bundle.rows[0], state)


def test_collect_recovery_rejects_source_root_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, row, preserved = _recovery_fixture(tmp_path)
    original_copy = collection_recovery._copy_member_no_follow
    raced = False

    def replace_root(*args: Any, **kwargs: Any) -> Any:
        nonlocal raced
        if not raced:
            raced = True
            preserved.rename(preserved.with_name("raced-original"))
            preserved.mkdir()
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(collection_recovery, "_copy_member_no_follow", replace_root)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(row.row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError, match="(replaced|identity changed) while copying"):
        driver.collect(bundle, row, _completed_recovery_state(bundle))


def test_collect_recovery_rejects_symlinked_destination_ancestor(tmp_path: Path) -> None:
    bundle, row, preserved = _recovery_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (bundle.run_set_dir / ".stage-attempts").symlink_to(outside, target_is_directory=True)
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(row.row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError, match="stage-attempts root"):
        driver.collect(bundle, row, _completed_recovery_state(bundle))
    assert list(outside.iterdir()) == []


def test_collect_recovery_rejects_destination_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, row, preserved = _recovery_fixture(tmp_path)
    original_copy = collection_recovery._copy_member_no_follow
    raced = False

    def replace_destination(*args: Any, **kwargs: Any) -> Any:
        nonlocal raced
        if not raced:
            raced = True
            attempts = bundle.run_set_dir / ".stage-attempts"
            attempt = attempts / "collect-recovery-6"
            attempt.rename(attempts / "raced-original")
            attempt.mkdir()
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(
        collection_recovery,
        "_copy_member_no_follow",
        replace_destination,
    )
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(),
        transport=FakeRunPodTransport(),
        collection_recovery_bindings=[CollectionRecoveryBinding(row.row_id, preserved)],
    )

    with pytest.raises(CollectionRecoveryError, match="attempt root identity changed while opening"):
        driver.collect(bundle, row, _completed_recovery_state(bundle))


def _recovery_fixture(tmp_path: Path) -> tuple[RunBundle, Any, Path]:
    bundle = _bundle(tmp_path, baseline=False)
    row = bundle.rows[0].model_copy(
        update={
            "launch": RowLaunchSpec(
                command=["python", "-m", "feedbax", "execute-training-run-spec", "spec.json"],
                collect=["manifest.json", "training-diagnostics.json", "checkpoints"],
            )
        }
    )
    bundle = bundle.model_copy(update={"rows": [row]})
    preserved = bundle.run_set_dir / "collected" / row.row_id
    checkpoints = preserved / "checkpoints" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (preserved / "manifest.json").write_text("{}\n")
    (preserved / "training-diagnostics.json").write_text("{}\n")
    (checkpoints / "latest.json").write_text("{}\n")
    return bundle, row, preserved


def test_teardown_remove_failure_stops_then_removes_owned_pod(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "busy"))
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory()
    driver = _owned_driver(transport=transport)

    result = driver.teardown(bundle, _owned_state(bundle))

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
    second_remove_index = transport.runpodctl_calls.index(("remove", "pod", "pod-123"), 1)
    assert 0 < transport.runpodctl_timeouts[second_remove_index] <= 60
    assert all(
        timeout is not None and 0 < timeout <= 60 for timeout in transport.runpodctl_timeouts
    )


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
    driver = _owned_driver(transport=transport)

    result = driver.teardown(bundle, _owned_state(bundle))

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
    driver = _owned_driver(transport=transport)

    result = driver.teardown(bundle, _owned_state(bundle))

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
        (CommandResult(0, '[{"id":"pod-other","name":"other"}]'), "non-empty"),
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
    driver = _owned_driver(transport=transport)
    engine = StageEngine(
        bundle=bundle,
        driver=driver,
        conformance_registry=CheckRegistry(
            {"fixture": lambda _row: CheckEntry(check_id="fixture", status="pass")}
        ),
    )

    state = engine._run_teardown(_owned_state(bundle), abort=False)

    assert state.stage("TEARDOWN").status == "completed"
    inventory = state.stage("TEARDOWN").outputs["final_pod_inventory"]
    assert inventory["verified"] is False
    assert inventory["outcome"] == expected_outcome
    with pytest.raises(
        OrchestrationStageError,
        match="globally empty provider resource inventory",
    ):
        engine._stage_register(state)


def test_teardown_fails_closed_when_remove_after_stop_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "busy"))
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "still busy"))
    driver = _owned_driver(transport=transport)

    with pytest.raises(RunPodDriverError, match="remove pod after stop failed"):
        driver.teardown(bundle, _owned_state(bundle))

    assert transport.runpodctl_calls == [
        ("remove", "pod", "pod-123"),
        ("stop", "pod", "pod-123"),
        ("remove", "pod", "pod-123"),
    ]


def test_teardown_confirms_absence_when_remove_reports_pod_not_found(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(
        ("remove", "pod", "pod-123"),
        CommandResult(1, "", "Attempted to remove pod that does not exist."),
    )
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory()
    driver = _owned_driver(transport=transport)

    result = driver.teardown(bundle, _owned_state(bundle))

    assert result["teardown"] == "already-absent"
    assert result["pod_absence"]["verified"] is True
    assert result["final_pod_inventory"]["verified"] is True
    assert ("stop", "pod", "pod-123") not in transport.runpodctl_calls


def test_teardown_polls_until_exact_owned_pod_is_absent(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    query = ("pod", "get", "pod-123", "--output", "json")
    transport.queue_runpodctl(query, CommandResult(0, '{"id":"pod-123"}'))
    transport.queue_runpodctl(query, CommandResult(1, "", "pod does not exist"))
    transport.queue_empty_global_inventory()
    clock = FakeClock()
    driver = _owned_driver(
        config=RunPodDriverConfig(
            poll_seconds=2,
            teardown_absence_timeout_seconds=10,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    result = driver.teardown(bundle, _owned_state(bundle))

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
    driver = _owned_driver(
        config=RunPodDriverConfig(
            poll_seconds=2,
            teardown_absence_timeout_seconds=4,
        ),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(RunPodDriverError, match="remained present for 4s"):
        driver.teardown(bundle, _owned_state(bundle))


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
    driver = _owned_driver(transport=transport)

    with pytest.raises(RunPodDriverError, match="ambiguous absence query"):
        driver.teardown(bundle, _owned_state(bundle))


def test_teardown_keep_alive_skips_owned_pod_query(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, keep_alive=True)
    transport = FakeRunPodTransport()
    driver = _owned_driver(transport=transport)

    result = driver.teardown(bundle, _owned_state(bundle))

    assert result["teardown"] == "skipped"
    assert result["ownership"]["kind"] == "orchestration_created"
    assert transport.runpodctl_calls == []


def test_teardown_never_removes_supplied_pod_id(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = RunPodOrchestrationDriver(
        config=RunPodDriverConfig(pod_id="pod-provided"),
        transport=transport,
    )
    state = _state(bundle).model_copy(
        update={
            "provision_record": {
                "pod_id": "pod-provided",
                "provided_pod": True,
                "provided_endpoint": False,
                "teardown_allowed": False,
            }
        }
    )

    result = driver.teardown(bundle, state)

    assert result["teardown"] == "skipped"
    assert result["skip_reason"] == "realized-capability-preserves-resources"
    assert result["capability_variant"] == "externally-managed"
    assert transport.runpodctl_calls == []


def test_auto_teardown_disabled_preserves_owned_pod_by_realized_capability(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    driver = _owned_driver(
        config=RunPodDriverConfig(auto_teardown=False),
        transport=transport,
    )

    result = driver.teardown(bundle, _owned_state(bundle))

    assert result["teardown"] == "skipped"
    assert result["skip_reason"] == "realized-capability-preserves-resources"
    assert result["capability_variant"] == "engine-acquired-preserved"
    assert transport.runpodctl_calls == []


def test_teardown_failure_persists_unresolved_owned_pod_and_primary_error(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "busy"))
    transport.queue_runpodctl(
        ("stop", "pod", "pod-123"), CommandResult(1, "", "provider unavailable")
    )
    driver = _owned_driver(transport=transport)
    engine = StageEngine(bundle=bundle, driver=driver)
    primary = RuntimeError("training failed first")

    try:
        raise primary
    except RuntimeError as raised:
        state = engine._run_teardown(_owned_state(bundle), abort=True)
        assert raised is primary

    teardown = state.stage("TEARDOWN")
    assert teardown.status == "failed"
    assert teardown.outputs["pod_absence"]["verified"] is False
    assert teardown.outputs["unresolved_owned_pod"] == {
        "pod_id": "pod-123",
        "last_known_state": "RUNNING",
        "reason": "runpodctl stop pod failed: provider unavailable",
    }


def test_teardown_cleanup_uses_one_hard_deadline(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    clock = FakeClock()

    class DeadlineConsumingTransport(FakeRunPodTransport):
        def runpodctl(
            self,
            *args: str,
            timeout_seconds: float | None = None,
        ) -> CommandResult:
            result = super().runpodctl(*args, timeout_seconds=timeout_seconds)
            if args[:2] == ("remove", "pod") and timeout_seconds is not None:
                clock.now += timeout_seconds
            return result

    transport = DeadlineConsumingTransport()
    transport.queue_runpodctl(("remove", "pod", "pod-123"), CommandResult(1, "", "busy"))
    driver = _owned_driver(
        config=RunPodDriverConfig(teardown_absence_timeout_seconds=4),
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    engine = StageEngine(bundle=bundle, driver=driver)

    state = engine._run_teardown(_owned_state(bundle), abort=True)

    assert clock.now == 4
    assert transport.runpodctl_calls == [("remove", "pod", "pod-123")]
    assert state.stage("TEARDOWN").status == "failed"
    assert "cleanup deadline expired" in state.stage("TEARDOWN").error


def test_run_command_timeout_kills_entire_child_process_group(tmp_path: Path) -> None:
    grandchild_pid = tmp_path / "grandchild.pid"
    script = (
        "import pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"pathlib.Path({str(grandchild_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )

    result = _run_command([sys.executable, "-c", script], timeout_seconds=0.3)

    assert result.returncode == 124
    pid = int(grandchild_pid.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"child process group member {pid} survived TERM/KILL escalation")


def test_interrupt_while_run_command_waits_does_not_wedge_parent(tmp_path: Path) -> None:
    # The child has to boot an interpreter and import feedbax before it can
    # signal readiness, and this suite runs with one worker per core, so the
    # setup wait is bounded generously: a tight deadline here measures how busy
    # the host is, not whether an interrupted parent wedges. The contract under
    # test is the wait *after* the signal — a wedged parent never returns at all,
    # so a generous bound still names the defect.
    startup_deadline_seconds = 120.0
    shutdown_deadline_seconds = 60.0
    child_pid = tmp_path / "child.pid"
    script = (
        "import pathlib, sys\n"
        "from feedbax.orchestration.drivers.runpod import _run_command\n"
        f"pathlib.Path({str(child_pid)!r}).write_text('starting')\n"
        "_run_command([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(Path.cwd()), os.environ.get("PYTHONPATH")))
            ),
        },
    )
    try:
        deadline = time.monotonic() + startup_deadline_seconds
        while not child_pid.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                pytest.fail(f"child exited before signalling readiness: {parent.returncode}")
            time.sleep(0.02)
        assert child_pid.exists()

        parent.send_signal(signal.SIGINT)

        assert parent.wait(timeout=shutdown_deadline_seconds) != 0
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=shutdown_deadline_seconds)


def test_abort_teardown_pulls_failure_logs_before_pod_removal(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transport = FakeRunPodTransport()
    transport.rsync_result = CommandResult(1, "", "diagnostic pull failed")
    transport.queue_runpodctl(
        ("pod", "get", "pod-123", "--output", "json"),
        CommandResult(1, "", "pod not found"),
    )
    transport.queue_empty_global_inventory()
    driver = _owned_driver(
        config=RunPodDriverConfig(failure_log_pull_timeout_seconds=17),
        transport=transport,
    )
    engine = StageEngine(bundle=bundle, driver=driver)

    state = engine._run_teardown(_owned_state(bundle), abort=True)

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
