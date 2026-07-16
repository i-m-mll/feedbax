"""RunPod orchestration driver.

The driver keeps RunPod and SSH side effects behind an injectable transport so
unit tests can pin command shapes without contacting RunPod.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.native_execution import (
    inject_native_execution_context,
    is_native_training_command,
)
from feedbax.orchestration.state import PreflightCheckEntry, RunSetState


RUNPOD_CODE_EXCLUDES = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "_artifacts",
    "web/node_modules",
)
LATEST_BATCH_KEYS = (
    "completed_training_batches",
    "completed_batches",
    "completed_batch",
    "completedBatch",
)
METADATA_BATCH_KEYS = (
    "completed_training_batches",
    "completed_batches",
    "completed_batch",
    "completedBatch",
)


class RunPodDriverError(RuntimeError):
    """Raised when the RunPod driver cannot complete a requested action."""


@dataclass(frozen=True)
class CommandResult:
    """Captured command result from a RunPod transport."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    def check(self, context: str) -> "CommandResult":
        """Raise a driver error if the command failed."""
        if self.returncode != 0:
            detail = self.stderr.strip() or self.stdout.strip() or f"exit={self.returncode}"
            raise RunPodDriverError(f"{context} failed: {detail}")
        return self


class RunPodTransport(Protocol):
    """Transport surface used by :class:`RunPodOrchestrationDriver`."""

    def runpodctl(self, *args: str) -> CommandResult:
        """Run a ``runpodctl`` command."""

    def image_exists(self, image: str) -> bool:
        """Return whether a declared container image tag exists."""

    def ssh(self, command: str) -> CommandResult:
        """Run one command over SSH on the pod."""

    def rsync(
        self,
        source: str,
        target: str,
        *,
        delete: bool = False,
        excludes: Sequence[str] = (),
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Synchronize files between local and remote paths."""


@dataclass(frozen=True)
class SubprocessRunPodTransport:
    """Subprocess-backed RunPod transport."""

    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_key_path: Path | str = Path("~/.runpod/ssh/RunPod-Key-Go")
    ssh_user: str = "root"
    runpodctl_executable: str = "runpodctl"

    def runpodctl(self, *args: str) -> CommandResult:
        return _run_command([self.runpodctl_executable, *args])

    def image_exists(self, image: str) -> bool:
        repository, separator, tag = image.rpartition(":")
        if not separator or not repository or not tag or "/" not in repository:
            return False
        namespace, name = repository.split("/", 1)
        url = f"https://hub.docker.com/v2/repositories/{namespace}/{name}/tags/{tag}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError):
            return False

    def ssh(self, command: str) -> CommandResult:
        host = self._require_host()
        return _run_command([*self._ssh_base(detach_stdin=True), host, command])

    def rsync(
        self,
        source: str,
        target: str,
        *,
        delete: bool = False,
        excludes: Sequence[str] = (),
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        host = self._require_host()
        rsync_target = target
        source_is_local = Path(source.rstrip("/")).exists()
        if source_is_local and _looks_remote_path(target):
            rsync_target = f"{host}:{target}"
        rsync_source = source
        if not source_is_local and _looks_remote_path(source):
            rsync_source = f"{host}:{source}"
        args = ["rsync", "-az", "--no-owner", "--no-group", "--progress", "--stats"]
        if delete:
            args.append("--delete")
        for exclude in excludes:
            args.extend(["--exclude", exclude])
        args.extend(
            [
                "-e",
                " ".join(shlex.quote(part) for part in self._ssh_base()),
                rsync_source,
                rsync_target,
            ]
        )
        return _run_command(args, timeout_seconds=timeout_seconds)

    def _require_host(self) -> str:
        if not self.ssh_host or self.ssh_port is None:
            raise RunPodDriverError("ssh host and port are required")
        return f"{self.ssh_user}@{self.ssh_host}"

    def _ssh_base(self, *, detach_stdin: bool = False) -> list[str]:
        key = str(Path(self.ssh_key_path).expanduser())
        if self.ssh_port is None:
            raise RunPodDriverError("ssh port is required")
        args = ["ssh"]
        if detach_stdin:
            args.append("-n")
        args.extend(
            [
                "-i",
                key,
                "-p",
                str(self.ssh_port),
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
            ]
        )
        return args


@dataclass(frozen=True)
class PodStateClassification:
    """Pure classification for ``runpodctl pod get`` JSON."""

    status: Literal["ready", "not_ready", "dead", "unknown"]
    ip: str | None = None
    port: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class EndpointClassification:
    """Shape of a RunPod SSH endpoint declaration."""

    kind: Literal["ssh_object", "ssh_command", "partial", "missing"]
    ip: str | None = None
    port: int | None = None
    ssh_command: str | None = None


@dataclass(frozen=True)
class BaselineEntry:
    """Declared baseline checkpoint that must be staged remotely."""

    path: str
    completed_batch: str
    label: str


@dataclass(frozen=True)
class RunPodDriverConfig:
    """Configuration for the RunPod orchestration driver."""

    pod_id: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    gpu_id: str | None = None
    datacenters: tuple[str, ...] = ()
    min_balance_usd: float = 5.0
    image: str = "runpod/pytorch:latest"
    pod_name_prefix: str = "feedbax-orchestration"
    max_acquire_seconds: float = 900.0
    poll_seconds: float = 5.0
    env_step_timeout_seconds: float = 1800.0
    failure_log_pull_timeout_seconds: float = 60.0
    volume_mount: str = "/workspace"
    remote_repo_root: str = "/workspace"
    remote_run_root: str = "/workspace/feedbax_runs"
    remote_artifacts_dir: str = "/workspace/_artifacts"
    local_repos: Mapping[str, Path | str] = field(default_factory=dict)
    remote_repos: Mapping[str, str] = field(default_factory=dict)
    path_patches: tuple[tuple[str, str, str], ...] = ()
    overlay_steps: tuple[str, ...] = ("uv pip install -U 'jax[cuda12]'",)
    auto_teardown: bool = True


class RunPodOrchestrationDriver:
    """Synchronous RunPod implementation of the orchestration driver protocol."""

    poll_interval_seconds = 5.0

    def __init__(
        self,
        *,
        config: RunPodDriverConfig | None = None,
        transport: RunPodTransport | None = None,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.config = config or RunPodDriverConfig()
        self.transport = transport or SubprocessRunPodTransport(
            ssh_host=self.config.ssh_host,
            ssh_port=self.config.ssh_port,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._preflight_passed = False
        self._pod_id = self.config.pod_id
        self._provided_endpoint = bool(self.config.ssh_host and self.config.ssh_port)
        self._endpoint: EndpointClassification | None = (
            EndpointClassification("ssh_object", self.config.ssh_host, self.config.ssh_port)
            if self._provided_endpoint
            else None
        )

    def provision(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Acquire or reuse a pod and verify SSH/GPU readiness."""
        if self._provided_endpoint:
            self._require_gpu_ready()
            return {
                "driver": "runpod",
                "provided_endpoint": True,
                "ssh_host": self.config.ssh_host,
                "ssh_port": self.config.ssh_port,
                "teardown_allowed": False,
            }

        if self._pod_id:
            pod = self._pod_get(self._pod_id)
            endpoint = self._wait_for_endpoint(self._pod_id, pod)
            self._endpoint = endpoint
            self._configure_subprocess_endpoint(endpoint)
            self._require_gpu_ready()
            return self._provision_record(pod, provided_pod=True)

        if not self._preflight_passed:
            raise RunPodDriverError(
                "RunPod creation requires passing named driver PREFLIGHT checks first"
            )
        pod_id = self._create_pod(bundle)
        self._pod_id = pod_id
        try:
            pod = self._pod_get(pod_id)
            self._endpoint = self._wait_for_endpoint(pod_id, pod)
            self._configure_subprocess_endpoint(self._endpoint)
            self._require_gpu_ready()
            return self._provision_record(pod, provided_pod=False)
        except Exception as exc:
            try:
                self.teardown(bundle, state)
            except Exception as teardown_exc:
                raise RunPodDriverError(
                    f"{exc}; automatic teardown failed: {teardown_exc}"
                ) from exc
            raise

    def preflight_checks(self, bundle: RunBundle) -> list[PreflightCheckEntry]:
        """Run named, non-mutating RunPod checks before any billable action."""
        checks: list[PreflightCheckEntry] = []
        image = bundle.environment.image_id or self.config.image
        image_exists = self.transport.image_exists(image)
        checks.append(_preflight_check("runpod-image-tag-exists", image_exists, observed=image))

        gpu_policy = bool(self._provided_endpoint or self._pod_id or self.config.gpu_id)
        checks.append(
            _preflight_check(
                "runpod-gpu-policy-declared",
                gpu_policy,
                observed={
                    "gpu_id": self.config.gpu_id,
                    "datacenter_fallbacks": list(self.config.datacenters),
                    "provided_target": bool(self._provided_endpoint or self._pod_id),
                },
            )
        )

        credentials_required = not self._provided_endpoint or bundle.deadman_enabled
        user_result = (
            self.transport.runpodctl("user", "--output", "json")
            if credentials_required
            else CommandResult(0, '{"provided_endpoint": true}')
        )
        credentials_ok = user_result.returncode == 0
        checks.append(
            _preflight_check(
                "runpod-credentials",
                credentials_ok,
                detail=None if credentials_ok else (user_result.stderr or user_result.stdout),
                observed="verified" if credentials_required else "not-required-provided-endpoint",
            )
        )

        balance_required = not self._provided_endpoint and not self._pod_id
        try:
            user_payload = _json_object(user_result.stdout) if credentials_ok else {}
        except RunPodDriverError:
            user_payload = {}
            credentials_ok = False
            checks[-1] = _preflight_check(
                "runpod-credentials",
                False,
                detail="runpodctl user returned invalid JSON",
                observed="invalid-response",
            )
        balance = user_balance(user_payload)
        balance_ok = not balance_required or (
            balance is not None and balance >= self.config.min_balance_usd
        )
        checks.append(
            _preflight_check(
                "runpod-balance-floor",
                balance_ok,
                detail=(
                    None
                    if balance_ok
                    else f"RunPod balance must be at least {self.config.min_balance_usd:g}"
                ),
                observed=balance if balance_required else "not-required-existing-target",
            )
        )
        self._preflight_passed = all(check.status == "pass" for check in checks)
        return checks

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        """Synchronize code and realize the remote Python environment."""
        fingerprint = compute_runpod_environment_fingerprint(bundle)
        remote_run_dir = self._remote_run_dir(bundle)
        self._ssh(
            f"mkdir -p {_sq(remote_run_dir)} {_sq(self._remote_sentinel_dir(bundle))} {_sq(remote_run_dir + '/logs')}"
        )
        self._ensure_deadman(bundle)
        if self._should_reuse_remote_environment(bundle, fingerprint):
            return fingerprint

        for name, local_root in self._local_repos().items():
            remote_root = self._remote_repos()[name]
            self._ssh(f"mkdir -p {_sq(remote_root)}")
            self._rsync_repo(str(Path(local_root)), remote_root)

        for remote_file, patch_from, patch_to in self.config.path_patches:
            self._ssh(build_literal_path_patch_command(remote_file, patch_from, patch_to))

        sentinel_dir = self._remote_sentinel_dir(bundle)
        logs_dir = f"{remote_run_dir}/logs"
        workdir = self._primary_workdir()
        self._remote_nohup_sentinel(
            label="uv sync",
            workdir=workdir,
            command="uv sync --frozen",
            done_file=f"{sentinel_dir}/uv-sync.done",
            failed_file=f"{sentinel_dir}/uv-sync.failed",
            log_file=f"{logs_dir}/uv-sync.log",
        )
        self._wait_for_remote_sentinel(
            label="uv sync",
            done_file=f"{sentinel_dir}/uv-sync.done",
            failed_file=f"{sentinel_dir}/uv-sync.failed",
            log_file=f"{logs_dir}/uv-sync.log",
        )
        for index, step in enumerate(
            (*bundle.environment.overlay_steps, *self.config.overlay_steps)
        ):
            self._remote_nohup_sentinel(
                label=f"overlay {index}",
                workdir=workdir,
                command=step,
                done_file=f"{sentinel_dir}/overlay-{index}.done",
                failed_file=f"{sentinel_dir}/overlay-{index}.failed",
                log_file=f"{logs_dir}/overlay-{index}.log",
            )
            self._wait_for_remote_sentinel(
                label=f"overlay {index}",
                done_file=f"{sentinel_dir}/overlay-{index}.done",
                failed_file=f"{sentinel_dir}/overlay-{index}.failed",
                log_file=f"{logs_dir}/overlay-{index}.log",
            )
        self._ssh(f"printf %s {_sq(fingerprint)} > {_sq(self._remote_fingerprint_path(bundle))}")
        return fingerprint

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Stage declared baselines and custody pins on the pod."""
        del state
        baselines = declared_baselines(bundle)
        staged: list[dict[str, str]] = []
        for baseline in baselines:
            source = baseline_source_path(baseline.path, Path.cwd())
            target = baseline_remote_target(
                baseline.path, self.config.remote_artifacts_dir, Path.cwd()
            )
            verify_local_baseline(source, baseline)
            self._ssh(f"mkdir -p {_sq(str(Path(target).parent))}")
            self.transport.rsync(
                f"{source}/" if source.is_dir() else str(source),
                f"{target}/" if source.is_dir() else target,
                delete=source.is_dir(),
            ).check(f"stage baseline {baseline.label}")
            self._ssh(remote_baseline_verification_command(target, baseline.completed_batch))
            staged.append({"label": baseline.label, "source": str(source), "target": target})
        payloads: list[dict[str, str]] = []
        for row in bundle.rows:
            if row.launch.payload_routing.get("kind") != "registered-execution-payload":
                continue
            source = Path(row.execution.payload.uri or "")
            if not source.is_file():
                raise RunPodDriverError(
                    f"registered execution payload is not materialized for row {row.row_id!r}"
                )
            if _sha256_file(source) != row.execution.payload.sha256:
                raise RunPodDriverError(
                    f"registered execution payload digest mismatch for row {row.row_id!r}"
                )
            target = self._remote_payload_path(bundle, row)
            self._ssh(f"mkdir -p {_sq(str(Path(target).parent))}")
            self.transport.rsync(str(source), target).check(f"stage payload {row.row_id}")
            check_line = row.execution.payload.sha256 + "  " + target
            self._ssh(f"printf %s {_sq(check_line)} | sha256sum -c -")
            payloads.append({"row_id": row.row_id, "source": str(source), "target": target})
        if bundle.input_custody_pins:
            payload = json.dumps(
                [
                    pin.model_dump(mode="json", exclude_none=True)
                    for pin in bundle.input_custody_pins
                ],
                sort_keys=True,
            )
            self._ssh(
                f"mkdir -p {_sq(self._remote_run_dir(bundle) + '/inputs')} && "
                f"cat > {_sq(self._remote_run_dir(bundle) + '/inputs/custody_pins.json')} <<'JSON'\n"
                f"{payload}\nJSON"
            )
        return {
            "baseline_count": len(staged),
            "baselines": staged,
            "payload_count": len(payloads),
            "payloads": payloads,
        }

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Launch one row under nohup with sentinel files."""
        command = build_launch_row_command(
            bundle=bundle,
            row=row,
            remote_run_dir=self._remote_run_dir(bundle),
            remote_sentinel_dir=self._remote_sentinel_dir(bundle),
            workdir=self._row_workdir(row),
            env_fingerprint=state.environment_fingerprint or "",
            jax_cache_dir=f"{self.config.volume_mount}/jax_cache",
        )
        self._ssh(command)
        pid = self._read_remote_pid(bundle, row.row_id)
        return {"row_id": row.row_id, "pid": pid, "command": command}

    def probe(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> DriverRowProbe:
        """Return one SSH probe report for a row."""
        report = parse_probe_report(
            self._ssh(build_probe_command(self._remote_sentinel_dir(bundle), row.row_id)).stdout
        )
        row_report = report.get("rows", {}).get(row.row_id, {})
        status = str(row_report.get("status", "pending"))
        pid = row_report.get("pid")
        return DriverRowProbe(
            status=status,
            pid=pid if isinstance(pid, int) else None,
            detail=row_report.get("detail"),
            metadata=report,
        )

    def probe_rows(
        self,
        bundle: RunBundle,
        rows: Sequence[RunRowSpec],
        state: RunSetState,
    ) -> Mapping[str, DriverRowProbe]:
        """Probe every unfinished row in one SSH round trip."""
        row_ids = [row.row_id for row in rows]
        report = parse_probe_report(
            self._ssh(build_probe_command(self._remote_sentinel_dir(bundle), row_ids)).stdout
        )
        result: dict[str, DriverRowProbe] = {}
        for row_id in row_ids:
            row_report = report.get("rows", {}).get(row_id, {})
            pid = row_report.get("pid")
            result[row_id] = DriverRowProbe(
                status=str(row_report.get("status", "pending")),
                pid=pid if isinstance(pid, int) else None,
                detail=row_report.get("detail"),
                metadata=report,
            )
        return result

    def stop_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Stop one row by PID and mark its failed sentinel."""
        sentinel_dir = self._remote_sentinel_dir(bundle)
        command = (
            f"pid_file={_sq(sentinel_dir + '/' + row.row_id + '.pid')}; "
            f"failed={_sq(sentinel_dir + '/' + row.row_id + '.failed')}; "
            'if [ -f "$pid_file" ]; then pid=$(cat "$pid_file"); '
            'kill "$pid" 2>/dev/null || true; fi; '
            'touch "$failed"'
        )
        self._ssh(command)
        return {"row_id": row.row_id, "status": "stopped"}

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, str]:
        """Collect row outputs, events, and manifests to the local run-set directory."""
        dest_dir = bundle.run_set_dir / "collected" / row.row_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote_run_dir = self._remote_run_dir(bundle)
        sources = row.launch.collect or [
            f"rows/{row.row_id}",
            f"events/{row.row_id}.events.jsonl",
            "run-config.json",
            "bundle.json",
        ]
        collected: dict[str, str] = {}
        for source in sources:
            if source.startswith("/"):
                remote_source = source
            elif "/" not in source and is_native_training_command(row.launch.command):
                remote_source = f"{remote_run_dir}/rows/{row.row_id}/{source}"
            else:
                remote_source = f"{remote_run_dir}/{source}"
            target = dest_dir / Path(source).name
            self.transport.rsync(remote_source, str(target), delete=False).check(
                f"collect {row.row_id}:{source}"
            )
            collected[Path(source).name] = str(target)
        if is_native_training_command(row.launch.command):
            event_name = f"{row.row_id}.events.jsonl"
            event_target = bundle.run_set_dir / "events" / event_name
            event_target.parent.mkdir(parents=True, exist_ok=True)
            self.transport.rsync(
                f"{remote_run_dir}/events/{event_name}", str(event_target), delete=False
            ).check(f"collect {row.row_id}:events")
            collected[event_name] = str(event_target)
        payload_sha256 = row.launch.metadata.get("payload_sha256")
        if payload_sha256:
            verify_collected_payload(dest_dir, str(payload_sha256))
        return collected

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Remove the acquired pod unless teardown is disabled by policy."""
        if bundle.keep_alive or not self.config.auto_teardown or self._provided_endpoint:
            return {"driver": "runpod", "teardown": "skipped"}
        if not self._pod_id:
            return {"driver": "runpod", "teardown": "no-pod"}
        result = self.transport.runpodctl("remove", "pod", self._pod_id)
        if result.returncode != 0:
            self.transport.runpodctl("stop", "pod", self._pod_id).check("runpodctl stop pod")
            return {"driver": "runpod", "teardown": "stopped", "pod_id": self._pod_id}
        return {"driver": "runpod", "teardown": "removed", "pod_id": self._pod_id}

    def _pod_get(self, pod_id: str) -> Mapping[str, Any]:
        result = self.transport.runpodctl("pod", "get", pod_id, "--output", "json").check(
            "runpodctl pod get"
        )
        return _json_object(result.stdout)

    def _create_pod(self, bundle: RunBundle) -> str:
        name = f"{self.config.pod_name_prefix}-{bundle.run_set_id}"
        datacenters = self.config.datacenters or ("",)
        last_error = ""
        for dc in datacenters:
            args = [
                "pod",
                "create",
                "--name",
                name,
                "--image",
                self.config.image,
                "--ports",
                "22/tcp,8080/http",
            ]
            if self.config.gpu_id:
                args.extend(["--gpu-id", self.config.gpu_id])
            if dc:
                args.extend(["--data-center-ids", dc])
            result = self.transport.runpodctl(*args)
            if result.returncode == 0:
                payload = _json_object(result.stdout)
                pod_id = str(payload.get("id") or payload.get("podId") or "")
                if pod_id:
                    return pod_id
            last_error = result.stderr or result.stdout
        raise RunPodDriverError(f"pod create failed: {last_error.strip()}")

    def _wait_for_endpoint(
        self,
        pod_id: str,
        first_pod: Mapping[str, Any] | None = None,
    ) -> EndpointClassification:
        deadline = self._monotonic() + self.config.max_acquire_seconds
        pod = first_pod
        while self._monotonic() <= deadline:
            pod = pod or self._pod_get(pod_id)
            classification = classify_pod_state(pod)
            if classification.status == "ready":
                return EndpointClassification(
                    "ssh_object",
                    classification.ip,
                    classification.port,
                    endpoint_classification(pod).ssh_command,
                )
            if classification.status == "dead":
                raise RunPodDriverError(f"pod entered dead state: {classification.reason}")
            self._sleep(self.config.poll_seconds)
            pod = None
        raise RunPodDriverError("timed out waiting for RunPod SSH endpoint")

    def _require_gpu_ready(self) -> None:
        self._ssh("nvidia-smi >/dev/null").check("nvidia-smi readiness")

    def _configure_subprocess_endpoint(self, endpoint: EndpointClassification) -> None:
        if isinstance(self.transport, SubprocessRunPodTransport):
            self.transport = replace(
                self.transport,
                ssh_host=endpoint.ip,
                ssh_port=endpoint.port,
            )

    def _ensure_deadman(self, bundle: RunBundle) -> None:
        if not bundle.deadman_enabled or bundle.keep_alive:
            return
        if not self._pod_id:
            raise RunPodDriverError("dead-man switch requires a RunPod pod id")
        self._ssh(
            "command -v runpodctl >/dev/null && runpodctl user --output json >/dev/null"
        ).check("in-pod runpodctl presence and authentication")
        self._ssh(
            build_deadman_watchdog_command(
                pod_id=self._pod_id,
                remote_run_dir=self._remote_run_dir(bundle),
                remote_sentinel_dir=self._remote_sentinel_dir(bundle),
                events_dir=self._remote_events_dir(bundle),
                silence_seconds=bundle.deadman_silence_seconds,
            )
        )

    def _provision_record(self, pod: Mapping[str, Any], *, provided_pod: bool) -> dict[str, Any]:
        endpoint = self._endpoint or endpoint_classification(pod)
        return {
            "driver": "runpod",
            "pod_id": self._pod_id,
            "provided_pod": provided_pod,
            "provided_endpoint": False,
            "ssh_host": endpoint.ip,
            "ssh_port": endpoint.port,
            "billing_started_at": pod.get("createdAt") or pod.get("created_at"),
            "teardown_allowed": not provided_pod,
        }

    def _ssh(self, command: str) -> CommandResult:
        return self.transport.ssh(command).check("ssh")

    def _rsync_repo(self, source: str, target: str) -> None:
        self.transport.rsync(
            source.rstrip("/") + "/",
            target.rstrip("/") + "/",
            delete=True,
            excludes=RUNPOD_CODE_EXCLUDES,
        ).check(f"rsync repo {source}")

    def _should_reuse_remote_environment(self, bundle: RunBundle, fingerprint: str) -> bool:
        result = self.transport.ssh(
            f"test -f {_sq(self._remote_fingerprint_path(bundle))} && "
            f"cat {_sq(self._remote_fingerprint_path(bundle))}"
        )
        return result.returncode == 0 and result.stdout.strip() == fingerprint

    def _read_remote_pid(self, bundle: RunBundle, row_id: str) -> int | None:
        result = self.transport.ssh(
            f"cat {_sq(self._remote_sentinel_dir(bundle) + '/' + row_id + '.pid')}"
        )
        try:
            return int(result.stdout.strip())
        except Exception:
            return None

    def _local_repos(self) -> Mapping[str, Path | str]:
        if self.config.local_repos:
            return self.config.local_repos
        return {"feedbax": Path.cwd()}

    def _remote_repos(self) -> Mapping[str, str]:
        if self.config.remote_repos:
            return self.config.remote_repos
        return {"feedbax": f"{self.config.remote_repo_root}/feedbax"}

    def _primary_workdir(self) -> str:
        remote_repos = self._remote_repos()
        return (
            remote_repos.get("rlrmp") or remote_repos.get("feedbax") or self.config.remote_repo_root
        )

    def _row_workdir(self, row: RunRowSpec) -> str:
        workdir = row.launch.metadata.get("workdir")
        return str(workdir) if workdir else self._primary_workdir()

    def _remote_run_dir(self, bundle: RunBundle) -> str:
        return f"{self.config.remote_run_root.rstrip('/')}/{bundle.run_set_id}"

    def _remote_sentinel_dir(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/sentinels"

    def _remote_payload_path(self, bundle: RunBundle, row: RunRowSpec) -> str:
        return f"{self._remote_run_dir(bundle)}/inputs/{row.row_id}.json"

    def _remote_events_dir(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/events"

    def _remote_fingerprint_path(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/env-fingerprint.txt"

    def _remote_nohup_sentinel(
        self,
        *,
        label: str,
        workdir: str,
        command: str,
        done_file: str,
        failed_file: str,
        log_file: str,
    ) -> None:
        self._ssh(
            build_remote_nohup_sentinel_command(
                workdir=workdir,
                command=command,
                done_file=done_file,
                failed_file=failed_file,
                log_file=log_file,
            )
        ).check(label)

    def _wait_for_remote_sentinel(
        self,
        *,
        label: str,
        done_file: str,
        failed_file: str,
        log_file: str,
    ) -> None:
        deadline = self._monotonic() + self.config.env_step_timeout_seconds
        probe = (
            f"if [ -f {_sq(failed_file)} ]; then printf failed; "
            f"elif [ -f {_sq(done_file)} ]; then printf done; "
            "else printf pending; fi"
        )
        while True:
            result = self.transport.ssh(probe)
            status = result.stdout.strip()
            if result.returncode == 0 and status == "done":
                return
            if result.returncode == 0 and status == "failed":
                raise RunPodDriverError(
                    f"{label} failed; remote log tail:\n{self._remote_log_tail(log_file)}"
                )
            if self._monotonic() >= deadline:
                raise RunPodDriverError(
                    f"{label} timed out after {self.config.env_step_timeout_seconds:g}s; "
                    f"remote log tail:\n{self._remote_log_tail(log_file)}"
                )
            self._sleep(self.config.poll_seconds)

    def _remote_log_tail(self, log_file: str) -> str:
        result = self.transport.ssh(f"tail -n 50 -- {_sq(log_file)}")
        detail = result.stdout.strip() or result.stderr.strip()
        return detail or "<remote log unavailable>"

    def collect_failure_logs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, str]:
        """Pull remote logs before failure teardown destroys the pod."""
        del state
        destination = bundle.run_set_dir / "failure-logs"
        destination.mkdir(parents=True, exist_ok=True)
        self.transport.rsync(
            f"{self._remote_run_dir(bundle)}/logs/",
            str(destination) + "/",
            delete=False,
            timeout_seconds=self.config.failure_log_pull_timeout_seconds,
        ).check("collect failure logs")
        return {"failure_logs": str(destination)}


def classify_pod_state(pod: Mapping[str, Any] | None) -> PodStateClassification:
    """Classify a RunPod pod object as ready, not-ready, dead, or unknown."""
    if not isinstance(pod, Mapping):
        return PodStateClassification("unknown")
    desired = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
    ssh = pod.get("ssh") if isinstance(pod.get("ssh"), Mapping) else {}
    ssh_status = str(ssh.get("status") or "").upper() if isinstance(ssh, Mapping) else ""
    if desired in {"EXITED", "TERMINATED", "FAILED"}:
        reason = str(pod.get("lastStatusChange") or ssh.get("error") or "exited")
        return PodStateClassification("dead", reason=_safe_reason(reason))
    if ssh_status in {"EXITED", "TERMINATED", "FAILED"}:
        return PodStateClassification("dead", reason=f"ssh_{ssh_status.lower()}")
    endpoint = endpoint_classification(pod)
    if endpoint.ip and endpoint.port:
        return PodStateClassification("ready", endpoint.ip, endpoint.port)
    return PodStateClassification("not_ready")


def endpoint_classification(pod: Mapping[str, Any]) -> EndpointClassification:
    """Classify RunPod's secure-cloud SSH endpoint shape."""
    ssh = pod.get("ssh") if isinstance(pod.get("ssh"), Mapping) else {}
    ip = _string_or_none(ssh.get("ip") or ssh.get("host")) if isinstance(ssh, Mapping) else None
    port = _int_or_none(ssh.get("port")) if isinstance(ssh, Mapping) else None
    command = (
        _string_or_none(ssh.get("ssh_command") or ssh.get("sshCommand") or ssh.get("command"))
        if isinstance(ssh, Mapping)
        else None
    )
    parsed_ip, parsed_port = parse_ssh_command_endpoint(command or "")
    ip = ip or parsed_ip
    port = port or parsed_port
    if ip and port and command:
        return EndpointClassification("ssh_command", ip, port, command)
    if ip and port:
        return EndpointClassification("ssh_object", ip, port, command)
    if ip or port or command:
        return EndpointClassification("partial", ip, port, command)
    return EndpointClassification("missing")


def parse_ssh_command_endpoint(command: str) -> tuple[str | None, int | None]:
    """Extract host and port from a RunPod ``ssh_command`` string."""
    if not command:
        return None, None
    parts = shlex.split(command)
    host = None
    port = None
    for index, part in enumerate(parts):
        if part == "-p" and index + 1 < len(parts):
            port = _int_or_none(parts[index + 1])
        elif part.startswith("-p") and len(part) > 2:
            port = _int_or_none(part[2:])
        elif "@" in part and not part.startswith("-"):
            host = part.rsplit("@", 1)[1]
    return host, port


def rank_datacenters_for_gpu(
    datacenters: Sequence[Mapping[str, Any]],
    gpu_id: str,
) -> list[str]:
    """Rank datacenter IDs by RunPod stock status for one GPU type."""
    rank = {"High": 3, "Medium": 2, "Low": 1}
    candidates: list[tuple[int, str]] = []
    for dc in datacenters:
        dc_id = _string_or_none(dc.get("id"))
        if not dc_id:
            continue
        score: int | None = None
        for availability in dc.get("gpuAvailability") or ():
            if not isinstance(availability, Mapping) or availability.get("gpuId") != gpu_id:
                continue
            availability_score = rank.get(str(availability.get("stockStatus")), 0)
            score = availability_score if score is None else max(score, availability_score)
        if score is not None:
            candidates.append((-score, dc_id))
    return [dc_id for _, dc_id in sorted(candidates)]


def user_balance(user_payload: Mapping[str, Any]) -> float | None:
    """Return the RunPod client balance when present and numeric."""
    value = user_payload.get("clientBalance", user_payload.get("balance"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _preflight_check(
    name: str,
    passed: bool,
    *,
    detail: str | None = None,
    observed: Any = None,
) -> PreflightCheckEntry:
    return PreflightCheckEntry(
        name=name,
        status="pass" if passed else "fail",
        detail=detail,
        observed=observed,
    )


def compute_runpod_environment_fingerprint(bundle: RunBundle) -> str:
    """Compute the declared remote environment fingerprint."""
    payload = {
        "python_version": bundle.environment.python_version,
        "repo_revisions": [
            revision.model_dump(mode="json", exclude_none=True)
            for revision in bundle.environment.repo_revisions
        ],
        "lockfile_hashes": dict(sorted(bundle.environment.lockfile_hashes.items())),
        "overlay_steps": list(bundle.environment.overlay_steps),
        "image_id": bundle.environment.image_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_literal_path_patch_command(
    remote_file: str,
    patch_from: str,
    patch_to: str,
) -> str:
    """Build one literal, explicitly file-scoped path-patch command."""
    expression = "s/\\Q$ENV{PATCH_FROM}\\E/$ENV{PATCH_TO}/g"
    return (
        f"PATCH_FROM={_sq(patch_from)} PATCH_TO={_sq(patch_to)} "
        f"perl -0pi -e {_sq(expression)} {_sq(remote_file)}"
    )


def declared_baselines(bundle: RunBundle) -> list[BaselineEntry]:
    """Extract baseline declarations from inline row specs."""
    entries: list[BaselineEntry] = []
    for row in bundle.rows:
        payload = _registered_row_payload(row)
        if isinstance(payload, Mapping):
            entries.extend(_baseline_entries(payload, f"row:{row.row_id}"))
    return entries


def _baseline_entries(payload: Mapping[str, Any], label: str) -> list[BaselineEntry]:
    entries: list[BaselineEntry] = []
    path = _first_string(
        payload,
        "baseline_checkpoint_path",
        "baseline_checkpoint",
        "checkpoint_path",
    )
    batch = _first_scalar(
        payload,
        "baseline_completed_batches",
        "baseline_completed_batch",
        "completed_batches",
        "completed_batch",
    )
    if path:
        entries.append(BaselineEntry(path=path, completed_batch=str(batch or ""), label=label))
    resume = payload.get("resume")
    if isinstance(resume, Mapping):
        path = _first_string(
            resume,
            "baseline_checkpoint_path",
            "baseline_checkpoint",
            "checkpoint_path",
            "checkpoint",
        )
        batch = _first_scalar(
            resume,
            "baseline_completed_batches",
            "baseline_completed_batch",
            "completed_batches",
            "completed_batch",
        )
        if path:
            entries.append(BaselineEntry(path=path, completed_batch=str(batch or ""), label=label))
    training_config = payload.get("training_config")
    if isinstance(training_config, Mapping):
        entries.extend(_baseline_entries(training_config, f"{label}:training_config"))
    return entries


def baseline_source_path(source: str, repo_root: Path) -> Path:
    """Resolve a baseline source path using the deploy-script rules."""
    path = Path(source)
    return path if path.is_absolute() else repo_root / path


def baseline_remote_target(source: str, remote_artifacts_dir: str, repo_root: Path) -> str:
    """Return the remote target path for a baseline source."""
    source_path = baseline_source_path(source, repo_root)
    if source.startswith("_artifacts/"):
        return f"{remote_artifacts_dir.rstrip('/')}/{source.removeprefix('_artifacts/')}"
    artifacts_prefix = str(repo_root / "_artifacts") + "/"
    source_text = str(source_path)
    if source_text.startswith(artifacts_prefix):
        return f"{remote_artifacts_dir.rstrip('/')}/{source_text.removeprefix(artifacts_prefix)}"
    return f"{remote_artifacts_dir.rstrip('/')}/baselines/{source_path.name}"


def verify_local_baseline(source: Path, baseline: BaselineEntry) -> None:
    """Verify a local baseline checkpoint before staging."""
    if not source.exists():
        raise RunPodDriverError(
            f"baseline preflight failed: source checkpoint not found for {baseline.label}: {source}"
        )
    latest = source / "latest.json"
    if not latest.is_file():
        raise RunPodDriverError(
            f"baseline preflight failed: custody latest.json not found for {baseline.label}: {latest}"
        )
    if not baseline.completed_batch or baseline.completed_batch == "null":
        raise RunPodDriverError(
            f"baseline preflight failed: declared completed_batch missing for {baseline.label}"
        )
    actual = latest_pointer_completed_batches(json.loads(latest.read_text(encoding="utf-8")))
    if actual != baseline.completed_batch:
        raise RunPodDriverError(
            "baseline preflight failed: completed_batch mismatch for "
            f"{baseline.label}: declared {baseline.completed_batch} but latest.json has {actual}"
        )


def latest_pointer_completed_batches(payload: Mapping[str, Any]) -> str | None:
    """Extract completed-batch progress from a custody ``latest.json`` payload."""
    for key in LATEST_BATCH_KEYS:
        value = payload.get(key)
        if value is not None:
            return str(value)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in METADATA_BATCH_KEYS:
            value = metadata.get(key)
            if value is not None:
                return str(value)
    return None


def remote_baseline_verification_command(target: str, expected_batch: str) -> str:
    """Build a remote command that verifies a staged baseline."""
    latest = f"{target.rstrip('/')}/latest.json"
    jq_filter = (
        ".completed_training_batches // .completed_batches // .completed_batch // "
        ".completedBatch // .metadata.completed_training_batches // "
        ".metadata.completed_batches // .metadata.completed_batch // .metadata.completedBatch // "
        "empty"
    )
    return (
        f"test -e {_sq(target)} && test -f {_sq(latest)} && "
        f"actual=$(jq -r {_sq(jq_filter)} {_sq(latest)}) && "
        f'test -n "$actual" && test "$actual" = {_sq(expected_batch)}'
    )


def build_remote_nohup_sentinel_command(
    *,
    workdir: str,
    command: str,
    done_file: str,
    failed_file: str,
    log_file: str,
) -> str:
    """Build a remote nohup command with done/failed sentinel semantics."""
    sentinel_command = (
        f"cd {_sq(workdir)} && success=0; child=; "
        'mark_failed() { rc=$?; if [ -n "$child" ]; then kill "$child" 2>/dev/null || true; fi; '
        f'if [ "$success" -ne 1 ]; then touch {_sq(failed_file)}; fi; exit "$rc"; }}; '
        'signal_failed() { rc=$1; if [ -n "$child" ]; then kill "$child" 2>/dev/null || true; fi; '
        f'touch {_sq(failed_file)}; exit "$rc"; }}; '
        "trap mark_failed EXIT; trap 'signal_failed 130' INT; "
        "trap 'signal_failed 143' TERM; trap 'signal_failed 129' HUP; "
        f'{{ {command}; }} & child=$!; wait "$child"; rc=$?; child=; '
        f'if [ "$rc" -eq 0 ]; then success=1; touch {_sq(done_file)}; '
        f'else touch {_sq(failed_file)}; exit "$rc"; fi'
    )
    return (
        f"mkdir -p {_sq(str(Path(done_file).parent))} {_sq(str(Path(log_file).parent))} && "
        f"rm -f {_sq(done_file)} {_sq(failed_file)} && "
        f"nohup bash -lc {_sq(sentinel_command)} </dev/null >{_sq(log_file)} 2>&1 &"
    )


def build_launch_row_command(
    *,
    bundle: RunBundle,
    row: RunRowSpec,
    remote_run_dir: str,
    remote_sentinel_dir: str,
    workdir: str,
    env_fingerprint: str,
    jax_cache_dir: str,
) -> str:
    """Build the row launch command with RunPod sentinel and event exports."""
    done_file = f"{remote_sentinel_dir}/{row.row_id}.done"
    failed_file = f"{remote_sentinel_dir}/{row.row_id}.failed"
    started_file = f"{remote_sentinel_dir}/{row.row_id}.started"
    pid_file = f"{remote_sentinel_dir}/{row.row_id}.pid"
    log_file = f"{remote_run_dir}/logs/{row.row_id}.log"
    events_dir = f"{remote_run_dir}/events"
    row_dir = f"{remote_run_dir}/rows/{row.row_id}"
    command_parts = (
        [str(part) for part in row.launch.command]
        if row.launch.command
        else ["uv", "run", "--no-sync", "python", row.launch.entry or ""]
    )
    if (
        row.launch.payload_routing.get("kind") == "registered-execution-payload"
        and is_native_training_command(command_parts)
    ):
        command_index = command_parts.index("execute-training-run-spec")
        if command_index + 1 == len(command_parts) or command_parts[
            command_index + 1
        ].startswith("-"):
            command_parts.insert(command_index + 1, f"{remote_run_dir}/inputs/{row.row_id}.json")
    command_parts = inject_native_execution_context(
        command_parts,
        row=row,
        environment_fingerprint=env_fingerprint,
        collection_root=row_dir,
    )
    command = " ".join(shlex.quote(part) for part in command_parts)
    inner = (
        f"cd {_sq(workdir)} && success=0; child=; "
        'mark_failed() { rc=$?; if [ -n "$child" ]; then kill "$child" 2>/dev/null || true; fi; '
        f'if [ "$success" -ne 1 ]; then touch {_sq(failed_file)}; fi; exit "$rc"; }}; '
        'signal_failed() { rc=$1; if [ -n "$child" ]; then kill "$child" 2>/dev/null || true; fi; '
        f'touch {_sq(failed_file)}; exit "$rc"; }}; '
        "trap mark_failed EXIT; trap 'signal_failed 130' INT; trap 'signal_failed 143' TERM; "
        "trap 'signal_failed 129' HUP; "
        f"echo $$ > {_sq(pid_file)} && "
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"JAX_COMPILATION_CACHE_DIR={_sq(jax_cache_dir)} "
        f"FEEDBAX_RUN_SET_ID={_sq(bundle.run_set_id)} "
        f"FEEDBAX_ROW_ID={_sq(row.row_id)} "
        f"FEEDBAX_RUN_EVENTS_DIR={_sq(events_dir)} "
        f"FEEDBAX_ENV_FINGERPRINT={_sq(env_fingerprint)} "
        f"FEEDBAX_ROW_DIR={_sq(row_dir)} && "
        f'( {command} ) & child=$!; wait "$child"; rc=$?; child=; '
        f'if [ "$rc" -eq 0 ]; then success=1; touch {_sq(done_file)}; '
        f'else touch {_sq(failed_file)}; exit "$rc"; fi'
    )
    return (
        f"mkdir -p {_sq(remote_sentinel_dir)} {_sq(remote_run_dir + '/logs')} "
        f"{_sq(events_dir)} {_sq(row_dir)} {_sq(jax_cache_dir)} && "
        f"if [ -f {_sq(done_file)} ] || [ -f {_sq(failed_file)} ]; then exit 0; fi && "
        f"if [ -f {_sq(started_file)} ]; then "
        f"pid=$(cat {_sq(pid_file)} 2>/dev/null || true); "
        'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi; '
        f"echo 'orphaned launch: started sentinel present, process dead, "
        f"no terminal sentinel' > {_sq(failed_file)}; exit 0; fi && "
        f"touch {_sq(started_file)} && "
        f"nohup bash -lc {_sq(inner)} </dev/null >{_sq(log_file)} 2>&1 &"
    )


def build_probe_command(remote_sentinel_dir: str, row_ids: str | Sequence[str]) -> str:
    """Build a compact remote probe command for one or more rows."""
    rows = [row_ids] if isinstance(row_ids, str) else list(row_ids)
    return (
        "python - <<'PY'\n"
        "import json, os, subprocess\n"
        f"sdir={remote_sentinel_dir!r}\n"
        f"rows={rows!r}\n"
        "gpu=subprocess.run(['bash','-lc','nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true'],capture_output=True,text=True).stdout.strip()\n"
        "reports={}\n"
        "for row in rows:\n"
        "    base=os.path.join(sdir,row)\n"
        "    pid=None\n"
        "    pid_path=base+'.pid'\n"
        "    if os.path.exists(pid_path):\n"
        "        try: pid=int(open(pid_path).read().strip())\n"
        "        except Exception: pid=None\n"
        "    status='pending'\n"
        "    detail=None\n"
        "    if os.path.exists(base+'.done'): status='completed'\n"
        "    elif os.path.exists(base+'.failed'): status='failed'\n"
        "    elif os.path.exists(base+'.started'):\n"
        "        if pid:\n"
        "            alive=subprocess.run(['bash','-lc',f'kill -0 {pid} 2>/dev/null']).returncode==0\n"
        "            status='running' if alive else 'failed'\n"
        "            if not alive: detail='pid exited without sentinel'\n"
        "        else: status='running'\n"
        "    reports[row]={'status':status,'pid':pid,'detail':detail}\n"
        "print(json.dumps({'gpu':gpu,'rows':reports}, sort_keys=True))\n"
        "PY"
    )


def parse_probe_report(output: str) -> dict[str, Any]:
    """Parse the JSON report returned by :func:`build_probe_command`."""
    return _json_object(output or "{}")


def build_deadman_watchdog_command(
    *,
    pod_id: str,
    remote_run_dir: str,
    remote_sentinel_dir: str,
    events_dir: str,
    silence_seconds: int,
) -> str:
    """Build the optional in-pod dead-man watchdog command."""
    warning = f"{remote_run_dir}/deadman-warning.txt"
    pid_file = f"{remote_run_dir}/deadman.pid"
    script = (
        f"pod_id={_sq(pod_id)}; run_dir={_sq(remote_run_dir)}; "
        f"sdir={_sq(remote_sentinel_dir)}; edir={_sq(events_dir)}; "
        f"silence={int(silence_seconds)}; warning={_sq(warning)}; "
        "while true; do "
        "live=0; newest=0; now=$(date +%s); "
        'for started in "$sdir"/*.started; do [ -e "$started" ] || continue; '
        'base=${started%.started}; [ -f "$base.done" ] || [ -f "$base.failed" ] || live=1; done; '
        'for path in "$edir"/*.jsonl "$sdir"/*; do [ -e "$path" ] || continue; '
        'mtime=$(stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || echo 0); '
        '[ "$mtime" -gt "$newest" ] && newest=$mtime; done; '
        "age=$((now-newest)); "
        'if [ "$live" -eq 0 ] && [ "$newest" -gt 0 ] && [ "$age" -ge "$silence" ]; then '
        'printf \'deadman removing pod after %ss silence\\n\' "$age" > "$warning"; '
        'runpodctl remove pod "$pod_id"; exit $?; fi; '
        "sleep 30; done"
    )
    return (
        f"pid_file={_sq(pid_file)}; "
        'if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then exit 0; fi; '
        f"nohup bash -lc {_sq(script)} </dev/null "
        f'>>{_sq(remote_run_dir + "/logs/deadman.log")} 2>&1 & echo $! > "$pid_file"'
    )


def verify_collected_payload(dest_dir: Path, expected_sha256: str) -> None:
    """Verify that at least one collected file matches the expected SHA-256."""
    for path in dest_dir.rglob("*"):
        if path.is_file() and _sha256_file(path) == expected_sha256:
            return
    raise RunPodDriverError(f"payload sha256 mismatch under {dest_dir}")


def _registered_row_payload(row: RunRowSpec) -> dict[str, Any] | None:
    ref = row.execution.payload
    if ref.uri is None:
        return None
    path = Path(ref.uri)
    if _sha256_file(path) != ref.sha256:
        raise RunPodDriverError(f"registered payload digest mismatch for row {row.row_id!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RunPodDriverError("registered row payload must be a JSON object")
    if payload.get("schema_id") != ref.schema_id or payload.get("schema_version") != ref.schema_version:
        raise RunPodDriverError("registered row payload schema does not match its reference")
    return dict(payload)


def _run_command(args: Sequence[str], *, timeout_seconds: float | None = None) -> CommandResult:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(124, exc.stdout or "", f"timed out after {timeout_seconds:g}s")
    except OSError as exc:
        return CommandResult(127, "", str(exc))
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _json_object(payload: str) -> dict[str, Any]:
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RunPodDriverError(f"invalid JSON payload: {exc}") from exc
    if isinstance(loaded, Mapping):
        return dict(loaded)
    raise RunPodDriverError("expected JSON object")


def _safe_reason(reason: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.:/=@,+-" else "_" for ch in reason).rstrip("_")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_string(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _first_scalar(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _looks_remote_path(value: str) -> bool:
    return value.startswith("/")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sq(value: str) -> str:
    return shlex.quote(str(value))
