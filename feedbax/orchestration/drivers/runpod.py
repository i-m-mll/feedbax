"""RunPod orchestration driver.

The driver keeps RunPod and SSH side effects behind an injectable transport so
unit tests can pin command shapes without contacting RunPod.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from feedbax.contracts.run_matrix import (
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID_V1,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2,
    TrainingRunMatrixPreflightBinding,
    training_run_matrix_preflight_binding_sha256,
)
from feedbax.orchestration.bundle import (
    ResolvedAssemblyInput,
    RunBundle,
    RunRowSpec,
    canonical_run_bundle_sha256,
    environment_declaration_identity_projection,
)
from feedbax.orchestration.collection_recovery import (
    CollectionRecoveryBinding,
    recover_collected_outputs,
)
from feedbax.orchestration.drivers.base import DriverRowProbe, ProvisioningAttemptError
from feedbax.orchestration.drivers.native_execution import (
    bind_native_execution_command,
    inject_native_execution_context,
    is_native_training_command,
    native_resume_checkpoint_authority_json,
    native_resume_checkpoint_source,
    SECURE_CHECKPOINT_SEED_SCRIPT,
)
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    InputProviderRootBinding,
    materialize_bundle_inputs,
    preflight_input_provider_bindings,
)
from feedbax.orchestration.matrix_authority import (
    MatrixAuthorityError,
    build_training_run_matrix_authority,
    is_training_matrix_bundle,
)
from feedbax.orchestration.repo_snapshot import (
    RepoSnapshotError,
    RepoSnapshotManifest,
    SealedRepoSnapshots,
    restore_repo_snapshots,
    seal_repo_snapshots,
    snapshot_manifest_digest,
    verify_repo_snapshot,
)
from feedbax.orchestration.state import PreflightCheckEntry, RunSetState, utc_now


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
RUNPOD_ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION = (
    "feedbax.runpod_environment_fingerprint.v1"
)
_RUNPOD_PREFLIGHT_CHECK_NAMES = frozenset(
    {
        "input-provider-bindings",
        "runpod-repo-snapshots",
        "runpod-image-immutable",
        "runpod-image-tag-exists",
        "runpod-lockfiles-declared",
        "runpod-python-version-declared",
        "runpod-gpu-policy-declared",
        "runpod-credentials",
        "runpod-balance-floor",
        "runpod-deadman-credentials",
    }
)
_IMMUTABLE_IMAGE_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_POD_NOT_FOUND_MARKERS = ("not found", "does not exist", "404")
_SAFE_POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_RUNPOD_GO_UTC_PATTERN = re.compile(
    r"^(?P<instant>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \+0000 UTC$"
)
_REMOTE_ENVIRONMENT_PROBE = r"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

declaration = json.loads(sys.argv[1])
declared_python = declaration["python_version"]
observed_python = platform.python_version()
python_matches = observed_python == declared_python or (
    declared_python is not None
    and declared_python.count(".") == 1
    and observed_python.startswith(declared_python + ".")
)
if not python_matches:
    raise RuntimeError(
        f"Python version mismatch: expected {declared_python}, observed {observed_python}"
    )

observed_lockfiles = {}
for relative_path, expected in declaration["lockfile_hashes"].items():
    path = Path(relative_path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(
            f"lockfile digest mismatch for {relative_path}: expected {expected}, observed {observed}"
        )
    observed_lockfiles[relative_path] = observed

import equinox
import jax
import jaxlib

devices = jax.devices()
if not devices:
    raise RuntimeError("JAX reported no runtime devices")
plugins = []
for entry_point in importlib.metadata.entry_points(group="feedbax.plugins"):
    entry_point.load()
    distribution = entry_point.dist
    plugins.append(
        {
            "distribution": distribution.name if distribution is not None else None,
            "distribution_version": (
                distribution.version if distribution is not None else None
            ),
            "name": entry_point.name,
            "value": entry_point.value,
        }
    )
primary_device = devices[0]
client = getattr(primary_device, "client", None)
if getattr(primary_device, "platform", None) not in {"cuda", "gpu"}:
    raise RuntimeError(
        f"JAX CUDA backend is unavailable: observed platform {primary_device.platform!r}"
    )
payload = {
    "schema_version": "feedbax.runpod_environment_fingerprint.v1",
    "declaration_sha256": declaration["declaration_sha256"],
    "image_id": declaration["image_id"],
    "lockfile_hashes": observed_lockfiles,
    "runtime": {
        "device_count": len(devices),
        "device_kind": getattr(primary_device, "device_kind", None),
        "jax": jax.__version__,
        "jax_platform": getattr(primary_device, "platform", None),
        "jax_platform_version": getattr(client, "platform_version", None),
        "jaxlib": jaxlib.__version__,
        "python": observed_python,
        "python_implementation": platform.python_implementation(),
        "equinox": equinox.__version__,
        "feedbax": importlib.metadata.version("feedbax"),
    },
    "feedbax_plugins": sorted(plugins, key=lambda item: (item["name"], item["value"])),
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
_REMOTE_ATOMIC_DIRECTORY_PUBLISH = r"""
import ctypes
import errno
import os
import sys

source, destination = sys.argv[1:]
libc = ctypes.CDLL(None, use_errno=True)
try:
    renameat2 = libc.renameat2
except AttributeError as exc:
    raise RuntimeError("atomic no-replace directory publication is unavailable") from exc
renameat2.argtypes = (
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint,
)
renameat2.restype = ctypes.c_int
if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"input publication target already exists: {destination}")
    raise OSError(error, os.strerror(error), destination)
"""


class RunPodDriverError(RuntimeError):
    """Raised when the RunPod driver cannot complete a requested action."""


class _ProvisioningIdentityError(RunPodDriverError):
    """Raised when provider observations conflict during one acquisition."""


def _canonical_runpod_timestamp(value: Any) -> str | None:
    """Return a canonical UTC instant for a RunPod client timestamp."""
    if not isinstance(value, str) or not value:
        return None
    match = _RUNPOD_GO_UTC_PATTERN.fullmatch(value)
    candidate = f"{match.group('instant')}+00:00" if match is not None else value
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


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

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
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
    rsync_executable: str = "rsync"

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        return _run_command(
            [self.runpodctl_executable, *args],
            timeout_seconds=timeout_seconds,
        )

    def image_exists(self, image: str) -> bool:
        tagged_image = image.split("@", 1)[0]
        repository, separator, tag = tagged_image.rpartition(":")
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
        rsync_executable, secluded_args = self._resolve_rsync_capability()
        rsync_target = target
        source_is_local = Path(source.rstrip("/")).exists()
        if source_is_local and _looks_remote_path(target):
            remote_target = target if secluded_args else shlex.quote(target)
            rsync_target = f"{host}:{remote_target}"
        rsync_source = source
        if not source_is_local and _looks_remote_path(source):
            remote_source = source if secluded_args else shlex.quote(source)
            rsync_source = f"{host}:{remote_source}"
        args = [
            rsync_executable,
            "-az",
            "--no-owner",
            "--no-group",
        ]
        if secluded_args:
            args.append("--secluded-args")
        args.extend(["--progress", "--stats"])
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

    def _resolve_rsync_capability(self) -> tuple[str, bool]:
        executable = shutil.which(self.rsync_executable)
        if executable is None:
            raise RunPodDriverError(f"rsync executable is unavailable: {self.rsync_executable!r}")
        version = _run_command([executable, "--version"])
        if version.returncode != 0:
            detail = (
                version.stderr.strip() or version.stdout.strip() or f"exit={version.returncode}"
            )
            raise RunPodDriverError(f"rsync executable is unusable: {detail}")
        secluded_probe = _run_command([executable, "--secluded-args", "--version"])
        if secluded_probe.returncode == 0:
            return executable, True
        detail = secluded_probe.stderr.strip() or secluded_probe.stdout.strip()
        unsupported_markers = (
            "unrecognized option `--secluded-args'",
            'unknown option "--secluded-args"',
            "unknown option --secluded-args",
        )
        if any(marker in detail for marker in unsupported_markers):
            return executable, False
        raise RunPodDriverError(
            "could not determine rsync secluded-argument support: "
            + (detail or f"exit={secluded_probe.returncode}")
        )

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
class _PodAcquisition:
    """Provider-observed identity returned by one accepted pod-create request."""

    pod_id: str
    accepted_datacenter: str | None


@dataclass(frozen=True)
class RunPodDriverConfig:
    """Configuration for the RunPod orchestration driver."""

    pod_id: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    gpu_id: str | None = None
    datacenters: tuple[str, ...] = ()
    api_key: str | None = field(default=None, repr=False)
    min_balance_usd: float = 5.0
    image: str = "runpod/pytorch:latest"
    pod_name_prefix: str = "feedbax-orchestration"
    max_acquire_seconds: float = 900.0
    poll_seconds: float = 5.0
    env_step_timeout_seconds: float = 1800.0
    failure_log_pull_timeout_seconds: float = 60.0
    teardown_absence_timeout_seconds: float = 60.0
    volume_mount: str = "/workspace"
    remote_repo_root: str = "/workspace"
    remote_run_root: str = "/workspace/feedbax_runs"
    remote_artifacts_dir: str = "/workspace/_artifacts"
    local_repos: Mapping[str, Path | str] = field(default_factory=dict)
    remote_repos: Mapping[str, str] = field(default_factory=dict)
    protected_refs: Mapping[str, str] = field(default_factory=dict)
    path_patches: tuple[tuple[str, str, str], ...] = ()
    overlay_steps: tuple[str, ...] = ("uv pip install \"jax[cuda12]==$(uv run --no-sync python -c 'import jax; print(jax.__version__)')\"",)
    auto_teardown: bool = True


class RunPodOrchestrationDriver:
    """Synchronous RunPod implementation of the orchestration driver protocol."""

    poll_interval_seconds = 5.0
    govern_provisioning_retries = True

    def __init__(
        self,
        *,
        config: RunPodDriverConfig | None = None,
        transport: RunPodTransport | None = None,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
        input_provider_bindings: Sequence[InputProviderRootBinding] = (),
        collection_recovery_bindings: Sequence[CollectionRecoveryBinding] = (),
    ) -> None:
        self.config = config or RunPodDriverConfig()
        self.transport = transport or SubprocessRunPodTransport(
            ssh_host=self.config.ssh_host,
            ssh_port=self.config.ssh_port,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self.input_provider_bindings = tuple(input_provider_bindings)
        self.collection_recovery_bindings = tuple(collection_recovery_bindings)
        self._collection_recovery_evidence: dict[str, Mapping[str, object]] = {}
        self._repo_snapshots: SealedRepoSnapshots | None = None
        self._preflight_passed = False
        self._pod_id = self.config.pod_id
        self._provided_endpoint = bool(self.config.ssh_host and self.config.ssh_port)
        self._endpoint: EndpointClassification | None = (
            EndpointClassification("ssh_object", self.config.ssh_host, self.config.ssh_port)
            if self._provided_endpoint
            else None
        )
        self._last_provision_pod: Mapping[str, Any] | None = None

    def provision(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Perform one acquisition attempt; the stage engine owns retries."""
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
            endpoint, pod = self._wait_for_endpoint(self._pod_id)
            self._endpoint = endpoint
            self._configure_subprocess_endpoint(endpoint)
            self._require_gpu_ready()
            return self._provision_record(pod, provided_pod=True)

        if not self._preflight_passed:
            raise ProvisioningAttemptError(
                "RunPod creation requires passing named driver PREFLIGHT checks first",
                retryable=False,
                attempt_record={"driver": "runpod", "acquired": False},
            )
        pod: Mapping[str, Any] | None = None
        self._last_provision_pod = None
        acquired = False
        acquisition: _PodAcquisition | None = None
        pod_id: str | None = None
        try:
            acquisition = self._create_pod(bundle)
            pod_id = acquisition.pod_id
            acquired = True
            self._pod_id = pod_id
            self._endpoint, pod = self._wait_for_endpoint(pod_id)
            self._configure_subprocess_endpoint(self._endpoint)
            self._require_gpu_ready()
            return self._provision_record(
                pod,
                provided_pod=False,
                accepted_datacenter=acquisition.accepted_datacenter,
            )
        except Exception as exc:
            if isinstance(exc, ProvisioningAttemptError):
                raise
            record: dict[str, Any] = {"driver": "runpod", "acquired": acquired, "pod_id": pod_id}
            pod = pod or self._last_provision_pod
            if pod is not None:
                record.update(project_runpod_provision_facts(pod))
            if acquired:
                try:
                    record["cleanup"] = dict(self.teardown(bundle, state))
                except Exception as teardown_exc:
                    record["cleanup_error"] = str(teardown_exc)
                    self._pod_id = None
                    self._endpoint = None
                    raise ProvisioningAttemptError(
                        f"{exc}; automatic teardown failed: {teardown_exc}",
                        retryable=False,
                        attempt_record=record,
                        stop_reason="teardown-failure",
                    ) from exc
            self._pod_id = None
            self._endpoint = None
            raise ProvisioningAttemptError(
                str(exc),
                retryable=not isinstance(
                    exc,
                    (ValueError, TypeError, _ProvisioningIdentityError),
                ),
                attempt_record=record,
            ) from exc

    @property
    def provision_retry_delay_seconds(self) -> float:
        """Configured positive delay between governed acquisition attempts."""
        return self.config.poll_seconds

    def preflight_checks(self, bundle: RunBundle) -> list[PreflightCheckEntry]:
        """Run named, non-mutating RunPod checks before any billable action."""
        if is_training_matrix_bundle(bundle):
            try:
                build_training_run_matrix_authority(
                    bundle,
                    local_repos=self.config.local_repos,
                    protected_refs=self.config.protected_refs,
                )
            except (OSError, ValueError, RunPodDriverError) as exc:
                self._preflight_passed = False
                return [
                    _preflight_check(
                        "training-matrix-authority",
                        False,
                        detail=str(exc),
                    )
                ]
        try:
            self.seal_repo_snapshots(bundle)
        except RepoSnapshotError as exc:
            self._preflight_passed = False
            return [
                _preflight_check(
                    "runpod-repo-snapshots",
                    False,
                    detail=str(exc),
                )
            ]
        snapshot_check = _preflight_check(
            "runpod-repo-snapshots",
            True,
            observed=self.repo_snapshot_manifest().model_dump(mode="json"),
        )
        failures, observed = preflight_input_provider_bindings(
            bundle, self.input_provider_bindings
        )
        binding_check = _preflight_check(
            "input-provider-bindings",
            not failures,
            detail="; ".join(failures) if failures else None,
            observed=observed or "no-resolved-inputs",
        )
        if failures:
            self._preflight_passed = False
            return [binding_check]
        checks: list[PreflightCheckEntry] = [snapshot_check, binding_check]
        image = bundle.environment.image_id
        image_is_immutable = is_immutable_runpod_image_id(image)
        checks.append(
            _preflight_check(
                "runpod-image-immutable",
                image_is_immutable,
                detail=(
                    None
                    if image_is_immutable
                    else "environment.image_id must be an OCI image pinned by @sha256:<64 hex>"
                ),
                observed=image,
            )
        )
        image_exists = bool(image_is_immutable and self.transport.image_exists(image or ""))
        checks.append(
            _preflight_check(
                "runpod-image-tag-exists",
                image_exists,
                observed=image,
            )
        )
        lockfile_error = runpod_lockfile_declaration_error(
            bundle.environment.lockfile_hashes
        )
        checks.append(
            _preflight_check(
                "runpod-lockfiles-declared",
                lockfile_error is None,
                detail=lockfile_error,
                observed=dict(sorted(bundle.environment.lockfile_hashes.items())),
            )
        )
        checks.append(
            _preflight_check(
                "runpod-python-version-declared",
                bool(bundle.environment.python_version),
                detail=(
                    None
                    if bundle.environment.python_version
                    else "environment.python_version is required for deterministic realization"
                ),
                observed=bundle.environment.python_version,
            )
        )

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
        checks.append(
            _preflight_check(
                "runpod-deadman-credentials",
                not bundle.deadman_enabled or bool(self.config.api_key),
                observed="available" if self.config.api_key else "not-required-or-missing",
            )
        )
        self._preflight_passed = all(check.status == "pass" for check in checks)
        return checks

    def preflight_evidence(
        self,
        bundle: RunBundle,
        state: RunSetState,
        checks: Sequence[PreflightCheckEntry],
    ) -> Mapping[str, Any]:
        """Create durable authority for a completed, named RunPod preflight."""
        legacy_payload = self._preflight_evidence_payload(bundle, state, checks)
        legacy = {**legacy_payload, "sha256": _sha256_json(legacy_payload)}
        if not is_training_matrix_bundle(bundle):
            return legacy
        return self._matrix_preflight_evidence(bundle, legacy)

    def seal_repo_snapshots(self, bundle: RunBundle) -> RepoSnapshotManifest:
        """Seal configured tracked working trees before any provisioning action."""
        self._repo_snapshots = seal_repo_snapshots(
            self._local_repos(),
            snapshot_parent=bundle.run_set_dir / ".repo-snapshots",
        )
        return self._repo_snapshots.manifest

    def repo_snapshot_manifest(self) -> RepoSnapshotManifest | None:
        """Return the current sealed transfer authority, if available."""
        return self._repo_snapshots.manifest if self._repo_snapshots is not None else None

    def restore_completed_preflight(self, bundle: RunBundle, state: RunSetState) -> None:
        """Restore only cryptographically bound completed preflight authority.

        This is deliberately offline: it validates persisted evidence without
        querying RunPod or repeating mutable credential/balance observations.
        """
        if state.repo_snapshot_manifest is None:
            raise RunPodDriverError("completed PREFLIGHT lacks sealed repo snapshot authority")
        try:
            self._repo_snapshots = restore_repo_snapshots(
                self._local_repos(),
                state.repo_snapshot_manifest,
                snapshot_parent=bundle.run_set_dir / ".repo-snapshots",
            )
        except RepoSnapshotError as exc:
            raise RunPodDriverError(str(exc)) from exc
        stage = state.stage("PREFLIGHT")
        evidence = stage.outputs.get("driver_evidence")
        persisted_checks = stage.outputs.get("checks")
        if persisted_checks != [check.model_dump(mode="json") for check in stage.checks]:
            raise RunPodDriverError("completed PREFLIGHT checks are internally inconsistent")
        if state.run_set_id != bundle.run_set_id:
            raise RunPodDriverError("completed PREFLIGHT run-set binding is mismatched")
        assemble_completed = state.stage("ASSEMBLE").completed_at
        if (
            assemble_completed is None
            or stage.started_at is None
            or stage.completed_at is None
            or not assemble_completed <= stage.started_at <= stage.completed_at
        ):
            raise RunPodDriverError("completed PREFLIGHT timestamps are internally inconsistent")
        expected_payload = self._preflight_evidence_payload(bundle, state, stage.checks)
        expected_v1 = {**expected_payload, "sha256": _sha256_json(expected_payload)}
        matrix_bundle = is_training_matrix_bundle(bundle)
        if evidence is None and not matrix_bundle:
            self._preflight_passed = True
            return
        if not isinstance(evidence, Mapping):
            raise RunPodDriverError("completed PREFLIGHT RunPod evidence has invalid shape")
        if matrix_bundle:
            expected_v2 = self._matrix_preflight_evidence(bundle, expected_v1)
            if evidence.get("sha256") != expected_v2["sha256"]:
                raise RunPodDriverError("completed matrix PREFLIGHT evidence digest mismatch")
            if dict(evidence) != expected_v2:
                raise RunPodDriverError("completed matrix PREFLIGHT evidence is not canonical")
        else:
            if evidence.get("sha256") != expected_v1["sha256"]:
                raise RunPodDriverError("completed PREFLIGHT evidence digest mismatch")
            if dict(evidence) != expected_v1:
                raise RunPodDriverError("completed PREFLIGHT evidence is not canonical")
        self._preflight_passed = True

    def _matrix_preflight_evidence(
        self, bundle: RunBundle, legacy: Mapping[str, Any]
    ) -> dict[str, Any]:
        binding = self._matrix_preflight_binding(
            bundle, nested_evidence_sha256=str(legacy["sha256"])
        )
        payload = {
            "schema_id": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID,
            "schema_version": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2,
            "v1": dict(legacy),
            "matrix_binding": binding.model_dump(mode="json", exclude_none=True),
            "matrix_binding_sha256": training_run_matrix_preflight_binding_sha256(binding),
        }
        return {**payload, "sha256": _sha256_json(payload)}

    def _matrix_preflight_binding(
        self,
        bundle: RunBundle,
        *,
        nested_evidence_sha256: str,
    ) -> TrainingRunMatrixPreflightBinding:
        """Bind completed provider evidence to recomputed neutral authority."""
        try:
            authority = build_training_run_matrix_authority(
                bundle,
                local_repos=self.config.local_repos,
                protected_refs=self.config.protected_refs,
            )
        except MatrixAuthorityError as exc:
            raise RunPodDriverError(str(exc)) from exc
        return TrainingRunMatrixPreflightBinding(
            matrix=authority.matrix,
            rows=authority.rows,
            resolved_inputs=authority.resolved_inputs,
            code_authorities=authority.code_authorities,
            monitor=authority.monitor,
            bundle_sha256=authority.bundle_sha256,
            nested_preflight_evidence_sha256=nested_evidence_sha256,
        )

    def _preflight_evidence_payload(
        self,
        bundle: RunBundle,
        state: RunSetState,
        checks: Sequence[PreflightCheckEntry],
    ) -> dict[str, Any]:
        assemble_sha256 = state.stage("ASSEMBLE").outputs.get("bundle_sha256")
        bundle_sha256 = canonical_run_bundle_sha256(bundle)
        if not isinstance(assemble_sha256, str) or assemble_sha256 != bundle_sha256:
            raise RunPodDriverError("completed PREFLIGHT bundle binding is missing or mismatched")
        self._validate_preflight_checks(bundle, checks)
        snapshot_manifest = state.repo_snapshot_manifest or self.repo_snapshot_manifest()
        if snapshot_manifest is None:
            raise RunPodDriverError("completed PREFLIGHT lacks sealed repo snapshot authority")
        return {
            "schema_id": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID_V1,
            "schema_version": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
            "run_set_id": bundle.run_set_id,
            "bundle_sha256": bundle_sha256,
            "checks_sha256": _sha256_json([check.model_dump(mode="json") for check in checks]),
            "driver_contract": self._preflight_driver_contract(),
            "repo_snapshot_manifest_sha256": snapshot_manifest_digest(snapshot_manifest),
        }

    def _preflight_driver_contract(self) -> dict[str, Any]:
        """Project the complete effective RunPod config without exposing secrets."""
        return {
            "pod_id": self.config.pod_id,
            "ssh_host": self.config.ssh_host,
            "ssh_port": self.config.ssh_port,
            "gpu_id": self.config.gpu_id,
            "datacenters": list(self.config.datacenters),
            "api_key_sha256": (
                hashlib.sha256(self.config.api_key.encode("utf-8")).hexdigest()
                if self.config.api_key
                else None
            ),
            "min_balance_usd": self.config.min_balance_usd,
            "image": self.config.image,
            "pod_name_prefix": self.config.pod_name_prefix,
            "max_acquire_seconds": self.config.max_acquire_seconds,
            "poll_seconds": self.config.poll_seconds,
            "env_step_timeout_seconds": self.config.env_step_timeout_seconds,
            "failure_log_pull_timeout_seconds": self.config.failure_log_pull_timeout_seconds,
            "teardown_absence_timeout_seconds": self.config.teardown_absence_timeout_seconds,
            "volume_mount": self.config.volume_mount,
            "remote_repo_root": self.config.remote_repo_root,
            "remote_run_root": self.config.remote_run_root,
            "remote_artifacts_dir": self.config.remote_artifacts_dir,
            "local_repos_sha256": _sha256_json(
                {str(name): str(path) for name, path in sorted(self.config.local_repos.items())}
            ),
            "remote_repos": {
                str(name): str(path) for name, path in sorted(self.config.remote_repos.items())
            },
            "path_patches": [list(patch) for patch in self.config.path_patches],
            "overlay_steps_sha256": _sha256_json(list(self.config.overlay_steps)),
            "auto_teardown": self.config.auto_teardown,
            "provided_endpoint": self._provided_endpoint,
            "input_provider_bindings": sorted(
                binding.name for binding in self.input_provider_bindings
            ),
        }

    def _validate_preflight_checks(
        self, bundle: RunBundle, checks: Sequence[PreflightCheckEntry]
    ) -> None:
        """Validate persisted RunPod checks without provider or input access."""
        named = {check.name: check for check in checks if check.name in _RUNPOD_PREFLIGHT_CHECK_NAMES}
        if len(named) != len(_RUNPOD_PREFLIGHT_CHECK_NAMES) or any(
            check.name in _RUNPOD_PREFLIGHT_CHECK_NAMES
            and sum(item.name == check.name for item in checks) != 1
            for check in named.values()
        ):
            raise RunPodDriverError("completed PREFLIGHT lacks a unique RunPod check set")
        if any(check.status != "pass" for check in checks):
            raise RunPodDriverError("completed PREFLIGHT includes a failing check")

        failures, resolved_observed = preflight_input_provider_bindings(
            bundle, self.input_provider_bindings
        )
        if failures:
            raise RunPodDriverError("current bundle no longer has valid resolved input declarations")
        current_bindings = {binding.name for binding in self.input_provider_bindings}
        required_bindings = {item.custody.provider_binding for item in bundle.resolved_inputs}
        if current_bindings != required_bindings:
            raise RunPodDriverError("current input-provider bindings do not cover the bundle")
        expected_observed = resolved_observed or "no-resolved-inputs"
        if named["input-provider-bindings"].observed != expected_observed:
            raise RunPodDriverError("input-provider preflight observation does not match the bundle")

        image = bundle.environment.image_id
        if self.config.image != image:
            raise RunPodDriverError("RunPod driver image does not match the bundle")
        expected_checks = {
            "runpod-repo-snapshots": (
                self.repo_snapshot_manifest().model_dump(mode="json")
                if self.repo_snapshot_manifest() is not None
                else None
            ),
            "runpod-image-immutable": image,
            "runpod-image-tag-exists": image,
            "runpod-lockfiles-declared": dict(sorted(bundle.environment.lockfile_hashes.items())),
            "runpod-python-version-declared": bundle.environment.python_version,
            "runpod-gpu-policy-declared": {
                "gpu_id": self.config.gpu_id,
                "datacenter_fallbacks": list(self.config.datacenters),
                "provided_target": bool(self._provided_endpoint or self._pod_id),
            },
        }
        for name, observed in expected_checks.items():
            if named[name].observed != observed:
                raise RunPodDriverError(f"{name} observation does not match current declarations")

        credentials_required = not self._provided_endpoint or bundle.deadman_enabled
        expected_credentials = "verified" if credentials_required else "not-required-provided-endpoint"
        if named["runpod-credentials"].observed != expected_credentials:
            raise RunPodDriverError("credential preflight observation is inconsistent")
        balance_required = not self._provided_endpoint and not self._pod_id
        balance = named["runpod-balance-floor"].observed
        if balance_required:
            if isinstance(balance, bool) or not isinstance(balance, (int, float)):
                raise RunPodDriverError("balance preflight observation is not numeric")
            if not math.isfinite(float(balance)) or balance < self.config.min_balance_usd:
                raise RunPodDriverError("balance preflight observation does not meet the current threshold")
        elif balance != "not-required-existing-target":
            raise RunPodDriverError("balance preflight observation is inconsistent")
        expected_deadman = "available" if self.config.api_key else "not-required-or-missing"
        if named["runpod-deadman-credentials"].observed != expected_deadman:
            raise RunPodDriverError("deadman credential observation is inconsistent")
        if bundle.deadman_enabled and not self.config.api_key:
            raise RunPodDriverError("deadman requires an available API key")

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        """Synchronize code and realize the remote Python environment."""
        require_deterministic_runpod_environment(bundle)
        snapshot_manifest = state.repo_snapshot_manifest or self.repo_snapshot_manifest()
        if snapshot_manifest is None:
            raise RunPodDriverError(
                "RunPod REALIZE_ENV requires repo snapshots sealed during PREFLIGHT"
            )
        try:
            self._repo_snapshots = restore_repo_snapshots(
                self._local_repos(),
                snapshot_manifest,
                snapshot_parent=bundle.run_set_dir / ".repo-snapshots",
            )
        except RepoSnapshotError as exc:
            raise RunPodDriverError(str(exc)) from exc
        declaration_fingerprint = compute_runpod_environment_fingerprint(
            bundle, self._repo_snapshots.manifest
        )
        remote_run_dir = self._remote_run_dir(bundle)
        self._ssh(
            f"mkdir -p {_sq(remote_run_dir)} {_sq(self._remote_sentinel_dir(bundle))} "
            f"{_sq(remote_run_dir + '/logs')} && command -v setsid >/dev/null"
        )
        self._ensure_deadman(bundle)
        reused_fingerprint = self._reused_remote_environment_fingerprint(
            bundle, declaration_fingerprint
        )
        if reused_fingerprint is not None:
            return reused_fingerprint

        for name, snapshot in self._repo_snapshots.snapshots.items():
            remote_root = self._remote_repos()[name]
            self._ssh(
                f"mkdir -p {_sq(remote_root)} && chmod -R u+w {_sq(remote_root)}"
            )
            try:
                verify_repo_snapshot(
                    snapshot.staging_root,
                    content_sha256=snapshot.record.content_sha256,
                    file_count=snapshot.record.file_count,
                )
            except RepoSnapshotError as exc:
                raise RunPodDriverError(str(exc)) from exc
            self._rsync_repo(str(snapshot.staging_root), remote_root)

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
        realized_fingerprint = self._probe_realized_environment(
            bundle, declaration_fingerprint
        )
        self._ssh(
            f"printf %s {_sq(declaration_fingerprint)} > "
            f"{_sq(self._remote_declaration_fingerprint_path(bundle))} && "
            f"printf %s {_sq(realized_fingerprint)} > "
            f"{_sq(self._remote_fingerprint_path(bundle))}"
        )
        return realized_fingerprint

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Publish one verified input tree without exposing partial final paths."""
        attempt = state.stage("STAGE_INPUTS").attempts
        attempt_root = (
            bundle.run_set_dir / ".stage-attempts" / f"stage-inputs-{attempt}"
        )
        try:
            staged_inputs = materialize_bundle_inputs(
                bundle,
                destination_root=attempt_root,
                provider_bindings=self.input_provider_bindings,
            )
        except InputMaterializationError as exc:
            raise RunPodDriverError(str(exc)) from exc
        attempt_inputs = attempt_root / "inputs"
        payloads: list[dict[str, str]] = []
        payload_hashes: list[tuple[str, str]] = []
        for row in bundle.rows:
            if row.launch.payload_routing.get("kind") != "registered-execution-payload":
                continue
            source = Path(row.execution.payload.uri or "")
            if not source.is_file():
                raise RunPodDriverError(
                    f"registered execution payload is not materialized for row {row.row_id!r}"
                )
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != row.execution.payload.sha256:
                raise RunPodDriverError(
                    f"registered execution payload digest mismatch for row {row.row_id!r}"
                )
            local_target = attempt_inputs / f"{row.row_id}.json"
            try:
                with local_target.open("xb") as handle:
                    handle.write(data)
            except FileExistsError as exc:
                raise RunPodDriverError(
                    f"input attempt path already exists for row {row.row_id!r}"
                ) from exc
            payload_hashes.append((local_target.name, row.execution.payload.sha256))
            payloads.append(
                {
                    "row_id": row.row_id,
                    "source": str(source),
                    "target": self._remote_payload_path(bundle, row),
                }
            )

        remote_run_dir = self._remote_run_dir(bundle)
        remote_attempt_root = (
            f"{remote_run_dir}/.stage-attempts/stage-inputs-{attempt}"
        )
        remote_attempt_inputs = f"{remote_attempt_root}/inputs"
        self._ssh(
            f"mkdir -p {_sq(remote_run_dir + '/.stage-attempts')} && "
            f"mkdir -- {_sq(remote_attempt_root)}"
        )
        self.transport.rsync(
            str(attempt_inputs) + "/",
            remote_attempt_inputs + "/",
            delete=True,
        ).check("stage authenticated input tree")

        transferred: list[dict[str, Any]] = []
        for staged in staged_inputs:
            target = f"{remote_run_dir}/inputs/{staged.target_role}"
            for item in staged.files:
                relative_path = PurePosixPath(item.relative_path).relative_to("inputs")
                remote_path = f"{remote_attempt_inputs}/{relative_path}"
                check_line = f"{item.sha256}  {remote_path}"
                self._ssh(f"printf %s {_sq(check_line)} | sha256sum -c -")
            transferred.append(
                {
                    "target_role": staged.target_role,
                    "source": str(staged.destination),
                    "target": target,
                    "file_count": len(staged.files),
                }
            )
        for name, digest in payload_hashes:
            check_line = f"{digest}  {remote_attempt_inputs}/{name}"
            self._ssh(f"printf %s {_sq(check_line)} | sha256sum -c -")
        self._ssh(
            build_atomic_directory_publish_command(
                remote_attempt_inputs,
                f"{remote_run_dir}/inputs",
            )
        )
        return {
            "input_count": len(transferred),
            "inputs": transferred,
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
        if self.collection_recovery_bindings and (
            self._pod_id is not None or self._provided_endpoint
        ):
            raise RunPodDriverError(
                "collection recovery refuses a configured live or provider-backed target"
            )
        recovered = recover_collected_outputs(
            bundle,
            row,
            state,
            bindings=self.collection_recovery_bindings,
        )
        if recovered is not None:
            self._collection_recovery_evidence[row.row_id] = recovered.evidence
            return recovered.outputs
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
            source_kind = self._remote_collection_source_kind(remote_source)
            delete = False
            if source_kind == "directory":
                if os.path.lexists(target):
                    if target.is_symlink() or not target.is_dir():
                        raise RunPodDriverError(
                            f"collection directory target is unsafe: {target}"
                        )
                else:
                    target.mkdir()
                remote_source = remote_source.rstrip("/") + "/"
                rsync_target = str(target) + "/"
                delete = True
            else:
                rsync_target = str(target)
            self.transport.rsync(remote_source, rsync_target, delete=delete).check(
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

    def collection_recovery_evidence(self, row_id: str) -> Mapping[str, object] | None:
        """Return evidence for an explicit provider-free collection recovery."""
        return self._collection_recovery_evidence.get(row_id)

    def _remote_collection_source_kind(self, source: str) -> Literal["file", "directory"]:
        """Classify one remote output without following its final path component."""
        source_path = source.rstrip("/") or "/"
        command = (
            f"path={_sq(source_path)}; "
            'if [ -L "$path" ]; then printf symlink; '
            'elif [ -f "$path" ]; then printf file; '
            'elif [ -d "$path" ]; then printf directory; '
            'elif [ -e "$path" ]; then printf unsupported; '
            "else printf missing; fi"
        )
        kind = self._ssh(command).stdout.strip()
        if kind in {"file", "directory"}:
            return kind
        if kind == "missing":
            raise RunPodDriverError(f"declared collection output is absent: {source}")
        if kind == "symlink":
            raise RunPodDriverError(f"declared collection output is a symlink: {source}")
        if kind == "unsupported":
            raise RunPodDriverError(
                f"declared collection output is not a regular file or directory: {source}"
            )
        raise RunPodDriverError(
            f"could not classify declared collection output {source!r}: {kind!r}"
        )

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        """Remove the acquired pod unless teardown is disabled by policy."""
        if bundle.keep_alive or not self.config.auto_teardown or self._provided_endpoint:
            return {"driver": "runpod", "teardown": "skipped"}
        if not self._pod_id:
            return {"driver": "runpod", "teardown": "no-pod"}
        pod_id = self._pod_id
        result = self.transport.runpodctl("remove", "pod", pod_id)
        if result.returncode != 0:
            self.transport.runpodctl("stop", "pod", pod_id).check("runpodctl stop pod")
            self.transport.runpodctl(
                "remove",
                "pod",
                pod_id,
                timeout_seconds=self.config.teardown_absence_timeout_seconds,
            ).check("runpodctl remove pod after stop")
            action = "stopped-then-removed"
        else:
            action = "removed"
        absence = self._wait_for_pod_absence(pod_id)
        self._pod_id = None
        final_inventory = self._observe_global_pod_inventory()
        return {
            "driver": "runpod",
            "teardown": action,
            "pod_id": pod_id,
            "pod_absence": absence,
            "final_pod_inventory": final_inventory,
        }

    def _observe_global_pod_inventory(self) -> Mapping[str, Any]:
        """Return sanitized evidence from the provider-wide RunPod pod inventory."""
        result = self.transport.runpodctl("pod", "list", "--output", "json")
        observed_at = utc_now().isoformat()
        basis = "runpodctl pod list --output json"
        if result.returncode != 0:
            return {
                "scope": "provider-account",
                "verified": False,
                "observed_at": observed_at,
                "observation_basis": basis,
                "outcome": "unavailable",
                "pod_count": None,
                "pod_ids": [],
            }
        try:
            pod_ids = _parse_runpod_pod_inventory(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                "scope": "provider-account",
                "verified": False,
                "observed_at": observed_at,
                "observation_basis": basis,
                "outcome": "invalid",
                "pod_count": None,
                "pod_ids": [],
            }
        if pod_ids:
            return {
                "scope": "provider-account",
                "verified": False,
                "observed_at": observed_at,
                "observation_basis": basis,
                "outcome": "non-empty",
                "pod_count": len(pod_ids),
                "pod_ids": list(pod_ids),
            }
        return {
            "scope": "provider-account",
            "verified": True,
            "observed_at": observed_at,
            "observation_basis": basis,
            "outcome": "empty",
            "pod_count": 0,
            "pod_ids": [],
        }

    def _wait_for_pod_absence(self, pod_id: str) -> Mapping[str, Any]:
        """Boundedly prove that one exact orchestration-owned pod is absent."""
        deadline = self._monotonic() + self.config.teardown_absence_timeout_seconds
        polls = 0
        while self._monotonic() < deadline:
            remaining = deadline - self._monotonic()
            result = self.transport.runpodctl(
                "pod",
                "get",
                pod_id,
                "--output",
                "json",
                timeout_seconds=remaining,
            )
            polls += 1
            if result.returncode != 0:
                detail = (result.stderr.strip() or result.stdout.strip()).lower()
                if any(marker in detail for marker in _POD_NOT_FOUND_MARKERS):
                    return {
                        "verified": True,
                        "pod_id": pod_id,
                        "polls": polls,
                        "terminal_observation": "not-found",
                    }
                raise RunPodDriverError(
                    f"ambiguous absence query for owned pod {pod_id!r}: "
                    f"{detail or f'exit={result.returncode}'}"
                )
            try:
                payload = _json_object(result.stdout)
            except (TypeError, ValueError) as exc:
                raise RunPodDriverError(
                    f"ambiguous absence query for owned pod {pod_id!r}: invalid JSON object"
                ) from exc
            observed_id = str(payload.get("id") or payload.get("podId") or "")
            if observed_id != pod_id:
                raise RunPodDriverError(
                    f"ambiguous absence query for owned pod {pod_id!r}: "
                    f"observed id {observed_id!r}"
                )
            remaining = deadline - self._monotonic()
            if remaining > 0:
                self._sleep(min(self.config.poll_seconds, remaining))
        raise RunPodDriverError(
            f"owned pod {pod_id!r} remained present for "
            f"{self.config.teardown_absence_timeout_seconds:g}s after teardown"
        )

    def _pod_get(
        self,
        pod_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        result = self.transport.runpodctl(
            "pod",
            "get",
            pod_id,
            "--output",
            "json",
            timeout_seconds=timeout_seconds,
        ).check("runpodctl pod get")
        return _json_object(result.stdout)

    def _create_pod(self, bundle: RunBundle) -> _PodAcquisition:
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
            if self.config.api_key:
                args.extend(["--env", json.dumps({"FEEDBAX_RUNPOD_API_KEY": self.config.api_key})])
            result = self.transport.runpodctl(*args)
            if result.returncode == 0:
                payload = _json_object(result.stdout)
                pod_id = str(payload.get("id") or payload.get("podId") or "")
                if pod_id:
                    return _PodAcquisition(
                        pod_id=pod_id,
                        accepted_datacenter=dc or None,
                    )
            classification, last_error = _classify_create_failure(result, self.config.api_key)
            if classification == "non-retryable":
                raise ProvisioningAttemptError(
                    last_error,
                    retryable=False,
                    attempt_record={"driver": "runpod", "acquired": False},
                )
        raise ProvisioningAttemptError(
            last_error,
            retryable=True,
            attempt_record={"driver": "runpod", "acquired": False},
        )

    def _wait_for_endpoint(
        self,
        pod_id: str,
    ) -> tuple[EndpointClassification, Mapping[str, Any]]:
        deadline = self._monotonic() + self.config.max_acquire_seconds
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                break
            pod = self._pod_get(pod_id, timeout_seconds=remaining)
            self._last_provision_pod = pod
            if self._monotonic() > deadline:
                break
            classification = classify_pod_state(pod)
            if classification.status == "ready":
                return (
                    EndpointClassification(
                        "ssh_object",
                        classification.ip,
                        classification.port,
                        endpoint_classification(pod).ssh_command,
                    ),
                    pod,
                )
            if classification.status == "dead":
                raise RunPodDriverError(f"pod entered dead state: {classification.reason}")
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                break
            self._sleep(min(self.config.poll_seconds, remaining))
        raise RunPodDriverError(
            "timed out waiting for RunPod SSH endpoint after "
            f"{self.config.max_acquire_seconds:g}s"
        )

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
        auth = "export RUNPOD_API_KEY=$(tr '\\0' '\\n' </proc/1/environ | sed -n 's/^FEEDBAX_RUNPOD_API_KEY=//p'); "
        self._ssh(
            auth + f"command -v runpodctl >/dev/null && runpodctl get pod {_sq(self._pod_id)} >/dev/null"
        ).check("in-pod runpodctl presence and authentication")
        self._ssh(
            auth + build_deadman_watchdog_command(
                pod_id=self._pod_id,
                remote_run_dir=self._remote_run_dir(bundle),
                remote_sentinel_dir=self._remote_sentinel_dir(bundle),
                events_dir=self._remote_events_dir(bundle),
                silence_seconds=bundle.deadman_silence_seconds,
            )
        )

    def _provision_record(
        self,
        pod: Mapping[str, Any],
        *,
        provided_pod: bool,
        accepted_datacenter: str | None = None,
    ) -> dict[str, Any]:
        endpoint = self._endpoint or endpoint_classification(pod)
        facts = project_runpod_provision_facts(pod)
        observed_datacenter = facts["region"]
        if (
            accepted_datacenter is not None
            and observed_datacenter is not None
            and observed_datacenter != accepted_datacenter
        ):
            raise _ProvisioningIdentityError(
                "RunPod pod response datacenter conflicts with the accepted create request: "
                f"accepted {accepted_datacenter!r}, observed {observed_datacenter!r}"
            )
        if accepted_datacenter is not None:
            facts["region"] = accepted_datacenter
            facts["provider_observation_basis"] = (
                "accepted singleton runpodctl pod create datacenter"
                + (
                    "; confirmed by runpodctl pod get response"
                    if observed_datacenter is not None
                    else "; runpodctl pod get response omitted datacenter"
                )
            )
        return {
            "driver": "runpod",
            **facts,
            "pod_id": self._pod_id,
            "provided_pod": provided_pod,
            "provided_endpoint": False,
            "ssh_host": endpoint.ip,
            "ssh_port": endpoint.port,
            "teardown_allowed": not provided_pod,
        }

    def _ssh(self, command: str) -> CommandResult:
        return self.transport.ssh(command).check("ssh")

    def _rsync_repo(self, source: str, target: str) -> None:
        self.transport.rsync(
            source.rstrip("/") + "/",
            target.rstrip("/") + "/",
            delete=True,
        ).check(f"rsync repo {source}")
        self._ssh(f"chmod -R u+w {_sq(target)}")

    def _reused_remote_environment_fingerprint(
        self,
        bundle: RunBundle,
        declaration_fingerprint: str,
    ) -> str | None:
        result = self.transport.ssh(
            f"test -f {_sq(self._remote_declaration_fingerprint_path(bundle))} && "
            f"cat {_sq(self._remote_declaration_fingerprint_path(bundle))}"
        )
        if result.returncode != 0 or result.stdout.strip() != declaration_fingerprint:
            return None
        realized = self.transport.ssh(
            f"test -f {_sq(self._remote_fingerprint_path(bundle))} && "
            f"cat {_sq(self._remote_fingerprint_path(bundle))}"
        ).check("read realized RunPod environment fingerprint").stdout.strip()
        validate_realized_runpod_environment_fingerprint(
            realized,
            bundle=bundle,
            declaration_fingerprint=declaration_fingerprint,
        )
        return realized

    def _probe_realized_environment(
        self,
        bundle: RunBundle,
        declaration_fingerprint: str,
    ) -> str:
        declaration = environment_declaration_identity_projection(bundle.environment)
        declaration.pop("repo_revisions")
        declaration.pop("overlay_steps")
        declaration["declaration_sha256"] = declaration_fingerprint
        command = (
            "uv run --no-sync python -c "
            f"{_sq(_REMOTE_ENVIRONMENT_PROBE)} "
            f"{_sq(json.dumps(declaration, sort_keys=True, separators=(',', ':')))}"
        )
        result = self._ssh(f"cd {_sq(self._primary_workdir())} && {command}")
        realized = result.stdout.strip()
        validate_realized_runpod_environment_fingerprint(
            realized,
            bundle=bundle,
            declaration_fingerprint=declaration_fingerprint,
        )
        return realized

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
            remote_repos.get("rlrmp2")
            or remote_repos.get("rlrmp")
            or remote_repos.get("feedbax")
            or self.config.remote_repo_root
        )

    def _row_workdir(self, row: RunRowSpec) -> str:
        return runpod_row_workdir(self.config, row)

    def _remote_run_dir(self, bundle: RunBundle) -> str:
        return f"{self.config.remote_run_root.rstrip('/')}/{bundle.run_set_id}"

    def _remote_sentinel_dir(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/sentinels"

    def _remote_payload_path(self, bundle: RunBundle, row: RunRowSpec) -> str:
        return f"{self._remote_run_dir(bundle)}/inputs/{row.row_id}.json"

    def _remote_events_dir(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/events"

    def _remote_fingerprint_path(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/env-fingerprint.json"

    def _remote_declaration_fingerprint_path(self, bundle: RunBundle) -> str:
        return f"{self._remote_run_dir(bundle)}/env-declaration-fingerprint.txt"

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


def project_runpod_provision_facts(pod: Mapping[str, Any]) -> dict[str, Any]:
    """Project provenance from an already-returned pod response without I/O."""

    def first(*paths: tuple[str, ...]) -> Any:
        for path in paths:
            value: Any = pod
            for part in path:
                if not isinstance(value, Mapping) or part not in value:
                    value = None
                    break
                value = value[part]
            if value not in (None, ""):
                return value
        return None

    region = first(
        ("dataCenterId",),
        ("data_center_id",),
        ("dataCenter", "id"),
        ("machine", "dataCenterId"),
    )
    immutable_image_id = first(
        ("imageName",),
        ("image_name",),
        ("containerImage",),
        ("container", "image"),
        ("template", "imageName"),
        ("template", "image_name"),
    )
    hourly_rate_raw = first(
        ("costPerHr",),
        ("costPerHour",),
        ("hourlyRate",),
        ("machine", "costPerHr"),
        ("machine", "costPerHour"),
    )
    billing_started_at_raw = first(("createdAt",), ("created_at",))
    try:
        parsed_hourly_rate = float(hourly_rate_raw) if hourly_rate_raw is not None else None
    except (TypeError, ValueError):
        parsed_hourly_rate = None
    hourly_rate = (
        parsed_hourly_rate
        if parsed_hourly_rate is not None and math.isfinite(parsed_hourly_rate)
        else None
    )
    raw_rate_observation = hourly_rate_raw
    if isinstance(raw_rate_observation, float) and not math.isfinite(raw_rate_observation):
        raw_rate_observation = repr(raw_rate_observation)
    return {
        "provider": "runpod",
        "region": str(region) if region is not None else None,
        "immutable_image_id": (
            str(immutable_image_id) if immutable_image_id is not None else None
        ),
        "hourly_rate": hourly_rate,
        "hourly_rate_raw": raw_rate_observation,
        "billing_started_at": _canonical_runpod_timestamp(billing_started_at_raw),
        "billing_started_at_raw": billing_started_at_raw,
        "currency": "USD" if hourly_rate is not None else None,
        "provider_observation_basis": "runpodctl pod get response",
    }


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


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_immutable_runpod_image_id(image_id: str | None) -> bool:
    """Return whether an image identity is pinned to an OCI SHA-256 digest."""
    return bool(image_id and _IMMUTABLE_IMAGE_PATTERN.fullmatch(image_id))


def runpod_lockfile_declaration_error(lockfile_hashes: Mapping[str, str]) -> str | None:
    """Validate remote lockfile paths and their expected SHA-256 digests."""
    if not lockfile_hashes:
        return "environment.lockfile_hashes must declare at least one locked dependency file"
    for path_text, digest in lockfile_hashes.items():
        path = PurePosixPath(path_text)
        if (
            not path_text
            or path.is_absolute()
            or path_text != path.as_posix()
            or ".." in path.parts
        ):
            return (
                "environment.lockfile_hashes keys must be safe paths relative to the "
                f"primary remote workdir: {path_text!r}"
            )
        if not _SHA256_PATTERN.fullmatch(digest):
            return f"invalid SHA-256 digest for lockfile {path_text!r}: {digest!r}"
    return None


def require_deterministic_runpod_environment(bundle: RunBundle) -> None:
    """Reject RunPod declarations that cannot realize an immutable environment."""
    if not is_immutable_runpod_image_id(bundle.environment.image_id):
        raise RunPodDriverError(
            "RunPod REALIZE_ENV requires environment.image_id pinned by @sha256:<64 hex>"
        )
    lockfile_error = runpod_lockfile_declaration_error(
        bundle.environment.lockfile_hashes
    )
    if lockfile_error is not None:
        raise RunPodDriverError(lockfile_error)
    if not bundle.environment.python_version:
        raise RunPodDriverError(
            "RunPod REALIZE_ENV requires environment.python_version"
        )


def validate_realized_runpod_environment_fingerprint(
    fingerprint: str,
    *,
    bundle: RunBundle,
    declaration_fingerprint: str,
) -> None:
    """Fail closed unless a realized fingerprint binds the declared environment."""
    try:
        payload = json.loads(fingerprint)
    except json.JSONDecodeError as exc:
        raise RunPodDriverError("realized RunPod environment probe returned invalid JSON") from exc
    declared_environment = environment_declaration_identity_projection(bundle.environment)
    expected = {
        "schema_version": RUNPOD_ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION,
        "declaration_sha256": declaration_fingerprint,
    }
    for field_name in ("image_id", "lockfile_hashes"):
        expected[field_name] = declared_environment[field_name]
    for field_name, expected_value in expected.items():
        if payload.get(field_name) != expected_value:
            raise RunPodDriverError(
                "realized RunPod environment fingerprint mismatch for "
                f"{field_name}: expected {expected_value!r}, observed {payload.get(field_name)!r}"
            )
    runtime = payload.get("runtime")
    required_runtime = {
        "device_count",
        "device_kind",
        "equinox",
        "feedbax",
        "jax",
        "jax_platform",
        "jax_platform_version",
        "jaxlib",
        "python",
        "python_implementation",
    }
    if not isinstance(runtime, Mapping) or not required_runtime <= runtime.keys():
        raise RunPodDriverError(
            "realized RunPod environment fingerprint lacks required runtime provenance"
        )
    if not _python_version_matches(
        str(declared_environment["python_version"]), str(runtime["python"])
    ):
        raise RunPodDriverError(
            "realized RunPod environment fingerprint Python version does not match declaration"
        )
    if runtime["jax_platform"] not in {"cuda", "gpu"}:
        raise RunPodDriverError(
            "realized RunPod environment fingerprint does not prove a JAX CUDA backend"
        )
    plugins = payload.get("feedbax_plugins")
    if not isinstance(plugins, list) or any(not isinstance(item, Mapping) for item in plugins):
        raise RunPodDriverError(
            "realized RunPod environment fingerprint lacks Feedbax plugin provenance"
        )


def _python_version_matches(declared: str, observed: str) -> bool:
    return observed == declared or (
        declared.count(".") == 1 and observed.startswith(declared + ".")
    )


def compute_runpod_environment_fingerprint(
    bundle: RunBundle,
    snapshot_manifest: RepoSnapshotManifest,
) -> str:
    """Compute the declared remote environment fingerprint."""
    payload = environment_declaration_identity_projection(bundle.environment)
    payload["repo_snapshot_content_sha256"] = {
        name: record.content_sha256
        for name, record in sorted(snapshot_manifest.repos.items())
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


def build_atomic_directory_publish_command(source: str, destination: str) -> str:
    """Build a fail-closed Linux atomic directory publish with no replacement."""
    return (
        f"python3 -c {_sq(_REMOTE_ATOMIC_DIRECTORY_PUBLISH)} "
        f"{_sq(source)} {_sq(destination)}"
    )


def build_native_resume_seed_command(
    source: str, attempt: str, target: str, resolved: ResolvedAssemblyInput
) -> str:
    """Build the shared secure clone plus atomic no-replace publication protocol."""

    return (
        f"python3 -c {_sq(SECURE_CHECKPOINT_SEED_SCRIPT)} {_sq(source)} {_sq(attempt)} "
        f"{_sq(target)} {_sq(native_resume_checkpoint_authority_json(resolved))}"
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
        f"setsid -f bash -lc {_sq(sentinel_command)} </dev/null >{_sq(log_file)} 2>&1"
    )


def _normalize_explicit_native_launch_command(command: Sequence[str]) -> list[str]:
    """Run an explicit native executor command in the realized uv environment."""
    normalized = [str(part) for part in command]
    if not is_native_training_command(normalized):
        return normalized
    if not normalized or Path(normalized[0]).name != "uv":
        return ["uv", "run", "--no-sync", *normalized]
    if len(normalized) < 2 or normalized[1] != "run":
        raise RunPodDriverError(
            "explicit native launch command beginning with uv must use 'uv run'"
        )
    if len(normalized) < 3 or normalized[2] != "--no-sync":
        normalized.insert(2, "--no-sync")
    return normalized


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
    checkpoint_source = native_resume_checkpoint_source(bundle, row)
    command_parts = (
        [str(part) for part in row.launch.command]
        if row.launch.command
        else ["uv", "run", "--no-sync", "python", row.launch.entry or ""]
    )
    command_parts, row = bind_native_execution_command(
        command_parts,
        row=row,
        payload_path=f"{remote_run_dir}/inputs/{row.row_id}.json",
        collection_root=row_dir,
    )
    command_parts = inject_native_execution_context(
        command_parts,
        row=row,
        environment_fingerprint=env_fingerprint,
        collection_root=row_dir,
    )
    if row.launch.command:
        command_parts = _normalize_explicit_native_launch_command(command_parts)
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
    seed_command = ""
    if checkpoint_source is not None:
        source = f"{remote_run_dir}/inputs/{checkpoint_source.custody.target_role}"
        attempt = f"{row_dir}/.checkpoint-seed-attempt"
        target = f"{row_dir}/checkpoints"
        seed_command = (
            f"{build_native_resume_seed_command(source, attempt, target, checkpoint_source)} && "
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
        f"{seed_command}"
        f"rm -f {_sq(pid_file)} && touch {_sq(started_file)} && "
        f"setsid -f bash -lc {_sq(inner)} </dev/null >{_sq(log_file)} 2>&1 && "
        f"i=0; while [ ! -s {_sq(pid_file)} ] && [ \"$i\" -lt 40 ]; do "
        'i=$((i+1)); sleep 0.05; done; '
        f"[ -s {_sq(pid_file)} ]"
    )


def runpod_row_workdir(config: RunPodDriverConfig, row: RunRowSpec) -> str:
    """Resolve the same row workdir used by live and dry-run RunPod launches."""
    workdir = row.launch.metadata.get("workdir")
    if workdir:
        return str(workdir)
    remote_repos = config.remote_repos or {
        "feedbax": f"{config.remote_repo_root}/feedbax"
    }
    return (
        remote_repos.get("rlrmp2")
        or remote_repos.get("rlrmp")
        or remote_repos.get("feedbax")
        or config.remote_repo_root
    )


def dry_run_launch_bundle(
    bundle: RunBundle,
    config: RunPodDriverConfig,
    input_provider_bindings: Sequence[InputProviderRootBinding] = (),
) -> tuple[str, ...]:
    """Bind all RunPod launch rows without constructing a transport."""
    failures, _ = preflight_input_provider_bindings(bundle, input_provider_bindings)
    if failures:
        raise RunPodDriverError("; ".join(failures))
    remote_run_dir = f"{config.remote_run_root.rstrip('/')}/{bundle.run_set_id}"
    remote_sentinel_dir = f"{remote_run_dir}/sentinels"
    return tuple(
        build_launch_row_command(
            bundle=bundle,
            row=row,
            remote_run_dir=remote_run_dir,
            remote_sentinel_dir=remote_sentinel_dir,
            workdir=runpod_row_workdir(config, row),
            env_fingerprint="dry-run-unrealized-environment",
            jax_cache_dir=f"{config.volume_mount}/jax_cache",
        )
        for row in bundle.rows
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
        f"echo $$ > {_sq(pid_file)}; "
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
        'rm -f "$pid_file"; '
        f"setsid -f bash -lc {_sq(script)} </dev/null "
        f'>>{_sq(remote_run_dir + "/logs/deadman.log")} 2>&1; '
        'i=0; while [ ! -s "$pid_file" ] && [ "$i" -lt 40 ]; do '
        'i=$((i+1)); sleep 0.05; done; [ -s "$pid_file" ]'
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


def load_runpod_api_key(config_path: Path | str = "~/.runpod/config.toml") -> str | None:
    """Load the RunPod key without exposing it through durable orchestration state."""
    if key := os.environ.get("RUNPOD_API_KEY"):
        return key
    try:
        payload = tomllib.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    key = payload.get("apikey")
    return str(key) if key else None


def _redact_secret(value: str, secret: str | None) -> str:
    return value.replace(secret, "<redacted>") if secret else value


def _classify_create_failure(
    result: CommandResult, secret: str | None
) -> tuple[Literal["retryable", "non-retryable"], str]:
    """Classify only sanitized provider authorization and request failures."""
    detail = _redact_secret(result.stderr or result.stdout, secret).strip()
    try:
        payload = json.loads(result.stdout or result.stderr)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        code = str(payload.get("statusCode") or payload.get("code") or "").lower()
        if code in {"400", "401", "403", "422", "unauthorized", "forbidden", "invalid_request"}:
            return "non-retryable", detail
    if result.returncode in {400, 401, 403, 422}:
        return "non-retryable", detail
    return "retryable", detail


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


def _parse_runpod_pod_inventory(payload: str) -> tuple[str, ...]:
    """Parse supported ``runpodctl pod list`` shapes into sanitized pod IDs."""
    loaded = json.loads(payload)
    if isinstance(loaded, list):
        pods = loaded
    elif isinstance(loaded, Mapping):
        candidates: list[Any] = []
        if "pods" in loaded:
            candidates.append(loaded["pods"])
        if "data" in loaded:
            data = loaded["data"]
            if isinstance(data, list):
                candidates.append(data)
            elif isinstance(data, Mapping) and "pods" in data:
                candidates.append(data["pods"])
            else:
                raise ValueError("unsupported RunPod inventory data wrapper")
        if len(candidates) != 1:
            raise ValueError("ambiguous RunPod inventory wrapper")
        pods = candidates[0]
    else:
        raise TypeError("RunPod inventory must be a list or wrapper object")
    if not isinstance(pods, list):
        raise TypeError("RunPod inventory wrapper must contain a list")

    pod_ids: list[str] = []
    for pod in pods:
        if not isinstance(pod, Mapping):
            raise TypeError("RunPod inventory entries must be objects")
        candidates = [pod.get("id"), pod.get("podId")]
        nested = pod.get("pod")
        if isinstance(nested, Mapping):
            candidates.append(nested.get("id"))
        identities = {
            value for value in candidates if isinstance(value, str) and value
        }
        if len(identities) != 1:
            raise ValueError("RunPod inventory entry has ambiguous pod identity")
        pod_id = identities.pop()
        if _SAFE_POD_ID_PATTERN.fullmatch(pod_id) is None:
            raise ValueError("RunPod inventory entry has unsafe pod identity")
        pod_ids.append(pod_id)
    if len(pod_ids) != len(set(pod_ids)):
        raise ValueError("RunPod inventory contains duplicate pod identities")
    return tuple(sorted(pod_ids))


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
