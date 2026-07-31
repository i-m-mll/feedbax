"""RunPod orchestration driver.

The driver keeps RunPod and SSH side effects behind an injectable transport so
unit tests can pin command shapes without contacting RunPod.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from feedbax.contracts.training import (
    TrainingMethodRegistry,
    TrainingRunSpec,
    resolve_training_run_spec,
)
from feedbax.contracts.run_matrix import (
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID,
    RUNPOD_PREFLIGHT_BASE_EVIDENCE_SCHEMA_ID,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V3,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V4,
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
from feedbax.orchestration.drivers.base import (
    AcquisitionCreateError,
    AcquisitionResult,
    DriverRowProbe,
    ProviderPodInventoryRecord,
)
from feedbax.orchestration.drivers.capabilities import (
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverConstructionContext,
    DriverHook,
    DriverRegistration,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RecoverySemantics,
    RealizedDriverCapabilities,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)
from feedbax.orchestration.drivers.native_execution import (
    inject_native_execution_context,
    is_native_training_command,
    native_resume_checkpoint_authority_json,
    native_resume_checkpoint_source,
    SECURE_CHECKPOINT_SEED_SCRIPT,
)
from feedbax.orchestration.executor_family import executor_family_adapter
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    InputProviderRootBinding,
    materialize_bundle_inputs,
    preflight_bundle_input_bindings,
)
from feedbax.orchestration.matrix_authority import (
    MatrixAuthorityError,
    build_training_run_matrix_authority,
    is_training_matrix_bundle,
)
from feedbax.orchestration.repo_snapshot import (
    RepoSnapshotError,
    SealedRepoSnapshots,
    restore_repo_snapshots,
    verify_repo_snapshot,
)
from feedbax.orchestration.staged_root_custody import StagedRootSnapshotBinding
from feedbax.orchestration.repo_realization import (
    EditableSourceResolution,
    RepoRealizationEntry,
    RepoRealizationError,
    RepoRealizationPlan,
    read_sealed_lock_bytes,
    seal_local_repo_realizations,
    validate_non_overlapping_remote_roots,
)
from feedbax.orchestration.schedule_eval import compare_continuation_schedule_projections
from feedbax.orchestration.state import (
    DEPENDENCY_SKIP_OUTCOME,
    PreflightCheckEntry,
    RunSetState,
    dependency_skip_observed,
    utc_now,
)
from feedbax.training.checkpoint_custody import (
    authenticated_run_contract_source_projection,
    load_checkpoint_custody_documents,
    validate_checkpoint_continuation_source_count,
)
from feedbax.training.diagnostics import NativeExecutionProducerContext


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
RUNPOD_ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION = "feedbax.runpod_environment_fingerprint.v1"
_RUNPOD_PREFLIGHT_CHECK_ORDER = (
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
)
_RUNPOD_PREFLIGHT_CHECK_NAMES = frozenset(_RUNPOD_PREFLIGHT_CHECK_ORDER)
_IMMUTABLE_IMAGE_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_LOCAL_LOCK_SOURCE_FORMS = frozenset({"editable", "path"})
_UNSUPPORTED_LOCAL_LOCK_SOURCE_FORMS = frozenset({"directory", "virtual", "workspace"})
_SUPPORTED_REMOTE_LOCK_SOURCE_FORMS = frozenset({"git", "registry", "url"})
_SUPPORTED_UV_LOCK_VERSIONS = frozenset({1})
_DEPENDENCY_FILE_NAMES = frozenset(
    {
        "uv.lock",
        "pyproject.toml",
        "poetry.lock",
        "pdm.lock",
        "pipfile",
        "pipfile.lock",
        "pixi.lock",
        "environment.yml",
        "environment.yaml",
    }
)
_POD_NOT_FOUND_MARKERS = ("not found", "does not exist", "404")
_SAFE_POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_CHILD_TERMINATION_GRACE_SECONDS = 1.0
_RUNPOD_GO_UTC_PATTERN = re.compile(
    r"^(?P<instant>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \+0000 UTC$"
)
_REMOTE_ENVIRONMENT_PROBE = r"""
import asyncio
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
from feedbax.plugins.composition import compose_application

devices = jax.devices()
if not devices:
    raise RuntimeError("JAX reported no runtime devices")
bootstrap_state = asyncio.run(compose_application())
plugins = [
    {
        "distribution": provenance.distribution,
        "distribution_version": provenance.distribution_version,
        "name": provenance.entry_point_name,
        "value": provenance.entry_point_value,
    }
    for provenance in bootstrap_state.provenance
]
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


class RunPodTeardownError(RunPodDriverError):
    """A failed teardown with durable, sanitized unresolved-pod evidence."""

    def __init__(self, message: str, *, teardown_outputs: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.teardown_outputs = dict(teardown_outputs)


class RunPodRemoteSmokeError(RunPodDriverError):
    """A bounded remote smoke failure with stage-persistable evidence."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


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
    primary_repo: str | None = None
    protected_refs: Mapping[str, str] = field(default_factory=dict)
    path_patches: tuple[tuple[str, str, str], ...] = ()
    overlay_steps: tuple[str, ...] = (
        "uv pip install \"jax[cuda12]==$(uv run --no-sync python -c 'import jax; print(jax.__version__)')\"",
    )
    auto_teardown: bool = True


@dataclass(frozen=True)
class RunPodExecutionNamespace:
    """Orchestration-owned paths and identity for one native execution.

    Both the durable launch and its bounded smoke use this contract.  Keeping
    every writable path here makes scratch confinement reviewable at the call
    site instead of relying on independent command-line overrides.
    """

    row_root: str
    manifest_root: str
    checkpoint_root: str
    events_dir: str
    sentinel_dir: str
    log_path: str
    payload_path: str
    run_identity: str
    sentinel_stem: str
    seed_source: str | None = None
    seed_attempt: str | None = None
    seed_target: str | None = None
    env_exports: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.manifest_root != f"{self.row_root}/manifests":
            raise ValueError("manifest_root must be namespaced below row_root")
        if self.checkpoint_root != f"{self.row_root}/checkpoints":
            raise ValueError("checkpoint_root must be namespaced below row_root")
        seed_values = (self.seed_source, self.seed_attempt, self.seed_target)
        if any(value is not None for value in seed_values) and not all(
            value is not None for value in seed_values
        ):
            raise ValueError(
                "checkpoint seed source, attempt, and target must be supplied together"
            )


class RunPodOrchestrationDriver:
    """Synchronous RunPod implementation of the orchestration driver protocol."""

    _COMMON_HOOKS = frozenset(
        {
            DriverHook.RESTORE_FROM_PROVISION_RECORD,
            DriverHook.RESTORE_COMPLETED_PREFLIGHT,
            DriverHook.STATIC_PREFLIGHT_CHECKS,
            DriverHook.PREFLIGHT_CHECKS,
            DriverHook.REPO_REALIZATION_PLAN,
            DriverHook.PREFLIGHT_EVIDENCE,
            DriverHook.REMOTE_SMOKE,
            DriverHook.SMOKE_FAILURE_EVIDENCE,
            DriverHook.COLLECTION_RECOVERY_EVIDENCE,
            DriverHook.COLLECT_FAILURE_LOGS,
            DriverHook.TEARDOWN_OWNERSHIP,
            DriverHook.BATCH_PROBE,
            DriverHook.DRY_RUN_LAUNCH,
        }
    )
    _ACQUISITION_HOOKS = frozenset(
        {
            DriverHook.HAS_PENDING_OWNED_RESOURCE,
            DriverHook.GOVERN_PROVISIONING_RETRIES,
            DriverHook.ENGINE_ACQUISITION,
            DriverHook.PROVISION_RETRY_DELAY,
        }
    )
    capability_envelope = DriverCapabilityEnvelope(
        driver_name="runpod",
        variants={
            "externally-managed": DriverCapabilityFacts(
                variant_id="externally-managed",
                venue=DriverVenue.CLOUD_RESOURCE,
                resources=ResourceSemantics.EXTERNALLY_MANAGED,
                spend=SpendSemantics.EXTERNALLY_MANAGED,
                authorization=AuthorizationSemantics.OPTIONAL_CALLER_CREDENTIAL,
                environment=EnvironmentSemantics.REMOTE_REALIZATION,
                monitoring=MonitoringSemantics.PROVIDER_INVENTORY,
                recovery=RecoverySemantics.DURABLE_REMOTE,
                retry=RetrySemantics.NONE,
                acquisition=AcquisitionSemantics.EXTERNALLY_PROVIDED,
                teardown=TeardownSemantics.RESOURCES_PRESERVED,
                custody=CustodySemantics.EPHEMERAL_REMOTE_RESOURCE,
                optional_hooks=_COMMON_HOOKS,
            ),
            "engine-acquired": DriverCapabilityFacts(
                variant_id="engine-acquired",
                venue=DriverVenue.CLOUD_RESOURCE,
                resources=ResourceSemantics.DRIVER_OWNED,
                spend=SpendSemantics.DRIVER_OBSERVED,
                authorization=AuthorizationSemantics.CLOUD_AND_SPEND_REQUIRED,
                environment=EnvironmentSemantics.REMOTE_REALIZATION,
                monitoring=MonitoringSemantics.PROVIDER_INVENTORY,
                recovery=RecoverySemantics.DURABLE_REMOTE,
                retry=RetrySemantics.DRIVER_GOVERNED,
                acquisition=AcquisitionSemantics.ENGINE_GOVERNED,
                teardown=TeardownSemantics.VERIFIED_RESOURCE_ABSENCE,
                custody=CustodySemantics.EPHEMERAL_REMOTE_RESOURCE,
                optional_hooks=_COMMON_HOOKS
                | _ACQUISITION_HOOKS
                | frozenset({DriverHook.GLOBAL_RESOURCE_INVENTORY}),
            ),
            "engine-acquired-preserved": DriverCapabilityFacts(
                variant_id="engine-acquired-preserved",
                venue=DriverVenue.CLOUD_RESOURCE,
                resources=ResourceSemantics.DRIVER_OWNED,
                spend=SpendSemantics.DRIVER_OBSERVED,
                authorization=AuthorizationSemantics.CLOUD_AND_SPEND_REQUIRED,
                environment=EnvironmentSemantics.REMOTE_REALIZATION,
                monitoring=MonitoringSemantics.PROVIDER_INVENTORY,
                recovery=RecoverySemantics.DURABLE_REMOTE,
                retry=RetrySemantics.DRIVER_GOVERNED,
                acquisition=AcquisitionSemantics.ENGINE_GOVERNED,
                teardown=TeardownSemantics.RESOURCES_PRESERVED,
                custody=CustodySemantics.EPHEMERAL_REMOTE_RESOURCE,
                optional_hooks=_COMMON_HOOKS | _ACQUISITION_HOOKS,
            ),
        },
    )

    poll_interval_seconds = 5.0
    driver_name = "runpod"

    def govern_provisioning_retries(self) -> bool:
        """Confirm that this variant supplies governed provisioning retry hooks."""
        return True

    def engine_acquisition_required(self) -> bool:
        """Return whether the engine must run the create WAL protocol."""
        return not self._provided_endpoint and not self._provided_pod

    def acquisition_candidates(self, bundle: RunBundle) -> tuple[str | None, ...]:
        """Return the ordered singleton-create datacenter candidates."""
        del bundle
        return tuple(self.config.datacenters) or (None,)

    @property
    def reconciliation_timeout_seconds(self) -> float:
        """Bound engine reconciliation by the existing teardown ceiling."""
        return self.config.teardown_absence_timeout_seconds

    def acquisition_pod_name(self, intent_id: str) -> str:
        """Return the exact provider name tag bound to one durable intent."""
        return f"{self.config.pod_name_prefix}-{intent_id}"

    def acquisition_config_identity(self, bundle: RunBundle) -> str:
        """Return a secret-free identity for the acquisition-relevant configuration."""
        payload = {
            "run_set_id": bundle.run_set_id,
            "image": self.config.image,
            "gpu_id": self.config.gpu_id,
            "datacenters": list(self.config.datacenters),
            "pod_name_prefix": self.config.pod_name_prefix,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def create_pod_once(
        self,
        bundle: RunBundle,
        candidate: str | None,
        intent_id: str,
    ) -> AcquisitionResult:
        """Invoke the provider exactly once for one engine-persisted intent."""
        if not self._preflight_passed:
            raise AcquisitionCreateError(
                "RunPod creation requires passing named driver PREFLIGHT checks first",
                clean_rejection=True,
                evidence={"returncode": None, "classification": "preflight-not-passed"},
            )
        name = self.acquisition_pod_name(intent_id)
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
        if candidate:
            args.extend(["--data-center-ids", candidate])
        if self.config.api_key:
            args.extend(["--env", json.dumps({"FEEDBAX_RUNPOD_API_KEY": self.config.api_key})])
        result = self.transport.runpodctl(*args)
        detail = _redact_secret(result.stderr or result.stdout, self.config.api_key).strip()
        if result.returncode == 0:
            try:
                payload = _json_object(result.stdout)
            except (TypeError, ValueError, RunPodDriverError) as exc:
                raise AcquisitionCreateError(
                    str(exc),
                    clean_rejection=False,
                    evidence={
                        "returncode": result.returncode,
                        "classification": "ambiguous-invalid-success-payload",
                        "detail": detail,
                    },
                ) from exc
            pod_id = str(payload.get("id") or payload.get("podId") or "")
            if pod_id and _SAFE_POD_ID_PATTERN.fullmatch(pod_id):
                return AcquisitionResult(pod_id, candidate)
            raise AcquisitionCreateError(
                "successful RunPod create response omitted a safe pod identity",
                clean_rejection=False,
                evidence={
                    "returncode": result.returncode,
                    "classification": "ambiguous-missing-pod-id",
                    "detail": detail,
                },
            )
        classification, classified_detail = _classify_create_failure(result, self.config.api_key)
        clean_rejection = classification == "non-retryable"
        raise AcquisitionCreateError(
            classified_detail or f"runpodctl pod create exited {result.returncode}",
            clean_rejection=clean_rejection,
            evidence={
                "returncode": result.returncode,
                "classification": (
                    "classified-clean-provider-rejection" if clean_rejection else "ambiguous"
                ),
                "detail": classified_detail,
            },
        )

    def finish_acquired_pod(
        self,
        bundle: RunBundle,
        acquisition: AcquisitionResult,
        intent_id: str,
    ) -> Mapping[str, Any]:
        """Configure one acquired pod, install its watchdog, and prove GPU readiness."""
        self._pod_id = acquisition.pod_id
        self._provided_pod = False
        self._last_provision_pod = None
        self._endpoint, pod = self._wait_for_endpoint(acquisition.pod_id)
        self._configure_subprocess_endpoint(self._endpoint)
        remote_run_dir = self._remote_run_dir(bundle)
        self._ssh(
            f"mkdir -p {_sq(remote_run_dir)} {_sq(self._remote_sentinel_dir(bundle))} "
            f"{_sq(remote_run_dir + '/logs')} && command -v setsid >/dev/null"
        )
        self._ensure_deadman(bundle)
        self._require_gpu_ready()
        return self._provision_record(
            pod,
            provided_pod=False,
            accepted_datacenter=acquisition.accepted_datacenter,
            intent_id=intent_id,
        )

    def restore_from_provision_record(self, record: Mapping[str, Any]) -> None:
        """Restore process-local pod and endpoint identity from durable state."""
        if record.get("driver") != self.driver_name:
            return
        pod_id = record.get("pod_id")
        host = record.get("ssh_host")
        port = record.get("ssh_port")
        if not isinstance(host, str) or not isinstance(port, int):
            raise RunPodDriverError("persisted RunPod provision record lacks a usable endpoint")
        if record.get("provided_endpoint") is not True and (
            not isinstance(pod_id, str) or not pod_id
        ):
            raise RunPodDriverError("persisted RunPod provision record lacks a pod identity")
        self._pod_id = pod_id if isinstance(pod_id, str) and pod_id else None
        self._provided_pod = record.get("provided_pod") is True
        self._provided_endpoint = record.get("provided_endpoint") is True
        self._endpoint = EndpointClassification("ssh_object", host, port)
        self._configure_subprocess_endpoint(self._endpoint)

    def adopt_owned_pod(
        self,
        pod_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Adopt one intent-matched pod for the standard owned-pod teardown path."""
        self._pod_id = pod_id
        self._provided_pod = False
        self._provided_endpoint = False
        self._endpoint = None
        self._adopted_teardown_timeout_seconds = timeout_seconds

    def adopted_provision_record(self, intent_id: str) -> Mapping[str, Any]:
        """Project an adopted intent-matched pod through the canonical record builder."""
        if self._pod_id is None:
            raise RunPodDriverError("cannot record an adopted pod before adopting its identity")
        return self._provision_record(
            {"id": self._pod_id},
            provided_pod=False,
            intent_id=intent_id,
        )

    def acquisition_failure_evidence(self) -> Mapping[str, Any]:
        """Return sanitized provider facts observed after an acquired transition."""
        if self._last_provision_pod is None:
            return {}
        return project_runpod_provision_facts(self._last_provision_pod)

    def __init__(
        self,
        *,
        config: RunPodDriverConfig | None = None,
        transport: RunPodTransport | None = None,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
        input_provider_bindings: Sequence[InputProviderRootBinding] = (),
        staged_root_bindings: Sequence[StagedRootSnapshotBinding] = (),
        collection_recovery_bindings: Sequence[CollectionRecoveryBinding] = (),
        training_method_registry: TrainingMethodRegistry | None = None,
        realized_capabilities: RealizedDriverCapabilities | None = None,
    ) -> None:
        self.config = config or RunPodDriverConfig()
        self.transport = transport or SubprocessRunPodTransport(
            ssh_host=self.config.ssh_host,
            ssh_port=self.config.ssh_port,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self.input_provider_bindings = tuple(input_provider_bindings)
        self.staged_root_bindings = tuple(staged_root_bindings)
        self.collection_recovery_bindings = tuple(collection_recovery_bindings)
        self.training_method_registry = training_method_registry
        self._collection_recovery_evidence: dict[str, Mapping[str, object]] = {}
        self._repo_snapshots: SealedRepoSnapshots | None = None
        self._repo_realization_plan: RepoRealizationPlan | None = None
        self._preflight_passed = False
        self._pod_id = self.config.pod_id
        self._provided_pod = self.config.pod_id is not None
        self._provided_endpoint = bool(self.config.ssh_host and self.config.ssh_port)
        default_variant = (
            "externally-managed"
            if self._provided_endpoint or self._provided_pod
            else ("engine-acquired" if self.config.auto_teardown else "engine-acquired-preserved")
        )
        self.realized_capabilities = realized_capabilities or self.capability_envelope.realize(
            default_variant
        )
        self._endpoint: EndpointClassification | None = (
            EndpointClassification("ssh_object", self.config.ssh_host, self.config.ssh_port)
            if self._provided_endpoint
            else None
        )
        self._last_provision_pod: Mapping[str, Any] | None = None
        self._adopted_teardown_timeout_seconds: float | None = None

    def dry_run_launch(self, bundle: RunBundle) -> tuple[str, ...]:
        """Bind launch commands without contacting RunPod."""
        return dry_run_launch_bundle(
            bundle,
            self.config,
            self.input_provider_bindings,
            self.staged_root_bindings,
        )

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

        raise RunPodDriverError(
            "new RunPod resources must be acquired by StageEngine's durable intent protocol"
        )

    @property
    def provision_retry_delay_seconds(self) -> float:
        """Configured positive delay between governed acquisition attempts."""
        return self.config.poll_seconds

    def provision_retry_delay(self) -> float:
        """Return the configured positive delay between provisioning attempts."""
        return self.config.poll_seconds

    def preflight_checks(self, bundle: RunBundle) -> list[PreflightCheckEntry]:
        """Run named, non-mutating RunPod checks before any billable action."""
        return self._run_preflight_checks(bundle)

    def static_preflight_checks(
        self,
        bundle: RunBundle,
        *,
        upstream_failures: Sequence[str],
    ) -> list[PreflightCheckEntry]:
        """Run local checks while recording provider checks blocked by core failures."""
        return self._run_preflight_checks(
            bundle,
            provider_checks_allowed=False,
            upstream_failures=upstream_failures,
        )

    def _run_preflight_checks(
        self,
        bundle: RunBundle,
        *,
        provider_checks_allowed: bool = True,
        upstream_failures: Sequence[str] = (),
    ) -> list[PreflightCheckEntry]:
        checks_by_name: dict[str, PreflightCheckEntry] = {}
        conditional_checks: list[PreflightCheckEntry] = []
        lockfile_error = runpod_lockfile_declaration_error(bundle.environment.lockfile_hashes)
        if is_training_matrix_bundle(bundle):
            try:
                build_training_run_matrix_authority(
                    bundle,
                    local_repos=self.config.local_repos,
                    protected_refs=self.config.protected_refs,
                )
            except (OSError, ValueError, RunPodDriverError) as exc:
                conditional_checks.append(
                    _preflight_check(
                        "training-matrix-authority",
                        False,
                        detail=str(exc),
                    )
                )

        plan_error: tuple[str, str] | None = None
        if lockfile_error is not None:
            plan_error = ("runpod-lockfiles-declared", lockfile_error)
            checks_by_name["runpod-repo-snapshots"] = _dependency_skipped_preflight_check(
                "runpod-repo-snapshots", "runpod-lockfiles-declared"
            )
        else:
            try:
                self.seal_repo_realization_plan(bundle)
            except RunPodDriverError as exc:
                plan_error = ("runpod-remote-layout-vs-lock", str(exc))
                checks_by_name["runpod-repo-snapshots"] = _dependency_skipped_preflight_check(
                    "runpod-repo-snapshots", "repo-realization-plan-sealing"
                )
            except (RepoSnapshotError, RepoRealizationError) as exc:
                plan_error = ("runpod-repo-snapshots", str(exc))
                checks_by_name["runpod-repo-snapshots"] = _preflight_check(
                    "runpod-repo-snapshots", False, detail=str(exc)
                )
            else:
                plan = self.repo_realization_plan()
                assert plan is not None
                checks_by_name["runpod-repo-snapshots"] = _preflight_check(
                    "runpod-repo-snapshots",
                    True,
                    observed=plan.model_dump(mode="json"),
                )

        failures, observed = preflight_bundle_input_bindings(
            bundle,
            provider_bindings=self.input_provider_bindings,
            staged_root_bindings=self.staged_root_bindings,
        )
        checks_by_name["input-provider-bindings"] = _preflight_check(
            "input-provider-bindings",
            not failures,
            detail="; ".join(failures) if failures else None,
            observed=observed or "no-resolved-inputs",
        )
        non_native_smoke_rows = [
            row.row_id
            for row in bundle.rows
            if bundle.smoke_enabled
            and (
                (
                    row.execution_family == "native-training"
                    and not _row_uses_registered_native_execution(row)
                )
                or (
                    row.execution_family == "evaluation-matrix"
                    and "matrix-harness" not in _row_launch_command_parts(row)
                )
            )
        ]
        checks_by_name["runpod-remote-smoke-applicability"] = _preflight_check(
            "runpod-remote-smoke-applicability",
            not non_native_smoke_rows,
            detail=(
                None
                if not non_native_smoke_rows
                else "remote smoke requires registered native execution; non-native rows: "
                + ", ".join(repr(row_id) for row_id in non_native_smoke_rows)
            ),
            observed={
                "smoke_enabled": bundle.smoke_enabled,
                "non_native_rows": non_native_smoke_rows,
            },
        )

        schedule_failures, schedule_observed = _preflight_continuation_schedule_consistency(
            bundle,
            self.input_provider_bindings,
            training_method_registry=self.training_method_registry,
            input_bindings_valid=not failures,
        )
        if schedule_observed.get("outcome") == "skipped-due-to-dependency":
            checks_by_name["continuation-schedule-consistency"] = (
                _dependency_skipped_preflight_check(
                    "continuation-schedule-consistency", "input-provider-bindings"
                )
            )
        else:
            checks_by_name["continuation-schedule-consistency"] = _preflight_check(
                "continuation-schedule-consistency",
                not schedule_failures,
                detail="; ".join(schedule_failures) if schedule_failures else None,
                observed=schedule_observed or "no-continuations",
            )

        checks_by_name["runpod-lockfiles-declared"] = _preflight_check(
            "runpod-lockfiles-declared",
            lockfile_error is None,
            detail=lockfile_error,
            observed=dict(sorted(bundle.environment.lockfile_hashes.items())),
        )

        if plan_error is not None and plan_error[0] == "runpod-remote-layout-vs-lock":
            checks_by_name["runpod-remote-layout-vs-lock"] = _preflight_check(
                "runpod-remote-layout-vs-lock",
                False,
                detail=plan_error[1],
            )
        elif plan_error is not None and plan_error[0] == "runpod-repo-snapshots":
            checks_by_name["runpod-remote-layout-vs-lock"] = _dependency_skipped_preflight_check(
                "runpod-remote-layout-vs-lock", "runpod-repo-snapshots"
            )
        elif plan_error is not None:
            checks_by_name["runpod-remote-layout-vs-lock"] = _dependency_skipped_preflight_check(
                "runpod-remote-layout-vs-lock", "runpod-lockfiles-declared"
            )
        else:
            layout_error, layout_observed = validate_runpod_repo_realization_plan(
                bundle,
                self.config,
                self.repo_realization_plan(),
                self._repo_snapshots,
            )
            checks_by_name["runpod-remote-layout-vs-lock"] = _preflight_check(
                "runpod-remote-layout-vs-lock",
                layout_error is None,
                detail=layout_error,
                observed=layout_observed,
            )

        image = bundle.environment.image_id
        image_is_immutable = is_immutable_runpod_image_id(image)
        checks_by_name["runpod-image-immutable"] = _preflight_check(
            "runpod-image-immutable",
            image_is_immutable,
            detail=(
                None
                if image_is_immutable
                else "environment.image_id must be an OCI image pinned by @sha256:<64 hex>"
            ),
            observed=image,
        )
        checks_by_name["runpod-python-version-declared"] = _preflight_check(
            "runpod-python-version-declared",
            bool(bundle.environment.python_version),
            detail=(
                None
                if bundle.environment.python_version
                else "environment.python_version is required for deterministic realization"
            ),
            observed=bundle.environment.python_version,
        )
        gpu_policy = bool(self._provided_endpoint or self._pod_id or self.config.gpu_id)
        checks_by_name["runpod-gpu-policy-declared"] = _preflight_check(
            "runpod-gpu-policy-declared",
            gpu_policy,
            detail=None if gpu_policy else "a GPU policy or existing target is required",
            observed={
                "gpu_id": self.config.gpu_id,
                "datacenter_fallbacks": list(self.config.datacenters),
                "provided_target": bool(self._provided_endpoint or self._pod_id),
            },
        )
        checks_by_name["runpod-deadman-credentials"] = _preflight_check(
            "runpod-deadman-credentials",
            not bundle.deadman_enabled or bool(self.config.api_key),
            detail=(
                None
                if not bundle.deadman_enabled or self.config.api_key
                else "dead-man teardown requires an API credential"
            ),
            observed="available" if self.config.api_key else "not-required-or-missing",
        )

        structural_check_names = {
            "training-matrix-authority",
            "input-provider-bindings",
            "runpod-repo-snapshots",
            "runpod-remote-smoke-applicability",
            "continuation-schedule-consistency",
            "runpod-lockfiles-declared",
            "runpod-remote-layout-vs-lock",
        }
        structural_failure_names = [
            *upstream_failures,
            *(
                check.name
                for check in [*conditional_checks, *checks_by_name.values()]
                if check.status == "fail" and check.name in structural_check_names
            ),
        ]
        image_dependencies = [
            *structural_failure_names,
            *([] if image_is_immutable else ["runpod-image-immutable"]),
        ]
        if not provider_checks_allowed or image_dependencies:
            checks_by_name["runpod-image-tag-exists"] = _dependency_skipped_preflight_check(
                "runpod-image-tag-exists", *image_dependencies
            )
        else:
            try:
                image_exists = bool(self.transport.image_exists(image or ""))
            except Exception as exc:
                checks_by_name["runpod-image-tag-exists"] = _preflight_check(
                    "runpod-image-tag-exists",
                    False,
                    detail=f"image existence query raised {type(exc).__name__}: {exc}",
                    observed=image,
                )
            else:
                checks_by_name["runpod-image-tag-exists"] = _preflight_check(
                    "runpod-image-tag-exists",
                    image_exists,
                    detail=None if image_exists else "immutable image was not found by RunPod",
                    observed=image,
                )

        if not provider_checks_allowed or structural_failure_names:
            checks_by_name["runpod-credentials"] = _dependency_skipped_preflight_check(
                "runpod-credentials", *structural_failure_names
            )
            checks_by_name["runpod-balance-floor"] = _dependency_skipped_preflight_check(
                "runpod-balance-floor", "runpod-credentials"
            )
        else:
            credentials_required = not self._provided_endpoint or bundle.deadman_enabled
            try:
                user_result = (
                    self.transport.runpodctl("user", "--output", "json")
                    if credentials_required
                    else CommandResult(0, '{"provided_endpoint": true}')
                )
            except Exception as exc:
                user_result = CommandResult(
                    1,
                    "",
                    f"credential query raised {type(exc).__name__}: {exc}",
                )
            credentials_ok = user_result.returncode == 0
            credentials_check = _preflight_check(
                "runpod-credentials",
                credentials_ok,
                detail=(
                    None
                    if credentials_ok
                    else user_result.stderr
                    or user_result.stdout
                    or "runpodctl user failed without diagnostic output"
                ),
                observed="verified" if credentials_required else "not-required-provided-endpoint",
            )
            user_payload: Mapping[str, Any] = {}
            if credentials_ok:
                try:
                    user_payload = _json_object(user_result.stdout)
                except RunPodDriverError:
                    credentials_ok = False
                    credentials_check = _preflight_check(
                        "runpod-credentials",
                        False,
                        detail="runpodctl user returned invalid JSON",
                        observed="invalid-response",
                    )
            checks_by_name["runpod-credentials"] = credentials_check

            if not credentials_ok:
                checks_by_name["runpod-balance-floor"] = _dependency_skipped_preflight_check(
                    "runpod-balance-floor", "runpod-credentials"
                )
            else:
                balance_required = not self._provided_endpoint and not self._pod_id
                balance = user_balance(user_payload)
                balance_ok = not balance_required or (
                    balance is not None and balance >= self.config.min_balance_usd
                )
                checks_by_name["runpod-balance-floor"] = _preflight_check(
                    "runpod-balance-floor",
                    balance_ok,
                    detail=(
                        None
                        if balance_ok
                        else f"RunPod balance must be at least {self.config.min_balance_usd:g}"
                    ),
                    observed=balance if balance_required else "not-required-existing-target",
                )

        checks = [
            *conditional_checks,
            *(checks_by_name[name] for name in _RUNPOD_PREFLIGHT_CHECK_ORDER),
        ]
        self._preflight_passed = all(
            check.status == "pass" and not _is_dependency_skipped_preflight_check(check)
            for check in checks
        )
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

    def seal_repo_realization_plan(self, bundle: RunBundle) -> RepoRealizationPlan:
        """Build the complete provider-free realization authority before provisioning."""
        plan, snapshots = build_runpod_repo_realization_plan(
            bundle,
            self.config,
            snapshot_parent=bundle.run_set_dir / ".repo-snapshots",
        )
        self._repo_realization_plan = plan
        self._repo_snapshots = snapshots
        return plan

    def repo_realization_plan(self) -> RepoRealizationPlan | None:
        """Return the current complete realization authority, if available."""
        return self._repo_realization_plan

    def restore_completed_preflight(self, bundle: RunBundle, state: RunSetState) -> bool:
        """Restore only cryptographically bound completed preflight authority.

        This is deliberately offline: it validates persisted evidence without
        querying RunPod or repeating mutable credential/balance observations.
        """
        if state.repo_realization_plan is None:
            raise RunPodDriverError("completed PREFLIGHT lacks repo realization authority")
        self._repo_realization_plan = state.repo_realization_plan
        try:
            self._repo_snapshots = restore_repo_snapshots(
                {
                    name: entry.local_root
                    for name, entry in state.repo_realization_plan.repos.items()
                },
                state.repo_realization_plan.snapshot_manifest,
                snapshot_parent=bundle.run_set_dir / ".repo-snapshots",
            )
        except RepoSnapshotError as exc:
            raise RunPodDriverError(str(exc)) from exc
        stage = state.stage("PREFLIGHT")
        evidence = stage.outputs.get("driver_evidence")
        _require_preflight_plan_digest(evidence, state.repo_realization_plan.plan_digest)
        persisted_checks = stage.outputs.get("checks")
        if persisted_checks != [check.model_dump(mode="json") for check in stage.checks]:
            raise RunPodDriverError("completed PREFLIGHT checks are internally inconsistent")
        if state.run_set_id != bundle.run_set_id:
            raise RunPodDriverError("completed PREFLIGHT run-set binding is mismatched")
        if not any(check.name == "continuation-schedule-consistency" for check in stage.checks):
            return False
        if (
            is_training_matrix_bundle(bundle)
            and isinstance(evidence, Mapping)
            and evidence.get("schema_id") == RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID
            and evidence.get("schema_version")
            in {
                RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2,
                RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V3,
            }
        ):
            return False
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
            return True
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
        return True

    def _matrix_preflight_evidence(
        self, bundle: RunBundle, legacy: Mapping[str, Any]
    ) -> dict[str, Any]:
        binding = self._matrix_preflight_binding(
            bundle, nested_evidence_sha256=str(legacy["sha256"])
        )
        payload = {
            "schema_id": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID,
            "schema_version": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V4,
            "base": dict(legacy),
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
        realization_plan = state.repo_realization_plan or self.repo_realization_plan()
        if realization_plan is None:
            raise RunPodDriverError("completed PREFLIGHT lacks repo realization authority")
        return {
            "schema_id": RUNPOD_PREFLIGHT_BASE_EVIDENCE_SCHEMA_ID,
            "schema_version": RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
            "run_set_id": bundle.run_set_id,
            "bundle_sha256": bundle_sha256,
            "checks_sha256": _sha256_json([check.model_dump(mode="json") for check in checks]),
            "driver_contract": self._preflight_driver_contract(),
            "repo_realization_plan_digest": realization_plan.plan_digest,
        }

    def _preflight_driver_contract(self) -> dict[str, Any]:
        """Project the complete effective RunPod config without exposing secrets."""
        contract = {
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
            "primary_repo": self.config.primary_repo,
            "path_patches": [list(patch) for patch in self.config.path_patches],
            "overlay_steps_sha256": _sha256_json(list(self.config.overlay_steps)),
            "auto_teardown": self.config.auto_teardown,
            "provided_endpoint": self._provided_endpoint,
            "input_provider_bindings": sorted(
                binding.name for binding in self.input_provider_bindings
            ),
        }
        if self.staged_root_bindings:
            contract["staged_root_bindings"] = sorted(
                (binding.kind, binding.name) for binding in self.staged_root_bindings
            )
        return contract

    def _validate_preflight_checks(
        self, bundle: RunBundle, checks: Sequence[PreflightCheckEntry]
    ) -> None:
        """Validate persisted RunPod checks without provider or input access."""
        named = {
            check.name: check for check in checks if check.name in _RUNPOD_PREFLIGHT_CHECK_NAMES
        }
        if len(named) != len(_RUNPOD_PREFLIGHT_CHECK_NAMES) or any(
            check.name in _RUNPOD_PREFLIGHT_CHECK_NAMES
            and sum(item.name == check.name for item in checks) != 1
            for check in named.values()
        ):
            raise RunPodDriverError("completed PREFLIGHT lacks a unique RunPod check set")
        if any(
            check.status != "pass" or _is_dependency_skipped_preflight_check(check)
            for check in checks
        ):
            raise RunPodDriverError("completed PREFLIGHT includes a failing check")

        failures, resolved_observed = preflight_bundle_input_bindings(
            bundle,
            provider_bindings=self.input_provider_bindings,
            staged_root_bindings=self.staged_root_bindings,
        )
        if failures:
            raise RunPodDriverError(
                "current bundle no longer has valid resolved input declarations"
            )
        current_bindings = {binding.name for binding in self.input_provider_bindings}
        required_bindings = {item.custody.provider_binding for item in bundle.resolved_inputs}
        if current_bindings != required_bindings:
            raise RunPodDriverError("current input-provider bindings do not cover the bundle")
        expected_observed = resolved_observed or "no-resolved-inputs"
        if named["input-provider-bindings"].observed != expected_observed:
            raise RunPodDriverError(
                "input-provider preflight observation does not match the bundle"
            )

        image = bundle.environment.image_id
        if self.config.image != image:
            raise RunPodDriverError("RunPod driver image does not match the bundle")
        expected_checks = {
            "runpod-repo-snapshots": (
                self.repo_realization_plan().model_dump(mode="json")
                if self.repo_realization_plan() is not None
                else None
            ),
            "runpod-image-immutable": image,
            "runpod-image-tag-exists": image,
            "runpod-lockfiles-declared": dict(sorted(bundle.environment.lockfile_hashes.items())),
            "runpod-python-version-declared": bundle.environment.python_version,
            "runpod-gpu-policy-declared": {
                "gpu_id": self.config.gpu_id,
                "datacenter_fallbacks": list(self.config.datacenters),
                # provided_target is a user declaration observed at pre-acquisition
                # preflight, not acquisition state. Reconstruct it from the original
                # provided-target declarations so an acquired-pod run whose _pod_id was
                # restored from the provision record still matches its persisted
                # observation; a later-acquired pod must not flip this to True.
                "provided_target": bool(self._provided_endpoint or self._provided_pod),
            },
        }
        for name, observed in expected_checks.items():
            if named[name].observed != observed:
                raise RunPodDriverError(f"{name} observation does not match current declarations")

        layout_error, layout_observed = validate_runpod_repo_realization_plan(
            bundle,
            self.config,
            self.repo_realization_plan(),
            self._repo_snapshots,
        )
        if layout_error is not None:
            raise RunPodDriverError(
                f"current remote layout no longer satisfies the committed lock: {layout_error}"
            )
        if named["runpod-remote-layout-vs-lock"].observed != layout_observed:
            raise RunPodDriverError(
                "runpod-remote-layout-vs-lock observation does not match current layout"
            )

        credentials_required = not self._provided_endpoint or bundle.deadman_enabled
        expected_credentials = (
            "verified" if credentials_required else "not-required-provided-endpoint"
        )
        if named["runpod-credentials"].observed != expected_credentials:
            raise RunPodDriverError("credential preflight observation is inconsistent")
        # Whether a balance floor was required is a pre-acquisition declaration:
        # an acquiring run observes a numeric balance at preflight. Reconstruct it
        # from the provided-target declarations so a restored acquired-pod _pod_id
        # does not retroactively flip this to the existing-target branch.
        balance_required = not self._provided_endpoint and not self._provided_pod
        balance = named["runpod-balance-floor"].observed
        if balance_required:
            if isinstance(balance, bool) or not isinstance(balance, (int, float)):
                raise RunPodDriverError("balance preflight observation is not numeric")
            if not math.isfinite(float(balance)) or balance < self.config.min_balance_usd:
                raise RunPodDriverError(
                    "balance preflight observation does not meet the current threshold"
                )
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
        realization_plan = state.repo_realization_plan or self.repo_realization_plan()
        if realization_plan is None:
            raise RunPodDriverError(
                "RunPod REALIZE_ENV requires a repo realization plan from PREFLIGHT"
            )
        _require_preflight_plan_digest(
            state.stage("PREFLIGHT").outputs.get("driver_evidence"),
            realization_plan.plan_digest,
        )
        self._repo_realization_plan = realization_plan
        try:
            self._repo_snapshots = restore_repo_snapshots(
                {name: entry.local_root for name, entry in realization_plan.repos.items()},
                realization_plan.snapshot_manifest,
                snapshot_parent=bundle.run_set_dir / ".repo-snapshots",
            )
        except RepoSnapshotError as exc:
            raise RunPodDriverError(str(exc)) from exc
        declaration_fingerprint = compute_runpod_environment_fingerprint(bundle, realization_plan)
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
            remote_root = realization_plan.repos[name].remote_root
            self._ssh(f"mkdir -p {_sq(remote_root)} && chmod -R u+w {_sq(remote_root)}")
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
        realized_fingerprint = self._probe_realized_environment(bundle, declaration_fingerprint)
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
        attempt_root = bundle.run_set_dir / ".stage-attempts" / f"stage-inputs-{attempt}"
        try:
            staged_inputs = materialize_bundle_inputs(
                bundle,
                destination_root=attempt_root,
                provider_bindings=self.input_provider_bindings,
                staged_root_bindings=self.staged_root_bindings,
            )
        except InputMaterializationError as exc:
            raise RunPodDriverError(str(exc)) from exc
        attempt_inputs = attempt_root / "inputs"
        payloads: list[dict[str, str]] = []
        payload_hashes: list[tuple[str, str]] = []
        if bundle.execution_family == "evaluation-matrix":
            bundle_data = bundle.model_dump_json(exclude_none=True).encode("utf-8")
            bundle_target = attempt_inputs / "run-bundle.json"
            bundle_target.write_bytes(bundle_data)
            payload_hashes.append((bundle_target.name, hashlib.sha256(bundle_data).hexdigest()))
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
        remote_attempt_root = f"{remote_run_dir}/.stage-attempts/stage-inputs-{attempt}"
        remote_attempt_inputs = f"{remote_attempt_root}/inputs"
        self._ssh(
            f"mkdir -p {_sq(remote_run_dir + '/.stage-attempts')} && "
            f"mkdir -- {_sq(remote_attempt_root)}"
        )
        with self._stage_inputs_heartbeat(bundle, remote_attempt_root):
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

    @contextmanager
    def _stage_inputs_heartbeat(
        self,
        bundle: RunBundle,
        remote_attempt_root: str,
    ) -> Iterator[None]:
        """Keep the deadman informed only while this host actively stages inputs."""
        if not bundle.deadman_enabled:
            yield
            return

        heartbeat_path = f"{remote_attempt_root}/.host-active"
        heartbeat_command = f"touch -- {_sq(heartbeat_path)}"
        self._ssh(heartbeat_command)
        stop = threading.Event()
        failures: list[Exception] = []
        interval = max(
            0.1,
            min(self.config.poll_seconds, bundle.deadman_silence_seconds / 3),
        )

        def heartbeat() -> None:
            while not stop.wait(interval):
                try:
                    self._ssh(heartbeat_command)
                except Exception as exc:
                    failures.append(exc)
                    stop.set()

        thread = threading.Thread(
            target=heartbeat,
            name=f"feedbax-stage-inputs-{bundle.run_set_id}",
            daemon=True,
        )
        thread.start()
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop.set()
            thread.join()
            if body_error is None and failures:
                raise RunPodDriverError("stage-inputs host heartbeat failed") from failures[0]

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Launch one row under nohup with sentinel files."""
        namespace = build_runpod_execution_namespace(
            bundle=bundle,
            row=row,
            remote_run_dir=self._remote_run_dir(bundle),
            remote_sentinel_dir=self._remote_sentinel_dir(bundle),
            env_fingerprint=state.environment_fingerprint or "",
        )
        command = build_launch_row_command(
            bundle=bundle,
            row=row,
            workdir=self._row_workdir(row),
            env_fingerprint=state.environment_fingerprint or "",
            jax_cache_dir=f"{self.config.volume_mount}/jax_cache",
            execution_namespace=namespace,
        )
        self._ssh(command)
        pid = self._read_remote_pid(bundle, row.row_id)
        return {"row_id": row.row_id, "pid": pid, "command": command}

    def smoke_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Run one bounded native execution in an authenticated scratch namespace."""
        if row.execution_family == "evaluation-matrix":
            return {
                "row_id": row.row_id,
                "status": "opted-out",
                "update_budget": bundle.smoke_update_budget,
                "payload_binding_status": "not-run",
                "cleanup_status": "not-created",
                "deadline_seconds": bundle.smoke_deadline_seconds,
                "opt_out_reason": (
                    "evaluation-matrix smoke is the provider-free production shadow; "
                    "RunPod launch reuses the same bound public harness command"
                ),
            }
        if not _row_uses_registered_native_execution(row):
            raise RunPodDriverError(
                f"remote smoke row {row.row_id!r} is not registered native execution"
            )
        provenance = row.execution.row_provenance
        if provenance is None:  # Kept explicit for type narrowing and fail-closed diagnostics.
            raise RunPodDriverError(f"remote smoke row {row.row_id!r} lacks provenance")
        namespace, producer_context = self._smoke_execution_context(bundle, row, state)
        scratch_root = namespace.row_root
        derived_run_id = namespace.run_identity
        protected_before = self._protected_path_content_digests(bundle, row)
        command = build_launch_row_command(
            bundle=bundle,
            row=row,
            workdir=self._row_workdir(row),
            env_fingerprint=state.environment_fingerprint or "",
            jax_cache_dir=f"{self.config.volume_mount}/jax_cache",
            execution_namespace=namespace,
            update_budget=bundle.smoke_update_budget,
        )
        status = "running"
        result: Mapping[str, Any] = {
            "start_completed_batches": 0,
            "end_completed_batches": None,
            "payload_binding_status": "not-run",
            "executor_result_sha256": None,
        }
        detail = ""
        cleanup = "not-created"
        try:
            cleanup = "failed"
            self._ssh(command)
            deadline = self._monotonic() + bundle.smoke_deadline_seconds
            while status not in {"completed", "failed", "deadline_exceeded"}:
                probe = parse_probe_report(
                    self._ssh(
                        build_probe_command(namespace.sentinel_dir, namespace.sentinel_stem)
                    ).stdout
                )
                status = str(
                    probe.get("rows", {}).get(namespace.sentinel_stem, {}).get("status", "pending")
                )
                if status in {"completed", "failed"}:
                    break
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    self._ssh(build_bounded_remote_termination_command(namespace))
                    status = "deadline_exceeded"
                    break
                self._sleep(min(self.poll_interval_seconds, remaining))
            if status == "completed":
                result = self._read_smoke_result(namespace)
            else:
                detail = self.transport.ssh(
                    f"tail -n 80 -- {_sq(namespace.log_path)}"
                ).stdout.strip()
        except RunPodDriverError as exc:
            status = "failed"
            detail = str(exc)
        finally:
            try:
                cleanup = self._cleanup_smoke_namespace(namespace, scratch_root)
            except RunPodDriverError:
                cleanup = "failed"

        protected_after = self._protected_path_content_digests(bundle, row)
        evidence = {
            "row_id": row.row_id,
            "status": "passed" if status == "completed" else "failed",
            "derived_run_id": derived_run_id,
            "planned_run_id": provenance.planned_run_id,
            "derived_producer_context": producer_context,
            "scratch_namespace": scratch_root,
            "update_budget": bundle.smoke_update_budget,
            "deadline_seconds": bundle.smoke_deadline_seconds,
            "payload_binding_status": result["payload_binding_status"],
            "start_completed_batches": result["start_completed_batches"],
            "end_completed_batches": result["end_completed_batches"],
            "executor_result_sha256": result["executor_result_sha256"],
            "protected_paths_before": protected_before,
            "protected_paths_after": protected_after,
            "cleanup_status": cleanup,
        }
        if protected_before != protected_after:
            evidence["status"] = "failed"
            raise RunPodRemoteSmokeError(
                f"remote smoke changed protected launch paths for row {row.row_id!r}",
                evidence=evidence,
            )
        if status != "completed" or cleanup != "removed":
            if status == "deadline_exceeded":
                message = (
                    f"remote smoke row {row.row_id!r} exceeded wall-clock deadline "
                    f"{bundle.smoke_deadline_seconds:g}s; failed sentinel recorded"
                )
            elif cleanup != "removed":
                message = f"remote smoke cleanup failed for row {row.row_id!r}"
            else:
                message = f"remote smoke row {row.row_id!r} failed"
                if detail:
                    message += f": {detail}"
            raise RunPodRemoteSmokeError(message, evidence=evidence)
        return evidence

    def smoke_failure_evidence(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
        error: Exception,
    ) -> Mapping[str, Any]:
        """Return schema-valid evidence when an unexpected smoke exception escapes."""
        del error
        namespace, producer_context = self._smoke_execution_context(bundle, row, state)
        provenance = row.execution.row_provenance
        if provenance is None:
            raise RunPodDriverError(f"remote smoke row {row.row_id!r} lacks provenance")
        return {
            "row_id": row.row_id,
            "status": "failed",
            "planned_run_id": provenance.planned_run_id,
            "derived_run_id": namespace.run_identity,
            "derived_producer_context": producer_context,
            "scratch_namespace": namespace.row_root,
            "start_completed_batches": 0,
            "end_completed_batches": None,
            "update_budget": bundle.smoke_update_budget,
            "payload_binding_status": "not-run",
            "executor_result_sha256": None,
            "protected_paths_before": {},
            "protected_paths_after": {},
            "cleanup_status": "failed",
            "deadline_seconds": bundle.smoke_deadline_seconds,
        }

    def _smoke_execution_context(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> tuple[RunPodExecutionNamespace, Mapping[str, Any]]:
        """Build the identity and scratch namespace shared by smoke and fallback evidence."""
        provenance = row.execution.row_provenance
        if provenance is None:
            raise RunPodDriverError(f"remote smoke row {row.row_id!r} lacks provenance")
        remote_run_dir = self._remote_run_dir(bundle)
        namespace = build_runpod_execution_namespace(
            bundle=bundle,
            row=row,
            remote_run_dir=remote_run_dir,
            remote_sentinel_dir=self._remote_sentinel_dir(bundle),
            env_fingerprint=state.environment_fingerprint or "",
            scratch_root=f"{remote_run_dir}/smoke/{row.row_id}",
            run_identity=f"{provenance.planned_run_id}--smoke",
            sentinel_stem=f"smoke-{row.row_id}",
        )
        execution_row = _execution_row(row, namespace)
        producer_execution = execution_row.execution.model_copy(
            update={
                "payload": execution_row.execution.payload.model_copy(
                    update={"uri": namespace.payload_path}
                )
            }
        )
        producer_context = NativeExecutionProducerContext(
            execution=producer_execution,
            environment_fingerprint=state.environment_fingerprint or "",
            collection_root=namespace.row_root,
        ).model_dump(mode="json", exclude_none=True)
        return namespace, producer_context

    def _protected_path_content_digests(
        self, bundle: RunBundle, row: RunRowSpec
    ) -> Mapping[str, str]:
        remote_run_dir = self._remote_run_dir(bundle)
        paths = {
            "staged_inputs": f"{remote_run_dir}/inputs",
            "row_roots": f"{remote_run_dir}/rows",
            "events": f"{remote_run_dir}/events",
            "sentinels": self._remote_sentinel_dir(bundle),
        }
        result = self._ssh(build_remote_content_digest_command(paths))
        payload = _json_object(result.stdout)
        return {str(key): str(value) for key, value in payload.items()}

    def _read_smoke_result(self, namespace: RunPodExecutionNamespace) -> Mapping[str, Any]:
        result = self._ssh(build_remote_smoke_result_command(namespace.log_path))
        payload = _json_object(result.stdout)
        return {
            "start_completed_batches": int(payload["start_completed_batches"]),
            "end_completed_batches": int(payload["end_completed_batches"]),
            "payload_binding_status": str(payload["payload_binding_status"]),
            "executor_result_sha256": str(payload["executor_result_sha256"]),
        }

    def _cleanup_smoke_namespace(
        self, namespace: RunPodExecutionNamespace, scratch_root: str
    ) -> str:
        stem = f"{namespace.sentinel_dir}/{namespace.sentinel_stem}"
        self._ssh(
            f"rm -rf -- {_sq(scratch_root)} && "
            f"rm -f -- {_sq(stem + '.started')} {_sq(stem + '.done')} "
            f"{_sq(stem + '.failed')} {_sq(stem + '.pid')}"
        )
        return "removed"

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
            elif "/" not in source:
                remote_source = f"{remote_run_dir}/rows/{row.row_id}/{source}"
            else:
                remote_source = f"{remote_run_dir}/{source}"
            raw_evaluation_source = f"{remote_run_dir}/rows/{row.row_id}/evaluation"
            if bundle.execution_family == "evaluation-matrix" and (
                posixpath.normpath(source) == "evaluation"
                or posixpath.normpath(remote_source) == posixpath.normpath(raw_evaluation_source)
            ):
                # Older bundles may declare the raw working store under
                # row-relative, run-relative, or absolute spellings. Certified
                # compact products are the terminal collection contract.
                continue
            target = dest_dir / Path(source).name
            source_kind = self._remote_collection_source_kind(remote_source)
            delete = False
            if source_kind == "directory":
                if os.path.lexists(target):
                    if target.is_symlink() or not target.is_dir():
                        raise RunPodDriverError(f"collection directory target is unsafe: {target}")
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
        if (
            is_native_training_command(row.launch.command)
            or row.execution_family == "evaluation-matrix"
        ):
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
        """Boundedly remove a run-owned pod or record why it remains unresolved."""
        if self.realized_capabilities.facts.teardown is TeardownSemantics.RESOURCES_PRESERVED:
            return {
                "driver": "runpod",
                "teardown": "skipped",
                "skip_reason": "realized-capability-preserves-resources",
                "capability_variant": self.realized_capabilities.variant_id,
            }
        ownership = self.teardown_ownership(state)
        if not ownership["owned_by_run"]:
            return {
                "driver": "runpod",
                "teardown": "skipped",
                "skip_reason": ownership["kind"],
                "ownership": ownership,
            }
        if bundle.keep_alive or not self.config.auto_teardown:
            return {
                "driver": "runpod",
                "teardown": "skipped",
                "skip_reason": "keep_alive" if bundle.keep_alive else "auto_teardown_disabled",
                "ownership": ownership,
            }
        pod_id = ownership["pod_id"]
        if not isinstance(pod_id, str):
            return {
                "driver": "runpod",
                "teardown": "no-pod",
                "ownership": ownership,
            }
        timeout = self._adopted_teardown_timeout_seconds
        self._adopted_teardown_timeout_seconds = None
        deadline = self._monotonic() + (
            self.config.teardown_absence_timeout_seconds if timeout is None else timeout
        )
        action = "remove-requested"
        try:
            result = self.transport.runpodctl(
                "remove", "pod", pod_id, timeout_seconds=self._teardown_remaining(deadline)
            )
            if result.returncode != 0:
                detail = (result.stderr.strip() or result.stdout.strip()).lower()
                if any(marker in detail for marker in _POD_NOT_FOUND_MARKERS):
                    action = "already-absent"
                else:
                    self.transport.runpodctl(
                        "stop", "pod", pod_id, timeout_seconds=self._teardown_remaining(deadline)
                    ).check("runpodctl stop pod")
                    self.transport.runpodctl(
                        "remove",
                        "pod",
                        pod_id,
                        timeout_seconds=self._teardown_remaining(deadline),
                    ).check("runpodctl remove pod after stop")
                    action = "stopped-then-removed"
            else:
                action = "removed"
            absence = self._wait_for_pod_absence(pod_id, deadline=deadline)
            self._pod_id = None
            self._endpoint = None
            remaining = deadline - self._monotonic()
            final_inventory = (
                self._observe_global_pod_inventory(timeout_seconds=remaining)
                if remaining > 0
                else {
                    "scope": "provider-account",
                    "verified": False,
                    "observed_at": utc_now().isoformat(),
                    "observation_basis": "runpodctl pod list --output json",
                    "outcome": "cleanup-deadline-expired",
                    "pod_count": None,
                    "pod_ids": [],
                }
            )
            return {
                "driver": "runpod",
                "teardown": action,
                "pod_id": pod_id,
                "ownership": ownership,
                "pod_absence": absence,
                "final_pod_inventory": final_inventory,
            }
        except Exception as exc:
            reason = _redact_secret(str(exc), self.config.api_key)
            outputs = {
                "driver": "runpod",
                "teardown": "unresolved",
                "pod_id": pod_id,
                "ownership": ownership,
                "pod_absence": {
                    "verified": False,
                    "pod_id": pod_id,
                    "terminal_observation": "unresolved",
                    "reason": reason,
                },
                "unresolved_owned_pod": {
                    "pod_id": pod_id,
                    "last_known_state": self._last_known_pod_state(state),
                    "reason": reason,
                },
            }
            raise RunPodTeardownError(
                f"owned pod {pod_id!r} teardown is unresolved: {reason}",
                teardown_outputs=outputs,
            ) from exc

    def teardown_ownership(self, state: RunSetState) -> dict[str, Any]:
        """Describe provider-resource ownership without performing side effects."""
        record = state.provision_record or {}
        pod_id = self._pod_id or record.get("pod_id")
        provided_endpoint = self._provided_endpoint or record.get("provided_endpoint") is True
        provided_pod = (
            False
            if record.get("teardown_allowed") is True
            else self._provided_pod or record.get("provided_pod") is True
        )
        if provided_endpoint:
            kind = "provided_endpoint"
        elif provided_pod:
            kind = "provided_pod"
        elif isinstance(pod_id, str) and pod_id:
            kind = "orchestration_created"
        else:
            kind = "none"
        owned_by_run = kind == "orchestration_created"
        return {
            "kind": kind,
            "owned_by_run": owned_by_run,
            "teardown_allowed": owned_by_run,
            "pod_id": pod_id if isinstance(pod_id, str) else None,
            "resource_id": pod_id if isinstance(pod_id, str) else None,
            "endpoint": (
                f"ssh://{self._endpoint.ip}:{self._endpoint.port}"
                if self._endpoint is not None
                else None
            ),
        }

    def has_pending_owned_resource(self) -> bool:
        """Return whether this process still knows an owned pod needing cleanup."""
        return bool(self._pod_id and not self._provided_pod and not self._provided_endpoint)

    def _teardown_remaining(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise RunPodDriverError("RunPod teardown cleanup deadline expired")
        return remaining

    def _last_known_pod_state(self, state: RunSetState) -> str:
        for record in (self._last_provision_pod, state.provision_record):
            if not isinstance(record, Mapping):
                continue
            for key in ("status", "desiredStatus", "desired_status"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    return value[:128]
        return "unknown"

    def _observe_global_pod_inventory(
        self, *, timeout_seconds: float | None = None
    ) -> Mapping[str, Any]:
        """Return sanitized evidence from the provider-wide RunPod pod inventory."""
        result = self.transport.runpodctl(
            "pod", "list", "--output", "json", timeout_seconds=timeout_seconds
        )
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
            pods = _parse_runpod_pod_inventory(result.stdout)
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
        pod_ids = [pod.pod_id for pod in pods]
        if pod_ids:
            return {
                "scope": "provider-account",
                "verified": False,
                "observed_at": observed_at,
                "observation_basis": basis,
                "outcome": "non-empty",
                "pod_count": len(pod_ids),
                "pod_ids": pod_ids,
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

    def observe_pod_inventory(
        self, *, timeout_seconds: float | None = None
    ) -> tuple[tuple[ProviderPodInventoryRecord, ...], Mapping[str, Any]]:
        """Return typed provider inventory plus bounded observation evidence."""
        result = self.transport.runpodctl(
            "pod", "list", "--output", "json", timeout_seconds=timeout_seconds
        )
        observed_at = utc_now().isoformat()
        basis = "runpodctl pod list --output json"
        if result.returncode != 0:
            raise RunPodDriverError(
                "RunPod inventory observation is unavailable for acquisition reconciliation"
            )
        try:
            records = _parse_runpod_pod_inventory(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RunPodDriverError(
                "RunPod inventory observation is invalid for acquisition reconciliation"
            ) from exc
        evidence = {
            "scope": "provider-account",
            "verified": not records,
            "observed_at": observed_at,
            "observation_basis": basis,
            "outcome": "non-empty" if records else "empty",
            "pod_count": len(records),
            "pod_ids": [record.pod_id for record in records],
        }
        return records, evidence

    def observe_global_resource_inventory(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[tuple[ProviderPodInventoryRecord, ...], Mapping[str, Any]]:
        """Observe the provider-wide inventory through the generic capability hook."""
        return self.observe_pod_inventory(timeout_seconds=timeout_seconds)

    def _wait_for_pod_absence(
        self, pod_id: str, *, deadline: float | None = None
    ) -> Mapping[str, Any]:
        """Boundedly prove that one exact orchestration-owned pod is absent."""
        deadline = (
            deadline
            if deadline is not None
            else self._monotonic() + self.config.teardown_absence_timeout_seconds
        )
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
                    f"ambiguous absence query for owned pod {pod_id!r}: observed id {observed_id!r}"
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
            f"timed out waiting for RunPod SSH endpoint after {self.config.max_acquire_seconds:g}s"
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
            auth
            + f"command -v runpodctl >/dev/null && runpodctl get pod {_sq(self._pod_id)} >/dev/null"
        ).check("in-pod runpodctl presence and authentication")
        self._ssh(
            auth
            + build_deadman_watchdog_command(
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
        intent_id: str | None = None,
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
        record = {
            "driver": self.driver_name,
            **facts,
            "pod_id": self._pod_id,
            "provided_pod": provided_pod,
            "provided_endpoint": False,
            "ssh_host": endpoint.ip,
            "ssh_port": endpoint.port,
            "teardown_allowed": not provided_pod,
        }
        if intent_id is not None:
            record["intent_id"] = intent_id
        return record

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
        realized = (
            self.transport.ssh(
                f"test -f {_sq(self._remote_fingerprint_path(bundle))} && "
                f"cat {_sq(self._remote_fingerprint_path(bundle))}"
            )
            .check("read realized RunPod environment fingerprint")
            .stdout.strip()
        )
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
        return _local_repos(self.config)

    def _remote_repos(self) -> Mapping[str, str]:
        return _remote_repos(self.config)

    def _primary_workdir(self) -> str:
        remote_repos = self._remote_repos()
        primary_repo = _primary_repo_name(self.config, remote_repos.keys())
        return remote_repos[primary_repo]

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
        "immutable_image_id": (str(immutable_image_id) if immutable_image_id is not None else None),
        "hourly_rate": hourly_rate,
        "hourly_rate_raw": raw_rate_observation,
        "billing_started_at": _canonical_runpod_timestamp(billing_started_at_raw),
        "billing_started_at_raw": billing_started_at_raw,
        "currency": "USD" if hourly_rate is not None else None,
        "provider_observation_basis": "runpodctl pod get response",
    }


def _preflight_continuation_schedule_consistency(
    bundle: RunBundle,
    input_provider_bindings: Sequence[InputProviderRootBinding],
    *,
    training_method_registry: TrainingMethodRegistry | None,
    input_bindings_valid: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Authenticate and compare continuation schedules without provider transport calls."""
    failures: list[str] = []
    observed: dict[str, Any] = {}
    continuation_rows: list[tuple[RunRowSpec, TrainingRunSpec]] = []
    for row in bundle.rows:
        try:
            run_spec = _authenticated_row_training_spec(row)
        except (OSError, ValueError) as exc:
            failures.append(f"{row.row_id}: {exc}")
            continue
        if run_spec is None or run_spec.checkpoint_progress.continuation is None:
            continue
        continuation_rows.append((row, run_spec))
    if failures or not continuation_rows:
        return failures, observed
    if training_method_registry is None:
        return ["continuation schedule validation requires a training method registry"], observed
    if not input_bindings_valid:
        return [], {
            "outcome": "skipped-due-to-dependency",
            "dependencies": ["input-provider-bindings"],
        }

    with tempfile.TemporaryDirectory(prefix="feedbax-continuation-preflight-") as temp_dir:
        try:
            materialize_bundle_inputs(
                bundle,
                destination_root=Path(temp_dir),
                provider_bindings=input_provider_bindings,
            )
        except Exception as exc:
            return [f"authenticated source checkpoint materialization failed: {exc}"], observed
        for row, target_run_spec in continuation_rows:
            continuation = target_run_spec.checkpoint_progress.continuation
            assert continuation is not None
            try:
                source = native_resume_checkpoint_source(bundle, row)
                if source is None:
                    raise ValueError(
                        "declared continuation has no exact authenticated resume checkpoint source"
                    )
                checkpoint_root = Path(temp_dir) / "inputs" / source.custody.target_role
                documents = load_checkpoint_custody_documents(checkpoint_root)
                manifest = documents.manifest.document
                validate_checkpoint_continuation_source_count(manifest, continuation)
                source_run_spec, _source_phase_program = (
                    authenticated_run_contract_source_projection(manifest)
                )
                row_failures, row_observed = compare_continuation_schedule_projections(
                    source_run_spec=source_run_spec,
                    target_run_spec=target_run_spec,
                    source_manifest=manifest,
                    continuation=continuation,
                    source_resolved_method=resolve_training_run_spec(
                        source_run_spec, training_method_registry
                    ),
                    target_resolved_method=resolve_training_run_spec(
                        target_run_spec, training_method_registry
                    ),
                )
            except Exception as exc:
                failures.append(f"{row.row_id}: {exc}")
                continue
            observed[row.row_id] = row_observed
            failures.extend(f"{row.row_id}: {failure}" for failure in row_failures)
    return failures, observed


def _authenticated_row_training_spec(row: RunRowSpec) -> TrainingRunSpec | None:
    """Load one digest-authenticated inline TrainingRunSpec, when present."""
    ref = row.execution.payload
    if ref.schema_id != "feedbax.spec.training_run":
        return None
    if ref.uri is None:
        raise ValueError("training execution payload has no local URI")
    data = Path(ref.uri).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != ref.sha256:
        raise ValueError(
            f"training execution payload digest mismatch; expected={ref.sha256} actual={actual}"
        )
    payload = json.loads(data)
    if not isinstance(payload, Mapping):
        raise ValueError("training execution payload must be a JSON object")
    if (
        payload.get("schema_id") != ref.schema_id
        or payload.get("schema_version") != ref.schema_version
    ):
        raise ValueError("training execution payload schema does not match its artifact ref")
    return TrainingRunSpec.model_validate(payload)


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


def _dependency_skipped_preflight_check(
    name: str,
    *dependencies: str,
) -> PreflightCheckEntry:
    """Record a dependency skip without expanding the durable check-status schema."""
    ordered_dependencies = tuple(dict.fromkeys(dependencies))
    detail = "skipped-due-to-dependency: " + ", ".join(ordered_dependencies)
    return PreflightCheckEntry(
        name=name,
        status="pass",
        detail=detail,
        observed=dependency_skip_observed(*ordered_dependencies),
    )


def _is_dependency_skipped_preflight_check(check: PreflightCheckEntry) -> bool:
    observed = check.observed
    return isinstance(observed, Mapping) and observed.get("outcome") == DEPENDENCY_SKIP_OUTCOME


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


def build_runpod_repo_realization_plan(
    bundle: RunBundle,
    config: RunPodDriverConfig,
    *,
    snapshot_parent: Path | str,
) -> tuple[RepoRealizationPlan, SealedRepoSnapshots]:
    """Resolve all local repository identities once against sealed transfer bytes."""
    local_repos = dict(_local_repos(config))
    remote_repos = dict(_remote_repos(config))
    if set(local_repos) != set(remote_repos):
        raise RunPodDriverError(
            "local/remote repo key mismatch: "
            f"local={sorted(local_repos)!r}, remote={sorted(remote_repos)!r}"
        )
    primary_repo = _primary_repo_name(config, local_repos)
    normalized_remote_roots = validate_non_overlapping_remote_roots(remote_repos)
    lock_paths = {primary_repo: sorted(bundle.environment.lockfile_hashes)}
    expected_lock_digests = {primary_repo: dict(sorted(bundle.environment.lockfile_hashes.items()))}
    manifest, components = seal_local_repo_realizations(
        local_repos,
        lock_relative_paths=lock_paths,
        expected_lock_digests=expected_lock_digests,
        snapshot_parent=snapshot_parent,
    )
    snapshots = SealedRepoSnapshots(
        manifest=manifest,
        snapshots={name: component.snapshot for name, component in components.items()},
    )
    local_roots = {name: Path(root).expanduser().resolve() for name, root in local_repos.items()}
    resolutions: list[EditableSourceResolution] = []
    for lock_relative_path in sorted(bundle.environment.lockfile_hashes):
        component = components[primary_repo]
        lock_path = component.snapshot.staging_root / lock_relative_path
        lock_bytes = read_sealed_lock_bytes(component.snapshot.staging_root, lock_relative_path)
        try:
            document = tomllib.loads(lock_bytes.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise RunPodDriverError(
                f"malformed TOML in sealed lockfile {lock_path}: {exc}"
            ) from exc
        for form, spelling in _local_lock_sources(document, lock_path=lock_path):
            source_path = PurePosixPath(spelling)
            if source_path.is_absolute():
                raise RunPodDriverError(
                    f"absolute {form} source {spelling!r} in {lock_path} is unsupported"
                )
            if not spelling or spelling != source_path.as_posix():
                raise RunPodDriverError(
                    f"non-canonical {form} source spelling {spelling!r} in {lock_path}"
                )
            resolved_target = (local_roots[primary_repo] / Path(*source_path.parts)).resolve()
            matches = [
                name
                for name, repo_root in local_roots.items()
                if resolved_target == repo_root or resolved_target.is_relative_to(repo_root)
            ]
            if len(matches) != 1:
                qualifier = "ambiguous" if matches else "unmatched"
                roots = ", ".join(f"{name}={root}" for name, root in sorted(local_roots.items()))
                raise RunPodDriverError(
                    f"{qualifier} local target for lock source {spelling!r}: consumer "
                    f"{local_roots[primary_repo]} resolves to {resolved_target}; "
                    f"configured local repos are {roots}"
                )
            target_repo = matches[0]
            target_subpath = resolved_target.relative_to(local_roots[target_repo]).as_posix()
            resolutions.append(
                EditableSourceResolution(
                    consumer_repo=primary_repo,
                    lock_relative_path=lock_relative_path,
                    source_form=form,
                    spelling=spelling,
                    target_repo=target_repo,
                    target_subpath=target_subpath,
                )
            )
    entries = {
        name: RepoRealizationEntry(
            local_root=str(component.snapshot.source_root),
            staging_root=str(component.snapshot.staging_root),
            remote_root=normalized_remote_roots[name],
            snapshot=component.snapshot.record,
            sealed_lock_digests=dict(component.sealed_lock_digests),
        )
        for name, component in components.items()
    }
    plan = RepoRealizationPlan.create(
        primary_repo=primary_repo,
        repos=entries,
        editable_source_resolutions=resolutions,
        snapshot_manifest=manifest,
    )
    error, _observed = validate_runpod_repo_realization_plan(bundle, config, plan, snapshots)
    if error is not None:
        raise RunPodDriverError(error)
    return plan, snapshots


def validate_runpod_repo_realization_plan(
    bundle: RunBundle,
    config: RunPodDriverConfig,
    plan: RepoRealizationPlan | None,
    snapshots: SealedRepoSnapshots | None,
) -> tuple[str | None, Mapping[str, Any]]:
    """Validate a recorded plan against sealed roots and the keyed remote layout."""
    observed: dict[str, Any] = {"path_sources": []}
    if plan is None or snapshots is None:
        return "repo realization plan or sealed snapshots are unavailable", observed
    local_keys = set(_local_repos(config))
    remote_repos = dict(_remote_repos(config))
    if set(plan.repos) != local_keys or set(plan.repos) != set(remote_repos):
        return "repo realization plan keys no longer match configured repos", observed
    try:
        primary_repo = _primary_repo_name(config, local_keys)
        normalized_remote_roots = validate_non_overlapping_remote_roots(remote_repos)
    except (RunPodDriverError, RepoRealizationError, ValueError) as exc:
        return str(exc), observed
    if plan.primary_repo != primary_repo:
        return "repo realization plan primary repo no longer matches configuration", observed
    if {name: entry.remote_root for name, entry in plan.repos.items()} != normalized_remote_roots:
        return "repo realization plan remote roots no longer match configuration", observed
    declared = dict(sorted(bundle.environment.lockfile_hashes.items()))
    if plan.repos[primary_repo].sealed_lock_digests != declared:
        return "sealed lock digests do not match the environment declaration", observed
    if any(entry.sealed_lock_digests for name, entry in plan.repos.items() if name != primary_repo):
        return "non-primary repo carries undeclared sealed lock digests", observed

    remote_consumer_root = normalized_remote_roots[primary_repo]
    lock_paths = {
        posixpath.normpath(posixpath.join(remote_consumer_root, relative_path))
        for relative_path in declared
    }
    for remote_file, _patch_from, _patch_to in config.path_patches:
        normalized_file = posixpath.normpath(remote_file)
        basename = PurePosixPath(normalized_file).name.lower()
        dependency_target = (
            normalized_file in lock_paths
            or basename in _DEPENDENCY_FILE_NAMES
            or basename.startswith("requirements")
            and basename.endswith((".txt", ".in"))
        )
        if dependency_target:
            return (
                f"path_patches may not target a lock or dependency file: {remote_file!r}",
                observed,
            )

    evidence: list[dict[str, str]] = []
    for resolution in plan.editable_source_resolutions:
        target_snapshot = snapshots.snapshots[resolution.target_repo]
        target = target_snapshot.staging_root.joinpath(
            *PurePosixPath(resolution.target_subpath).parts
        )
        if not target.exists() and not target.is_symlink():
            return (
                f"recorded lock resolution target is absent from sealed repo "
                f"{resolution.target_repo!r}: {resolution.target_subpath!r}",
                observed,
            )
        planned_from_spelling = posixpath.normpath(
            posixpath.join(remote_consumer_root, resolution.spelling)
        )
        keyed_remote_target = posixpath.normpath(
            posixpath.join(
                normalized_remote_roots[resolution.target_repo],
                resolution.target_subpath,
            )
        )
        if planned_from_spelling != keyed_remote_target:
            return (
                f"lock source {resolution.spelling!r} resolves to remote path "
                f"{planned_from_spelling!r}, not keyed target {keyed_remote_target!r}",
                observed,
            )
        evidence.append(
            {
                "consumer_repo": resolution.consumer_repo,
                "lockfile": resolution.lock_relative_path,
                "form": resolution.source_form,
                "spelling": resolution.spelling,
                "target_repo": resolution.target_repo,
                "target_subpath": resolution.target_subpath,
                "planned_remote_target": keyed_remote_target,
            }
        )
    observed.update(
        {
            "primary_repo": primary_repo,
            "local_repo_keys": sorted(local_keys),
            "remote_repos": normalized_remote_roots,
            "lockfile_hashes": declared,
            "path_sources": evidence,
            "repo_realization_plan_digest": plan.plan_digest,
        }
    )
    return None, observed


def _local_lock_sources(
    document: Mapping[str, Any], *, lock_path: Path
) -> tuple[tuple[str, str], ...]:
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise RunPodDriverError(
            f"unsupported lock content in {lock_path}: integer version is required"
        )
    if version not in _SUPPORTED_UV_LOCK_VERSIONS:
        raise RunPodDriverError(
            f"unsupported lock version {version!r} in {lock_path}; "
            f"supported versions are {sorted(_SUPPORTED_UV_LOCK_VERSIONS)!r}"
        )
    packages = document.get("package", [])
    if not isinstance(packages, list) or any(not isinstance(item, Mapping) for item in packages):
        raise RunPodDriverError(
            f"unsupported lock content in {lock_path}: package must be an array of tables"
        )

    sources: list[tuple[str, str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            local_forms = set(value) & (
                _SUPPORTED_LOCAL_LOCK_SOURCE_FORMS | _UNSUPPORTED_LOCAL_LOCK_SOURCE_FORMS
            )
            if local_forms:
                unsupported = local_forms & _UNSUPPORTED_LOCAL_LOCK_SOURCE_FORMS
                if unsupported:
                    form = sorted(unsupported)[0]
                    raise RunPodDriverError(
                        f"unsupported local source form {form!r} in {lock_path}"
                    )
                if len(local_forms) != 1:
                    raise RunPodDriverError(
                        f"ambiguous local source forms {sorted(local_forms)!r} in {lock_path}"
                    )
                form = next(iter(local_forms))
                spelling = value[form]
                if not isinstance(spelling, str):
                    raise RunPodDriverError(
                        f"unsupported {form!r} source value in {lock_path}: {spelling!r}"
                    )
                sources.append((form, spelling))
            for key, child in value.items():
                if key in local_forms:
                    continue
                if key == "source":
                    if not isinstance(child, Mapping) or len(child) != 1:
                        forms = (
                            sorted(child) if isinstance(child, Mapping) else [type(child).__name__]
                        )
                        raise RunPodDriverError(
                            f"unsupported mixed source forms {forms!r} in {lock_path}"
                        )
                    form = next(iter(child))
                    known_forms = (
                        _SUPPORTED_LOCAL_LOCK_SOURCE_FORMS
                        | _UNSUPPORTED_LOCAL_LOCK_SOURCE_FORMS
                        | _SUPPORTED_REMOTE_LOCK_SOURCE_FORMS
                    )
                    if form not in known_forms:
                        raise RunPodDriverError(f"unsupported source form {form!r} in {lock_path}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return tuple(dict.fromkeys(sources))


def _local_repos(config: RunPodDriverConfig) -> Mapping[str, Path | str]:
    return config.local_repos or {"feedbax": Path.cwd()}


def _remote_repos(config: RunPodDriverConfig) -> Mapping[str, str]:
    return config.remote_repos or {"feedbax": f"{config.remote_repo_root}/feedbax"}


def _primary_repo_name(config: RunPodDriverConfig, repo_keys: Iterable[str]) -> str:
    keys = set(repo_keys)
    if config.primary_repo is not None:
        if config.primary_repo not in keys:
            raise RunPodDriverError(
                f"primary_repo {config.primary_repo!r} is not a configured repo key: "
                f"{sorted(keys)!r}"
            )
        return config.primary_repo
    preferred = [name for name in ("rlrmp2", "rlrmp", "feedbax") if name in keys]
    if len(preferred) == 1:
        return preferred[0]
    if len(keys) == 1:
        return next(iter(keys))
    raise RunPodDriverError(
        "RunPod primary repo is ambiguous; set RunPodDriverConfig.primary_repo for keys "
        f"{sorted(keys)!r}"
    )


def require_deterministic_runpod_environment(bundle: RunBundle) -> None:
    """Reject RunPod declarations that cannot realize an immutable environment."""
    if not is_immutable_runpod_image_id(bundle.environment.image_id):
        raise RunPodDriverError(
            "RunPod REALIZE_ENV requires environment.image_id pinned by @sha256:<64 hex>"
        )
    lockfile_error = runpod_lockfile_declaration_error(bundle.environment.lockfile_hashes)
    if lockfile_error is not None:
        raise RunPodDriverError(lockfile_error)
    if not bundle.environment.python_version:
        raise RunPodDriverError("RunPod REALIZE_ENV requires environment.python_version")


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


def _require_preflight_plan_digest(evidence: Any, plan_digest: str) -> None:
    """Fail closed unless persisted preflight evidence binds the exact plan."""
    if not isinstance(evidence, Mapping):
        raise RunPodDriverError("PREFLIGHT evidence lacks repo realization plan binding")
    payload: Mapping[str, Any] = evidence
    if evidence.get("schema_id") == RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_ID:
        nested = evidence.get("base")
        if not isinstance(nested, Mapping):
            raise RunPodDriverError("matrix PREFLIGHT evidence lacks its base payload")
        payload = nested
    observed = payload.get("repo_realization_plan_digest")
    if observed != plan_digest:
        raise RunPodDriverError(
            "repo realization plan digest mismatch between PREFLIGHT and REALIZE_ENV: "
            f"expected {plan_digest}, observed {observed!r}"
        )


def compute_runpod_environment_fingerprint(
    bundle: RunBundle,
    realization_plan: RepoRealizationPlan,
) -> str:
    """Compute the declared remote environment fingerprint."""
    payload = environment_declaration_identity_projection(bundle.environment)
    payload["repo_realization_plan_digest"] = realization_plan.plan_digest
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
    return f"python3 -c {_sq(_REMOTE_ATOMIC_DIRECTORY_PUBLISH)} {_sq(source)} {_sq(destination)}"


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
    if not is_native_training_command(normalized) and "matrix-harness" not in normalized:
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


def _row_launch_command_parts(row: RunRowSpec) -> list[str]:
    return (
        [str(part) for part in row.launch.command]
        if row.launch.command
        else ["uv", "run", "--no-sync", "python", row.launch.entry or ""]
    )


def _row_uses_registered_native_execution(row: RunRowSpec) -> bool:
    return (
        row.launch.payload_routing.get("kind") == "registered-execution-payload"
        and is_native_training_command(_row_launch_command_parts(row))
        and row.execution.row_provenance is not None
    )


def build_runpod_execution_namespace(
    *,
    bundle: RunBundle,
    row: RunRowSpec,
    remote_run_dir: str,
    remote_sentinel_dir: str,
    env_fingerprint: str,
    scratch_root: str | None = None,
    run_identity: str | None = None,
    sentinel_stem: str | None = None,
) -> RunPodExecutionNamespace:
    """Construct the sole writable-path contract for a real or smoke execution."""
    provenance = row.execution.row_provenance
    row_root = scratch_root or f"{remote_run_dir}/rows/{row.row_id}"
    checkpoint_source = native_resume_checkpoint_source(bundle, row)
    seed_source = (
        f"{remote_run_dir}/inputs/{checkpoint_source.custody.target_role}"
        if checkpoint_source is not None
        else None
    )
    stem = sentinel_stem or row.row_id
    return RunPodExecutionNamespace(
        row_root=row_root,
        manifest_root=f"{row_root}/manifests",
        checkpoint_root=f"{row_root}/checkpoints",
        events_dir=(
            f"{row_root}/events" if scratch_root is not None else f"{remote_run_dir}/events"
        ),
        sentinel_dir=remote_sentinel_dir,
        log_path=(
            f"{row_root}/smoke.log"
            if scratch_root is not None
            else f"{remote_run_dir}/logs/{row.row_id}.log"
        ),
        payload_path=f"{remote_run_dir}/inputs/{row.row_id}.json",
        run_identity=run_identity
        or (provenance.planned_run_id if provenance is not None else row.row_id),
        sentinel_stem=stem,
        seed_source=seed_source,
        seed_attempt=f"{row_root}/.checkpoint-seed-attempt" if seed_source else None,
        seed_target=f"{row_root}/checkpoints" if seed_source else None,
        env_exports=(
            ("FEEDBAX_RUN_SET_ID", bundle.run_set_id),
            ("FEEDBAX_ROW_ID", row.row_id),
            (
                "FEEDBAX_RUN_EVENTS_DIR",
                f"{row_root}/events" if scratch_root else f"{remote_run_dir}/events",
            ),
            ("FEEDBAX_ENV_FINGERPRINT", env_fingerprint),
            ("FEEDBAX_ROW_DIR", row_root),
        ),
    )


def _execution_row(row: RunRowSpec, namespace: RunPodExecutionNamespace) -> RunRowSpec:
    """Derive producer context identity without mutating the real planned identity."""
    provenance = row.execution.row_provenance
    if not _row_uses_registered_native_execution(row):
        return row
    if provenance is None:
        raise RunPodDriverError(f"native execution row {row.row_id!r} lacks provenance")
    if namespace.run_identity == provenance.planned_run_id:
        return row
    return row.model_copy(
        update={
            "execution": row.execution.model_copy(
                update={
                    "row_provenance": provenance.model_copy(
                        update={"planned_run_id": namespace.run_identity}
                    )
                }
            )
        }
    )


def build_launch_row_command(
    *,
    bundle: RunBundle,
    row: RunRowSpec,
    workdir: str,
    env_fingerprint: str,
    jax_cache_dir: str,
    execution_namespace: RunPodExecutionNamespace,
    update_budget: int | None = None,
) -> str:
    """Build the row launch command with RunPod sentinel and event exports."""
    namespace = execution_namespace
    stem = f"{namespace.sentinel_dir}/{namespace.sentinel_stem}"
    done_file = f"{stem}.done"
    failed_file = f"{stem}.failed"
    started_file = f"{stem}.started"
    pid_file = f"{stem}.pid"
    log_file = namespace.log_path
    events_dir = namespace.events_dir
    row_dir = namespace.row_root
    checkpoint_source = native_resume_checkpoint_source(bundle, row)
    command_parts = _row_launch_command_parts(row)
    execution_row = _execution_row(row, namespace)
    command_parts, execution_row = executor_family_adapter(row.execution_family).bind_command(
        command_parts,
        bundle=bundle,
        row=execution_row,
        payload_path=namespace.payload_path,
        collection_root=row_dir,
        inputs_root=str(PurePosixPath(namespace.payload_path).parent),
        repo_root=workdir,
        environment_fingerprint=env_fingerprint,
        update_budget=update_budget,
        native_context_injector=(
            inject_native_execution_context if row.execution_family == "native-training" else None
        ),
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
        + " ".join(f"{key}={_sq(value)}" for key, value in namespace.env_exports)
        + " && "
        f'( {command} ) & child=$!; wait "$child"; rc=$?; child=; '
        f'if [ "$rc" -eq 0 ]; then success=1; touch {_sq(done_file)}; '
        f'else touch {_sq(failed_file)}; exit "$rc"; fi'
    )
    seed_command = ""
    if checkpoint_source is not None:
        assert namespace.seed_source is not None
        assert namespace.seed_attempt is not None
        assert namespace.seed_target is not None
        seed_command = (
            f"{build_native_resume_seed_command(namespace.seed_source, namespace.seed_attempt, namespace.seed_target, checkpoint_source)} "
            "&& "
        )
    return (
        f"mkdir -p {_sq(namespace.sentinel_dir)} {_sq(str(PurePosixPath(log_file).parent))} "
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
        f'i=0; while [ ! -s {_sq(pid_file)} ] && [ "$i" -lt 40 ]; do '
        "i=$((i+1)); sleep 0.05; done; "
        f"[ -s {_sq(pid_file)} ]"
    )


def build_bounded_remote_termination_command(namespace: RunPodExecutionNamespace) -> str:
    """Build bounded process-group TERM-to-KILL escalation for a smoke deadline."""
    stem = f"{namespace.sentinel_dir}/{namespace.sentinel_stem}"
    pid_file = f"{stem}.pid"
    return (
        f"pid=$(cat {_sq(pid_file)} 2>/dev/null || true); "
        'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then '
        'kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; '
        'i=0; while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do '
        "i=$((i+1)); sleep 0.05; done; "
        'kill -KILL -- "-$pid" 2>/dev/null || '
        'kill -KILL "$pid" 2>/dev/null || true; fi; '
        f"touch {_sq(stem + '.failed')}"
    )


def build_remote_content_digest_command(paths: Mapping[str, str]) -> str:
    """Hash path names, entry kinds, symlink targets, and file bytes, not metadata."""
    script = r"""
import hashlib,json,os,sys
paths=json.loads(sys.argv[1])
def digest(root):
    h=hashlib.sha256()
    if not os.path.lexists(root):
        h.update(b"missing\0")
        return h.hexdigest()
    if os.path.isfile(root) and not os.path.islink(root):
        entries=[("file",".",root)]
    else:
        entries=[]
        for base,dirs,files in os.walk(root,followlinks=False):
            dirs.sort(); files.sort()
            for name in dirs+files:
                path=os.path.join(base,name)
                rel=os.path.relpath(path,root)
                kind="link" if os.path.islink(path) else ("dir" if os.path.isdir(path) else "file")
                entries.append((kind,rel,path))
    for kind,rel,path in entries:
        h.update(kind.encode()+b"\0"+rel.encode()+b"\0")
        if kind=="file":
            with open(path,"rb") as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b""): h.update(chunk)
        elif kind=="link": h.update(os.readlink(path).encode())
        h.update(b"\0")
    return h.hexdigest()
print(json.dumps({name:digest(path) for name,path in paths.items()},sort_keys=True))
""".strip()
    payload = json.dumps(dict(paths), sort_keys=True, separators=(",", ":"))
    return f"python3 -c {_sq(script)} {_sq(payload)}"


def build_remote_smoke_result_command(log_path: str) -> str:
    """Extract and hash the native executor's JSON result from its smoke log."""
    script = r"""
import hashlib,json,sys
text=open(sys.argv[1],encoding="utf-8",errors="replace").read()
decoder=json.JSONDecoder(); candidates=[]
for index,char in enumerate(text):
    if char!="{": continue
    try: value,_=decoder.raw_decode(text[index:])
    except json.JSONDecodeError: continue
    if isinstance(value,dict) and {"start_completed_batches","end_completed_batches"}<=value.keys():
        candidates.append(value)
if not candidates: raise RuntimeError("smoke executor log lacks a typed result")
result=candidates[-1]
canonical=json.dumps(result,sort_keys=True,separators=(",",":")).encode()
print(json.dumps({
    "start_completed_batches":result["start_completed_batches"],
    "end_completed_batches":result["end_completed_batches"],
    "payload_binding_status":result.get("payload_binding_status"),
    "executor_result_sha256":hashlib.sha256(canonical).hexdigest(),
},sort_keys=True))
""".strip()
    return f"python3 -c {_sq(script)} {_sq(log_path)}"


def runpod_row_workdir(config: RunPodDriverConfig, row: RunRowSpec) -> str:
    """Resolve the same row workdir used by live and dry-run RunPod launches."""
    workdir = row.launch.metadata.get("workdir")
    if workdir:
        return str(workdir)
    remote_repos = _remote_repos(config)
    return remote_repos[_primary_repo_name(config, remote_repos.keys())]


def dry_run_launch_bundle(
    bundle: RunBundle,
    config: RunPodDriverConfig,
    input_provider_bindings: Sequence[InputProviderRootBinding] = (),
    staged_root_bindings: Sequence[StagedRootSnapshotBinding] = (),
) -> tuple[str, ...]:
    """Bind all RunPod launch rows without constructing a transport."""
    failures, _ = preflight_bundle_input_bindings(
        bundle,
        provider_bindings=input_provider_bindings,
        staged_root_bindings=staged_root_bindings,
    )
    if failures:
        raise RunPodDriverError("; ".join(failures))
    remote_run_dir = f"{config.remote_run_root.rstrip('/')}/{bundle.run_set_id}"
    remote_sentinel_dir = f"{remote_run_dir}/sentinels"
    return tuple(
        build_launch_row_command(
            bundle=bundle,
            row=row,
            workdir=runpod_row_workdir(config, row),
            env_fingerprint="dry-run-unrealized-environment",
            jax_cache_dir=f"{config.volume_mount}/jax_cache",
            execution_namespace=build_runpod_execution_namespace(
                bundle=bundle,
                row=row,
                remote_run_dir=remote_run_dir,
                remote_sentinel_dir=remote_sentinel_dir,
                env_fingerprint="dry-run-unrealized-environment",
            ),
        )
        for row in bundle.rows
    )


def runpod_config_for_bundle(
    bundle: RunBundle,
    *,
    api_key: str | None = None,
) -> RunPodDriverConfig:
    """Derive RunPod configuration from a validated bundle and explicit credential."""
    metadata = bundle.environment.metadata
    resources = bundle.deployment_policy.resources
    raw_patches = metadata.get("runpod_path_patches", ())
    path_patches = tuple(
        (str(item["remote_file"]), str(item["from"]), str(item["to"])) for item in raw_patches
    )
    return RunPodDriverConfig(
        pod_id=_string_or_none(metadata.get("runpod_pod_id")),
        ssh_host=_string_or_none(metadata.get("runpod_ssh_host")),
        ssh_port=_int_or_none(metadata.get("runpod_ssh_port")),
        gpu_id=resources.gpu_id,
        datacenters=tuple(resources.regions),
        api_key=api_key,
        min_balance_usd=float(metadata.get("runpod_min_balance_usd", 5.0)),
        image=bundle.environment.image_id or "runpod/pytorch:latest",
        local_repos={
            str(name): str(path) for name, path in metadata.get("runpod_local_repos", {}).items()
        },
        remote_repos={
            str(name): str(path) for name, path in metadata.get("runpod_remote_repos", {}).items()
        },
        primary_repo=_string_or_none(metadata.get("runpod_primary_repo")),
        protected_refs={
            str(name): str(ref) for name, ref in metadata.get("runpod_protected_refs", {}).items()
        },
        path_patches=path_patches,
    )


def runpod_driver_registration() -> DriverRegistration:
    """Return the context-sensitive built-in RunPod registration."""

    def config_for(context: DriverConstructionContext) -> RunPodDriverConfig:
        configuration = context.configuration
        explicit = configuration.get("driver_config")
        if explicit is not None:
            if not isinstance(explicit, RunPodDriverConfig):
                raise TypeError("runpod driver_config must be a RunPodDriverConfig")
            api_key = context.credentials.get("runpod_api_key")
            return replace(explicit, api_key=api_key) if api_key is not None else explicit
        bundle = configuration.get("bundle")
        if not isinstance(bundle, RunBundle):
            raise TypeError("runpod driver configuration requires a RunBundle")
        return runpod_config_for_bundle(
            bundle,
            api_key=context.credentials.get("runpod_api_key"),
        )

    def resolve(context: DriverConstructionContext):
        config = config_for(context)
        preserve_policy = context.configuration.get("preserve_owned_resources", False)
        if not isinstance(preserve_policy, bool):
            raise TypeError("preserve_owned_resources must be a bool")
        preserve_owned = not config.auto_teardown or preserve_policy
        if config.pod_id is not None or (config.ssh_host is not None and config.ssh_port):
            variant = "externally-managed"
        elif preserve_owned:
            variant = "engine-acquired-preserved"
        else:
            variant = "engine-acquired"
        return RunPodOrchestrationDriver.capability_envelope.realize(variant)

    def factory(context: DriverConstructionContext, realized):
        runtime = context.runtime_bindings
        if runtime.get("native_update_budget") is not None:
            raise ValueError(
                "remote capability variants do not support a local native update budget"
            )
        driver = RunPodOrchestrationDriver(
            config=config_for(context),
            transport=runtime.get("transport"),
            sleep=runtime.get("sleep", time.sleep),
            monotonic=runtime.get("monotonic", time.monotonic),
            input_provider_bindings=runtime.get("input_provider_bindings", ()),
            staged_root_bindings=runtime.get("staged_root_bindings", ()),
            collection_recovery_bindings=runtime.get("collection_recovery_bindings", ()),
            training_method_registry=runtime.get("training_method_registry"),
            realized_capabilities=realized,
        )
        if driver.realized_capabilities != realized:
            raise ValueError("RunPod factory realized a variant inconsistent with its context")
        return driver

    return DriverRegistration(
        name="runpod",
        supported_capabilities=RunPodOrchestrationDriver.capability_envelope,
        resolve_capabilities=resolve,
        factory=factory,
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
    installed_file = f"{remote_run_dir}/deadman.installed"
    script = (
        f"echo $$ > {_sq(pid_file)}; : > {_sq(installed_file)}; "
        f"pod_id={_sq(pod_id)}; run_dir={_sq(remote_run_dir)}; "
        f"sdir={_sq(remote_sentinel_dir)}; edir={_sq(events_dir)}; "
        f"silence={int(silence_seconds)}; warning={_sq(warning)}; "
        f"installed={_sq(installed_file)}; "
        "while true; do "
        'live=0; now=$(date +%s); newest=$(stat -c %Y "$installed" 2>/dev/null '
        '|| stat -f %m "$installed" 2>/dev/null || echo 0); '
        'for started in "$sdir"/*.started; do [ -e "$started" ] || continue; '
        'base=${started%.started}; [ -f "$base.done" ] || [ -f "$base.failed" ] || live=1; done; '
        'for path in "$edir"/*.jsonl "$sdir"/*; do [ -e "$path" ] || continue; '
        'mtime=$(stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || echo 0); '
        '[ "$mtime" -gt "$newest" ] && newest=$mtime; done; '
        "while IFS= read -r path; do "
        'mtime=$(stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || echo 0); '
        '[ "$mtime" -gt "$newest" ] && newest=$mtime; '
        'done < <(find "$run_dir/.stage-attempts" -type f -print 2>/dev/null); '
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
        f">>{_sq(remote_run_dir + '/logs/deadman.log')} 2>&1; "
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
    if (
        payload.get("schema_id") != ref.schema_id
        or payload.get("schema_version") != ref.schema_version
    ):
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


_NO_CAPACITY_MESSAGE_SUBSTRINGS = (
    "does not have the resources to deploy your pod",
    "no longer any instances available",
)


def _iter_structured_error_messages(text: str) -> Iterable[str]:
    """Yield RunPod ``error`` field values from a JSON-lines create response.

    ``runpodctl`` create failures interleave JSON error objects with plain-text
    usage/log lines, so each line is parsed independently rather than the
    whole stream at once.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            message = payload.get("error")
            if isinstance(message, str):
                yield message


def _is_no_capacity_create_response(stdout: str, stderr: str) -> bool:
    """Definitive provider rejection: no capacity, so no pod was created.

    Matches structurally on the provider's JSON ``error`` field rather than
    full-string equality against a specific message, since RunPod's wording
    varies by datacenter (e.g. "does not have the resources to deploy your
    pod" vs. "no longer any instances available"). Only structured error
    objects are considered so an incidental substring match in unparseable or
    lost-transport output stays ambiguous (fail-closed).
    """
    return any(
        substring in message
        for stream in (stdout, stderr)
        for message in _iter_structured_error_messages(stream)
        for substring in _NO_CAPACITY_MESSAGE_SUBSTRINGS
    )


def _classify_create_failure(
    result: CommandResult, secret: str | None
) -> tuple[Literal["retryable", "non-retryable"], str]:
    """Classify only sanitized provider authorization and request failures."""
    stdout = _redact_secret(result.stdout, secret)
    stderr = _redact_secret(result.stderr, secret)
    detail = (stderr or stdout).strip()
    if _is_no_capacity_create_response(stdout, stderr):
        return "non-retryable", detail
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
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return CommandResult(127, "", str(exc))
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, _stderr = _terminate_process_group(process)
        return CommandResult(124, stdout, f"timed out after {timeout_seconds:g}s")
    except BaseException:
        _terminate_process_group(process)
        raise
    return CommandResult(process.returncode, stdout, stderr)


def _terminate_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Boundedly terminate one supervised child process group and drain output."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=_CHILD_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate(timeout=_CHILD_TERMINATION_GRACE_SECONDS)


def _json_object(payload: str) -> dict[str, Any]:
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RunPodDriverError(f"invalid JSON payload: {exc}") from exc
    if isinstance(loaded, Mapping):
        return dict(loaded)
    raise RunPodDriverError("expected JSON object")


def _parse_runpod_pod_inventory(payload: str) -> tuple[ProviderPodInventoryRecord, ...]:
    """Parse supported inventory shapes without discarding provider pod names."""
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

    records: list[ProviderPodInventoryRecord] = []
    for pod in pods:
        if not isinstance(pod, Mapping):
            raise TypeError("RunPod inventory entries must be objects")
        candidates = [pod.get("id"), pod.get("podId")]
        nested = pod.get("pod")
        if isinstance(nested, Mapping):
            candidates.append(nested.get("id"))
        identities = {value for value in candidates if isinstance(value, str) and value}
        if len(identities) != 1:
            raise ValueError("RunPod inventory entry has ambiguous pod identity")
        pod_id = identities.pop()
        if _SAFE_POD_ID_PATTERN.fullmatch(pod_id) is None:
            raise ValueError("RunPod inventory entry has unsafe pod identity")
        names = [pod.get("name"), pod.get("podName")]
        if isinstance(nested, Mapping):
            names.append(nested.get("name"))
        pod_names = {value for value in names if isinstance(value, str) and value}
        if len(pod_names) != 1:
            raise ValueError("RunPod inventory entry has ambiguous or missing pod name")
        records.append(ProviderPodInventoryRecord(pod_id=pod_id, name=pod_names.pop()))
    pod_ids = [record.pod_id for record in records]
    if len(pod_ids) != len(set(pod_ids)):
        raise ValueError("RunPod inventory contains duplicate pod identities")
    return tuple(sorted(records, key=lambda record: record.pod_id))


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
