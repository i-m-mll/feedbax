"""RunPod client, transport, and provider response normalization."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from feedbax.contracts.strict_json import StrictJsonError, strict_json_loads


@dataclass(frozen=True)
class ProviderPodInventoryRecord:
    """Typed provider inventory row retaining both identity and name tag."""

    pod_id: str
    name: str


AcquisitionDisposition = Literal["terminal-rejection", "retryable-rejection", "ambiguous"]


_SAFE_POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


_CHILD_TERMINATION_GRACE_SECONDS = 1.0


_RUNPOD_GO_UTC_PATTERN = re.compile(
    r"^(?P<instant>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \+0000 UTC$"
)


class RunPodDriverError(RuntimeError):
    """Raised when the RunPod driver cannot complete a requested action."""


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


@dataclass(frozen=True)
class RunPodApiReply:
    """A response from the fixed RunPod control-plane origin."""

    status: int
    body: Mapping[str, Any] | None
    media_type: str


class RunPodTransport(Protocol):
    """Transport surface for the RunPod client."""

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Run a ``runpodctl`` command."""

    def image_exists(self, image: str) -> bool:
        """Return whether the exact declared container manifest exists."""

    def api_request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        *,
        api_key: str,
        params: Mapping[str, str | int] | None = None,
        body: Mapping[str, Any] | None = None,
        timeout_seconds: float = 8,
    ) -> RunPodApiReply:
        """Send one request to the RunPod v2 control plane."""

    def ssh(self, command: str, *, timeout_seconds: float | None = None) -> CommandResult:
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
    strict_ssh_host_key: bool = False
    ssh_known_hosts_path: str | None = None
    ssh_boot_public_key: str | None = None
    runpodctl_executable: str = "runpodctl"
    rsync_executable: str = "rsync"
    http_transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def runpodctl(
        self,
        *args: str,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        return _run_command(
            [self.runpodctl_executable, *args],
            timeout_seconds=timeout_seconds,
        )

    def api_request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        *,
        api_key: str,
        params: Mapping[str, str | int] | None = None,
        body: Mapping[str, Any] | None = None,
        timeout_seconds: float = 8,
    ) -> RunPodApiReply:
        if not path.startswith("/v2/") or not api_key or not 0 < timeout_seconds <= 60:
            raise RunPodDriverError("invalid RunPod API request configuration")
        try:
            with httpx.Client(
                base_url="https://api.runpod.io",
                timeout=timeout_seconds,
                trust_env=False,
                transport=self.http_transport,
            ) as client, client.stream(
                method,
                path,
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                params=params,
                json=body,
            ) as response:
                media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
                size_limit = 8192 if response.is_error else 4 * 1024 * 1024
                payload = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    if len(payload) + len(chunk) > size_limit:
                        return RunPodApiReply(response.status_code, None, media_type)
                    payload.extend(chunk)
                try:
                    decoded = strict_json_loads(bytes(payload)) if payload else None
                except (ValueError, RecursionError):
                    decoded = None
                return RunPodApiReply(
                    response.status_code,
                    decoded if isinstance(decoded, Mapping) else None,
                    media_type,
                )
        except httpx.HTTPError:
            raise RunPodDriverError("RunPod v2 transport failed") from None

    def image_exists(self, image: str) -> bool:
        match = re.fullmatch(
            r"(?:docker\.io/)?([a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*)"
            r"(?::[A-Za-z0-9_.-]+)?@(sha256:[0-9a-f]{64})",
            image,
        )
        if not match:
            return False
        repository, digest = match.groups()
        try:
            with httpx.Client(timeout=10, trust_env=False, transport=self.http_transport) as client:
                token = client.get(
                    "https://auth.docker.io/token",
                    params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
                ).raise_for_status().json()
                bearer = token.get("token") if isinstance(token, dict) else None
                if not isinstance(bearer, str) or not bearer:
                    return False
                manifest = client.head(
                    f"https://registry-1.docker.io/v2/{repository}/manifests/{digest}",
                    headers={
                        "Authorization": f"Bearer {bearer}",
                        "Accept": (
                            "application/vnd.oci.image.manifest.v1+json, "
                            "application/vnd.oci.image.index.v1+json, "
                            "application/vnd.docker.distribution.manifest.v2+json, "
                            "application/vnd.docker.distribution.manifest.list.v2+json"
                        ),
                    },
                )
                return manifest.status_code == 200 and manifest.headers.get(
                    "Docker-Content-Digest"
                ) == digest
        except (httpx.HTTPError, ValueError):
            return False

    def ssh(self, command: str, *, timeout_seconds: float | None = None) -> CommandResult:
        host = self._require_host()
        args = self._ssh_base(detach_stdin=True)
        if timeout_seconds is not None:
            args.extend(["-o", f"ConnectTimeout={max(1, math.ceil(timeout_seconds))}"])
        return _run_command([*args, host, command], timeout_seconds=timeout_seconds)

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
        args.extend(["-i", key, "-p", str(self.ssh_port)])
        if self.strict_ssh_host_key:
            args.extend(["-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes"])
            if self.ssh_known_hosts_path is not None:
                args.extend([
                    "-F", "/dev/null", "-o", "IdentitiesOnly=yes",
                    "-o", "GlobalKnownHostsFile=/dev/null",
                    "-o", f'UserKnownHostsFile="{self.ssh_known_hosts_path}"',
                ])
            if self.ssh_boot_public_key is not None:
                self._write_boot_host_pin()
                args.extend(["-o", "HostKeyAlgorithms=ssh-ed25519", "-o", "UpdateHostKeys=no"])
        else:
            args.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        return args

    def _write_boot_host_pin(self) -> None:
        host, port, public = self.ssh_host, self.ssh_port, self.ssh_boot_public_key
        if (
            not host or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host)
            or port is None or not 1 <= port <= 65535
            or not public or not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/]+={0,2}", public)
            or self.ssh_known_hosts_path is None
            or not re.fullmatch(r"/[A-Za-z0-9/_.-]+", self.ssh_known_hosts_path)
        ):
            raise RunPodDriverError("missing or invalid boot SSH host pin")
        endpoint = host if port == 22 else f"[{host}]:{port}"
        path = Path(self.ssh_known_hosts_path)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(f"{endpoint} {public}\n")
                stream.flush()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


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
    if isinstance(ssh, Mapping) and "direct" in ssh:
        direct = ssh.get("direct")
        if not isinstance(direct, Mapping) or direct.get("username") != "root":
            return EndpointClassification("missing")
        host, port = direct.get("host"), direct.get("port")
        if (
            isinstance(host, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host)
            and type(port) is int
            and 1 <= port <= 65535
        ):
            return EndpointClassification("ssh_object", host, port)
        return EndpointClassification("missing")
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
        ("image",),
    )
    hourly_rate_raw = first(
        ("costPerHr",),
        ("costPerHour",),
        ("hourlyRate",),
        ("machine", "costPerHr"),
        ("machine", "costPerHour"),
        ("cost",),
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
            payload = strict_json_loads(line)
        except (json.JSONDecodeError, StrictJsonError):
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


def classify_create_failure(
    result: CommandResult, secret: str | None
) -> tuple[AcquisitionDisposition, str]:
    """Separate confirmed capacity and request rejections from ambiguous failures."""
    stdout = _redact_secret(result.stdout, secret)
    stderr = _redact_secret(result.stderr, secret)
    detail = (stderr or stdout).strip()
    try:
        payload = strict_json_loads(result.stdout or result.stderr)
    except (json.JSONDecodeError, StrictJsonError):
        payload = None
    code = (
        str(payload.get("statusCode") or payload.get("code") or "").lower()
        if isinstance(payload, Mapping)
        else ""
    )
    if code in {"401", "403", "unauthorized", "forbidden"}:
        return "terminal-rejection", detail
    if _is_no_capacity_create_response(stdout, stderr):
        return "retryable-rejection", detail
    if code in {"400", "422", "invalid_request"}:
        return "terminal-rejection", detail
    return "ambiguous", detail


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


def _parse_runpod_pod_inventory(payload: str) -> tuple[ProviderPodInventoryRecord, ...]:
    """Parse supported inventory shapes without discarding provider pod names."""
    return _decode_runpod_pod_inventory(strict_json_loads(payload))


def _decode_runpod_pod_inventory(loaded: object) -> tuple[ProviderPodInventoryRecord, ...]:
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
