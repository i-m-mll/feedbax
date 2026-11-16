"""RunPod acquisition and teardown for the ordinary SessionDriver."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import re
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeGuard
from urllib.parse import quote
from uuid import UUID

from pydantic import Field, SecretStr, TypeAdapter

from feedbax.host.io import private_directory, sync_directory, write_json
from feedbax.host.protocol import JsonObject
from feedbax.orchestration.runpod import (
    RunPodApiReply,
    RunPodDriverError,
    RunPodTransport,
    SubprocessRunPodTransport,
    _decode_runpod_pod_inventory,
    _run_command,
    classify_pod_state,
    project_runpod_provision_facts,
)
from feedbax.orchestration.session import (
    Nonempty, Positive, ProviderMachineObservation, ProviderResource, ProviderUnreachable, SessionError, SessionLaunch, SessionModel,
    SessionOpeningFailed, SessionRequest,
)


_INTENT = re.compile(r"feedbax-session-[0-9a-f]{64}")
_RESOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PODS = "/v2/pods"
_SECRETS = "/v2/account/secrets"
_CREATE_REJECTIONS = frozenset({400, 401, 403, 409, 413, 422, 429})
_BOOT_KEY = "FEEDBAX_BOOT_SSH_KEY"
_BOOT_COMMAND = """set +x
set -euo pipefail
umask 077
key="${FEEDBAX_BOOT_SSH_KEY:-}"
unset FEEDBAX_BOOT_SSH_KEY LD_LIBRARY_PATH
"${@:4}"
unset RUNPOD_API_KEY
printf '%s\n' 'RunPod venue cutoff guard armed.' >&2
hold_for_guard() {
    printf '%s\n' 'RunPod SSH key setup failed; the active cutoff guard is still running.' >&2
    if [[ -n "${temporary:-}" ]]; then rm -f -- "$temporary"; fi
    exec /bin/sleep infinity
}
trap hold_for_guard ERR
test -n "$key"
temporary=$(mktemp "$2/.feedbax-host.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
printf '%s' "$key" | base64 -d >"$temporary"
unset key
derived=$(ssh-keygen -y -f "$temporary")
read -r algorithm blob _ <<<"$derived"
public="$algorithm $blob"
test "$public" = "$1"
mv -f -- "$temporary" "$2/ssh_host_ed25519_key"
printf '%s\n' "$public" >"$2/ssh_host_ed25519_key.pub"
trap - EXIT ERR
printf '%s\n' 'RunPod pinned SSH host key installed.' >&2
exec "$3"
"""
_GPU_PROBE = "nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv,noheader"


class _ControlPlaneUnavailable(ProviderUnreachable):
    """A bounded RunPod REST request failed in transport."""


class RunPodResources(SessionModel):
    """Exact single-device boot configuration and all-in hourly bound."""

    image: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")]
    gpu_id: Nonempty
    datacenter: Nonempty
    cuda_version: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    compute_capability: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    min_driver_major: Positive
    ephemeral_disk_gb: Positive
    hourly_microusd: Positive
    startup_seconds: Annotated[int, Field(gt=0, le=300, strict=True)] = 60


class RunPodCutoffArm(SessionModel):
    create_terms: JsonObject
    boot_argv: Annotated[tuple[Nonempty, ...], Field(min_length=1)]


class RunPodRunningCutoff(Protocol):
    """Bind the running session deadline before submitting pod creation.

    The capability covers the interval before authenticated SSH is available.
    Its image-local boot command arms the active guard and returns before sshd
    starts. Both its boot and termination commands act only on the owned pod.
    Neither saved command nor create terms may contain credentials. Repeated
    calls for one intent must preserve its original cutoff.
    """

    def arm(self, launch: SessionLaunch) -> RunPodCutoffArm: ...

    def terminate_argv(self, resource: ProviderResource, launch: SessionLaunch) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class RunPodEndpoint:
    host: str
    ssh_argv: tuple[str, ...]
    placement: JsonObject
    opening_deadline: float | None = None


class _Intent(SessionModel):
    launch: SessionLaunch
    public_key: Annotated[str, Field(pattern=r"^ssh-ed25519 [A-Za-z0-9+/]+={0,2}$")]
    cutoff: RunPodCutoffArm
    opening_aborted: bool = False


class _SecretRequested(_Intent):
    state: Literal["secret-requested"] = "secret-requested"
    secret_id: Nonempty | None = None


class _PodRequested(_Intent):
    state: Literal["pod-requested"] = "pod-requested"
    secret_id: Nonempty
    pod_id: str | None = None
    startup_deadline_ms: Positive | None = None
    boot_verified_at_ms: Positive | None = None
    completion_requested: bool | None = None


_Journal = TypeAdapter(Annotated[_SecretRequested | _PodRequested, Field(discriminator="state")])


def _absolute_command(argv: tuple[str, ...]) -> tuple[str, ...]:
    if not argv or not argv[0].startswith("/") or any("\0" in part for part in argv):
        raise SessionError("RunPod cutoff requires an absolute venue executable.")
    return argv


def _secret_metadata(intent: str) -> dict[str, str]:
    return {
        "name": f"feedbax-host-{hashlib.sha256(intent.encode()).hexdigest()}",
        "description": f"Feedbax acquisition {intent}",
    }


def _identity(value: object) -> str:
    if not isinstance(value, str) or not _RESOURCE_ID.fullmatch(value):
        raise SessionError("RunPod returned an invalid resource identity.")
    return value


def _secret_resource(body: Mapping[str, object]) -> ProviderResource:
    description = body.get("description")
    intent = description.removeprefix("Feedbax acquisition ") if isinstance(description, str) else ""
    owned = bool(_INTENT.fullmatch(intent)) and all(
        body.get(key) == value for key, value in _secret_metadata(intent).items()
    )
    return ProviderResource(id=_identity(body.get("id")), intent=intent if owned else None, kind="secret")


def _fresh_boot_host_key() -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="feedbax-ssh-key-") as directory:
        key = Path(directory) / "host"
        result = _run_command(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "", "-f", str(key)],
            timeout_seconds=5,
        )
        if result.returncode:
            raise SessionError("Could not generate an ephemeral RunPod host key.")
        public = " ".join(key.with_suffix(".pub").read_text(encoding="ascii").split()[:2])
        return base64.b64encode(key.read_bytes()).decode("ascii"), public


def _rate_within_cap(rate: object, cap_microusd: int) -> TypeGuard[int | float]:
    if type(rate) not in (int, float):
        return False
    value = Decimal(str(rate))
    return value.is_finite() and 0 < value * 1_000_000 <= cap_microusd


def _epoch_ms() -> int:
    return time.time_ns() // 1_000_000


class RunPodAcquisition:
    target = "runpod"

    def __init__(
        self, root: Path, transport: RunPodTransport, *, api_key: str, ssh_key_path: Path,
        running_cutoff: RunPodRunningCutoff | None = None, ssh_executable: str = "ssh",
    ) -> None:
        self.root = private_directory(root.resolve())
        sync_directory(self.root.parent)
        self.transport = transport
        self._api_key = SecretStr(api_key)
        self._ssh_key_path = ssh_key_path
        self._running_cutoff = running_cutoff
        self._ssh_executable = ssh_executable

    @property
    def enforces_cutoff(self) -> bool:
        return self._running_cutoff is not None

    @property
    def running_cutoff(self) -> RunPodRunningCutoff | None:
        return self._running_cutoff

    def terminate_argv(self, resource: ProviderResource, launch: SessionLaunch) -> tuple[str, ...]:
        if self._running_cutoff is None or resource.kind != "venue" or resource.intent != launch.intent:
            raise SessionError("RunPod termination requires the owned pod and scoped cutoff capability.")
        _identity(resource.id)
        return _absolute_command(self._running_cutoff.terminate_argv(resource, launch))

    def hourly_microusd(self, request: SessionRequest) -> int:
        return RunPodResources.model_validate(request.resources).hourly_microusd

    def review_quote(self, request: SessionRequest) -> JsonObject:
        """Fetch exact GPU compute pricing without creating provider resources."""
        if request.target != self.target:
            raise SessionError("RunPod quote requires a RunPod session request.")
        return self._verify_quote(RunPodResources.model_validate(request.resources), timeout_seconds=8)

    def _api(
        self, method: Literal["GET", "POST", "DELETE"], path: str, *,
        params: Mapping[str, str | int] | None = None, body: Mapping[str, object] | None = None,
        timeout_seconds: float = 8,
    ) -> RunPodApiReply:
        try:
            return self.transport.api_request(
                method, path, api_key=self._api_key.get_secret_value(), params=params,
                body=body, timeout_seconds=timeout_seconds,
            )
        except (OSError, RunPodDriverError):
            raise _ControlPlaneUnavailable("RunPod control-plane transport failed.") from None

    def _pages(self, path: str, key: str, *, paginated: bool) -> Iterator[object]:
        params: dict[str, str | int] = {"limit": 100} if paginated else {}
        seen: set[str] = set()
        deadline = time.monotonic() + 15
        for _ in range(5):
            reply = self._api("GET", path, params=params, timeout_seconds=self._remaining(deadline))
            body = reply.body or {}
            page, pagination = body.get(key), body.get("pagination")
            if reply.status != 200 or not isinstance(page, list):
                raise SessionError(f"RunPod did not return a complete {key} inventory.")
            yield from page
            if pagination is None and not paginated:
                return
            if isinstance(pagination, Mapping):
                if pagination.get("hasNextPage") is False:
                    return
                cursor = pagination.get("nextCursor")
                if pagination.get("hasNextPage") is True and isinstance(cursor, str) and cursor and cursor not in seen:
                    seen.add(cursor)
                    params["cursor"] = cursor
                    continue
            raise SessionError(f"RunPod returned an invalid {key} inventory cursor.")
        raise SessionError(f"RunPod {key} inventory exceeded five pages.")

    def _venues(self) -> tuple[ProviderResource, ...]:
        try:
            pods = _decode_runpod_pod_inventory(list(self._pages(_PODS, "pods", paginated=True)))
        except (ValueError, TypeError):
            raise SessionError("RunPod pod inventory contains invalid entries.") from None
        return tuple(ProviderResource(
            id=_identity(row.pod_id), intent=row.name if _INTENT.fullmatch(row.name) else None, kind="venue",
        ) for row in pods)

    def inventory(self) -> tuple[ProviderResource, ...]:
        pods = self._venues()
        rows = list(self._pages(_SECRETS, "secrets", paginated=False))
        if any(not isinstance(row, Mapping) for row in rows):
            raise SessionError("RunPod secret inventory contains invalid entries.")
        secrets = tuple(_secret_resource(row) for row in rows if isinstance(row, Mapping))
        if len({item.id for item in secrets}) != len(secrets):
            raise SessionError("RunPod secret inventory repeats a resource identity.")
        return (*pods, *secrets)

    def _journal_path(self, intent: str) -> Path:
        if not _INTENT.fullmatch(intent):
            raise SessionError("RunPod resource has an invalid acquisition intent.")
        return self.root / f"{intent}.json"

    @contextmanager
    def _exclusive(self, intent: str) -> Iterator[Path]:
        path = self._journal_path(intent)
        deadline = time.monotonic() + 8
        with path.with_suffix(".lock").open("ab") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.05, self._remaining(deadline)))
            yield path

    def _secret(self, intent: str, *, timeout_seconds: float = 8) -> ProviderResource:
        reply = self._api("GET", _SECRETS, params={"name": _secret_metadata(intent)["name"]},
                          timeout_seconds=timeout_seconds)
        rows = (reply.body or {}).get("secrets")
        if reply.status != 200 or not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise SessionError("RunPod boot secret creation remains unresolved for this intent.")
        resource = _secret_resource(rows[0])
        if resource.intent != intent:
            raise SessionError("RunPod boot secret does not match the acquisition intent.")
        return resource

    @staticmethod
    def _cutoff_remaining(launch: SessionLaunch) -> float:
        seconds = (launch.ends_at_ms - _epoch_ms()) / 1000
        if seconds <= 0:
            raise SessionOpeningFailed("RunPod acquisition exhausted the authorized cutoff.")
        return seconds

    @classmethod
    def _before_cutoff(cls, launch: SessionLaunch) -> float:
        return min(8, cls._cutoff_remaining(launch))

    @staticmethod
    def _remaining(deadline: float) -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise SessionError("RunPod operation exceeded its deadline.")
        return min(8, seconds)

    def _verify_quote(self, config: RunPodResources, *, timeout_seconds: float) -> JsonObject:
        reply = self._api("GET", f"/v2/catalog/gpus/{quote(config.gpu_id, safe='')}", params={
            "include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": 1,
            "cudaVersions": config.cuda_version,
        }, timeout_seconds=timeout_seconds)
        if reply.status in (401, 403, 408, 429) or reply.status >= 500:
            raise ProviderUnreachable(f"RunPod could not answer the quote request (HTTP {reply.status}).")
        body = reply.body or {}
        price = body.get("price")
        hourly = price.get("secure") if isinstance(price, Mapping) else None
        versions = body.get("cudaVersions")
        regions = body.get("dataCenters")
        stock = [row.get("availability") for row in regions if isinstance(row, Mapping)
                 and row.get("id") == config.datacenter] if isinstance(regions, list) else []
        if not (
            reply.status == 200 and body.get("id") == config.gpu_id
            and _rate_within_cap(hourly, config.hourly_microusd)
            and isinstance(versions, list) and any(isinstance(row, Mapping) and row.get("available") is True
                and row.get("version") == config.cuda_version for row in versions)
            and len(stock) == 1 and stock[0] in ("LOW", "MEDIUM", "HIGH")
        ):
            raise SessionError("RunPod quote did not establish the authorized GPU, rate, CUDA and region.")
        return {
            "gpu_id": config.gpu_id, "datacenter": config.datacenter, "cuda_version": config.cuda_version,
            "gpu_hourly_usd": hourly, "ephemeral_disk_gb": config.ephemeral_disk_gb,
            "authorized_total_hourly_microusd": config.hourly_microusd,
        }

    @staticmethod
    def _pod_body(state: _Intent, config: RunPodResources) -> JsonObject:
        secret = _secret_metadata(state.launch.intent)["name"]
        exact: JsonObject = {
            "name": state.launch.intent, "image": config.image, "cloud": "SECURE",
            "disk": config.ephemeral_disk_gb, "gpu": {
                "id": config.gpu_id, "count": 1, "allowedCudaVersions": [config.cuda_version],
            },
            "dataCenterIds": [config.datacenter], "ports": ["22/tcp"], "startSsh": True,
            "env": {_BOOT_KEY: "{{ RUNPOD_SECRET_" + secret + " }}"},
            "entrypoint": ["/bin/bash", "-c", _BOOT_COMMAND, "feedbax-host-key",
                           state.public_key, "/etc/ssh", "/start.sh", *_absolute_command(state.cutoff.boot_argv)],
            "cmd": [],
        }
        protected = exact.keys() | {
            "queue", "queueId", "mounts", "volumes", "volumeId", "volumeInGb", "volumeMountPath",
            "networkVolumeId", "networkVolumeIds", "template", "templateId",
            "environment", "dockerEnv", "args", "dockerArgs", "dockerEntrypoint", "dockerStartCmd",
        }
        if protected & state.cutoff.create_terms.keys():
            raise SessionError("RunPod cutoff terms conflict with the authorized pod configuration.")
        return exact | state.cutoff.create_terms

    def acquire(self, launch: SessionLaunch) -> ProviderResource:
        """Create each resource once for a durable SessionDriver launch intent."""
        if self._running_cutoff is None:
            raise SessionError("RunPod has no verified independent preboot cutoff capability.")
        config = RunPodResources.model_validate(launch.request.resources)
        if launch.request.target != self.target or launch.hourly_microusd != config.hourly_microusd:
            raise SessionError("RunPod launch differs from the authorized target or hourly rate.")
        self._before_cutoff(launch)
        with self._exclusive(launch.intent) as path:
            try:
                state = _Journal.validate_json(path.read_bytes())
            except FileNotFoundError:
                if any(item.intent == launch.intent for item in self.inventory()):
                    raise SessionError("RunPod already lists a resource for an unjournaled launch intent.")
                if not self.transport.image_exists(config.image):
                    raise SessionError("RunPod could not verify the authorized immutable image.")
                self._verify_quote(config, timeout_seconds=self._before_cutoff(launch))
                self._before_cutoff(launch)
                cutoff = self._running_cutoff.arm(launch)
                private, public = _fresh_boot_host_key()
                state = _SecretRequested(launch=launch, public_key=public, cutoff=cutoff)
                self._pod_body(state, config)
                self._before_cutoff(launch)
                write_json(path, state.model_dump(mode="json"))
                reply = self._api("POST", _SECRETS, body={**_secret_metadata(launch.intent), "value": private},
                                  timeout_seconds=self._before_cutoff(launch))
                if reply.status in _CREATE_REJECTIONS - {409}:
                    write_json(path, state.model_copy(update={"opening_aborted": True}).model_dump(mode="json"))
                    raise SessionOpeningFailed(f"RunPod rejected boot secret creation (HTTP {reply.status}).")
                if reply.status != 201 or reply.body is None:
                    raise SessionError("RunPod boot secret creation remains unresolved for this intent.")
                secret = _secret_resource(reply.body)
                if secret.intent != launch.intent:
                    raise SessionError("Created RunPod boot secret lacks the acquisition intent.")
            else:
                if state.launch != launch:
                    raise SessionError("RunPod journal is bound to a different session launch.")
                if state.opening_aborted:
                    raise SessionOpeningFailed("RunPod opening was already closed.")
                if isinstance(state, _PodRequested):
                    return self._reconcile_pod(state, path)
                secret = self._secret(launch.intent, timeout_seconds=self._before_cutoff(launch))

            if state.secret_id is not None and secret.id != state.secret_id:
                raise SessionError("RunPod boot secret differs from the recorded identity.")
            if state.secret_id is None:
                state = state.model_copy(update={"secret_id": secret.id})
                write_json(path, state.model_dump(mode="json"))
            try:
                self._verify_quote(config, timeout_seconds=self._before_cutoff(launch))
            except SessionError as error:
                raise SessionOpeningFailed(f"RunPod did not submit pod creation: {error}") from error
            pod_body = self._pod_body(state, config)
            submitted = _PodRequested(
                launch=state.launch, public_key=state.public_key, cutoff=state.cutoff, secret_id=secret.id,
                startup_deadline_ms=min(launch.ends_at_ms, _epoch_ms() + 1000 * config.startup_seconds),
                completion_requested=False,
            )
            self._before_cutoff(launch)
            write_json(path, submitted.model_dump(mode="json"))
            reply = self._api("POST", _PODS, body=pod_body, timeout_seconds=self._before_cutoff(launch))
            if reply.status in _CREATE_REJECTIONS:
                write_json(path, state.model_copy(update={"opening_aborted": True}).model_dump(mode="json"))
                raise SessionOpeningFailed(f"RunPod rejected pod creation (HTTP {reply.status}).")
            if reply.status != 201 or reply.body is None:
                raise SessionError("RunPod pod creation remains unresolved for this intent.")
            pod_id = _identity(reply.body.get("id"))
            write_json(path, submitted.model_copy(update={"pod_id": pod_id}).model_dump(mode="json"))
            return ProviderResource(id=pod_id, intent=launch.intent, kind="venue")

    def _reconcile_pod(self, state: _PodRequested, path: Path) -> ProviderResource:
        matches = [row for row in self._venues() if row.intent == state.launch.intent]
        if len(matches) != 1 or (state.pod_id is not None and matches[0].id != state.pod_id):
            raise SessionError("RunPod pod creation remains unresolved for this intent.")
        resource = matches[0]
        if state.pod_id is None:
            write_json(path, state.model_copy(update={"pod_id": resource.id}).model_dump(mode="json"))
        return resource

    def _owned_detail(
        self, item: ProviderResource, path: str, *, timeout_seconds: float = 8,
    ) -> Mapping[str, object] | None:
        reply = self._api("GET", path, timeout_seconds=timeout_seconds)
        if reply.status == 404:
            return None
        body = reply.body
        if reply.status != 200 or body is None or body.get("id") != item.id:
            raise SessionError("RunPod detail did not establish the exact resource identity.")
        intent = body.get("name") if item.kind == "venue" else _secret_resource(body).intent
        if intent != item.intent:
            raise SessionError("RunPod resource intent changed before use.")
        return body

    def machine_observation(self, resource: ProviderResource) -> ProviderMachineObservation:
        if resource.kind != "venue" or resource.intent is None or _INTENT.fullmatch(resource.intent) is None:
            raise SessionError("Machine status requires an owned RunPod venue.")
        pod = self._owned_detail(resource, f"{_PODS}/{_identity(resource.id)}")
        observed = _epoch_ms()
        if pod is None:
            return ProviderMachineObservation(state="off", observed_at_ms=observed)
        actual = pod.get("status")
        if actual in ("EXITED", "TERMINATED", "FAILED"):
            state = "off"
        elif actual == "RUNNING":
            state = "on"
        elif actual in ("CREATED", "STARTING", "PENDING", "INITIALIZING"):
            state = "starting"
        else:
            raise SessionError("RunPod did not report the venue's current power state.")
        started = project_runpod_provision_facts(pod)["billing_started_at"]
        return ProviderMachineObservation(
            state=state, observed_at_ms=observed,
            billing_started_at_ms=round(datetime.fromisoformat(started).timestamp() * 1000)
            if isinstance(started, str) else None,
        )

    @staticmethod
    def _pod_state(resource: ProviderResource, path: Path) -> _PodRequested:
        if resource.kind != "venue" or resource.intent is None:
            raise SessionError("RunPod SSH requires an owned venue.")
        pod_id = _identity(resource.id)
        try:
            state = _Journal.validate_json(path.read_bytes())
        except FileNotFoundError:
            raise SessionError("RunPod SSH has no persisted boot key pin for this intent.") from None
        if not isinstance(state, _PodRequested) or state.launch.intent != resource.intent:
            raise SessionError("RunPod SSH has no pod submission for this intent.")
        if state.pod_id is not None and state.pod_id != pod_id:
            raise SessionError("RunPod venue differs from the recorded pod identity.")
        return state

    def begin_completion(self, resource: ProviderResource, launch: SessionLaunch) -> None:
        """Record that the Host fence may be initialized before sending the request."""
        with self._exclusive(launch.intent) as path:
            state = self._pod_state(resource, path)
            if state.launch != launch:
                raise SessionError("RunPod completion requires the exact launch.")
            if state.opening_aborted:
                raise SessionOpeningFailed("RunPod opening was already closed.")
            self._before_cutoff(launch)
            if state.completion_requested is not True:
                write_json(path, state.model_copy(update={"completion_requested": True}).model_dump(mode="json"))

    def claim_uninitialized(
        self, resource: ProviderResource | None, launch: SessionLaunch,
    ) -> tuple[ProviderResource, ...] | None:
        """Fence completion locally if its remote request was never submitted."""
        with self._exclusive(launch.intent) as path:
            try:
                state = _Journal.validate_json(path.read_bytes())
            except FileNotFoundError:
                return None
            if state.launch != launch:
                raise SessionError("RunPod opening close requires the exact launch.")
            if isinstance(state, _PodRequested):
                if state.completion_requested is not False:
                    return None
                if resource is None:
                    if state.pod_id is None:
                        return None
                    resource = ProviderResource(id=state.pod_id, intent=launch.intent, kind="venue")
                if (
                    resource.intent != launch.intent or resource.kind != "venue"
                    or (state.pod_id is not None and resource.id != state.pod_id)
                ):
                    raise SessionError("RunPod opening close requires the exact venue identity.")
                resources = (resource, ProviderResource(id=state.secret_id, intent=launch.intent, kind="secret"))
            else:
                if resource is not None:
                    raise SessionError("RunPod observed a venue without a local pod submission.")
                secret_id = state.secret_id
                if secret_id is None and not state.opening_aborted:
                    secret_id = self._secret(launch.intent).id
                    state = state.model_copy(update={"secret_id": secret_id})
                resources = (
                    (ProviderResource(id=secret_id, intent=launch.intent, kind="secret"),)
                    if secret_id is not None else ()
                )
            if not state.opening_aborted:
                write_json(path, state.model_copy(update={"opening_aborted": True}).model_dump(mode="json"))
            return resources

    @staticmethod
    def _opening_remaining(deadline_ms: int, phase: str) -> float:
        remaining = (deadline_ms - _epoch_ms()) / 1000
        if remaining <= 0:
            raise SessionOpeningFailed(f"RunPod {phase} exceeded its opening deadline.")
        return remaining

    def endpoint(
        self, resource: ProviderResource, *, opening: bool = False, realization_seconds: int = 345,
    ) -> RunPodEndpoint:
        """Use a persisted readiness deadline for opening, or a fresh RPC budget for reconnection."""
        if resource.intent is None:
            raise SessionError("RunPod SSH requires an owned venue.")
        with self._exclusive(resource.intent) as path:
            state = self._pod_state(resource, path)
            pod_id = resource.id
            config = RunPodResources.model_validate(state.launch.request.resources)
            if opening and state.opening_aborted:
                raise SessionOpeningFailed("RunPod opening was already closed.")
            startup_deadline = min(
                state.launch.ends_at_ms,
                state.startup_deadline_ms or state.launch.acquired_at_ms + config.startup_seconds * 1000,
            )
            ready_at = state.boot_verified_at_ms
            if opening and ready_at is None:
                deadline = time.monotonic() + self._opening_remaining(startup_deadline, "SSH readiness")
            else:
                deadline = time.monotonic() + 8

            def remaining() -> float:
                seconds = deadline - time.monotonic()
                if opening and ready_at is None:
                    if seconds <= 0:
                        raise SessionOpeningFailed("RunPod SSH readiness exceeded its opening deadline.")
                    seconds = min(seconds, self._opening_remaining(startup_deadline, "SSH readiness"))
                elif opening and ready_at is not None:
                    realization_deadline = min(state.launch.ends_at_ms, ready_at + 1000 * realization_seconds)
                    seconds = min(seconds, self._opening_remaining(realization_deadline, "realization"))
                if seconds <= 0:
                    raise SessionError("RunPod SSH reconnect exceeded its RPC deadline.")
                return min(8, seconds)

            while True:
                try:
                    pod = self._owned_detail(resource, f"{_PODS}/{pod_id}", timeout_seconds=remaining())
                except _ControlPlaneUnavailable:
                    if not opening or ready_at is not None:
                        raise
                    time.sleep(min(0.25, remaining()))
                    continue
                if pod is None:
                    raise SessionOpeningFailed("RunPod venue is absent.")
                status = classify_pod_state(pod)
                if status.status == "dead":
                    raise SessionOpeningFailed("RunPod venue exited during SSH readiness.")
                ssh = pod.get("ssh")
                # RunPod can report RUNNING before it publishes the direct SSH mapping.
                if (
                    pod.get("status") == "RUNNING" and isinstance(ssh, Mapping) and "direct" in ssh
                    and status.ip is not None and status.port is not None
                ):
                    if ":" in status.ip:
                        raise SessionOpeningFailed("RunPod did not expose an authenticated direct SSH endpoint.")
                    endpoint = self._probe_endpoint(state, config, resource, pod, status.ip, status.port, remaining())
                    if endpoint is not None:
                        remaining()
                        if opening:
                            verified_at = ready_at if ready_at is not None else _epoch_ms()
                            realization_deadline = min(state.launch.ends_at_ms, verified_at + 1000 * realization_seconds)
                            endpoint = RunPodEndpoint(
                                endpoint.host, endpoint.ssh_argv, endpoint.placement,
                                opening_deadline=time.monotonic() + self._opening_remaining(
                                    realization_deadline, "realization",
                                ),
                            )
                        else:
                            verified_at = ready_at
                        if state.pod_id is None or ready_at != verified_at:
                            updated = state.model_copy(update={"pod_id": pod_id, "boot_verified_at_ms": verified_at})
                            write_json(path, updated.model_dump(mode="json"))
                        return endpoint
                time.sleep(min(0.25, remaining()))

    def _probe_endpoint(
        self, state: _PodRequested, config: RunPodResources, resource: ProviderResource,
        pod: Mapping[str, object], ip: str, port: int, seconds: float,
    ) -> RunPodEndpoint | None:
        gpu, mounts = pod.get("gpu"), pod.get("mounts")
        if not (
            pod.get("image") == config.image and pod.get("cloud") == "SECURE"
            and pod.get("dataCenterId") == config.datacenter and pod.get("cudaVersion") == config.cuda_version
            and type(pod.get("disk")) is int and pod["disk"] == config.ephemeral_disk_gb
            and pod.get("ports") == ["22/tcp"] and isinstance(gpu, Mapping) and gpu.get("id") == config.gpu_id
            and type(gpu.get("count")) is int and gpu["count"] == 1 and isinstance(mounts, Mapping)
            and not any(mounts.get(kind) for kind in ("persistent", "network", "global"))
            and _rate_within_cap(pod.get("cost"), config.hourly_microusd)
        ):
            raise SessionOpeningFailed("RunPod venue differs from the authorized exact CUDA placement.")
        transport = SubprocessRunPodTransport(
            ssh_host=ip, ssh_port=port, ssh_key_path=self._ssh_key_path,
            strict_ssh_host_key=True, ssh_known_hosts_path=str(self.root / f"{resource.intent}.known-hosts"),
            ssh_boot_public_key=state.public_key, ssh_executable=self._ssh_executable,
        )
        host = f"root@{ip}"
        result = transport.ssh(_GPU_PROBE, timeout_seconds=seconds)
        if result.returncode == 124 or (result.returncode == 255 and any(
            message in result.stderr.lower() for message in (
                "connection refused", "connection timed out", "connection reset", "no route to host",
            )
        )):
            return None
        lines = result.stdout.strip().splitlines()
        fields = [part.strip() for part in lines[0].split(",")] if len(lines) == 1 else []
        if result.returncode or len(fields) != 4:
            raise SessionOpeningFailed("Pinned RunPod SSH could not attest exactly one GPU.")
        name, driver, capability, device_id = fields
        if not (
            name == config.gpu_id and capability == config.compute_capability
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver)
            and int(driver.split(".", 1)[0]) >= config.min_driver_major
            and device_id.startswith("GPU-")
        ):
            raise SessionOpeningFailed("RunPod device probe differs from the authorized GPU and driver.")
        try:
            UUID(device_id.removeprefix("GPU-"))
        except ValueError:
            raise SessionOpeningFailed("RunPod device probe did not report a GPU UUID.") from None
        facts = project_runpod_provision_facts(pod)
        return RunPodEndpoint(host, transport.pinned_ssh_argv(), {
            "pod_id": resource.id, "intent": resource.intent, "device": name, "device_uuid": device_id,
            "driver": driver, "compute_capability": capability, "region": config.datacenter,
            "cuda_version": config.cuda_version, "image": config.image, "ephemeral_disk_gb": config.ephemeral_disk_gb,
            "hourly_microusd": config.hourly_microusd, "billing_started_at": facts["billing_started_at"],
        })

    def remove(self, resource: ProviderResource) -> None:
        """Delete exactly one owned resource and verify that it is gone."""
        if resource.intent is None or resource.kind == "volume":
            raise SessionError("RunPod removal requires an owned pod or boot secret.")
        prefix = _PODS if resource.kind == "venue" else _SECRETS
        path = f"{prefix}/{_identity(resource.id)}"
        with self._exclusive(resource.intent):
            deadline = time.monotonic() + 15
            if self._owned_detail(resource, path, timeout_seconds=self._remaining(deadline)) is None:
                return
            if resource.kind == "secret" and any(row.intent == resource.intent for row in self._venues()):
                raise SessionError("RunPod cannot release the boot secret while an owned pod remains.")
            try:
                reply = self._api("DELETE", path, timeout_seconds=self._remaining(deadline))
            except SessionError:
                if self._owned_detail(resource, path, timeout_seconds=self._remaining(deadline)) is None:
                    return
                raise
            if reply.status not in {204, 404}:
                raise SessionError(f"RunPod resource deletion returned HTTP {reply.status}.")
            while self._owned_detail(resource, path, timeout_seconds=self._remaining(deadline)) is not None:
                time.sleep(min(0.25, self._remaining(deadline)))
