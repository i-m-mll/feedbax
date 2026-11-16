"""RunPod realization and transport for the ordinary Host session driver."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.execution_target import OpeningStep
from feedbax.host import watchdog
from feedbax.host.client import HostClient, StdioAttachment
from feedbax.host.protocol import HostError, JsonObject, SessionAccess
from feedbax.orchestration import repo_tree, runpod_bootstrap
from feedbax.orchestration.repo_snapshot import RepoSnapshotRecord, SealedRepoSnapshot, verify_repo_snapshot
from feedbax.orchestration.runpod_cutoff import NativeRunPodCutoff
from feedbax.orchestration.runpod_session_provider import RunPodAcquisition
from feedbax.orchestration.session import (
    Digest, Positive, ProviderMachineObservation, ProviderResource, SessionDriver, SessionError, SessionLaunch, SessionModel,
    SessionOpeningFailed, SessionRealization, SessionRequest,
)
from feedbax.orchestration.session_jobs import SessionJobs
from feedbax.orchestration.venue_jobs import VenueJobs
from feedbax.persistence.ssh_store import SSHStoreConnection


type DependencyName = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]


class RunPodEnvironment(SessionModel):
    workspace: str
    bootstrap_python: str
    uv: str | None = None
    uv_sha256: Digest | None = None
    uv_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    snapshot: RepoSnapshotRecord
    backend: Literal["cpu", "gpu"]
    groups: tuple[DependencyName, ...] = ()
    extras: tuple[DependencyName, ...] = ()
    enable_x64: bool = False
    bootstrap_modules: tuple[str, ...] = ()

    @field_validator("workspace", "bootstrap_python", "uv")
    @classmethod
    def _absolute(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or path.as_posix() != value or "\x00" in value:
            raise ValueError("RunPod environment requires canonical absolute venue paths.")
        return value

    @model_validator(mode="after")
    def _uv_source(self) -> RunPodEnvironment:
        if (self.uv is None) == (self.uv_sha256 is None):
            raise ValueError("Declare an image uv path or a staged uv SHA-256.")
        return self

    @property
    def digest(self) -> str:
        uv_source = "uv_sha256" if self.uv_sha256 is None else "uv"
        declaration = self.model_dump(mode="json", exclude={uv_source})
        return hashlib.sha256(canonical_json_v2_bytes(declaration)).hexdigest()

    @property
    def worker_environment(self) -> dict[str, str]:
        return {
            "JAX_PLATFORMS": "cuda" if self.backend == "gpu" else "cpu",
            "JAX_ENABLE_X64": str(int(self.enable_x64)),
        }


@dataclass(frozen=True)
class _Layout:
    root: PurePosixPath

    @property
    def python(self) -> str:
        return str(self.root / "venv" / "bin" / "python")

    @property
    def host(self) -> PurePosixPath:
        return self.root / "host"


class RunPodSessionProvider:
    """Realize a session over its pinned SSH connection and expose ordinary Host RPCs."""

    target = "runpod"
    supports_idle_close = False
    supports_suspension = False

    def __init__(
        self, acquisition: RunPodAcquisition, environment: RunPodEnvironment,
        snapshot: SealedRepoSnapshot, *,
        uv_binary: Path | None = None,
        rsync_executable: str = "rsync", install_timeout: Positive = 300,
    ):
        if snapshot.record != environment.snapshot:
            raise SessionError("RunPod repository snapshot differs from the declared environment.")
        if type(install_timeout) is not int or install_timeout <= 0:
            raise ValueError("RunPod environment installation timeout must be positive.")
        self._uv_binary: bytes | None = None
        if uv_binary is not None:
            if environment.uv_sha256 is None:
                raise SessionError("A local uv binary requires a staged uv SHA-256.")
            if not uv_binary.is_file():
                raise SessionError("RunPod staged uv requires a regular local binary.")
            binary = uv_binary.read_bytes()
            if hashlib.sha256(binary).hexdigest() != environment.uv_sha256:
                raise SessionError("Local uv binary differs from the authorized SHA-256.")
            self._uv_binary = binary
        cutoff = acquisition.running_cutoff
        self._native_cutoff = cutoff if isinstance(cutoff, NativeRunPodCutoff) else None
        if self._native_cutoff is not None and (
            self._native_cutoff.workspace != environment.workspace
            or self._native_cutoff.bootstrap_python != environment.bootstrap_python
        ):
            raise SessionError("Native RunPod cutoff must use the declared venue root and interpreter.")
        self.acquisition = acquisition
        self.environment = environment
        self.snapshot = snapshot
        self._rsync = rsync_executable
        self._install_timeout = install_timeout
        self._bootstrap = "\n".join((
            Path(repo_tree.__file__).read_text(), Path(runpod_bootstrap.__file__).read_text(),
            "bootstrap(snapshot_tree_identity)",
        ))
        self._watchdog = Path(watchdog.__file__).read_text()

    @property
    def enforces_cutoff(self) -> bool:
        return self.acquisition.enforces_cutoff

    def hourly_microusd(self, request: SessionRequest) -> int:
        self._check_environment(request)
        return self.acquisition.hourly_microusd(request)

    def review_rate(self, request: SessionRequest) -> int:
        rate = self.hourly_microusd(request)
        self.acquisition.review_quote(request)
        return rate

    def _check_environment(self, request: SessionRequest) -> None:
        if self.environment.digest != request.environment:
            raise SessionError("The RunPod session names a different environment declaration.")

    def inventory(self) -> tuple[ProviderResource, ...]:
        return tuple(self.acquisition.inventory())

    def machine_observation(self, resource: ProviderResource) -> ProviderMachineObservation:
        return self.acquisition.machine_observation(resource)

    def acquire(self, launch: SessionLaunch) -> ProviderResource:
        self._check_environment(launch.request)
        self._check_uv_binary()
        if not self.enforces_cutoff:
            raise SessionError("RunPod running-session cutoff is not qualified.")
        return self.acquisition.acquire(launch)

    def _check_uv_binary(self) -> None:
        if self.environment.uv_sha256 is not None and self._uv_binary is None:
            raise SessionError("Opening requires the local binary for the authorized staged uv.")

    def remove(self, resource: ProviderResource) -> None:
        self.acquisition.remove(resource)

    def claim_opening_close(
        self, resource: ProviderResource | None, launch: SessionLaunch,
    ) -> tuple[ProviderResource, ...] | None:
        if resource is not None and resource.intent != launch.intent:
            raise SessionError("RunPod setup close requires the exact acquisition intent.")
        unopened = self.acquisition.claim_uninitialized(resource, launch)
        if unopened is not None or resource is None:
            return unopened
        raw = self._call(self._connection(resource), self._layout(resource), "close-opening")
        return (resource,) if json.loads(raw) is True else None

    def _layout(self, resource: ProviderResource) -> _Layout:
        if resource.kind != "venue" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resource.id):
            raise SessionError("Expected the exact RunPod venue identity.")
        return _Layout(PurePosixPath(self.environment.workspace) / "feedbax-sessions" / resource.id)

    def _connection(self, resource: ProviderResource) -> SSHStoreConnection:
        endpoint = self.acquisition.endpoint(resource)
        return SSHStoreConnection(endpoint.host, endpoint.ssh_argv, self._rsync)

    def _call(
        self, connection: SSHStoreConnection, layout: _Layout, operation: str,
        settings: dict[str, object] | None = None,
    ) -> bytes:
        return connection.run(
            [self.environment.bootstrap_python, "-I", "-c", self._bootstrap, operation, str(layout.root)],
            json.dumps(settings, allow_nan=False).encode(),
        )

    def _read(self, connection: SSHStoreConnection, layout: _Layout) -> SessionRealization | None:
        raw = self._call(connection, layout, "read")
        return SessionRealization.model_validate_json(raw) if raw else None

    def realization(self, resource: ProviderResource) -> SessionRealization:
        realization = self._read(self._connection(resource), self._layout(resource))
        if realization is None or realization.launch.intent != resource.intent:
            raise SessionError("RunPod venue has no matching realized session.")
        return realization

    def realize(
        self, resource: ProviderResource, launch: SessionLaunch, *,
        adoption_token: str, controller_token: str, opening: Callable[[OpeningStep], None],
    ) -> SessionRealization:
        self._check_environment(launch.request)
        if resource.intent != launch.intent or not self.enforces_cutoff:
            raise SessionError("RunPod realization requires the owned venue and scoped cutoff authority.")
        endpoint = self.acquisition.endpoint(
            resource, opening=True, realization_seconds=self._install_timeout + 45,
        )
        connection = SSHStoreConnection(endpoint.host, endpoint.ssh_argv, self._rsync)
        deadline = endpoint.opening_deadline
        assert deadline is not None
        try:
            return self._realize(
                resource, launch, connection, endpoint.placement, deadline,
                adoption_token=adoption_token, controller_token=controller_token, opening=opening,
            )
        except SessionOpeningFailed:
            raise
        except (SessionError, OSError, ValueError) as error:
            raise SessionOpeningFailed(f"RunPod realization failed: {error}") from error

    def _realize(
        self, resource: ProviderResource, launch: SessionLaunch, connection: SSHStoreConnection,
        placement: JsonObject, deadline: float, *, adoption_token: str, controller_token: str,
        opening: Callable[[OpeningStep], None],
    ) -> SessionRealization:
        def remaining() -> float:
            seconds = min(deadline - time.monotonic(), launch.ends_at_ms / 1000 - time.time())
            if seconds <= 0:
                raise SessionOpeningFailed("RunPod realization exceeded its opening deadline.")
            return seconds

        def bounded() -> SSHStoreConnection:
            return replace(connection, timeout_seconds=min(connection.timeout_seconds, remaining()))

        layout = self._layout(resource)
        native = self._native_cutoff
        cutoff_proof = native.read_proof(
            bounded().run(native.proof_argv(resource, launch), b""), resource, launch,
        ) if native is not None else None
        realized = self._read(bounded(), layout)
        if realized is None:
            self._check_uv_binary()
            prepare: dict[str, object] = {}
            if native is None:
                prepare = {
                    "watchdog": self._watchdog,
                    "worker_env": self.environment.worker_environment,
                    "policy": {
                        "ends_at_ms": launch.ends_at_ms,
                        "host_root": str(layout.host),
                        "host_python": layout.python,
                        "ready_file": str(layout.root / "ready"),
                        "terminate_argv": self.acquisition.terminate_argv(resource, launch),
                    },
                }
            prepare["uv_sha256"] = self.environment.uv_sha256
            prepared = json.loads(self._call(bounded(), layout, "prepare", prepare))
            if prepared["send_uv"] or prepared["send"]:
                opening("copying")
            if prepared["send_uv"]:
                digest, binary = self.environment.uv_sha256, self._uv_binary
                assert digest is not None and binary is not None
                bounded().run([
                    self.environment.bootstrap_python, "-I", "-c", self._bootstrap,
                    "stage-uv", str(layout.root), digest,
                ], binary)
            if prepared["send"]:
                record = self.snapshot.record
                verify_repo_snapshot(
                    self.snapshot.staging_root, content_sha256=record.content_sha256, file_count=record.file_count,
                )
                bounded().copy_tree(self.snapshot.staging_root, layout.root / "incoming")
            opening("installing")
            install_seconds = remaining()
            probe = self._call(replace(connection, timeout_seconds=install_seconds), layout, "install", {
                "content_sha256": self.environment.snapshot.content_sha256,
                "file_count": self.environment.snapshot.file_count,
                "timeout": min(self._install_timeout, install_seconds),
                "uv": self.environment.uv,
                "uv_sha256": self.environment.uv_sha256,
                "uv_version": self.environment.uv_version,
                "groups": self.environment.groups,
                "extras": self.environment.extras,
                "backend": self.environment.backend,
                "worker_env": self.environment.worker_environment,
            })
            evidence = json.loads(probe) | {"placement": placement}
            if self.environment.uv_sha256 is not None:
                evidence["uv_sha256"] = self.environment.uv_sha256
            if cutoff_proof is not None:
                evidence["running_cutoff"] = cutoff_proof
            realized = SessionRealization(
                launch=launch, environment=self.environment.digest, probe=cast(JsonObject, evidence),
            )
        if realized.launch != launch or realized.environment != self.environment.digest:
            raise SessionError("Existing RunPod realization identifies a different authorized launch.")
        if native is not None:
            proof = native.read_proof(
                bounded().run(native.proof_argv(resource, launch), b""), resource, launch,
            )
            if realized.probe.get("running_cutoff") != proof:
                raise SessionError("RunPod realization has different running cutoff evidence.")
        capability = SessionAccess(str(launch.request.id), 0, controller_token)
        opening("starting_host")
        remaining()
        self.acquisition.begin_completion(resource, launch)
        self._call(bounded(), layout, "complete", {
            "fence": {
                "session_id": capability.session_id, "operator": launch.authorization.operator,
                "expires_at": launch.ends_at_ms / 1000,
                "adoption_token": adoption_token, "controller_token": controller_token,
            },
            "realization": realized.model_dump_json(),
        })
        host = self._client(connection, layout, capability)
        ping_deadline = time.monotonic() + min(8, remaining())
        while True:
            try:
                timeout = min(5, remaining(), ping_deadline - time.monotonic())
                if timeout <= 0:
                    raise SessionOpeningFailed("RunPod Host did not respond during opening.")
                host.ping(timeout=timeout)
                remaining()
                return realized
            except HostError as error:
                if error.code != "disconnected":
                    raise
                time.sleep(min(.05, remaining(), max(0, ping_deadline - time.monotonic())))

    @staticmethod
    def _client(connection: SSHStoreConnection, layout: _Layout, access: SessionAccess | None) -> HostClient:
        attach = connection.command([
            layout.python, "-I", "-m", "feedbax.host", "--root", str(layout.host), "attach", "--no-start",
        ])
        return HostClient(attachment=StdioAttachment(lambda _: attach), session=access)

    def client(self, resource: ProviderResource, access: SessionAccess | None) -> HostClient:
        return self._client(self._connection(resource), self._layout(resource), access)

    def jobs(self, driver: SessionDriver, session_id: UUID) -> SessionJobs:
        resource = driver.record(session_id).venue
        if driver.provider is not self or resource is None:
            raise SessionError("Saved-plan submission requires this RunPod venue.")
        layout = self._layout(resource)
        return SessionJobs(driver, session_id, venue_jobs=VenueJobs(
            self._connection(resource), python=layout.python, cwd=layout.root / "repo",
            host_root=layout.host, projects_root=layout.root / "projects", stores_root=layout.root / "stores",
            bootstrap_modules=self.environment.bootstrap_modules, worker_environment=self.environment.worker_environment,
        ))
