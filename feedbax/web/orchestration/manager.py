"""Manages the lifecycle of a cloud training instance."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from feedbax.web.orchestration.gcp import (
    InstanceConfig,
    InstanceInfo,
    InstanceStatus,
    create_instance,
    delete_instance,
    get_instance,
    list_instances,
)
from feedbax.web.worker.client import wait_for_health

# Seconds between status polls when waiting for instance to reach RUNNING.
_POLL_INTERVAL = 10.0
# Maximum seconds to wait for instance to reach RUNNING state.
_INSTANCE_STARTUP_TIMEOUT = 300.0
# Maximum seconds to wait for the worker HTTP server to respond.
_WORKER_HEALTH_TIMEOUT = 120.0
_WORKER_HEALTH_REFRESH_TIMEOUT = 2.0
_WORKER_HEALTH_FAILURE_THRESHOLD = 2
_ACTIVE_STATUSES = {"creating", "connecting", "running"}

logger = logging.getLogger(__name__)


def _default_state_path() -> Path:
    """Return the durable state path used by the process-local singleton."""
    return Path.home() / ".cache" / "feedbax" / "orchestration-state.json"


@dataclass
class OrchestrationState:
    """Snapshot of the current cloud orchestration state.

    Args:
        instance: The :class:`~feedbax.web.orchestration.gcp.InstanceInfo` for
            the active instance, or ``None`` if no instance is running.
        status: Human-readable lifecycle status string.  One of:
            ``"idle"``, ``"creating"``, ``"connecting"``, ``"running"``,
            ``"preempted"``, ``"error"``.
        error: Error message when *status* is ``"error"``, otherwise ``None``.
        worker_url: Base URL for the running worker, e.g.
            ``"http://1.2.3.4:8765"``.  Set once the worker is healthy.
        project: GCP project ID stored so that terminate/refresh can use it
            without requiring the caller to track it separately.
        zone: GCP zone stored alongside the project for the same reason.
        worker_port: Worker port for reconstructing worker URL after restart.
        auth_token: In-memory worker auth token.  It is deliberately not
            serialized to durable state.
        worker_health_failures: Consecutive failed health checks observed
            while the VM itself is still running.
        orphaned_instance: Instance name that could not be deleted after a
            failed launch attempt, if cleanup failed.
    """

    instance: Optional[InstanceInfo] = None
    status: str = "idle"
    error: Optional[str] = None
    worker_url: Optional[str] = None
    project: Optional[str] = None
    zone: Optional[str] = None
    worker_port: Optional[int] = None
    auth_token: Optional[str] = None
    worker_health_failures: int = 0
    orphaned_instance: Optional[str] = None


class OrchestrationManager:
    """Manages GCP instance lifecycle and connects the TrainingService.

    Thread-safety is provided via an :class:`asyncio.Lock` so that concurrent
    requests cannot interleave launch/terminate operations.
    """

    def __init__(self, state_path: str | Path | None = None) -> None:
        self._state_path = Path(state_path) if state_path is not None else _default_state_path()
        self._state = self._load_state()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> OrchestrationState:
        """Return the current orchestration state (read-only snapshot)."""
        return self._state

    def _load_state(self) -> OrchestrationState:
        """Load durable orchestration state from disk if present."""
        try:
            payload = json.loads(self._state_path.read_text())
        except FileNotFoundError:
            return OrchestrationState()
        except Exception:
            logger.exception("Failed to read orchestration state from %s", self._state_path)
            return OrchestrationState(status="error", error="Failed to read orchestration state")
        try:
            return _state_from_json(payload)
        except Exception:
            logger.exception("Failed to parse orchestration state from %s", self._state_path)
            return OrchestrationState(status="error", error="Failed to parse orchestration state")

    def _persist_state_unlocked(self) -> None:
        """Write the current state to disk without persisting secrets."""
        payload = _state_to_json(self._state)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self._state_path)

    async def _set_state(self, state: OrchestrationState) -> None:
        async with self._lock:
            self._state = state
            self._persist_state_unlocked()

    async def reserve_launch(
        self,
        config: InstanceConfig,
        instance_name: str,
    ) -> OrchestrationState:
        """Atomically reserve the launch slot before billable provisioning."""
        async with self._lock:
            if self._state.status in _ACTIVE_STATUSES:
                raise RuntimeError(
                    f"orchestration launch already active with status {self._state.status!r}"
                )
            self._state = OrchestrationState(
                instance=InstanceInfo(
                    name=instance_name,
                    status=InstanceStatus.CREATING,
                    zone=config.zone,
                    machine_type=config.machine_type,
                ),
                status="creating",
                project=config.project,
                zone=config.zone,
                worker_port=config.worker_port,
                auth_token=config.auth_token,
            )
            self._persist_state_unlocked()
            return self._state

    async def launch(
        self,
        config: InstanceConfig,
        training_service,
        instance_name: str,
        *,
        reserved: bool = False,
    ) -> OrchestrationState:
        """Create a GCP instance and connect *training_service* to its worker.

        The method is intentionally designed to be called from a background
        ``asyncio`` task so that the HTTP response can be returned immediately
        while the long-running provisioning happens in the background.

        Steps:

        1. Create the GCP instance (startup script installs the worker).
        2. Poll :func:`~feedbax.web.orchestration.gcp.get_instance` until
           ``status == RUNNING`` (timeout :data:`_INSTANCE_STARTUP_TIMEOUT`).
        3. Compute ``worker_url`` from ``external_ip`` and ``config.worker_port``.
        4. Poll :func:`~feedbax.web.worker.client.wait_for_health` with a
           generous timeout (:data:`_WORKER_HEALTH_TIMEOUT`) to allow the
           startup script to finish installing dependencies.
        5. Call ``training_service.connect_remote(worker_url, config.auth_token)``.
        6. Set ``state.status = "running"``.

        Args:
            config: Instance configuration.
            training_service: A :class:`~feedbax.web.services.training_service.TrainingService`
                instance.  Must expose ``connect_remote(url, auth_token)``.
            instance_name: Name for the GCP instance.

        Returns:
            The updated :class:`OrchestrationState`.
        """
        if not reserved:
            await self.reserve_launch(config, instance_name)

        created_instance: Optional[InstanceInfo] = None

        try:
            # Step 1 — create instance.
            info = await create_instance(config, instance_name)
            created_instance = info
            async with self._lock:
                self._state.instance = info
                self._state.worker_port = config.worker_port
                self._state.auth_token = config.auth_token
                self._persist_state_unlocked()

            # Step 2 — wait for RUNNING.
            deadline = asyncio.get_event_loop().time() + _INSTANCE_STARTUP_TIMEOUT
            while True:
                info = await get_instance(config.project, config.zone, instance_name)
                async with self._lock:
                    self._state.instance = info
                    self._persist_state_unlocked()

                if info.status == InstanceStatus.RUNNING:
                    break
                if info.status in (
                    InstanceStatus.TERMINATED,
                    InstanceStatus.STOPPING,
                    InstanceStatus.PREEMPTED,
                ):
                    raise RuntimeError(
                        f"Instance entered terminal state: {info.status.value}"
                    )
                if asyncio.get_event_loop().time() >= deadline:
                    raise RuntimeError(
                        "Timed out waiting for instance to reach RUNNING state"
                    )
                await asyncio.sleep(_POLL_INTERVAL)

            # Step 3 — compute worker URL.
            if info.external_ip is None:
                raise RuntimeError("Instance has no external IP after reaching RUNNING state")
            worker_url = f"http://{info.external_ip}:{config.worker_port}"

            # Step 4 — wait for worker to be healthy.
            async with self._lock:
                self._state.status = "connecting"
                self._state.worker_url = worker_url
                self._persist_state_unlocked()

            await wait_for_health(
                worker_url,
                timeout=_WORKER_HEALTH_TIMEOUT,
                interval=5.0,
                auth_token=config.auth_token,
            )

            # Step 5 — connect TrainingService.
            training_service.connect_remote(worker_url, config.auth_token)

            # Step 6 — mark running.
            async with self._lock:
                self._state.status = "running"
                self._state.worker_health_failures = 0
                self._persist_state_unlocked()

        except Exception as exc:
            orphaned_instance: Optional[str] = None
            if created_instance is not None:
                try:
                    await delete_instance(config.project, config.zone, created_instance.name)
                except Exception:
                    orphaned_instance = created_instance.name
                    logger.exception(
                        "Failed to delete cloud instance %s after launch failure",
                        created_instance.name,
                    )

            async with self._lock:
                self._state.status = "error"
                self._state.error = str(exc)
                self._state.orphaned_instance = orphaned_instance
                if orphaned_instance is None:
                    self._state.instance = None
                self._persist_state_unlocked()

        return self._state

    async def terminate(self, training_service) -> None:
        """Destroy the current instance and disconnect TrainingService.

        Uses the project and zone stored in :attr:`state` when the instance
        was launched.

        Args:
            training_service: A :class:`~feedbax.web.services.training_service.TrainingService`
                instance.
        """
        async with self._lock:
            instance = self._state.instance
            project = self._state.project or ""
            zone = self._state.zone or ""

        delete_error: Optional[Exception] = None
        if instance is not None:
            try:
                await delete_instance(project, zone, instance.name)
            except Exception as exc:
                delete_error = exc
                logger.exception("Failed to delete cloud instance %s", instance.name)

        # Always disconnect the training service, regardless of deletion outcome.
        disconnect_error: Optional[Exception] = None
        try:
            training_service._terminate_worker()
        except Exception as exc:
            disconnect_error = exc
            logger.exception("Failed to disconnect training service from worker")

        if delete_error is not None:
            async with self._lock:
                self._state.status = "error"
                self._state.error = f"Failed to delete instance {instance.name}: {delete_error}"
                self._state.orphaned_instance = instance.name
                self._persist_state_unlocked()
            raise RuntimeError(self._state.error) from delete_error

        if disconnect_error is not None:
            async with self._lock:
                self._state = OrchestrationState(
                    status="error",
                    error=f"Failed to disconnect worker: {disconnect_error}",
                    project=project or None,
                    zone=zone or None,
                )
                self._persist_state_unlocked()
            raise RuntimeError(self._state.error) from disconnect_error

        async with self._lock:
            self._state = OrchestrationState(status="idle")
            self._persist_state_unlocked()

    async def refresh_status(self) -> OrchestrationState:
        """Poll GCP for the current instance status and detect preemption.

        Uses the project and zone stored in :attr:`state` when the instance
        was launched.  If the instance has been preempted or terminated
        externally, the orchestration state is updated to reflect this.

        Returns:
            The updated :class:`OrchestrationState`.
        """
        async with self._lock:
            instance = self._state.instance
            current_status = self._state.status
            project = self._state.project or ""
            zone = self._state.zone or ""
            worker_url = self._state.worker_url
            auth_token = self._state.auth_token

        if instance is None or current_status in ("idle", "error"):
            return self._state

        try:
            info = await get_instance(project, zone, instance.name)
        except Exception:
            # gcloud call failed — preserve existing state rather than masking.
            return self._state

        async with self._lock:
            self._state.instance = info
            if info.status == InstanceStatus.PREEMPTED:
                self._state.status = "preempted"
                self._state.error = "Instance was preempted"
            elif info.status in (InstanceStatus.TERMINATED, InstanceStatus.STOPPING):
                if self._state.status == "running":
                    self._state.status = "error"
                    self._state.error = f"Instance entered state: {info.status.value}"
            self._persist_state_unlocked()

        if (
            current_status == "running"
            and info.status == InstanceStatus.RUNNING
            and worker_url is not None
        ):
            try:
                await wait_for_health(
                    worker_url,
                    timeout=_WORKER_HEALTH_REFRESH_TIMEOUT,
                    interval=0.2,
                    auth_token=auth_token,
                )
            except Exception as exc:
                async with self._lock:
                    self._state.worker_health_failures += 1
                    if self._state.worker_health_failures >= _WORKER_HEALTH_FAILURE_THRESHOLD:
                        self._state.status = "error"
                        self._state.error = f"Worker health check failed: {exc}"
                    self._persist_state_unlocked()
            else:
                async with self._lock:
                    self._state.worker_health_failures = 0
                    self._state.error = None
                    self._persist_state_unlocked()

        return self._state

    async def reconcile_from_provider(self) -> OrchestrationState:
        """Reconcile persisted state with provider-visible Feedbax instances."""
        async with self._lock:
            state = self._state
            project = state.project
            zone = state.zone
            expected_name = state.instance.name if state.instance is not None else None
            worker_port = state.worker_port

        if not project or not zone or state.status not in _ACTIVE_STATUSES:
            return self._state

        instances = await list_instances(project, zone)
        match = next(
            (instance for instance in instances if expected_name is None or instance.name == expected_name),
            None,
        )
        if match is None:
            await self._set_state(
                OrchestrationState(
                    status="error",
                    error="Persisted cloud instance was not found during startup reconcile",
                    project=project,
                    zone=zone,
                    worker_port=worker_port,
                )
            )
            return self._state

        worker_url = self._state.worker_url
        if worker_url is None and match.external_ip is not None and worker_port is not None:
            worker_url = f"http://{match.external_ip}:{worker_port}"

        status = "running" if match.status == InstanceStatus.RUNNING else "creating"
        if match.status in (InstanceStatus.TERMINATED, InstanceStatus.STOPPING):
            status = "error"
        if match.status == InstanceStatus.PREEMPTED:
            status = "preempted"

        await self._set_state(
            OrchestrationState(
                instance=match,
                status=status,
                error=(
                    f"Instance entered state: {match.status.value}"
                    if status == "error"
                    else None
                ),
                worker_url=worker_url,
                project=project,
                zone=zone,
                worker_port=worker_port,
            )
        )
        return self._state


def _instance_to_json(instance: InstanceInfo | None) -> dict[str, object] | None:
    if instance is None:
        return None
    return {
        "name": instance.name,
        "status": instance.status.value,
        "internal_ip": instance.internal_ip,
        "external_ip": instance.external_ip,
        "tailscale_ip": instance.tailscale_ip,
        "zone": instance.zone,
        "machine_type": instance.machine_type,
    }


def _instance_from_json(payload: dict[str, object] | None) -> InstanceInfo | None:
    if payload is None:
        return None
    status = InstanceStatus(str(payload.get("status", InstanceStatus.UNKNOWN.value)))
    return InstanceInfo(
        name=str(payload.get("name", "")),
        status=status,
        internal_ip=payload.get("internal_ip") if isinstance(payload.get("internal_ip"), str) else None,
        external_ip=payload.get("external_ip") if isinstance(payload.get("external_ip"), str) else None,
        tailscale_ip=payload.get("tailscale_ip") if isinstance(payload.get("tailscale_ip"), str) else None,
        zone=str(payload.get("zone", "")),
        machine_type=str(payload.get("machine_type", "")),
    )


def _state_to_json(state: OrchestrationState) -> dict[str, object]:
    return {
        "schema_version": "feedbax.orchestration.state.v1",
        "instance": _instance_to_json(state.instance),
        "status": state.status,
        "error": state.error,
        "worker_url": state.worker_url,
        "project": state.project,
        "zone": state.zone,
        "worker_port": state.worker_port,
        "worker_health_failures": state.worker_health_failures,
        "orphaned_instance": state.orphaned_instance,
    }


def _state_from_json(payload: dict[str, object]) -> OrchestrationState:
    if payload.get("schema_version") != "feedbax.orchestration.state.v1":
        raise ValueError("unsupported orchestration state schema")
    instance_payload = payload.get("instance")
    return OrchestrationState(
        instance=_instance_from_json(
            instance_payload if isinstance(instance_payload, dict) else None
        ),
        status=str(payload.get("status", "idle")),
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        worker_url=payload.get("worker_url")
        if isinstance(payload.get("worker_url"), str)
        else None,
        project=payload.get("project") if isinstance(payload.get("project"), str) else None,
        zone=payload.get("zone") if isinstance(payload.get("zone"), str) else None,
        worker_port=payload.get("worker_port")
        if isinstance(payload.get("worker_port"), int)
        else None,
        worker_health_failures=(
            int(payload.get("worker_health_failures", 0))
            if isinstance(payload.get("worker_health_failures", 0), int)
            else 0
        ),
        orphaned_instance=payload.get("orphaned_instance")
        if isinstance(payload.get("orphaned_instance"), str)
        else None,
    )


orchestration_manager = OrchestrationManager()
