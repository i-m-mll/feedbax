"""Manages the lifecycle of a cloud training instance."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from feedbax.web.orchestration.gcp import (
    InstanceConfig,
    InstanceInfo,
    InstanceStatus,
    create_instance,
    delete_instance,
    get_instance,
)
from feedbax.web.worker.client import wait_for_health

# Seconds between status polls when waiting for instance to reach RUNNING.
_POLL_INTERVAL = 10.0
# Maximum seconds to wait for instance to reach RUNNING state.
_INSTANCE_STARTUP_TIMEOUT = 300.0
# Maximum seconds to wait for the worker HTTP server to respond.
_WORKER_HEALTH_TIMEOUT = 120.0


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
    """

    instance: Optional[InstanceInfo] = None
    status: str = "idle"
    error: Optional[str] = None
    worker_url: Optional[str] = None
    project: Optional[str] = None
    zone: Optional[str] = None


class OrchestrationManager:
    """Manages GCP instance lifecycle and connects the TrainingService.

    Thread-safety is provided via an :class:`asyncio.Lock` so that concurrent
    requests cannot interleave launch/terminate operations.
    """

    def __init__(self) -> None:
        self._state = OrchestrationState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> OrchestrationState:
        """Return the current orchestration state (read-only snapshot)."""
        return self._state

    async def launch(
        self,
        config: InstanceConfig,
        training_service,
        instance_name: str,
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
        async with self._lock:
            self._state = OrchestrationState(
                status="creating",
                project=config.project,
                zone=config.zone,
            )

        try:
            # Step 1 — create instance.
            info = await create_instance(config, instance_name)
            async with self._lock:
                self._state.instance = info

            # Step 2 — wait for RUNNING.
            deadline = asyncio.get_event_loop().time() + _INSTANCE_STARTUP_TIMEOUT
            while True:
                info = await get_instance(config.project, config.zone, instance_name)
                async with self._lock:
                    self._state.instance = info

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

        except Exception as exc:
            async with self._lock:
                self._state.status = "error"
                self._state.error = str(exc)

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

        if instance is not None:
            try:
                await delete_instance(project, zone, instance.name)
            except Exception:
                pass

        # Always disconnect the training service, regardless of deletion outcome.
        try:
            training_service._terminate_worker()
        except Exception:
            pass

        async with self._lock:
            self._state = OrchestrationState(status="idle")

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

        return self._state


orchestration_manager = OrchestrationManager()
