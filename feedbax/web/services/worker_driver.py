"""Orchestration driver that wraps the existing Studio training worker HTTP API."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.studio_training import StudioTrainingAssemblySpec
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.capabilities import (
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverConstructionContext,
    DriverRegistration,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RecoverySemantics,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)
from feedbax.orchestration.events import RUN_EVENT_TERMINAL_TYPES, RunEvent
from feedbax.orchestration.state import RunSetState
from feedbax.web.worker.identity import require_worker_job_id


@dataclass
class _OwnedStream:
    """Locally owned state for one worker SSE stream."""

    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    response: Any | None = None


class WorkerStreamTeardownError(RuntimeError):
    """Raised after best-effort cleanup leaves stream failures or survivors."""


class WorkerHttpDriver:
    """Drive one Studio training worker through the orchestration interface."""

    poll_interval_seconds = 0.05
    stream_join_timeout_seconds = 1.0

    capability_envelope = DriverCapabilityEnvelope.single(
        "worker-http",
        DriverCapabilityFacts(
            variant_id="external-service",
            venue=DriverVenue.REMOTE_SERVICE,
            resources=ResourceSemantics.EXTERNALLY_MANAGED,
            spend=SpendSemantics.EXTERNALLY_MANAGED,
            authorization=AuthorizationSemantics.OPTIONAL_CALLER_CREDENTIAL,
            environment=EnvironmentSemantics.OPAQUE_DRIVER_IDENTITY,
            monitoring=MonitoringSemantics.EVENT_STREAM_AND_ROW_POLL,
            recovery=RecoverySemantics.NONE,
            retry=RetrySemantics.NONE,
            acquisition=AcquisitionSemantics.EXTERNALLY_PROVIDED,
            teardown=TeardownSemantics.RESOURCES_PRESERVED,
            custody=CustodySemantics.EXTERNAL_SERVICE,
        ),
    )
    realized_capabilities = capability_envelope.realize("external-service")

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str | None = None,
        request_timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.request_timeout = request_timeout
        self._stream_lock = threading.Lock()
        self._streams: dict[str, _OwnedStream] = {}
        self._stream_errors: dict[str, str] = {}

    def provision(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        del state
        for dirname in ("events", "sentinels", "rows", "collected", "inputs"):
            (bundle.run_set_dir / dirname).mkdir(parents=True, exist_ok=True)
        return {
            "driver": "worker-http",
            "worker_url": self.base_url,
            "run_set_dir": str(bundle.run_set_dir),
        }

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        del bundle, state
        return "worker-http"

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        del state
        inputs_dir = bundle.run_set_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        return {"inputs_dir": str(inputs_dir)}

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        import httpx

        del state
        row_id = require_worker_job_id(row.row_id)
        paths = _row_paths(bundle, row_id)
        if paths["done"].exists():
            return {"row_id": row.row_id, "status": "completed"}
        if paths["failed"].exists():
            return {"row_id": row.row_id, "status": "failed"}

        paths["events"].mkdir(parents=True, exist_ok=True)
        paths["sentinels"].mkdir(parents=True, exist_ok=True)
        paths["row_dir"].mkdir(parents=True, exist_ok=True)
        paths["started"].write_text(str(time.time()), encoding="utf-8")

        body = _worker_start_body(bundle, row)
        response = httpx.post(
            f"{self.base_url}/start",
            json=body,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        self._ensure_stream_thread(bundle, row)
        return {"row_id": row.row_id, "status": "running"}

    def probe(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> DriverRowProbe:
        import httpx

        del state
        row_id = require_worker_job_id(row.row_id)
        paths = _row_paths(bundle, row_id)
        if paths["done"].exists():
            return DriverRowProbe(status="completed")
        if paths["failed"].exists():
            return DriverRowProbe(
                status="failed",
                detail=paths["failed"].read_text(encoding="utf-8").strip() or None,
            )
        try:
            response = httpx.get(
                f"{self.base_url}/jobs/{row.row_id}/status",
                headers=self._headers(),
                timeout=2.0,
            )
            if response.status_code == 404:
                return self._orphan_probe(row.row_id)
            response.raise_for_status()
        except Exception as exc:
            return self._orphan_probe(row.row_id, detail=str(exc))

        payload = response.json()
        status = payload.get("status")
        if status == "completed":
            paths["done"].write_text("status=completed\n", encoding="utf-8")
            return DriverRowProbe(status="completed")
        if status == "error":
            detail = str(payload.get("error") or "worker status=error")
            paths["failed"].write_text(detail + "\n", encoding="utf-8")
            return DriverRowProbe(status="failed", detail=detail)
        self._ensure_stream_thread(bundle, row)
        return DriverRowProbe(status="running")

    def stop_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        import httpx

        del state
        row_id = require_worker_job_id(row.row_id)
        try:
            httpx.post(
                f"{self.base_url}/jobs/{row_id}/stop",
                headers=self._headers(),
                timeout=5.0,
            ).raise_for_status()
        finally:
            failed = _row_paths(bundle, row_id)["failed"]
            failed.write_text("stopped\n", encoding="utf-8")
        return {"row_id": row.row_id, "status": "stopped"}

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, str]:
        del state
        row_id = require_worker_job_id(row.row_id)
        paths = _row_paths(bundle, row_id)
        dest_dir = bundle.run_set_dir / "collected" / row_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not paths["event_log"].exists():
            return {}
        dest = dest_dir / paths["event_log"].name
        dest.write_text(paths["event_log"].read_text(encoding="utf-8"), encoding="utf-8")
        return {paths["event_log"].name: str(dest)}

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        del bundle, state
        with self._stream_lock:
            streams = tuple(self._streams.items())
            for _, stream in streams:
                stream.cancel.set()
            self._stream_errors.clear()

        failures: list[str] = []
        for row_id, stream in streams:
            response = stream.response
            if response is not None:
                try:
                    response.close()
                except Exception as exc:
                    failures.append(f"{row_id}: response close failed: {exc}")
        for row_id, stream in streams:
            thread = stream.thread
            if thread is None:
                continue
            if thread is not threading.current_thread():
                thread.join(timeout=self.stream_join_timeout_seconds)
            if thread.is_alive():
                failures.append(f"{row_id}: stream thread did not terminate")

        with self._stream_lock:
            for row_id, stream in streams:
                thread = stream.thread
                if (
                    self._streams.get(row_id) is stream
                    and thread is not None
                    and not thread.is_alive()
                ):
                    del self._streams[row_id]

        if failures:
            raise WorkerStreamTeardownError("; ".join(failures))
        return {"driver": "worker-http"}

    def _headers(self) -> dict[str, str]:
        if self.auth_token is None:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}

    def _ensure_stream_thread(self, bundle: RunBundle, row: RunRowSpec) -> None:
        row_id = require_worker_job_id(row.row_id)
        with self._stream_lock:
            if row_id in self._streams:
                return
            stream = _OwnedStream()
            thread = threading.Thread(
                target=self._stream_row_events,
                args=(bundle, row, stream),
                name=f"feedbax-worker-http-events-{row.row_id}",
                daemon=True,
            )
            stream.thread = thread
            self._streams[row_id] = stream
            try:
                thread.start()
            except BaseException:
                if self._streams.get(row_id) is stream:
                    del self._streams[row_id]
                raise

    def _stream_row_events(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        stream: _OwnedStream | None = None,
    ) -> None:
        import httpx

        row_id = require_worker_job_id(row.row_id)
        paths = _row_paths(bundle, row_id)
        if stream is None:
            stream = _OwnedStream()
        try:
            with httpx.stream(
                "GET",
                f"{self.base_url}/jobs/{row_id}/stream",
                headers=self._headers(),
                timeout=self.request_timeout,
            ) as response:
                with self._stream_lock:
                    stream.response = response
                if stream.cancel.is_set():
                    response.close()
                    return
                response.raise_for_status()
                response.request.extensions["timeout"]["read"] = None
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    event = RunEvent.model_validate_json(payload)
                    with paths["event_log"].open("a", encoding="utf-8") as handle:
                        handle.write(event.model_dump_json(exclude_none=True) + "\n")
                    if event.type == "complete":
                        paths["done"].write_text("event=complete\n", encoding="utf-8")
                    elif event.type == "failed":
                        paths["failed"].write_text("event=failed\n", encoding="utf-8")
                    if event.type in RUN_EVENT_TERMINAL_TYPES:
                        return
        except Exception as exc:
            if not stream.cancel.is_set():
                with self._stream_lock:
                    self._stream_errors[row_id] = str(exc)
        finally:
            with self._stream_lock:
                stream.response = None
                if self._streams.get(row_id) is stream:
                    del self._streams[row_id]

    def _orphan_probe(self, row_id: str, detail: str | None = None) -> DriverRowProbe:
        row_id = require_worker_job_id(row_id)
        with self._stream_lock:
            stream = self._streams.get(row_id)
            thread = stream.thread if stream is not None else None
            error = self._stream_errors.get(row_id)
        if thread is not None and thread.is_alive():
            return DriverRowProbe(status="running")
        error = error or detail or "worker row is no longer reachable"
        return DriverRowProbe(status="failed", detail=f"orphaned: {error}")


def worker_http_driver_registration() -> DriverRegistration:
    """Return the built-in context-aware Studio worker driver registration."""

    def resolve(context: DriverConstructionContext):
        del context
        return WorkerHttpDriver.realized_capabilities

    def factory(context: DriverConstructionContext, realized):
        configuration = context.configuration
        if context.recovery_inputs:
            raise ValueError("worker-http capability variant does not support recovery inputs")
        base_url = configuration.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("worker-http driver configuration requires a non-empty base_url")
        request_timeout = configuration.get("request_timeout", 10.0)
        if not isinstance(request_timeout, (int, float)):
            raise TypeError("worker-http request_timeout must be numeric")
        driver = WorkerHttpDriver(
            base_url=base_url,
            auth_token=context.credentials.get("worker_http_token"),
            request_timeout=float(request_timeout),
        )
        if driver.realized_capabilities != realized:
            raise ValueError("worker-http factory received inconsistent realized capabilities")
        return driver

    return DriverRegistration(
        name="worker-http",
        supported_capabilities=WorkerHttpDriver.capability_envelope,
        resolve_capabilities=resolve,
        factory=factory,
    )


def _worker_start_body(bundle: RunBundle, row: RunRowSpec) -> dict[str, Any]:
    require_worker_job_id(row.row_id)
    body = load_worker_execution_payload(row)
    body["job_id"] = row.row_id
    body["run_set_id"] = bundle.run_set_id
    return body


def load_worker_execution_payload(row: RunRowSpec) -> dict[str, Any]:
    """Resolve and validate a row's registered Studio worker payload."""
    payload_ref = row.execution.payload
    family = default_spec_registry.resolve("StudioTrainingAssemblySpec")
    if family.identity != payload_ref.schema_id:
        raise ValueError(
            "WorkerHttpDriver execution payload schema does not match its registered family: "
            f"expected={family.identity!r}, actual={payload_ref.schema_id!r}"
        )
    if payload_ref.uri is None:
        raise ValueError("WorkerHttpDriver execution payload requires a materialization URI")
    payload_path = Path(payload_ref.uri).expanduser()
    data = payload_path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != payload_ref.sha256:
        raise ValueError(
            "WorkerHttpDriver execution payload byte digest mismatch: "
            f"expected={payload_ref.sha256!r}, actual={actual_sha256!r}"
        )
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError("WorkerHttpDriver execution payload must decode to a JSON object")
    migrated = default_spec_registry.migrate("StudioTrainingAssemblySpec", raw)
    return StudioTrainingAssemblySpec.model_validate(migrated.payload).worker_payload()


def _row_paths(bundle: RunBundle, row_id: str) -> dict[str, Path]:
    row_id = require_worker_job_id(row_id)
    run_set_dir = bundle.run_set_dir
    sentinels = run_set_dir / "sentinels"
    events = run_set_dir / "events"
    row_dir = run_set_dir / "rows" / row_id
    return {
        "sentinels": sentinels,
        "events": events,
        "row_dir": row_dir,
        "started": sentinels / f"{row_id}.started",
        "done": sentinels / f"{row_id}.done",
        "failed": sentinels / f"{row_id}.failed",
        "event_log": events / f"{row_id}.events.jsonl",
    }
