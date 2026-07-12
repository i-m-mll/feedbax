"""Orchestration driver that wraps the existing Studio training worker HTTP API."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from feedbax.orchestration.bundle import RunBundle, RunRowSpec
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.studio_training import StudioTrainingAssemblySpec
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.events import RUN_EVENT_TERMINAL_TYPES, RunEvent
from feedbax.orchestration.state import RunSetState


class WorkerHttpDriver:
    """Drive one Studio training worker through the orchestration interface."""

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
        self._streams: dict[str, threading.Thread] = {}
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
        del state
        paths = _row_paths(bundle, row.row_id)
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
        del state
        paths = _row_paths(bundle, row.row_id)
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
        del state
        try:
            httpx.post(
                f"{self.base_url}/jobs/{row.row_id}/stop",
                headers=self._headers(),
                timeout=5.0,
            ).raise_for_status()
        finally:
            failed = _row_paths(bundle, row.row_id)["failed"]
            failed.write_text("stopped\n", encoding="utf-8")
        return {"row_id": row.row_id, "status": "stopped"}

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, str]:
        del state
        paths = _row_paths(bundle, row.row_id)
        dest_dir = bundle.run_set_dir / "collected" / row.row_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not paths["event_log"].exists():
            return {}
        dest = dest_dir / paths["event_log"].name
        dest.write_text(paths["event_log"].read_text(encoding="utf-8"), encoding="utf-8")
        return {paths["event_log"].name: str(dest)}

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        del bundle, state
        return {"driver": "worker-http"}

    def _headers(self) -> dict[str, str]:
        if self.auth_token is None:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}

    def _ensure_stream_thread(self, bundle: RunBundle, row: RunRowSpec) -> None:
        thread = self._streams.get(row.row_id)
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=self._stream_row_events,
            args=(bundle, row),
            name=f"feedbax-worker-http-events-{row.row_id}",
            daemon=True,
        )
        self._streams[row.row_id] = thread
        thread.start()

    def _stream_row_events(self, bundle: RunBundle, row: RunRowSpec) -> None:
        paths = _row_paths(bundle, row.row_id)
        try:
            with httpx.stream(
                "GET",
                f"{self.base_url}/jobs/{row.row_id}/stream",
                headers=self._headers(),
                timeout=None,
            ) as response:
                response.raise_for_status()
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
            self._stream_errors[row.row_id] = str(exc)

    def _orphan_probe(self, row_id: str, detail: str | None = None) -> DriverRowProbe:
        thread = self._streams.get(row_id)
        if thread is not None and thread.is_alive():
            return DriverRowProbe(status="running")
        error = self._stream_errors.get(row_id) or detail or "worker row is no longer reachable"
        return DriverRowProbe(status="failed", detail=f"orphaned: {error}")


def _worker_start_body(bundle: RunBundle, row: RunRowSpec) -> dict[str, Any]:
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
