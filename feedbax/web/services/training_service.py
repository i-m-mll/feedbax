"""Studio custody for retained HTTP worker runs."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence

from feedbax.contracts.base import utc_now
from feedbax.contracts.studio_training import STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID
from feedbax.orchestration.bundle import RunBundle
from feedbax.orchestration.events import RunEvent, RunEventReader
from feedbax.orchestration.stages import STAGE_ASSEMBLE
from feedbax.orchestration.state import RunSetState, RunSetStateStore
from feedbax.web.services.worker_driver import load_worker_execution_payload
from feedbax.web.worker.identity import require_worker_job_id
from feedbax.web.worker.transport import (
    WorkerConfigurationError,
    WorkerEndpoint,
    WorkerResponseError,
    request_json,
)


logger = logging.getLogger(__name__)
_OWNER_EXIT = "Studio restarted before the local worker finished"
_FORCED_STOP = "Studio force-stopped the local worker"


class RunStateCorruptionError(RuntimeError):
    """A persisted Studio orchestration document cannot be admitted safely."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__("Persisted Studio run state is corrupt")
        self.path = path
        self.reason = reason


class _RunStateDocumentUnavailable(FileNotFoundError):
    """A persisted document disappeared during a concurrent lifecycle transition."""


@dataclass(frozen=True)
class _JobRef:
    """Rebuildable pointer from a Studio job id to durable orchestration state."""

    job_id: str
    run_set_id: str
    state_path: Path
    bundle_path: Path


def _orchestration_parent_root() -> Path:
    configured = os.environ.get("FEEDBAX_ORCHESTRATION_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "feedbax" / "orchestration"


def _allowed_worker_origins_from_env() -> tuple[str, ...]:
    raw = os.environ.get("FEEDBAX_WORKER_ALLOWED_ORIGINS")
    if raw is None:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerConfigurationError(
            "FEEDBAX_WORKER_ALLOWED_ORIGINS must be a JSON array of exact origins"
        ) from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise WorkerConfigurationError(
            "FEEDBAX_WORKER_ALLOWED_ORIGINS must be a JSON array of exact origins"
        )
    return tuple(value)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TrainingService:
    """Recover, list and stop retained Studio worker runs."""

    def __init__(self, *, allowed_remote_origins: Sequence[str] | None = None) -> None:
        self._endpoint: WorkerEndpoint | None = None
        self._allowed_remote_origins = tuple(
            allowed_remote_origins
            if allowed_remote_origins is not None
            else _allowed_worker_origins_from_env()
        )
        self._job_refs_by_job: dict[str, _JobRef] = {}

        env_url = os.environ.get("FEEDBAX_WORKER_URL")
        if env_url:
            self.connect_remote(env_url, os.environ.get("FEEDBAX_WORKER_AUTH_TOKEN"))

    def connect_remote(self, url: str, auth_token: Optional[str] = None) -> None:
        """Configure the exact worker origin for retained cancellation."""
        endpoint = WorkerEndpoint.create(
            url,
            credential=auth_token,
            allowed_origins=self._allowed_remote_origins,
        )
        self._endpoint = endpoint

    def reconcile_desktop_worker(self, jobs: Path, *, forced: bool = False) -> None:
        """Settle incomplete legacy rows after their inherited worker lock is released."""
        jobs.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 5
        with (jobs / "worker.lock").open("a+b") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("previous Studio training worker has not exited")
                    time.sleep(0.05)
            self.rebuild_cache_from_state_docs()
            for ref in self._job_refs_by_job.values():
                status = self._status_from_state(ref.job_id)
                if status is None or status["status"] not in ("pending", "ready", "launched", "running"):
                    continue
                store = RunSetStateStore(ref.state_path)
                with store.lock(break_stale=True):
                    state = store.load()
                    row = state.rows[ref.job_id].model_copy(update={
                        "status": "failed", "error": _FORCED_STOP if forced else _OWNER_EXIT,
                        "completed_at": utc_now(),
                    })
                    store.save(state.with_row(ref.job_id, row))

    def _terminate_worker(self) -> None:
        """Release the configured worker connection."""
        self._endpoint = None

    async def stop_training(self, job_id: str) -> None:
        """Request a cooperative stop. The worker publishes the actual outcome."""
        ref = self._job_ref_for(job_id)
        if self._endpoint is None:
            if ref is None:
                raise ValueError(f"Unknown job {job_id!r}")
            return
        try:
            require_worker_job_id(job_id)
            await request_json(self._endpoint, "POST", f"/jobs/{job_id}/stop")
        except WorkerResponseError as exc:
            if exc.status_code == 404:
                raise ValueError(f"Unknown job {job_id!r}") from exc
            raise

    def rebuild_cache_from_state_docs(self) -> None:
        """Rebuild the in-memory job index from persisted orchestration state."""
        refs: dict[str, _JobRef] = {}
        for state_path in self._iter_state_paths():
            ref = _JobRef("", "", state_path, state_path.with_name("bundle.json"))
            try:
                state = self._load_state(ref)
            except _RunStateDocumentUnavailable:
                continue
            bundle_path = state_path.with_name("bundle.json")
            if not bundle_path.exists():
                if state.stage(STAGE_ASSEMBLE).status == "completed":
                    self._raise_run_state_corruption(
                        bundle_path,
                        "completed ASSEMBLE state is missing its bundle document",
                    )
                continue
            try:
                bundle_text = self._read_document(bundle_path)
            except _RunStateDocumentUnavailable:
                continue
            try:
                bundle = RunBundle.model_validate_json(bundle_text)
            except Exception as exc:
                self._raise_run_state_corruption(
                    bundle_path,
                    f"bundle validation failed: {type(exc).__name__}: {exc}",
                    cause=exc,
                )
            self._validate_state_bundle(
                state_path,
                state,
                bundle.run_set_id,
                [row.row_id for row in bundle.rows],
            )
            for row in bundle.rows:
                if row.execution.payload.schema_id != STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID:
                    continue
                refs[row.row_id] = _JobRef(
                    job_id=row.row_id,
                    run_set_id=bundle.run_set_id,
                    state_path=state_path,
                    bundle_path=bundle_path,
                )
        self._job_refs_by_job = refs

    def list_live_training_runs(self) -> list[dict[str, Any]]:
        """Return state-backed rows that may not have durable manifests yet."""
        self.rebuild_cache_from_state_docs()
        rows: list[dict[str, Any]] = []
        for ref in self._job_refs_by_job.values():
            status = self._status_from_state(ref.job_id)
            if status is None:
                continue
            rows.append(
                {
                    "id": status.get("manifest_id") or ref.job_id,
                    "name": ref.job_id,
                    "created_at": status.get("created_at", ""),
                    "status": status.get("status", "unknown"),
                    "hyperparams": {"run_set_id": ref.run_set_id},
                    "metrics": {"last_loss": status.get("last_loss")},
                    "provenance_id": ref.run_set_id,
                    "job_id": ref.job_id,
                    "run_set_id": ref.run_set_id,
                    "error": status.get("error"),
                    "completed_batches": status["batch"],
                    "total_batches": status["total_batches"],
                }
            )
        return rows

    def _job_ref_for(self, job_id: str) -> _JobRef | None:
        ref = self._job_refs_by_job.get(job_id)
        if ref is not None:
            return ref
        self.rebuild_cache_from_state_docs()
        return self._job_refs_by_job.get(job_id)

    def _load_bundle(self, ref: _JobRef) -> RunBundle:
        try:
            return RunBundle.model_validate_json(self._read_document(ref.bundle_path))
        except _RunStateDocumentUnavailable:
            raise
        except Exception as exc:
            self._raise_run_state_corruption(
                ref.bundle_path,
                f"bundle validation failed: {type(exc).__name__}: {exc}",
                cause=exc,
            )

    def _load_state(self, ref: _JobRef) -> RunSetState:
        try:
            return RunSetState.model_validate_json(self._read_document(ref.state_path))
        except _RunStateDocumentUnavailable:
            raise
        except Exception as exc:
            self._raise_run_state_corruption(
                ref.state_path,
                f"state validation failed: {type(exc).__name__}: {exc}",
                cause=exc,
            )

    @staticmethod
    def _read_document(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise _RunStateDocumentUnavailable(path) from exc

    @staticmethod
    def _raise_run_state_corruption(
        path: Path,
        reason: str,
        *,
        cause: Exception | None = None,
    ) -> NoReturn:
        logger.error("Corrupt Studio run-state document at %s: %s", path, reason)
        raise RunStateCorruptionError(path, reason) from cause

    def _validate_state_bundle(
        self,
        state_path: Path,
        state: RunSetState,
        bundle_run_set_id: str,
        bundle_row_ids: list[str],
    ) -> None:
        if state.run_set_id != bundle_run_set_id:
            self._raise_run_state_corruption(
                state_path,
                "state and bundle run_set_id values disagree",
            )
        state_rows = set(state.rows)
        bundle_rows = set(bundle_row_ids)
        if state_rows != bundle_rows:
            self._raise_run_state_corruption(
                state_path,
                "state and bundle row identities disagree",
            )

    def _status_from_state(self, job_id: str) -> dict[str, Any] | None:
        ref = self._job_ref_for(job_id)
        if ref is None:
            return None
        try:
            state = self._load_state(ref)
        except _RunStateDocumentUnavailable:
            return None
        row = state.rows.get(job_id)
        if row is None:
            return None
        try:
            bundle = self._load_bundle(ref)
        except _RunStateDocumentUnavailable:
            return None
        self._validate_state_bundle(
            ref.state_path,
            state,
            bundle.run_set_id,
            [row.row_id for row in bundle.rows],
        )
        total_batches = load_worker_execution_payload(bundle.row(job_id)).get("total_batches", 0)
        events = self._read_job_events(bundle, job_id)
        latest = events[-1] if events else None
        payload = dict(latest.payload) if latest is not None else {}
        status = row.status
        if latest is not None and latest.type == "complete":
            status = "stopped" if payload.get("status") == "cancelled" else "completed"
        elif status == "failed":
            status = "interrupted" if row.error in (_OWNER_EXIT, _FORCED_STOP) else "error"
        elif status in ("launched", "ready"):
            status = "running"
        return {
            "status": status,
            "error": row.error,
            "batch": int(payload.get("batch", 0) or 0),
            "program_step": payload.get("program_step"),
            "total_batches": int(payload.get("total_batches", total_batches) or 0),
            "last_loss": float(payload.get("loss", 0.0) or 0.0),
            "job_id": job_id,
            "run_set_id": ref.run_set_id,
            "manifest_path": payload.get("manifest_path"),
            "manifest_id": payload.get("manifest_id"),
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "last_event_type": row.last_event_type,
            "event_seq_high_water_mark": row.event_seq_high_water_mark,
            "event_discrepancies": row.event_discrepancies,
        }

    def _read_job_events(
        self, bundle: RunBundle, job_id: str, *, from_seq: int | None = None
    ) -> list[RunEvent]:
        path = bundle.run_set_dir / "events" / f"{job_id}.events.jsonl"
        try:
            return RunEventReader(path).read_all(from_seq=from_seq)
        except Exception:
            return []

    def _iter_state_paths(self) -> list[Path]:
        root = _orchestration_parent_root()
        if not root.exists():
            return []
        return sorted(root.glob("*/state.json"))


training_service = TrainingService()
