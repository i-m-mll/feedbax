"""Studio training service — spawns a worker subprocess and relays its SSE stream."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence

import httpx

from feedbax.contracts.studio_api import (
    STUDIO_API_TRANSPORT_SCHEMA_ID,
    STUDIO_API_TRANSPORT_SCHEMA_VERSION,
)
from feedbax.contracts.spec_storage import store_canonical_json_artifact
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_COMPILER_ID,
    STUDIO_TRAINING_COMPILER_VERSION,
    StudioTrainingAssemblySpec,
    register_studio_training_compiler,
)
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompilerIdentity,
    RunAssemblyRequest,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RunBundle,
    SchemaArtifactRef,
)
from feedbax.orchestration.revision import resolve_feedbax_revision
from feedbax.orchestration.conformance import CheckRegistry
from feedbax.orchestration.drivers.capabilities import (
    DriverConstructionContext,
    DriverRegistry,
)
from feedbax.orchestration.events import (
    RUN_EVENT_SCHEMA_ID,
    RUN_EVENT_TERMINAL_TYPES,
    RunEvent,
    RunEventReader,
    legacy_worker_event_from_run_event,
)
from feedbax.orchestration.stages import (
    STAGE_CERTIFY,
    STAGE_COLLECT,
    STAGE_MONITOR,
    STAGE_REGISTER,
    StageEngine,
)
from feedbax.orchestration.state import RunSetState, RunSetStateStore, utc_now
import feedbax.web.worker.client as worker_client
from feedbax.web.services.worker_driver import load_worker_execution_payload


# ---------------------------------------------------------------------------
# Public event type
# ---------------------------------------------------------------------------


@dataclass
class TrainingEvent:
    """A single event relayed from the worker SSE stream."""

    raw: dict  # parsed JSON from the SSE data: line


@dataclass(frozen=True)
class _JobRef:
    """Rebuildable pointer from a Studio job id to durable orchestration state."""

    job_id: str
    run_set_id: str
    state_path: Path
    bundle_path: Path


# ---------------------------------------------------------------------------
# Port helper
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 to let the OS assign a free ephemeral port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker_stderr_excerpt(process: subprocess.Popen, *, limit: int = 4000) -> str:
    """Return captured stderr for an exited worker process."""
    if process.poll() is None or process.stderr is None:
        return ""
    try:
        _stdout, stderr = process.communicate(timeout=0.1)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if not isinstance(stderr, str):
        stderr = stderr.decode(errors="replace") if stderr else ""
    return stderr.strip()[-limit:]


def _orchestration_parent_root() -> Path:
    configured = os.environ.get("FEEDBAX_ORCHESTRATION_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "feedbax" / "orchestration"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TrainingService:
    """Manages the lifecycle of the headless training worker subprocess.

    Supports two operating modes:

    - **Local mode** (default): a worker subprocess is spawned on demand.
    - **Remote mode**: connects to a pre-existing worker at a given URL.
      Activated either by setting the ``FEEDBAX_WORKER_URL`` environment
      variable before construction, or by calling :meth:`connect_remote`.
    """

    def __init__(self) -> None:
        self._port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._remote: bool = False
        self._lock = asyncio.Lock()
        self._job_refs_by_job: dict[str, _JobRef] = {}
        self._stage_threads_by_run_set: dict[str, threading.Thread] = {}

        # Honour the FEEDBAX_WORKER_URL env var: skip subprocess and connect
        # directly to an external worker.
        env_url = os.environ.get("FEEDBAX_WORKER_URL")
        if env_url:
            self._base_url = env_url.rstrip("/")
            self._remote = True
        self.rebuild_cache_from_state_docs()

    # ------------------------------------------------------------------
    # Remote mode
    # ------------------------------------------------------------------

    def connect_remote(self, url: str, auth_token: Optional[str] = None) -> None:
        """Switch to remote worker mode.

        Terminates any running local subprocess and configures the service to
        forward all requests to the given URL.

        Args:
            url: Base URL of the remote worker, e.g. ``"http://100.1.2.3:8765"``.
            auth_token: Optional bearer token required by the remote worker.
        """
        self._terminate_worker()
        self._base_url = url.rstrip("/")
        self._auth_token = auth_token
        self._remote = True

    def worker_mode(self) -> str:
        """Return ``"remote"`` or ``"local"`` depending on current configuration."""
        return "remote" if self._remote else "local"

    # ------------------------------------------------------------------
    # Worker subprocess lifecycle
    # ------------------------------------------------------------------

    async def _ensure_worker(self) -> str:
        """Lazily start the worker subprocess and wait for it to be healthy.

        In remote mode the URL is already configured — this simply returns it.

        Returns:
            The worker base URL, e.g. ``"http://127.0.0.1:54321"``.

        Raises:
            RuntimeError: If the worker does not respond within 5 seconds.
        """
        async with self._lock:
            if self._remote:
                if self._base_url is None:
                    raise RuntimeError("Remote worker URL is not configured")
                return self._base_url

            if self._process is not None and self._process.poll() is None:
                # Worker is already alive.
                return self._base_url  # type: ignore[return-value]

            port = _find_free_port()
            self._port = port
            self._base_url = f"http://127.0.0.1:{port}"

            self._process = subprocess.Popen(
                [sys.executable, "-m", "feedbax.web.worker", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                await worker_client.wait_for_health(self._base_url, timeout=5.0, interval=0.1)
            except Exception as exc:
                stderr = _worker_stderr_excerpt(self._process)
                self._terminate_worker()
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"Worker subprocess failed health check{detail}") from exc
            return self._base_url

    def _terminate_worker(self) -> None:
        """Terminate the worker subprocess if it is running."""
        process = self._process
        self._process = None
        self._base_url = None
        self._port = None
        self._remote = False
        self._auth_token = None

        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
            wait = getattr(process, "wait", None)
            if callable(wait):
                try:
                    wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
            communicate = getattr(process, "communicate", None)
            if callable(communicate):
                try:
                    communicate(timeout=0.1)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    # ------------------------------------------------------------------
    # Public interface (mirrors the old TrainingService API)
    # ------------------------------------------------------------------

    async def start_training(
        self,
        total_batches: int,
        *,
        conformance_registry: CheckRegistry,
        driver_registry: DriverRegistry,
        plugin_provenance: Sequence[Any],
        training_config: Optional[dict] = None,
        training_spec: Optional[dict] = None,
        task_spec: Optional[dict] = None,
        task_binding_spec: Optional[dict] = None,
        graph_spec: Optional[dict] = None,
    ) -> str:
        """Start a training job on the worker.

        Args:
            total_batches: Number of training steps.
            training_config: Optional dict forwarded to the worker as the
                ``training_config`` key in the ``/start`` request body.
                Required by the worker for real JAX training.
            training_spec: Optional spec dict with optimizer/loss settings;
                forwarded to the worker for spec-driven configuration.
            task_spec: Optional task spec dict with task parameters;
                forwarded to the worker.

        Returns:
            The job ID assigned by the worker.
        """
        base_url = await self._ensure_worker()
        body = {
            "total_batches": total_batches,
            "training_config": training_config,
            "training_spec": training_spec,
            "task_spec": task_spec,
            "task_binding_spec": task_binding_spec,
            "graph_spec": graph_spec,
        }
        body = {key: value for key, value in body.items() if value is not None}
        request, context, registry = self._build_worker_assembly_request(worker_start=body)
        engine = StageEngine.from_request(
            request,
            context=context,
            registry=registry,
            driver_registry=driver_registry,
            driver_context=lambda _bundle: DriverConstructionContext(
                configuration={"base_url": base_url},
                credentials=(
                    {"worker_http_token": self._auth_token} if self._auth_token is not None else {}
                ),
            ),
            conformance_registry=conformance_registry,
            plugin_provenance=plugin_provenance,
        )
        engine.run(stop_after_stage="ASSEMBLE")
        bundle = engine.bundle
        if bundle is None:
            raise RuntimeError("Studio ASSEMBLE completed without producing a RunBundle")
        job_id = bundle.rows[0].row_id
        self._remember_job_ref(bundle)
        thread = threading.Thread(
            target=self._run_stage_engine,
            args=(engine,),
            name=f"feedbax-training-stage-engine-{bundle.run_set_id}",
            daemon=True,
        )
        self._stage_threads_by_run_set[bundle.run_set_id] = thread
        thread.start()
        return job_id

    async def stop_training(self, job_id: str) -> None:
        """Ask the worker to stop a job.

        Also kills the subprocess if the HTTP request fails (local mode only).
        """
        ref = self._job_ref_for(job_id)
        if self._base_url is None:
            if ref is None:
                raise ValueError(f"Unknown job {job_id!r}")
            return
        try:
            await worker_client.stop_job(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
            self._mark_row_stopped(job_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ValueError(f"Unknown job {job_id!r}") from exc
            raise
        except Exception:
            # If the HTTP call failed, forcibly kill the subprocess.
            if self._process is not None:
                try:
                    self._process.kill()
                except OSError:
                    pass

    async def worker_connected(self) -> bool:
        """Return whether the configured worker responds to health checks."""
        if self._base_url is None:
            return False
        try:
            await worker_client.wait_for_health(
                self._base_url,
                timeout=0.5,
                interval=0.05,
                auth_token=self._auth_token,
            )
            return True
        except Exception:
            return False

    async def get_status(self, job_id: str) -> Optional[dict]:
        """Return a job's worker status dict, or ``None`` if the job is unknown."""
        fallback = self._status_from_state(job_id)
        if fallback is not None:
            return fallback
        if self._base_url is None or (
            not self._remote and self._process is not None and self._process.poll() is not None
        ):
            return None
        try:
            status = await worker_client.get_status(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
            return status
        except Exception:
            return fallback

    def make_error_event(self, job_id: str, error: str) -> TrainingEvent:
        """Return a schema-versioned WebSocket error event for handler-originated failures."""
        return TrainingEvent(
            raw=self._normalize_training_event(
                job_id,
                {
                    "type": "training_error",
                    "job_id": job_id,
                    "error": error,
                    "diagnostics": [
                        {
                            "schema_id": "feedbax.diagnostic.domain",
                            "schema_version": "feedbax.diagnostic.domain.v1",
                            "severity": "error",
                            "code": "internal",
                            "message": error,
                            "node_ids": [],
                            "details": {"source": "training_ws"},
                        }
                    ],
                },
            )
        )

    def _normalize_training_event(self, job_id: str, event: dict) -> dict:
        """Add durable Studio protocol fields to a raw worker training event."""
        if event.get("schema_id") == RUN_EVENT_SCHEMA_ID:
            normalized = legacy_worker_event_from_run_event(event)
        else:
            normalized = dict(event)
        worker_seq = normalized.pop("seq", None)
        normalized["schema_id"] = STUDIO_API_TRANSPORT_SCHEMA_ID
        normalized["schema_version"] = STUDIO_API_TRANSPORT_SCHEMA_VERSION
        normalized["job_id"] = job_id
        normalized["seq"] = (
            int(worker_seq) if worker_seq is not None else self._next_event_seq(job_id)
        )
        normalized["emitted_at_ms"] = int(time.time() * 1000)
        if worker_seq is not None:
            normalized["worker_seq"] = int(worker_seq)
        if normalized.get("type") == "training_error":
            fallback_batch = (self._status_from_state(job_id) or {}).get("batch", 0)
            normalized.setdefault("batch", fallback_batch)
            normalized.setdefault("diagnostics", [])
        return normalized

    async def stream_progress(self, job_id: str) -> AsyncIterator[TrainingEvent]:
        """Relay the worker SSE stream as :class:`TrainingEvent` objects.

        Args:
            job_id: The job ID returned by :meth:`start_training`.

        Yields:
            :class:`TrainingEvent` instances wrapping raw event dicts.
        """
        if self._base_url is None:
            async for event in self._stream_from_event_log(job_id):
                yield event
            return
        async for event in worker_client.stream_events(
            self._base_url,
            job_id,
            auth_token=self._auth_token,
        ):
            event = self._normalize_training_event(job_id, event)
            yield TrainingEvent(raw=event)

    async def latest_checkpoint(self, job_id: str) -> Optional[dict]:
        """Return checkpoint metadata for the given job by querying the worker.

        Proxies to the worker's keyed checkpoint endpoint when a worker URL is
        configured.

        Args:
            job_id: The job ID.

        Returns:
            A dict with checkpoint metadata (keys: ``batch``, ``loss``,
            ``weights_available``), or ``None`` if the job is unknown.
        """
        if self._base_url is None:
            status = self._status_from_state(job_id)
            if status is None:
                return None
            return {
                "batch": status.get("batch", 0),
                "loss": status.get("last_loss", 0.0),
                "weights_available": False,
                "job_id": job_id,
            }
        try:
            data = await worker_client.get_checkpoint(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
            data["job_id"] = job_id
            return data
        except Exception:
            status = self._status_from_state(job_id)
            if status is None:
                return None
            return {
                "batch": status.get("batch", 0),
                "loss": status.get("last_loss", 0.0),
                "weights_available": False,
                "job_id": job_id,
            }

    async def latest_manifest(self, job_id: str) -> Optional[dict]:
        """Return the durable training manifest for *job_id* when available."""
        if self._base_url is None:
            return None
        try:
            return await worker_client.get_manifest(
                self._base_url,
                job_id,
                auth_token=self._auth_token,
            )
        except Exception:
            return None

    async def download_checkpoint(self, job_id: str, dest_path: str) -> None:
        """Download the serialized checkpoint from the worker to a local file.

        Args:
            job_id: The job ID whose checkpoint to download.
            dest_path: Local filesystem path to write the checkpoint file.

        Raises:
            ValueError: If *job_id* is unknown to the worker.
            RuntimeError: If no worker is configured.
        """
        if self._base_url is None:
            raise RuntimeError("No worker configured")
        try:
            await worker_client.download_checkpoint(
                self._base_url,
                job_id,
                dest_path,
                auth_token=self._auth_token,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ValueError(f"Unknown job {job_id!r}") from exc
            raise

    def last_loss(self, job_id: str) -> Optional[float]:
        """Return the last recorded loss for the given job.

        Args:
            job_id: The job ID.

        Returns:
            The last loss value, or ``None`` if unknown.
        """
        status = self._status_from_state(job_id)
        if status is None or status.get("last_loss") is None:
            return None
        return float(status["last_loss"])

    def rebuild_cache_from_state_docs(self) -> None:
        """Rebuild the in-memory job index from persisted orchestration state."""
        refs: dict[str, _JobRef] = {}
        for state_path in self._iter_state_paths():
            bundle_path = state_path.with_name("bundle.json")
            if not bundle_path.exists():
                continue
            try:
                bundle = RunBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
            except Exception:
                self._index_legacy_v2_bundle(refs, state_path, bundle_path)
                continue
            for row in bundle.rows:
                refs[row.row_id] = _JobRef(
                    job_id=row.row_id,
                    run_set_id=bundle.run_set_id,
                    state_path=state_path,
                    bundle_path=bundle_path,
                )
        self._job_refs_by_job = refs

    async def reconcile_from_state_docs(self) -> None:
        """Finalize terminal rows and fail truly orphaned rows after backend restart."""
        self.rebuild_cache_from_state_docs()
        for ref in list(self._job_refs_by_job.values()):
            try:
                bundle = self._load_bundle(ref)
                store = RunSetStateStore(ref.state_path)
                state = store.load()
            except Exception:
                continue
            changed = False
            for row in bundle.rows:
                row_state = state.rows.get(row.row_id)
                if row_state is None or row_state.status in ("completed", "failed", "stopped"):
                    continue
                event_path = bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl"
                done = bundle.run_set_dir / "sentinels" / f"{row.row_id}.done"
                failed = bundle.run_set_dir / "sentinels" / f"{row.row_id}.failed"
                reader = RunEventReader(event_path)
                events = reader.read_all()
                reconciled = reader.reconcile_sentinels(
                    done_sentinel=done,
                    failed_sentinel=failed,
                )
                high_water = events[-1].seq if events else row_state.event_seq_high_water_mark
                last_type = events[-1].type if events else row_state.last_event_type
                update: dict[str, Any] = {
                    "event_seq_high_water_mark": high_water,
                    "last_event_type": last_type,
                    "event_discrepancies": [dict(item) for item in reconciled.discrepancies],
                }
                if reconciled.status == "completed":
                    update.update({"status": "completed", "completed_at": utc_now()})
                elif reconciled.status in ("failed", "error"):
                    update.update(
                        {
                            "status": "failed",
                            "completed_at": utc_now(),
                            "error": "terminal event/sentinel reported failure",
                        }
                    )
                else:
                    discrepancies = list(update["event_discrepancies"])
                    discrepancies.append(
                        {
                            "code": "backend_restart_orphaned_row",
                            "detail": "row was non-terminal at startup with no live worker claim",
                        }
                    )
                    update.update(
                        {
                            "status": "failed",
                            "completed_at": utc_now(),
                            "error": "orphaned after backend restart",
                            "event_discrepancies": discrepancies,
                        }
                    )
                state = state.with_row(row.row_id, row_state.model_copy(update=update))
                changed = True
            if changed:
                state = self._finalize_terminal_state(state)
                store.save(state)

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
                }
            )
        return rows

    def _build_worker_assembly_request(
        self,
        *,
        worker_start: dict[str, Any],
    ) -> tuple[RunAssemblyRequest, AssemblyContext, AssemblyCompilerRegistry]:
        authored = StudioTrainingAssemblySpec.model_validate(worker_start)
        custody_root = _orchestration_parent_root() / "custody"
        artifact = store_canonical_json_artifact(
            authored,
            root=custody_root,
            role="studio_training_assembly",
            logical_name="studio-training-assembly.json",
        )
        authored_ref = SchemaArtifactRef(
            schema_id=authored.schema_id,
            schema_version=authored.schema_version,
            artifact_id=str(artifact.artifact_id),
            sha256=str(artifact.sha256),
            uri=artifact.uri,
        )
        request = RunAssemblyRequest(
            authored=authored_ref,
            compiler=CompilerIdentity(
                compiler_id=STUDIO_TRAINING_COMPILER_ID,
                compiler_version=STUDIO_TRAINING_COMPILER_VERSION,
            ),
            # Studio authors this request in the same process that will assemble it,
            # so the running package is itself the authority the gate then verifies.
            feedbax_revision=resolve_feedbax_revision(),
            deployment_policy=DeploymentPolicy(
                driver="worker-http",
                venue="remote",
                cloud_authorized=False,
                review_required=False,
                review_authorized=False,
            ),
            environment=EnvironmentDeclaration(python_version=sys.version.split()[0]),
            launch_policy=LaunchPolicy(max_parallel_rows=1),
            budget=BudgetPolicy(max_wall_clock_seconds=24 * 60 * 60),
            orchestration_root=str(_orchestration_parent_root()),
            metadata={"source": "studio-training-service"},
        )
        registry = AssemblyCompilerRegistry()
        register_studio_training_compiler(registry)
        context = AssemblyContext(
            custody_root=custody_root,
            repo_root=Path(__file__).resolve().parents[3],
        )
        return request, context, registry

    def _run_stage_engine(
        self,
        engine: StageEngine,
    ) -> None:
        try:
            engine.run(break_stale_lock=True)
        except Exception:
            # The durable state document carries the failed stage/row details.
            return

    def _remember_job_ref(self, bundle: RunBundle) -> None:
        state_path = bundle.run_set_dir / "state.json"
        bundle_path = bundle.run_set_dir / "bundle.json"
        for row in bundle.rows:
            self._job_refs_by_job[row.row_id] = _JobRef(
                job_id=row.row_id,
                run_set_id=bundle.run_set_id,
                state_path=state_path,
                bundle_path=bundle_path,
            )

    def _job_ref_for(self, job_id: str) -> _JobRef | None:
        ref = self._job_refs_by_job.get(job_id)
        if ref is not None:
            return ref
        self.rebuild_cache_from_state_docs()
        return self._job_refs_by_job.get(job_id)

    def _load_bundle(self, ref: _JobRef) -> RunBundle:
        return RunBundle.model_validate_json(ref.bundle_path.read_text(encoding="utf-8"))

    def _index_legacy_v2_bundle(
        self,
        refs: dict[str, _JobRef],
        state_path: Path,
        bundle_path: Path,
    ) -> None:
        """Index historical v2 rows for read-only status visibility only."""
        try:
            raw = json.loads(bundle_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if raw.get("schema_version") != "feedbax.orchestration.run_bundle.v2":
            return
        run_set_id = raw.get("run_set_id")
        if not isinstance(run_set_id, str):
            return
        for row in raw.get("rows", []):
            job_id = row.get("row_id") if isinstance(row, dict) else None
            if isinstance(job_id, str):
                refs[job_id] = _JobRef(job_id, run_set_id, state_path, bundle_path)

    def _status_from_state(self, job_id: str) -> dict[str, Any] | None:
        ref = self._job_ref_for(job_id)
        if ref is None or not ref.state_path.exists():
            return None
        try:
            state = RunSetStateStore(ref.state_path).load()
        except Exception:
            return None
        row = state.rows.get(job_id)
        if row is None:
            return None
        bundle: RunBundle | None
        legacy_worker_start: dict[str, Any] = {}
        try:
            bundle = self._load_bundle(ref)
            total_batches = load_worker_execution_payload(bundle.row(job_id)).get(
                "total_batches", 0
            )
        except Exception:
            bundle = None
            try:
                raw = json.loads(ref.bundle_path.read_text(encoding="utf-8"))
                legacy_row = next(
                    item for item in raw.get("rows", []) if item.get("row_id") == job_id
                )
                legacy_worker_start = dict(
                    (legacy_row.get("metadata") or {}).get("worker_start") or {}
                )
                total_batches = legacy_worker_start.get("total_batches", 0)
            except Exception:
                return None
        events = (
            self._read_job_events(bundle, job_id)
            if bundle is not None
            else RunEventReader(
                ref.state_path.parent / "events" / f"{job_id}.events.jsonl"
            ).read_all()
        )
        latest = events[-1] if events else None
        payload = dict(latest.payload) if latest is not None else {}
        status = row.status
        if status == "failed":
            status = "error"
        elif status in ("launched", "ready"):
            status = "running"
        return {
            "status": status,
            "batch": int(payload.get("batch", 0) or 0),
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

    def _next_event_seq(self, job_id: str) -> int:
        status = self._status_from_state(job_id)
        if status is None:
            return 0
        return int(status.get("event_seq_high_water_mark", -1)) + 1

    async def _stream_from_event_log(self, job_id: str) -> AsyncIterator[TrainingEvent]:
        ref = self._job_ref_for(job_id)
        if ref is None:
            return
        bundle = self._load_bundle(ref)
        for event in self._read_job_events(bundle, job_id):
            yield TrainingEvent(
                raw=self._normalize_training_event(job_id, event.model_dump(mode="json"))
            )
            if event.type in RUN_EVENT_TERMINAL_TYPES:
                return

    def _read_job_events(self, bundle: RunBundle, job_id: str) -> list[RunEvent]:
        path = bundle.run_set_dir / "events" / f"{job_id}.events.jsonl"
        try:
            return RunEventReader(path).read_all()
        except Exception:
            return []

    def _mark_row_stopped(self, job_id: str) -> None:
        ref = self._job_ref_for(job_id)
        if ref is None or not ref.state_path.exists():
            return
        store = RunSetStateStore(ref.state_path)
        state = store.load()
        row = state.rows.get(job_id)
        if row is None:
            return
        state = state.with_row(
            job_id,
            row.model_copy(update={"status": "stopped", "completed_at": utc_now()}),
        )
        store.save(state)

    def _finalize_terminal_state(self, state: RunSetState) -> RunSetState:
        if not state.rows:
            return state
        if not all(row.status in ("completed", "failed", "stopped") for row in state.rows.values()):
            return state
        for stage_id in (STAGE_MONITOR, STAGE_COLLECT, STAGE_CERTIFY, STAGE_REGISTER):
            stage = state.stage(stage_id)
            if stage.status != "completed":
                state = state.with_stage(
                    stage_id,
                    stage.model_copy(
                        update={
                            "status": "completed",
                            "started_at": stage.started_at or utc_now(),
                            "completed_at": utc_now(),
                            "outputs": {
                                **stage.outputs,
                                "reconciled_by": "TrainingService.reconcile_from_state_docs",
                            },
                            "error": None,
                        }
                    ),
                )
        return state

    def _iter_state_paths(self) -> list[Path]:
        root = _orchestration_parent_root()
        if not root.exists():
            return []
        return sorted(root.glob("*/state.json"))

    def __del__(self) -> None:
        self._terminate_worker()


training_service = TrainingService()
