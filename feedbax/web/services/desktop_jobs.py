"""Desktop service ownership of the existing Studio job mechanisms."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from feedbax.orchestration.bundle import RunBundle, canonical_run_bundle_sha256
from feedbax.orchestration.staged_training import (
    load_staged_training_receipt,
    require_staged_training_owner,
    training_receipt_lock,
)
from feedbax.orchestration.stages import (
    STAGE_ASSEMBLE,
    STAGE_LAUNCH,
    STAGE_TEARDOWN,
    StageEngine,
)
from feedbax.orchestration.state import RunSetState, RunSetStateStore
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.studio.native_training import (
    native_training_engine,
    prepare_native_training_dispatch,
    stage_native_checkpoint_resume,
)
from feedbax.studio.native_training_projection import project_owned_native_training
from feedbax.training.interruption import CancellationDecision
from feedbax.web.services.analysis_service import AnalysisJobService
from feedbax.web.services.training_service import TrainingService

logger = logging.getLogger(__name__)
StopDisposition = Literal["defer", "checkpoint", "force"]


class TrainingJobsUnavailable(RuntimeError):
    """The desktop service cannot accept a new training attempt."""


@dataclass
class NativeTrainingControl:
    _pending: CancellationDecision | None = None
    _requested: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request(self, action: Literal["stop", "terminate"]) -> None:
        with self._lock:
            if self._requested is None or (self._requested == "stop" and action == "terminate"):
                self._requested = action
                self._pending = CancellationDecision(action, "non_interactive", time.time())

    def poll(self) -> CancellationDecision | None:
        with self._lock:
            decision = self._pending
            self._pending = None
            return decision


@dataclass
class _OwnedAttempt:
    engine: StageEngine
    control: NativeTrainingControl
    thread: threading.Thread


class NativeTrainingJobs:
    def __init__(self, bootstrap: BootstrapState, orchestration_root: Path) -> None:
        self._bootstrap = bootstrap
        self._orchestration_root = orchestration_root
        self._attempts: dict[str, _OwnedAttempt] = {}
        self._lock = threading.Lock()
        self._accepting = True

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    @property
    def idle(self) -> bool:
        with self._lock:
            if self._attempts:
                return False
        return not self.unsettled()

    def unsettled(self) -> list[str]:
        return [
            bundle.run_set_id for bundle, state in self._records()
            if state.stage(STAGE_TEARDOWN).status != "completed"
        ]

    def _records(self) -> list[tuple[RunBundle, RunSetState]]:
        records = []
        for state_path in self._orchestration_root.glob("*/state.json"):
            bundle_path = state_path.with_name("bundle.json")
            try:
                bundle = RunBundle.model_validate_json(bundle_path.read_bytes())
            except FileNotFoundError:
                continue
            if bundle.staged_training is None or bundle.deployment_policy.driver != "local":
                continue
            state = RunSetStateStore(state_path).load()
            if (
                state.run_set_id != bundle.run_set_id
                or bundle.run_set_dir != state_path.parent.resolve()
                or set(state.rows) != {row.row_id for row in bundle.rows}
                or state.stage(STAGE_ASSEMBLE).outputs.get("bundle_sha256")
                != canonical_run_bundle_sha256(bundle)
            ):
                if state.stage(STAGE_ASSEMBLE).status == "completed":
                    raise ValueError(f"Invalid Studio training ownership at {state_path}")
                continue
            binding = bundle.staged_training
            receipt, _ = load_staged_training_receipt(Path(binding.manifest_root), binding.manifest_id)
            owner = receipt.metadata.get("staged_attempt")
            if not isinstance(owner, dict) or owner.get("run_set_id") != bundle.run_set_id:
                continue
            require_staged_training_owner(bundle, bundle.rows[0])
            records.append((bundle, state))
        return records

    def recover(self) -> None:
        """Resume monitoring and collection of verified, already admitted attempts."""
        for bundle, state in self._records():
            if state.stage(STAGE_TEARDOWN).status == "completed":
                continue
            control = NativeTrainingControl()
            engine = native_training_engine(
                bundle,
                registry_bundle=self._bootstrap.bundle,
                plugin_registrations=self._bootstrap.provenance,
                interruption_probe=control.poll,
            )
            with self._lock:
                self._start(engine, control, recover=True)

    def dispatch(self, engine: StageEngine, control: NativeTrainingControl) -> None:
        with self._lock:
            if not self._accepting:
                raise TrainingJobsUnavailable("Studio is stopping and cannot launch training")
            engine.run(stop_after_stage=STAGE_LAUNCH)
            self._start(engine, control, recover=False)

    def _start(
        self, engine: StageEngine, control: NativeTrainingControl, *, recover: bool
    ) -> None:
        assert engine.bundle is not None
        row_id = engine.bundle.rows[0].row_id
        if row_id in self._attempts:
            raise RuntimeError(f"Studio already supervises training row {row_id!r}")
        thread = threading.Thread(
            target=self._monitor,
            args=(row_id, engine, recover),
            name=f"feedbax-studio-native-{engine.run_set_id}",
            daemon=True,
        )
        self._attempts[row_id] = _OwnedAttempt(engine, control, thread)
        thread.start()

    def _monitor(self, row_id: str, engine: StageEngine, recover: bool) -> None:
        try:
            engine.run(break_stale_lock=recover, certify_and_register=False)
        except Exception:
            logger.exception("Studio native attempt %s stopped with an error", engine.run_set_id)
        finally:
            with self._lock:
                self._attempts.pop(row_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            for row_id, attempt in self._attempts.items():
                assert attempt.engine.bundle is not None
                binding = attempt.engine.bundle.staged_training
                if job_id == row_id or (binding is not None and job_id == binding.manifest_id):
                    attempt.control.request("stop")
                    return True
        return any(job_id in (job["job_id"], job["manifest_id"]) for job in self.snapshot())

    def resume(self, job_id: str) -> str:
        for bundle, state in self._records():
            row = bundle.rows[0]
            if row.row_id != job_id:
                continue
            binding = bundle.staged_training
            assert binding is not None
            root = Path(binding.manifest_root)
            with training_receipt_lock(root, binding.manifest_id):
                receipt, _ = load_staged_training_receipt(root, binding.manifest_id)
                projection = project_owned_native_training(binding.manifest_id, receipt.metadata, receipt_root=root)
                if (
                    projection is None or projection.status != "stopped" or projection.result is None
                    or state.stage(STAGE_TEARDOWN).status != "completed"
                ):
                    raise ValueError("Job has no settled checkpoint to resume")
                checkpoints = bundle.run_set_dir / "collected" / row.row_id / "checkpoints"
                successor, checkpoint = stage_native_checkpoint_resume(
                    bundle, receipt, projection.result, checkpoints,
                )
                if checkpoint is None:
                    return successor.metadata["staged_attempt"]["row_id"]
                control = NativeTrainingControl()
                engine = prepare_native_training_dispatch(
                    successor, root=root, generation=successor.metadata["staging_generation"],
                    registry_bundle=self._bootstrap.bundle, plugin_registrations=self._bootstrap.provenance,
                    interruption_probe=control.poll, checkpoint=checkpoint,
                )
                self.dispatch(engine, control)
                assert engine.bundle is not None
                return engine.bundle.rows[0].row_id
        raise KeyError(job_id)

    def begin_stop(self, *, force: bool) -> None:
        with self._lock:
            self._accepting = False
            for attempt in self._attempts.values():
                attempt.control.request("terminate" if force else "stop")

    def snapshot(self, *, recent: int = 50) -> list[dict[str, Any]]:
        records = sorted(self._records(), key=lambda record: record[1].updated_at, reverse=True)
        active = []
        terminal = []
        for bundle, state in records:
            binding = bundle.staged_training
            assert binding is not None
            row_id = bundle.rows[0].row_id
            row = state.rows[row_id]
            receipt = project_owned_native_training(
                binding.manifest_id,
                {"staged_attempt": {
                    "run_set_id": bundle.run_set_id,
                    "run_set_dir": str(bundle.run_set_dir),
                    "bundle_sha256": canonical_run_bundle_sha256(bundle),
                }},
                receipt_root=Path(binding.manifest_root),
            )
            status = receipt.status if receipt else row.status
            if state.stage(STAGE_TEARDOWN).status == "failed":
                status = "recovery_required"
            resumable = bool(receipt and receipt.result and receipt.result.checkpoint_custody
                             and status == "stopped" and not receipt.teardown_pending)
            error = (receipt.error if receipt else None) or row.error
            if status == "failed" and error and (
                "exited without sentinel" in error or "orphaned launch" in error
            ):
                status = "interrupted"
            item = {
                "job_id": row_id,
                "run_set_id": bundle.run_set_id,
                "manifest_id": binding.manifest_id,
                "manifest_root": binding.manifest_root,
                "status": status,
                "resumable": resumable,
                "error": error,
                "completed_batches": receipt.result.completed_batches if receipt and receipt.result else None,
                "event_sequence": row.event_seq_high_water_mark,
            }
            (active if status in ("running", "claimed", "pending") else terminal).append(item)
        return active + terminal[:recent]


class StudioJobLifecycle:
    """Coordinate service stop across the existing training and analysis owners."""

    def __init__(
        self, analysis: AnalysisJobService, training: NativeTrainingJobs, legacy: TrainingService,
        job_root: Path,
    ) -> None:
        self.analysis = analysis
        self.training = training
        self._legacy = legacy
        self._job_root = job_root
        self._lock = asyncio.Lock()
        self._closing = False

    async def start_legacy(self, *args: Any, **kwargs: Any) -> str:
        async with self._lock:
            if self._closing:
                raise TrainingJobsUnavailable("Studio is stopping and cannot launch training")
            return await self._legacy.start_training(*args, **kwargs)

    def snapshot(self) -> dict:
        return {
            "accepting": not self._closing,
            "training": self.training.snapshot() + self._legacy.list_live_training_runs()[:50],
            "analysis": self.analysis.snapshot(),
        }

    async def stop(
        self, disposition: StopDisposition, timeout_seconds: float = 30
    ) -> dict:
        async with self._lock:
            jobs = self.snapshot()
            legacy_active = [
                job["job_id"] for job in jobs["training"] if not job.get("manifest_root")
                and job["status"] in ("pending", "launched", "ready", "running")
            ]
            active = not self.training.idle or not self.analysis.idle or bool(legacy_active)
            if disposition == "defer" and active:
                return {"status": "deferred", "disposition": disposition, "jobs": jobs}
            self._closing = True
            force = disposition == "force"
            self.training.begin_stop(force=force)
            self.analysis.begin_stop(force=force)
            deadline = time.monotonic() + timeout_seconds
            if force:
                await asyncio.to_thread(self._legacy._terminate_worker)
                await asyncio.to_thread(self._legacy.reconcile_desktop_worker, self._job_root, forced=True)
            else:
                for job_id in legacy_active:
                    await self._legacy.stop_training(job_id)
            while True:
                jobs = self.snapshot()
                legacy_running = any(
                    job["job_id"] in legacy_active and job["status"] in (
                        "pending", "launched", "ready", "running"
                    ) for job in jobs["training"]
                )
                if self.training.idle and self.analysis.idle and (force or not legacy_running):
                    return {"status": "ready_to_stop", "disposition": disposition, "jobs": jobs}
                if time.monotonic() >= deadline:
                    return {
                        "status": "timed_out", "disposition": disposition, "jobs": jobs,
                        "unsettled_training": self.training.unsettled(),
                    }
                await asyncio.sleep(0.1)
