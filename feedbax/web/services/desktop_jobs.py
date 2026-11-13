"""Desktop service ownership of the existing Studio job mechanisms."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from feedbax.web.services.analysis_service import AnalysisJobService
from feedbax.web.services.training_service import TrainingService

if TYPE_CHECKING:
    from feedbax.jobs import JobService

StopDisposition = Literal["defer", "checkpoint", "force"]


class StudioJobLifecycle:
    """Quiesce jobs using the installed runtime before native service shutdown."""

    def __init__(
        self, analysis: AnalysisJobService, legacy: TrainingService,
        job_root: Path, hosted: JobService,
    ) -> None:
        self.analysis = analysis
        self._legacy = legacy
        self._job_root = job_root
        self._hosted = hosted
        self._lock = asyncio.Lock()
        self._closing = False

    def snapshot(self) -> dict:
        return {
            "accepting": not self._closing,
            "training": self._legacy.list_live_training_runs()[:50],
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
            active = not self.analysis.idle or bool(legacy_active)
            if disposition == "defer" and active:
                return {"status": "deferred", "disposition": disposition, "jobs": jobs}
            if not await asyncio.to_thread(self._hosted.begin_runtime_shutdown, disposition):
                jobs["hosted"] = [
                    job["id"] for job in await asyncio.to_thread(self._hosted.live_runtime_jobs)
                ]
                return {"status": "deferred", "disposition": disposition, "jobs": jobs}
            self._closing = True
            force = disposition == "force"
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
                hosted = await asyncio.to_thread(self._hosted.live_runtime_jobs)
                if self.analysis.idle and (force or not legacy_running) and not hosted:
                    return {"status": "ready_to_stop", "disposition": disposition, "jobs": jobs}
                if time.monotonic() >= deadline:
                    return {
                        "status": "timed_out", "disposition": disposition, "jobs": jobs,
                        "unsettled_training": [row["id"] for row in hosted],
                    }
                await asyncio.sleep(0.1)
