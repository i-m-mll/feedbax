"""Desktop service ownership of the existing Studio job mechanisms."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from feedbax.web.services.analysis_service import AnalysisJobService

if TYPE_CHECKING:
    from feedbax.jobs import JobService

StopDisposition = Literal["defer", "checkpoint", "force", "interrupt"]


class StudioJobLifecycle:
    """Quiesce jobs using the installed runtime before native service shutdown."""

    def __init__(
        self, analysis: AnalysisJobService, hosted: JobService, release: Path | None,
    ) -> None:
        self.analysis = analysis
        self._hosted = hosted
        self._release = release
        self._lock = asyncio.Lock()
        self._closing = False

    def snapshot(self) -> dict:
        return {
            "accepting": not self._closing,
            "analysis": self.analysis.snapshot(),
        }

    async def stop(
        self, disposition: StopDisposition, timeout_seconds: float = 30
    ) -> dict:
        if disposition == "interrupt":
            self._closing = True
            self.analysis.begin_stop(force=True)
            deadline = time.monotonic() + timeout_seconds
            while True:
                host = await self._retire_host(disposition)
                if self.analysis.idle and host["status"] in ("released", "absent"):
                    return {"status": "ready_to_stop", "disposition": disposition}
                if time.monotonic() >= deadline:
                    return {"status": "timed_out", "disposition": disposition, "jobs": host.get("jobs", [])}
                await asyncio.sleep(0.1)
        async with self._lock:
            if disposition == "defer" and not self.analysis.idle:
                return {"status": "deferred", "disposition": disposition, "jobs": self.snapshot()}
            host = await self._retire_host(disposition)
            if host["status"] == "deferred":
                jobs = self.snapshot() | {"hosted": host["jobs"]}
                return {"status": "deferred", "disposition": disposition, "jobs": jobs}
            self._closing = True
            force = disposition == "force"
            self.analysis.begin_stop(force=force)
            deadline = time.monotonic() + timeout_seconds
            while True:
                if self.analysis.idle and host["status"] in ("released", "absent"):
                    return {"status": "ready_to_stop", "disposition": disposition, "jobs": self.snapshot()}
                if time.monotonic() >= deadline:
                    return {
                        "status": "timed_out", "disposition": disposition, "jobs": self.snapshot(),
                        "unsettled_training": host.get("jobs", []),
                    }
                await asyncio.sleep(0.1)
                if host["status"] not in ("released", "absent"):
                    host = await self._retire_host(disposition)

    async def _retire_host(self, disposition: StopDisposition) -> dict:
        if self._release is None:
            return {"status": "released"}
        return await asyncio.to_thread(self._hosted.retire_runtime, str(self._release), disposition)
