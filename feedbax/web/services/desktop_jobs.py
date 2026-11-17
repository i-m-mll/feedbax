"""Release the managed runtime before native Studio service shutdown."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from feedbax.jobs import JobService

StopDisposition = Literal["defer", "checkpoint", "force", "interrupt"]


class StudioJobLifecycle:
    """Quiesce jobs using the installed runtime before native service shutdown."""

    def __init__(self, hosted: JobService, release: Path) -> None:
        self._hosted = hosted
        self._release = release
        self._lock = asyncio.Lock()
        self._closing = False

    def snapshot(self) -> dict:
        return {"accepting": not self._closing}

    async def stop(
        self, disposition: StopDisposition, timeout_seconds: float = 30
    ) -> dict:
        if disposition == "interrupt":
            return await self._stop_runtime(disposition, timeout_seconds)
        async with self._lock:
            return await self._stop_runtime(disposition, timeout_seconds)

    async def _stop_runtime(self, disposition: StopDisposition, timeout_seconds: float) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while True:
            host = await asyncio.to_thread(self._hosted.retire_runtime, str(self._release), disposition)
            if host["status"] == "deferred":
                return {
                    "status": "deferred", "disposition": disposition,
                    "jobs": self.snapshot() | {"hosted": host["jobs"]},
                }
            self._closing = True
            if host["status"] in ("released", "absent"):
                return {"status": "ready_to_stop", "disposition": disposition, "jobs": self.snapshot()}
            if time.monotonic() >= deadline:
                return {
                    "status": "timed_out", "disposition": disposition,
                    "jobs": self.snapshot(), "unsettled_training": host.get("jobs", []),
                }
            await asyncio.sleep(0.1)
