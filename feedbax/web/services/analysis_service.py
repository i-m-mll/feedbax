"""In-memory job tracker for demand-driven analysis generation.

Tracks the lifecycle of background analysis computation jobs triggered
by the ``POST /api/analysis/generate`` endpoint.  Storage is a plain
dict guarded by an ``asyncio.Lock`` -- not persistent across server
restarts, which is acceptable for this use case.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status values for an analysis generation job.

    These match the frontend ``FigureStatusResponse.status`` union:
    ``'pending' | 'running' | 'complete' | 'error'``.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class JobEntry:
    """State for a single analysis generation job."""

    request_id: str
    node_id: str
    status: JobStatus = JobStatus.PENDING
    figure_hashes: Optional[list[str]] = None
    error: Optional[str] = None


class AnalysisJobTracker:
    """Thread-safe, in-memory tracker for analysis generation jobs.

    All public methods are coroutines guarded by an ``asyncio.Lock`` so
    that concurrent endpoint handlers do not race on the internal dict.

    Completed and errored jobs are evicted when the tracker exceeds
    ``_MAX_JOBS`` entries, keeping the dict bounded.
    """

    _MAX_JOBS = 1000

    def __init__(self) -> None:
        self._jobs: dict[str, JobEntry] = {}
        self._lock = asyncio.Lock()

    def _evict_completed(self) -> None:
        """Remove oldest completed/errored jobs until under the limit.

        Must be called while ``self._lock`` is held.
        """
        if len(self._jobs) <= self._MAX_JOBS:
            return
        # Collect completed/errored entries (insertion-order = oldest first)
        evictable = [
            rid
            for rid, entry in self._jobs.items()
            if entry.status in (JobStatus.COMPLETE, JobStatus.ERROR)
        ]
        n_to_evict = len(self._jobs) - self._MAX_JOBS
        for rid in evictable[:n_to_evict]:
            del self._jobs[rid]
        if n_to_evict > 0:
            logger.info("Evicted %d completed/errored jobs", min(n_to_evict, len(evictable)))

    async def create_job(self, node_id: str) -> str:
        """Create a new pending job and return its ``request_id``."""
        request_id = str(uuid.uuid4())
        entry = JobEntry(request_id=request_id, node_id=node_id)
        async with self._lock:
            self._evict_completed()
            self._jobs[request_id] = entry
        logger.info("Created analysis job %s for node_id=%s", request_id, node_id)
        return request_id

    async def get_status(self, request_id: str) -> Optional[JobEntry]:
        """Return the job entry for *request_id*, or ``None`` if unknown."""
        async with self._lock:
            return self._jobs.get(request_id)

    async def update_status(
        self,
        request_id: str,
        status: JobStatus,
        *,
        figure_hashes: Optional[list[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the status (and optional results) of a tracked job."""
        async with self._lock:
            entry = self._jobs.get(request_id)
            if entry is None:
                logger.warning(
                    "Attempted to update unknown job %s -- ignoring", request_id
                )
                return
            entry.status = status
            if figure_hashes is not None:
                entry.figure_hashes = figure_hashes
            if error is not None:
                entry.error = error
        logger.info("Job %s -> %s", request_id, status.value)


# Module-level singleton shared across the application lifetime.
job_tracker = AnalysisJobTracker()
