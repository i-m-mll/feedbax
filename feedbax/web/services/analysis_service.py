"""App-scoped execution and transient status for Studio analysis jobs."""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class AnalysisJobResult:
    """Durable result identifiers produced by one Studio analysis job."""

    manifest_id: str
    manifest_path: str
    figure_hashes: list[str]
    artifact_ids: list[str]
    artifact_paths: list[str]


@dataclass(frozen=True)
class JobEntry:
    manifest_id: str
    status: JobStatus = JobStatus.PENDING
    result: AnalysisJobResult | None = None
    error: str | None = None


class AnalysisJobsUnavailable(RuntimeError):
    """The app cannot accept another analysis job."""


class AnalysisJobService:
    """Run one Studio job at a time, with room for one waiting submission."""

    _MAX_ACTIVE = 2
    _MAX_JOBS = 1000

    def __init__(self) -> None:
        # Serialize JAX compilations within this app to limit device memory use.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._lock = Lock()
        self._jobs: dict[str, JobEntry] = {}
        self._active = 0
        self._closed = False

    def submit(self, manifest_id: str, work: Callable[[], AnalysisJobResult]) -> str:
        """Admit and schedule work together, independently of the HTTP request."""
        with self._lock:
            if self._closed:
                raise AnalysisJobsUnavailable("Studio analysis jobs are unavailable")
            if self._active >= self._MAX_ACTIVE:
                raise AnalysisJobsUnavailable("Studio analysis job queue is full")
            request_id = str(uuid4())
            try:
                self._executor.submit(self._run, request_id, work)
            except RuntimeError as exc:
                # ThreadPoolExecutor can enqueue work before failing to start a thread.
                # Close the executor and discard that submission before it can run.
                self._stop_admission()
                self._executor.shutdown(wait=False, cancel_futures=True)
                logger.exception("Could not submit Studio analysis job")
                raise AnalysisJobsUnavailable("Studio analysis worker is unavailable") from exc
            if len(self._jobs) >= self._MAX_JOBS:
                oldest_terminal = next(
                    job_id
                    for job_id, entry in self._jobs.items()
                    if entry.status in (JobStatus.COMPLETE, JobStatus.ERROR)
                )
                del self._jobs[oldest_terminal]
            self._jobs[request_id] = JobEntry(manifest_id=manifest_id)
            self._active += 1
            return request_id

    def get_status(self, request_id: str) -> JobEntry | None:
        with self._lock:
            return self._jobs.get(request_id)

    def _run(self, request_id: str, work: Callable[[], AnalysisJobResult]) -> None:
        with self._lock:
            if self._closed:
                return
            entry = self._jobs[request_id]
            self._jobs[request_id] = replace(entry, status=JobStatus.RUNNING)

        try:
            result = work()
        except BaseException:
            # Executor threads capture even BaseException; settle the public job too.
            error = traceback.format_exc()
            logger.exception("Studio analysis job %s failed", request_id)
            terminal = replace(entry, status=JobStatus.ERROR, error=error)
        else:
            terminal = JobEntry(
                manifest_id=result.manifest_id, status=JobStatus.COMPLETE, result=result
            )

        with self._lock:
            self._jobs[request_id] = terminal
            self._active -= 1

    def _stop_admission(self) -> None:
        """Close admission and settle jobs that have yet to start, under the lock."""
        self._closed = True
        for request_id, entry in self._jobs.items():
            if entry.status is JobStatus.PENDING:
                self._jobs[request_id] = replace(
                    entry, status=JobStatus.ERROR, error="Studio stopped before this job started"
                )
                self._active -= 1

    async def aclose(self) -> None:
        with self._lock:
            self._stop_admission()
        join = asyncio.create_task(
            asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        )
        try:
            await asyncio.shield(join)
        except asyncio.CancelledError:
            await join
            raise
