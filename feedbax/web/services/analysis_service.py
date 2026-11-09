"""Admission, execution and status for Studio analysis jobs."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
import logging
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


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

    def __init__(self, *, job_root: Path | None = None) -> None:
        # Serialize JAX compilations within this app to limit device memory use.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._lock = Lock()
        self._jobs: dict[str, JobEntry] = {}
        self._active = 0
        self._closed = False
        self._job_root = job_root
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stop_requests: dict[str, JobStatus] = {}
        if job_root is not None:
            job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._recover()

    @property
    def owns_processes(self) -> bool:
        return self._job_root is not None

    def snapshot(self, *, recent: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            active = []
            terminal = []
            for request_id, entry in self._jobs.items():
                item = {"request_id": request_id, **asdict(entry)}
                (active if entry.status in (JobStatus.PENDING, JobStatus.RUNNING)
                 else terminal).append(item)
            return active + terminal[-recent:]

    def _record(self, request_id: str, entry: JobEntry) -> None:
        if self._job_root is not None:
            write_job_json(self._job_root / f"{request_id}.json", asdict(entry))
        self._jobs[request_id] = entry

    def _recover(self) -> None:
        assert self._job_root is not None
        for path in sorted(self._job_root.glob("*.json"), key=lambda file: file.stat().st_mtime_ns):
            raw = json.loads(path.read_bytes())
            raw["status"] = JobStatus(raw["status"])
            if raw["result"] is not None:
                raw["result"] = AnalysisJobResult(**raw["result"])
            entry = JobEntry(**raw)
            if entry.status in (JobStatus.PENDING, JobStatus.RUNNING):
                request_id = path.stem
                self._jobs[request_id] = entry
                with self._worker_lease(request_id):
                    result = self._completed_result(request_id)
                    if result is not None:
                        entry = result
                    else:
                        entry = replace(entry, status=JobStatus.INTERRUPTED,
                                        error="Studio exited before analysis published its result.")
                    self._record(request_id, entry)
            else:
                self._jobs[path.stem] = entry

    def _completed_result(self, request_id: str) -> JobEntry | None:
        assert self._job_root is not None
        path = self._job_root / request_id / "result.json"
        try:
            payload = json.loads(path.read_bytes())
        except FileNotFoundError:
            return None
        if "error" in payload:
            return JobEntry(self._jobs[request_id].manifest_id, JobStatus.ERROR,
                            error=payload["error"])
        result = AnalysisJobResult(**payload)
        return JobEntry(result.manifest_id, JobStatus.COMPLETE, result)

    def _worker_lease(self, request_id: str):
        assert self._job_root is not None
        return analysis_worker_lease(self._job_root / request_id)

    def submit(self, manifest_id: str, work: Callable[[], AnalysisJobResult] | _ProcessOperation) -> str:
        """Admit and schedule work together, independently of the HTTP request."""
        with self._lock:
            if self._closed:
                raise AnalysisJobsUnavailable("Studio analysis jobs are unavailable")
            if self._job_root is not None:
                for request_id, entry in self._jobs.items():
                    if entry.manifest_id == manifest_id and entry.status in (JobStatus.PENDING, JobStatus.RUNNING):
                        return request_id
            if self._active >= self._MAX_ACTIVE:
                raise AnalysisJobsUnavailable("Studio analysis job queue is full")
            request_id = str(uuid4())
            self._record(request_id, JobEntry(manifest_id=manifest_id))
            self._active += 1
            try:
                self._executor.submit(self._run, request_id, work)
            except RuntimeError as exc:
                # ThreadPoolExecutor can enqueue work before failing to start a thread.
                # Close the executor and discard that submission before it can run.
                self._stop_admission()
                self._executor.shutdown(wait=False, cancel_futures=True)
                logger.exception("Could not submit Studio analysis job")
                raise AnalysisJobsUnavailable("Studio analysis worker is unavailable") from exc
            if len(self._jobs) > self._MAX_JOBS:
                oldest_terminal = next(
                    job_id
                    for job_id, entry in self._jobs.items()
                    if entry.status not in (JobStatus.PENDING, JobStatus.RUNNING)
                )
                if self._job_root is not None:
                    (self._job_root / f"{oldest_terminal}.json").unlink()
                    shutil.rmtree(self._job_root / oldest_terminal, ignore_errors=True)
                del self._jobs[oldest_terminal]
            return request_id

    def get_status(self, request_id: str) -> JobEntry | None:
        with self._lock:
            return self._jobs.get(request_id)

    def _run(self, request_id: str, work: Callable[[], AnalysisJobResult] | _ProcessOperation) -> None:
        with self._lock:
            if self._closed:
                return
            entry = self._jobs[request_id]
            if entry.status is not JobStatus.PENDING:
                return
            self._record(request_id, replace(entry, status=JobStatus.RUNNING))

        try:
            result = work.run(request_id) if isinstance(work, _ProcessOperation) else work()
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
            stop_status = self._stop_requests.pop(request_id, None)
            if stop_status is not None:
                terminal = self._completed_result(request_id) or replace(
                    entry, status=stop_status, error="Analysis was stopped before it published a result."
                )
            self._record(request_id, terminal)
            self._active -= 1

    def submit_operation(
        self, manifest_id: str, *, operation: dict[str, Any], bootstrap_modules: tuple[str, ...]
    ) -> str:
        """Run a validated Studio operation in a cancellable child of this service."""
        assert self._job_root is not None
        return self.submit(
            manifest_id, _ProcessOperation(self, operation, bootstrap_modules)
        )

    def _run_operation(
        self, request_id: str, operation: dict[str, Any], bootstrap_modules: tuple[str, ...]
    ) -> AnalysisJobResult:
        assert self._job_root is not None
        job_dir = self._job_root / request_id
        with self._worker_lease(request_id) as lease_fd:
            write_job_json(job_dir / "operation.json", {
                "operation": operation, "bootstrap_modules": bootstrap_modules,
            })
            read_fd, write_fd = os.pipe()
            try:
                with self._lock:
                    if request_id in self._stop_requests:
                        raise InterruptedError("Analysis was cancelled before process launch.")
                    process = subprocess.Popen(
                        [sys.executable, "-B", "-m", "feedbax.web.services.analysis_worker",
                         str(job_dir), str(read_fd), str(lease_fd)],
                        stdin=subprocess.DEVNULL,
                        pass_fds=(read_fd, lease_fd),
                    )
                    self._processes[request_id] = process
                process.wait()
            finally:
                os.close(read_fd)
                os.close(write_fd)
                with self._lock:
                    self._processes.pop(request_id, None)
        result = self._completed_result(request_id)
        if result is None or result.result is None:
            raise RuntimeError(result.error if result is not None else
                               f"Studio analysis process exited with status {process.returncode}")
        return result.result

    def cancel(self, request_id: str, *, status: JobStatus = JobStatus.CANCELLED) -> bool:
        with self._lock:
            entry = self._jobs.get(request_id)
            if entry is None:
                return False
            if entry.status is JobStatus.PENDING:
                self._record(request_id, replace(entry, status=status,
                                                  error="Analysis stopped before launch."))
                self._active -= 1
            elif entry.status is JobStatus.RUNNING:
                if not self.owns_processes:
                    raise AnalysisJobsUnavailable("This Studio instance cannot cancel running analysis")
                self._stop_requests[request_id] = status
                process = self._processes.get(request_id)
                if process is not None:
                    process.kill()
            return True

    def begin_stop(self, *, force: bool) -> None:
        with self._lock:
            self._stop_admission()
            running = [request_id for request_id, entry in self._jobs.items()
                       if entry.status is JobStatus.RUNNING]
        if force:
            for request_id in running:
                self.cancel(request_id, status=JobStatus.INTERRUPTED)

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._active == 0

    def _stop_admission(self) -> None:
        """Close admission and settle jobs that have yet to start, under the lock."""
        self._closed = True
        for request_id, entry in self._jobs.items():
            if entry.status is JobStatus.PENDING:
                status = JobStatus.INTERRUPTED if self.owns_processes else JobStatus.ERROR
                self._record(request_id, replace(
                    entry, status=status, error="Studio stopped before this job started"
                ))
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


@dataclass(frozen=True)
class _ProcessOperation:
    service: AnalysisJobService
    operation: dict[str, Any]
    bootstrap_modules: tuple[str, ...]

    def run(self, request_id: str) -> AnalysisJobResult:
        return self.service._run_operation(request_id, self.operation, self.bootstrap_modules)


@contextmanager
def analysis_worker_lease(job_dir: Path):
    import fcntl

    job_dir.mkdir(mode=0o700, exist_ok=True)
    with (job_dir / "worker.lock").open("ab") as lease:
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AnalysisJobsUnavailable("Previous Studio analysis worker is still alive")
                time.sleep(0.025)
        yield lease.fileno()


def write_job_json(path: Path, payload: object) -> None:
    from feedbax.desktop.control import write_private_json

    write_private_json(path, payload)
