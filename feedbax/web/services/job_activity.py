"""Studio's held view of host jobs, kept current by one follower of the host event stream."""

from __future__ import annotations

from collections import deque
from threading import Condition, Event, Thread
from time import monotonic

from feedbax.contracts.jobs import (
    StudioActivityJob,
    StudioJobEventHistory,
    StudioJobSnapshot,
    StudioLocalSession,
)
from feedbax.host import HostError
from feedbax.jobs import ActivityEntry, ActivityPage, JobService, history_events, stream_events

_FOLLOW_WAIT = 5.0
_RETRY_DELAY = 0.5
_LOG_LIMIT = 8192
_PAGE_LIMIT = 256


class JobActivity:
    """Answer job reads from state learned from the host's events, never from a read-time host call.

    One thread follows the host's event page stream. Each page is applied to the held
    Activity rows before its events join the log that every Studio event stream reads, so a
    held snapshot's cursor is at or past every event any window has received. Ranges older
    than the log are read from the host.
    """

    def __init__(self, service: JobService) -> None:
        self._service = service
        self._changed = Condition()
        self._stopped = Event()
        self._thread: Thread | None = None
        self._loaded = False
        self._error: HostError | None = None
        self._defect: Exception | None = None
        self._rows: dict[str, StudioActivityJob] = {}
        self._cursor = 0
        self._oldest_cursor = 0
        self._log: deque[ActivityEntry] = deque()
        self._log_floor = 0

    def snapshot(self, *, project: str | None = None) -> StudioJobSnapshot:
        with self._changed:
            self._await_loaded()
            jobs = [job for job in self._rows.values() if project is None or job.project_id == project]
            return StudioJobSnapshot(
                jobs=jobs, cursor=self._cursor, oldest_cursor=self._oldest_cursor,
                sessions=[StudioLocalSession()],
            )

    def events(
        self, *, after: int, projects: tuple[str, ...], wait: float, machine: bool,
    ) -> tuple[int, StudioJobEventHistory]:
        with self._changed:
            self._await_loaded()
            floor = self._log_floor
        if after < floor:
            head, page = self._service.events(after=after, projects=projects, wait=0, machine=machine)
            if page.cursor > floor:
                page = page.model_copy(update={
                    "cursor": floor, "events": [event for event in page.events if event.cursor <= floor],
                })
            return min(head, floor), page
        deadline = monotonic() + wait
        with self._changed:
            while self._cursor <= after and self._error is None and self._defect is None and (remaining := deadline - monotonic()) > 0:
                self._changed.wait(remaining)
            self._await_loaded()
            entries = [entry for entry in self._log if entry.event["cursor"] > after][:_PAGE_LIMIT]
            cursor = entries[-1].event["cursor"] if len(entries) == _PAGE_LIMIT else max(after, self._cursor)
            return self._cursor, StudioJobEventHistory(
                cursor=cursor, oldest_cursor=self._oldest_cursor,
                events=stream_events(entries, projects=projects, machine=machine),
            )

    def history(self, *, after: int, project: str, through: int) -> StudioJobEventHistory:
        with self._changed:
            self._await_loaded()
            if after >= self._log_floor and through <= self._cursor:
                return StudioJobEventHistory(
                    cursor=through, oldest_cursor=self._oldest_cursor,
                    events=history_events(
                        [entry for entry in self._log if after < entry.event["cursor"] <= through], project=project,
                    ),
                )
        return self._service.event_history(after=after, project=project, through=through)

    def advance(self) -> None:
        """Apply every event the host holds now, so this server's own writes are visible to its next read."""
        with self._changed:
            if not self._loaded:
                return
        while True:
            with self._changed:
                after = self._cursor
            try:
                page = self._service.activity_page(after=after, wait=0)
                self._apply(after, page)
            except HostError as error:
                self._fail(error)
                return
            if page.cursor >= page.head:
                return

    def put(self, job: StudioActivityJob) -> None:
        with self._changed:
            if job.job_id in self._rows:
                self._rows[job.job_id] = job

    def close(self) -> None:
        self._stopped.set()
        with self._changed:
            self._changed.notify_all()

    def _await_loaded(self) -> None:
        if self._thread is None:
            self._thread = Thread(target=self._follow, name="studio-job-activity", daemon=True)
            self._thread.start()
        while not self._loaded and self._error is None and self._defect is None:
            self._changed.wait()
        if self._defect is not None:
            raise RuntimeError("Studio stopped following host jobs.") from self._defect
        if self._error is not None:
            raise self._error

    def _follow(self) -> None:
        while not self._stopped.is_set():
            try:
                with self._changed:
                    loaded, after = self._loaded, self._cursor
                if not loaded:
                    self._load()
                    continue
                self._apply(after, self._service.activity_page(after=after, wait=_FOLLOW_WAIT))
            except HostError as error:
                self._fail(error)
                self._stopped.wait(_RETRY_DELAY)
            except Exception as defect:
                with self._changed:
                    self._defect = defect
                    self._changed.notify_all()
                raise

    def _fail(self, error: HostError) -> None:
        with self._changed:
            self._error = error
            self._changed.notify_all()

    def _load(self) -> None:
        snapshot = self._service.jobs()
        with self._changed:
            self._rows = {job.job_id: job for job in snapshot.jobs}
            self._cursor = self._log_floor = snapshot.cursor
            self._oldest_cursor = snapshot.oldest_cursor
            self._log.clear()
            self._loaded = True
            self._error = None
            self._changed.notify_all()

    def _apply(self, after: int, page: ActivityPage) -> None:
        if page.oldest_cursor > after or page.head < after:
            self._load()
            return
        with self._changed:
            known = self._rows.keys()
            admitted = {
                entry.activity.project_id for entry in page.entries
                if entry.activity is not None and entry.activity.job_id not in known
            }
        refreshed = [self._service.jobs(project=project) for project in sorted(admitted)]
        with self._changed:
            if self._cursor != after:
                return
            for entry in page.entries:
                if entry.activity is not None:
                    self._rows[entry.activity.job_id] = entry.activity
            for snapshot in refreshed:
                self._rows.update((job.job_id, job) for job in snapshot.jobs)
            self._log.extend(page.entries)
            while len(self._log) > _LOG_LIMIT:
                self._log_floor = self._log.popleft().event["cursor"]
            self._cursor = page.cursor
            self._oldest_cursor = page.oldest_cursor
            self._error = None
            self._changed.notify_all()
