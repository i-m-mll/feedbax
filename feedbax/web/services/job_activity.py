"""Studio's held view of host jobs, kept current by one follower of the host event stream."""

from __future__ import annotations

import logging
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
from feedbax.jobs import (
    ActivityEntry,
    ActivityPage,
    ActivitySnapshot,
    InvalidJobRecord,
    JobService,
    history_events,
    stream_events,
)

_FOLLOW_WAIT = 5.0
_RETRY_DELAY = 0.5
# While the host is unreachable, probe it this often so reads recover as soon as it answers.
_PROBE_DELAY = 0.05
_RESTART_DELAY_LIMIT = 30.0
_LOG_LIMIT = 8192
_PAGE_LIMIT = 256
_log = logging.getLogger(__name__)


class FollowerStopped(Exception):
    """The follower cannot answer: it met an error it is reloading after, or it has stopped."""


def _project_of(row: StudioActivityJob | InvalidJobRecord) -> str:
    return row.project if isinstance(row, InvalidJobRecord) else row.project_id


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
        self._rows: dict[str, StudioActivityJob | InvalidJobRecord] = {}
        self._cursor = 0
        self._oldest_cursor = 0
        self._log: deque[ActivityEntry] = deque()
        self._log_floor = 0

    def snapshot(self, *, project: str | None = None) -> StudioJobSnapshot:
        with self._changed:
            self._await_loaded()
            rows = [row for row in self._rows.values() if project is None or _project_of(row) == project]
            return StudioJobSnapshot(
                jobs=[row for row in rows if isinstance(row, StudioActivityJob)],
                unreadable=[row.unreadable() for row in rows if isinstance(row, InvalidJobRecord)],
                cursor=self._cursor, oldest_cursor=self._oldest_cursor,
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
            with self._changed:
                if self._cursor >= page.head:
                    return

    def put(self, job: StudioActivityJob) -> None:
        with self._changed:
            if job.job_id in self._rows:
                self._rows[job.job_id] = job

    def close(self) -> None:
        """Stop following and return once the follower has made its last host call."""
        with self._changed:
            self._stopped.set()
            thread = self._thread
            self._changed.notify_all()
        if thread is not None:
            thread.join()

    def _await_loaded(self) -> None:
        if self._stopped.is_set():
            raise FollowerStopped("Studio has stopped following host jobs.")
        if self._thread is None:
            self._thread = Thread(target=self._follow, name="studio-job-activity", daemon=True)
            self._thread.start()
        while not self._loaded and self._error is None and self._defect is None and not self._stopped.is_set():
            self._changed.wait()
        if self._stopped.is_set():
            raise FollowerStopped("Studio has stopped following host jobs.")
        if self._defect is not None:
            raise FollowerStopped(f"Studio is reloading host jobs after {self._defect!r}.") from self._defect
        if self._error is not None:
            raise self._error

    def _follow(self) -> None:
        delay = _RETRY_DELAY
        while not self._stopped.is_set():
            try:
                with self._changed:
                    loaded, after, failing = self._loaded, self._cursor, self._error is not None
                if not loaded:
                    self._load()
                    delay = _RETRY_DELAY
                    continue
                # A held error answers every read, so after one the next page is a probe that returns at once.
                self._apply(after, self._service.activity_page(after=after, wait=0 if failing else _FOLLOW_WAIT))
            except HostError as error:
                self._fail(error)
                self._stopped.wait(_PROBE_DELAY)
            except Exception as defect:
                _log.exception("Studio stopped following host jobs; reloading in %.1f s", delay)
                with self._changed:
                    self._defect = defect
                    self._loaded = False
                    self._changed.notify_all()
                self._stopped.wait(delay)
                delay = min(2 * delay, _RESTART_DELAY_LIMIT)

    def _fail(self, error: HostError) -> None:
        with self._changed:
            self._error = error
            self._changed.notify_all()

    def _load(self) -> None:
        snapshot = self._service.activity_snapshot()
        with self._changed:
            self._rows = {row.job_id: row for row in snapshot.rows}
            self._cursor = self._log_floor = snapshot.cursor
            self._oldest_cursor = snapshot.oldest_cursor
            self._log.clear()
            self._loaded = True
            self._error = None
            self._defect = None
            self._changed.notify_all()

    def _apply(self, after: int, page: ActivityPage) -> None:
        if page.oldest_cursor > after or page.head < after:
            self._load()
            return
        with self._changed:
            known = self._rows.keys()
            admitted = {
                _project_of(entry.activity) for entry in page.entries
                if entry.activity is not None and entry.activity.job_id not in known
            }
        refreshed: list[ActivitySnapshot] = [
            self._service.activity_snapshot(project=project) for project in sorted(admitted)
        ]
        with self._changed:
            if self._cursor != after:
                return
            # Rows are kept in the order of their latest change, so Activity can order projects by recency.
            for entry in page.entries:
                if entry.activity is not None:
                    self._rows.pop(entry.activity.job_id, None)
                    self._rows[entry.activity.job_id] = entry.activity
            for snapshot in refreshed:
                self._rows.update((row.job_id, row) for row in snapshot.rows)
            self._log.extend(page.entries)
            while len(self._log) > _LOG_LIMIT:
                self._log_floor = self._log.popleft().event["cursor"]
            self._cursor = page.cursor
            self._oldest_cursor = page.oldest_cursor
            self._error = None
            self._changed.notify_all()
