"""Saved-plan delegation, reconnect and result custody for an attached session."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import NamedTuple
from uuid import UUID

from feedbax.contracts.jobs import StudioJobSubmission
from feedbax.contracts.execution_target import SESSION_RUN_KINDS
from feedbax.contracts.project_plan import RunKey
from feedbax.host.client import HostClient
from feedbax.host.protocol import EventPage, JsonObject, JobsSnapshot, StopMode
from feedbax.jobs import JobService
from feedbax.orchestration.session import SessionDriver, SessionError, SessionModel
from feedbax.project_folder import ProjectFolder, SettledProjectPlan


class Delegation(SessionModel):
    project: str
    revision: str
    semantic_root: str
    run_keys: tuple[RunKey, ...]


class CollectedAttempt(NamedTuple):
    attempt_id: str
    sha256: str


class SessionJobs:
    """A controller journal for the existing saved-plan Host admission path."""

    def __init__(self, driver: SessionDriver, session_id: UUID, make_jobs: Callable[[HostClient], JobService]):
        self.driver = driver
        self.session_id = str(session_id)
        self.placement = driver.record(session_id).placement
        self.host = driver.attach(session_id)
        self._jobs = make_jobs(self.host)
        with self.driver.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_delegations(
                    session TEXT NOT NULL, token TEXT NOT NULL, request TEXT NOT NULL, submission TEXT,
                    PRIMARY KEY(session, token)
                );
                CREATE TABLE IF NOT EXISTS session_stops(
                    session TEXT NOT NULL, job TEXT NOT NULL, mode TEXT NOT NULL, reply TEXT,
                    PRIMARY KEY(session, job, mode)
                );
                CREATE TABLE IF NOT EXISTS session_observations(
                    session TEXT PRIMARY KEY, stream TEXT NOT NULL, cursor INTEGER NOT NULL, snapshot TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_events(
                    session TEXT NOT NULL, cursor INTEGER NOT NULL, event TEXT NOT NULL,
                    PRIMARY KEY(session, cursor)
                );
                """
            )

    def delegate(
        self, folders: ProjectFolder, reviewed: SettledProjectPlan, *,
        run_keys: Sequence[RunKey], token: str,
    ) -> StudioJobSubmission:
        if not run_keys or not token:
            raise SessionError("Delegation requires saved runs and a submission token.")
        selected = {run.key: run.kind for run in reviewed.plan.runs}
        if any(selected.get(key) not in SESSION_RUN_KINDS for key in run_keys):
            raise SessionError("This session accepts saved Train and Evaluate runs.")
        request = Delegation(
            project=reviewed.project_id, revision=reviewed.revision, semantic_root=reviewed.semantic_root,
            run_keys=tuple(run_keys),
        )
        with self.driver.connection() as db:
            db.execute(
                """INSERT INTO session_delegations(session, token, request) VALUES (?, ?, ?)
                   ON CONFLICT(session, token) DO NOTHING""",
                (self.session_id, token, request.model_dump_json()),
            )
            stored, submission = db.execute(
                "SELECT request, submission FROM session_delegations WHERE session=? AND token=?",
                (self.session_id, token),
            ).fetchone()
        if Delegation.model_validate_json(stored) != request:
            raise SessionError("This submission token belongs to a different saved delegation.")
        if submission is not None:
            return StudioJobSubmission.model_validate_json(submission)
        return self._deliver(folders, token, request)

    def _deliver(self, folders: ProjectFolder, token: str, request: Delegation) -> StudioJobSubmission:
        reviewed = folders.read_plan(request.project, revision=request.revision)
        if reviewed.semantic_root != request.semantic_root:
            raise SessionError("The venue's saved Plan differs from the delegated Plan.")
        submission = self._jobs.admit_plan(
            folders, reviewed, run_keys=request.run_keys, token=token, placement=self.placement,
        )
        with self.driver.connection() as db:
            db.execute(
                "UPDATE session_delegations SET submission=? WHERE session=? AND token=?",
                (submission.model_dump_json(), self.session_id, token),
            )
        return submission

    def queue_stop(self, job_id: str, mode: StopMode) -> None:
        """Persist an exact stop intent even while the Host is unreachable."""
        with self.driver.connection() as db:
            db.execute(
                """INSERT INTO session_stops(session, job, mode) VALUES (?, ?, ?)
                   ON CONFLICT(session, job, mode) DO NOTHING""",
                (self.session_id, job_id, mode),
            )

    def replay(self, folders: ProjectFolder) -> None:
        """Send unfinished deliveries with the original tokens and job IDs."""
        with self.driver.connection() as db:
            submissions = db.execute(
                """SELECT token, request FROM session_delegations
                   WHERE session=? AND submission IS NULL ORDER BY rowid""",
                (self.session_id,),
            ).fetchall()
            stops = db.execute(
                """SELECT job, mode FROM session_stops
                   WHERE session=? AND reply IS NULL ORDER BY rowid""",
                (self.session_id,),
            ).fetchall()
        for token, request in submissions:
            self._deliver(folders, token, Delegation.model_validate_json(request))
        for job_id, mode in stops:
            reply = self.host.stop((job_id,), mode)
            with self.driver.connection() as db:
                db.execute(
                    "UPDATE session_stops SET reply=? WHERE session=? AND job=? AND mode=?",
                    (json.dumps(reply), self.session_id, job_id, mode),
                )

    def _observation(self) -> tuple[str, int, JobsSnapshot] | None:
        with self.driver.connection() as db:
            row = db.execute(
                "SELECT stream, cursor, snapshot FROM session_observations WHERE session=?", (self.session_id,)
            ).fetchone()
        return (row[0], row[1], json.loads(row[2])) if row is not None else None

    def observation(self) -> tuple[int, JobsSnapshot | None]:
        if observed := self._observation():
            return observed[1], observed[2]
        return 0, None

    def poll(self, *, limit: int = 256, wait: float = 0) -> EventPage:
        """Mirror a bounded batch and its high-water mark in one commit."""
        observed = self._observation()
        stream = self.host.ping().get("stream_id")
        if not isinstance(stream, str) or (observed is not None and observed[0] != stream):
            raise SessionError("The venue Host no longer matches the saved event stream.")
        _, before, prior = observed if observed is not None else (stream, 0, None)
        page = self.host.events(before, limit=limit, wait=wait)
        snapshot = self.host.jobs_snapshot()
        if (
            before < page["oldest_cursor"] or not before <= page["cursor"] <= page["head"]
            or snapshot["cursor"] < page["cursor"]
            or (prior is not None and snapshot["cursor"] < prior["cursor"])
        ):
            raise SessionError("The venue Host event stream no longer matches the saved cursor.")
        with self.driver.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT stream, cursor, snapshot FROM session_observations WHERE session=?", (self.session_id,)
            ).fetchone()
            current = (current[0], current[1], json.loads(current[2])) if current is not None else None
            if current != observed:
                raise SessionError("Another poll advanced the session observation. Poll again.")
            db.executemany(
                "INSERT INTO session_events VALUES (?, ?, ?)",
                ((self.session_id, item["cursor"], json.dumps(item)) for item in page["events"]),
            )
            db.execute(
                """INSERT INTO session_observations VALUES (?, ?, ?, ?)
                   ON CONFLICT(session) DO UPDATE SET cursor=excluded.cursor, snapshot=excluded.snapshot""",
                (self.session_id, stream, page["cursor"], json.dumps(snapshot)),
            )
        return page

    def collect(self, collector: Callable[[JsonObject], CollectedAttempt]) -> tuple[CollectedAttempt, ...]:
        """Acknowledge an exact attempt only after the destination committed custody."""
        attempts = self.host.jobs_snapshot(include_launch=True)["jobs"]
        collected = []
        for attempt in attempts:
            outcome = attempt["outcome"]
            if not isinstance(outcome, dict) or not (outcome.get("store_success") or outcome.get("store_job_entry")):
                continue
            result = collector(attempt)
            if result.attempt_id != attempt["id"]:
                raise SessionError("Local custody acknowledged a different attempt.")
            self.host.acknowledge_collection(result.attempt_id, result.sha256)
            collected.append(result)
        return tuple(collected)
