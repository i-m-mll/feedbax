"""Saved-plan delegation, reconnect and result custody for an attached session."""

from __future__ import annotations

from base64 import b64encode
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.execution_target import SESSION_RUN_KINDS, SessionTarget
from feedbax.contracts.jobs import StudioJobSubmission
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_refs import RequestId
from feedbax.host.client import HostClient
from feedbax.host.protocol import Event, EventPage, JsonObject, JobsSnapshot, StopMode
from feedbax.jobs import JobService
from feedbax.orchestration.session import Digest, SessionDriver, SessionError, SessionModel
from feedbax.project_folder import ProjectFolder, SettledProjectPlan

if TYPE_CHECKING:
    from feedbax.orchestration.venue_jobs import VenueJobs


class Delegation(SessionModel):
    project: ProjectName
    revision: Digest
    semantic_root: Digest
    run_keys: tuple[RunKey, ...]
    request_ids: tuple[RequestId, ...] | None = None
    files: dict[str, str] | None = None
    local_store: str | None = None


def admit_saved_delegation(
    jobs: JobService, folders: ProjectFolder, request: Delegation, *, token: str, placement: SessionTarget,
) -> StudioJobSubmission:
    reviewed = folders.read_plan(request.project, revision=request.revision)
    actual = {run.key: run.request_id for run in reviewed.plan.runs if run.kind in SESSION_RUN_KINDS}
    if (
        reviewed.semantic_root != request.semantic_root
        or any(key not in actual for key in request.run_keys)
        or (request.request_ids is not None and tuple(actual[key] for key in request.run_keys) != request.request_ids)
    ):
        raise SessionError("The venue's saved Plan differs from the delegated Plan.")
    return jobs.admit_plan(folders, reviewed, run_keys=request.run_keys, token=token, placement=placement)


class CollectedAttempt(NamedTuple):
    attempt_id: str
    sha256: str


class SessionJobs:
    """A controller journal for the existing saved-plan Host admission path."""

    def __init__(
        self, driver: SessionDriver, session_id: UUID, make_jobs: Callable[[HostClient], JobService] | None = None,
        *, venue_jobs: VenueJobs | None = None,
    ) -> None:
        if (make_jobs is None) == (venue_jobs is None):
            raise ValueError("Provide one saved-plan admission service.")
        self.driver = driver
        self.session_id = str(session_id)
        self.placement = driver.record(session_id).placement
        self._access = driver.session_access(session_id) if venue_jobs is not None else None
        self.host = driver.attach(session_id, access=self._access)
        self._jobs = make_jobs(self.host) if make_jobs is not None else None
        self._venue_jobs = venue_jobs
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
        selected = {run.key: run.request_id for run in reviewed.plan.runs if run.kind in SESSION_RUN_KINDS}
        if any(key not in selected for key in run_keys):
            raise SessionError("This session accepts saved Train and Evaluate runs.")
        request = Delegation(
            project=reviewed.project_id, revision=reviewed.revision, semantic_root=reviewed.semantic_root,
            run_keys=tuple(run_keys), request_ids=tuple(selected[key] for key in run_keys),
            local_store=str(reviewed.store_path) if self._venue_jobs is not None else None,
        )
        with self.driver.connection() as db:
            select = "SELECT request, submission FROM session_delegations WHERE session=? AND token=?"
            key = (self.session_id, token)
            previous = db.execute(select, key).fetchone()
            if previous is None:
                pending = request
                if self._venue_jobs is not None:
                    text = folders.read_authored_snapshot(request.project, revision=request.revision)
                    pending = request.model_copy(update={
                        "files": {name: b64encode(data).decode("ascii") for name, data in text.files.items()},
                    })
                db.execute(
                    """INSERT INTO session_delegations(session, token, request) VALUES (?, ?, ?)
                       ON CONFLICT(session, token) DO NOTHING""",
                    (*key, pending.model_dump_json()),
                )
                previous = db.execute(select, key).fetchone()
        assert previous is not None
        stored, submission = previous
        pending = Delegation.model_validate_json(stored)
        prior = pending.model_copy(update={"files": None})
        if prior.request_ids is None and self._venue_jobs is None:
            prior = prior.model_copy(update={"request_ids": request.request_ids})
        if prior != request:
            raise SessionError("This submission token belongs to a different saved delegation.")
        if submission is not None:
            return StudioJobSubmission.model_validate_json(submission)
        return self._deliver(folders, token, pending)

    def _deliver(self, folders: ProjectFolder, token: str, request: Delegation) -> StudioJobSubmission:
        if self._venue_jobs is not None:
            assert self._access is not None
            submission = self._venue_jobs.admit(
                request.model_copy(update={"local_store": None}), token=token, access=self._access,
                placement=self.placement,
            )
        else:
            assert self._jobs is not None
            submission = admit_saved_delegation(
                self._jobs, folders, request, token=token, placement=self.placement,
            )
        if submission.target != self.placement:
            raise SessionError("The venue admission returned a different realized target.")
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
        stream = self._check_stream(observed)
        before = observed[1] if observed is not None else 0
        page = self.host.events(before, limit=limit, wait=wait)
        if before < page["oldest_cursor"] or not before <= page["cursor"] <= page["head"]:
            raise SessionError("The venue Host event stream no longer matches the saved cursor.")
        snapshot = self.host.jobs_snapshot(include_launch=True)
        self._record_observation(observed, stream, page["cursor"], snapshot, page["events"])
        return page

    def _check_stream(self, observed: tuple[str, int, JobsSnapshot] | None) -> str:
        stream = self.host.ping().get("stream_id")
        if not isinstance(stream, str) or (observed is not None and observed[0] != stream):
            raise SessionError("The venue Host no longer matches the saved event stream.")
        return stream

    def _record_observation(
        self, observed: tuple[str, int, JobsSnapshot] | None, stream: str, cursor: int,
        snapshot: JobsSnapshot, events: Sequence[Event] = (),
    ) -> None:
        if snapshot["cursor"] < cursor or (observed is not None and snapshot["cursor"] < observed[2]["cursor"]):
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
                ((self.session_id, item["cursor"], json.dumps(item)) for item in events),
            )
            db.execute(
                """INSERT INTO session_observations VALUES (?, ?, ?, ?)
                   ON CONFLICT(session) DO UPDATE SET cursor=excluded.cursor, snapshot=excluded.snapshot""",
                (self.session_id, stream, cursor, json.dumps(snapshot)),
            )

    def collect_project(
        self, local_jobs: JobService, reviewed: SettledProjectPlan, *, observed_only: bool = False,
    ) -> tuple[CollectedAttempt, ...]:
        """Collect this project's exact results into its reviewed local Store."""
        if self._venue_jobs is None:
            raise SessionError("Project collection requires a venue Store transport.")
        project, store = reviewed.project_id, reviewed.store_path
        if not store.is_absolute() or store.resolve() != store:
            raise SessionError("The reviewed local Store path must be canonical.")
        with self.driver.connection() as db:
            rows = db.execute(
                """SELECT json_extract(request, '$.local_store') FROM session_delegations
                   WHERE session=? AND json_extract(request, '$.project')=?
                   AND json_extract(request, '$.revision')=?""",
                (self.session_id, project, reviewed.revision),
            ).fetchall()
        if any(bound is not None and Path(bound) != store for (bound,) in rows):
            raise SessionError("The local Store differs from the saved delegation.")
        source = self._venue_jobs.store(project)

        def collect(attempt: JsonObject) -> CollectedAttempt:
            result = local_jobs.collect_session_attempt(source, attempt, project=project, store=store)
            return CollectedAttempt(result.activity.job_id, result.evidence_sha256)

        read = self.collect_observed if observed_only else self.collect
        return read(collect, project=project)

    def collect(
        self, collector: Callable[[JsonObject], CollectedAttempt], *, project: str | None = None,
    ) -> tuple[CollectedAttempt, ...]:
        """Acknowledge an exact attempt only after the destination committed custody."""
        observed = self._observation()
        stream = self._check_stream(observed)
        snapshot = self.host.jobs_snapshot(include_launch=True)
        self._record_observation(observed, stream, observed[1] if observed is not None else 0, snapshot)
        return self._collect(snapshot, collector, acknowledge=True, project=project)

    def collect_observed(
        self, collector: Callable[[JsonObject], CollectedAttempt], *, project: str | None = None,
    ) -> tuple[CollectedAttempt, ...]:
        """Copy results from the last authenticated snapshot before reconnecting to acknowledge them."""
        _, snapshot = self.observation()
        if snapshot is None:
            raise SessionError("This controller has no recorded venue Jobs observation.")
        return self._collect(snapshot, collector, acknowledge=False, project=project)

    def _collect(
        self, snapshot: JobsSnapshot, collector: Callable[[JsonObject], CollectedAttempt], *,
        acknowledge: bool, project: str | None,
    ) -> tuple[CollectedAttempt, ...]:
        collected = []
        for attempt in snapshot["jobs"]:
            if project is not None and attempt["project"] != project:
                continue
            outcome = attempt["outcome"]
            if not isinstance(outcome, dict):
                continue
            key = "store_job_entry" if attempt["state"] == "stopped" else "store_success"
            raw_ref = outcome.get(key)
            if raw_ref is None:
                continue
            observed_ref = ContentObjectRef.model_validate(raw_ref)
            result = collector(attempt)
            if (result.attempt_id, result.sha256) != (attempt["id"], observed_ref.sha256):
                raise SessionError("Local custody differs from the observed terminal attempt.")
            if acknowledge:
                self.host.acknowledge_collection(result.attempt_id, result.sha256)
            collected.append(result)
        return tuple(collected)
