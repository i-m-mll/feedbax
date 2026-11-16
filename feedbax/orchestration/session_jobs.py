"""Saved-plan delegation, reconnect and result custody for an attached session."""

from __future__ import annotations

from base64 import b64encode
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from typing import TYPE_CHECKING, NamedTuple, Never
from uuid import UUID

from feedbax.contracts.artifact_custody import ContentObjectRef, StoreTransferProgress
from feedbax.contracts.execution_target import SESSION_RUN_KINDS, OpeningStep, SessionTarget
from feedbax.contracts.jobs import JobStatus, StudioJobSubmission
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_refs import RequestId
from feedbax.host.client import HostClient
from feedbax.host.plan_history import plan_run_key
from feedbax.host.protocol import Event, EventPage, HostError, JsonObject, JobsSnapshot, StopMode
from feedbax.jobs import JobConflict, JobService
from feedbax.orchestration.session import (
    Digest, Nonnegative, SessionAuthorization, SessionDriver, SessionError, SessionModel,
    SessionOpeningFailed, SessionOpeningRejected, SessionRecord, SessionRequest,
)
from feedbax.orchestration.venue_history import VenueHistory
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


def _saved_delegation(
    reviewed: SettledProjectPlan, run_keys: Sequence[RunKey], *, destination: bool,
) -> Delegation:
    if not run_keys:
        raise SessionError("Delegation requires saved runs and a submission token.")
    selected = {run.key: run.request_id for run in reviewed.plan.runs if run.kind in SESSION_RUN_KINDS}
    if any(key not in selected for key in run_keys):
        raise SessionError("This session accepts saved Train and Evaluate runs.")
    return Delegation(
        project=reviewed.project_id, revision=reviewed.revision, semantic_root=reviewed.semantic_root,
        run_keys=tuple(run_keys), request_ids=tuple(selected[key] for key in run_keys),
        local_store=str(reviewed.store_path) if destination else None,
    )


def _freeze_delegation(folders: ProjectFolder, request: Delegation) -> Delegation:
    text = folders.read_authored_snapshot(request.project, revision=request.revision)
    return request.model_copy(update={
        "files": {name: b64encode(data).decode("ascii") for name, data in text.files.items()},
    })


def _create_journal(driver: SessionDriver) -> None:
    with driver.connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_delegations(
                session TEXT NOT NULL, token TEXT NOT NULL, request TEXT NOT NULL, submission TEXT,
                PRIMARY KEY(session, token)
            );
            CREATE TABLE IF NOT EXISTS session_openings(
                session TEXT PRIMARY KEY, approval TEXT NOT NULL, adoption_token TEXT NOT NULL,
                close_reason TEXT, failure TEXT
            );
            CREATE TABLE IF NOT EXISTS session_stops(
                session TEXT NOT NULL, job TEXT NOT NULL, mode TEXT NOT NULL, reply TEXT,
                PRIMARY KEY(session, job, mode)
            );
            """
        )


class _OpeningApproval(SessionModel):
    request: SessionRequest
    authorization: SessionAuthorization
    hourly_microusd: Nonnegative


class ApprovedRunClosed(SessionError):
    """The approved selection could not be delivered and its opening is settled."""


@dataclass(frozen=True)
class ApprovedRowStatus:
    key: RunKey
    request_id: RequestId
    status: JobStatus
    reason: str | None
    opening_step: OpeningStep | None
    job_id: str | None = None


@dataclass(frozen=True)
class ApprovedRunVisibility:
    request: SessionRequest
    project: str
    store: Path
    expires_at_ms: int
    hourly_microusd: int
    rows: tuple[ApprovedRowStatus, ...]


class ApprovedSessionRuns:
    """Stage a saved delegation before acquiring its machine, then advance it by session ID."""

    def __init__(self, driver: SessionDriver, attach: Callable[[SessionDriver, UUID], SessionJobs]) -> None:
        self.driver = driver
        self._attach = attach
        _create_journal(driver)

    def stage(
        self, folders: ProjectFolder, reviewed: SettledProjectPlan, *, run_keys: Sequence[RunKey],
        request: SessionRequest, authorization: SessionAuthorization, hourly_microusd: int,
    ) -> None:
        self.driver._validate(request, authorization)
        runnable = {
            run.key: run for run in reviewed.plan.runs
            if run.kind in SESSION_RUN_KINDS and run.admission_problem is None
        }
        if len(set(run_keys)) != len(run_keys) or any(key not in runnable for key in run_keys):
            raise SessionError("Machine approval requires distinct runnable saved Train and Evaluate rows.")
        approval = _OpeningApproval(request=request, authorization=authorization, hourly_microusd=hourly_microusd)
        delegated = _freeze_delegation(folders, _saved_delegation(reviewed, run_keys, destination=True))
        identity = str(request.id)
        with self.driver.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                """SELECT a.approval, d.request FROM session_openings a
                   JOIN session_delegations d ON d.session=a.session AND d.token=a.session WHERE a.session=?""",
                (identity,),
            ).fetchone()
            if prior is None:
                if authorization.expires_at_ms <= self.driver._clock_ms():
                    raise SessionError("The machine quote has expired.")
                db.execute(
                    "INSERT INTO session_delegations(session, token, request) VALUES (?, ?, ?)",
                    (identity, identity, delegated.model_dump_json()),
                )
                db.execute(
                    "INSERT INTO session_openings(session, approval, adoption_token) VALUES (?, ?, ?)",
                    (identity, approval.model_dump_json(), token_urlsafe()),
                )
            elif (
                _OpeningApproval.model_validate_json(prior[0]) != approval
                or Delegation.model_validate_json(prior[1]) != delegated
            ):
                raise SessionError("This approval belongs to a different saved delegation.")

    def staged(self, session_id: UUID, *, project: str) -> bool:
        with self.driver.connection() as db:
            return db.execute(
                """SELECT 1 FROM session_openings a
                   JOIN session_delegations d ON d.session=a.session AND d.token=a.session
                   WHERE a.session=? AND json_extract(d.request, '$.project')=?""",
                (str(session_id), project),
            ).fetchone() is not None

    def pending(self) -> tuple[UUID, ...]:
        with self.driver.connection() as db:
            return tuple(UUID(session) for (session,) in db.execute(
                """SELECT a.session FROM session_openings a
                   JOIN session_delegations d ON d.session=a.session AND d.token=a.session
                   WHERE a.failure IS NULL AND (d.submission IS NULL OR a.close_reason IS NOT NULL) ORDER BY a.rowid""",
            ))

    def visibility(self, records: Mapping[UUID, SessionRecord]) -> tuple[ApprovedRunVisibility, ...]:
        """Read approved row state from the staged delegation, before a Host attempt is visible."""
        with self.driver.connection() as db:
            staged = db.execute(
                """SELECT json_extract(a.approval,'$.request'),
                          json_extract(a.approval,'$.authorization.expires_at_ms'),
                          json_extract(a.approval,'$.hourly_microusd'), a.failure, a.close_reason,
                          json_remove(d.request,'$.files'), d.submission
                   FROM session_openings a JOIN session_delegations d ON d.session=a.session AND d.token=a.session
                   ORDER BY a.rowid""",
            ).fetchall()
        visible = []
        for request_json, expiry, rate, failure, close_reason, delegation_json, submission_json in staged:
            request = SessionRequest.model_validate_json(request_json)
            delegated = Delegation.model_validate_json(delegation_json)
            assert delegated.local_store is not None and delegated.request_ids is not None
            record = records.get(request.id)
            failure = failure or close_reason or (record.opening_failure if record is not None else None)
            events = record.opening_observations if record is not None else ()
            step = events[-1].step if events else None
            if failure is not None:
                status, reason = JobStatus.FAILED, failure
            elif record is None:
                status, reason = JobStatus.QUEUED, "Waiting for the machine to open."
            elif record.phase == "open":
                status, reason, step = JobStatus.STARTING, "Sending runs to the machine.", None
            else:
                status, reason = JobStatus.STARTING, None if step else "Opening the machine."
            items = {}
            if submission_json is not None:
                submission = StudioJobSubmission.model_validate_json(submission_json)
                items = {item.run_key: item for item in submission.items}
            rows = []
            for key, request_id in zip(delegated.run_keys, delegated.request_ids, strict=True):
                item = items.get(plan_run_key(key.variant, key.step, key.row))
                if item is not None and item.disposition == "rejected":
                    rows.append(ApprovedRowStatus(key, request_id, JobStatus.FAILED, item.reason or failure, step))
                else:
                    rows.append(ApprovedRowStatus(key, request_id, status, reason, step, item.job_id if item else None))
            visible.append(ApprovedRunVisibility(
                request, delegated.project, Path(delegated.local_store), expiry, rate, tuple(rows),
            ))
        return tuple(visible)

    def _reject(self, session_id: UUID, message: str) -> Never:
        with self.driver.connection() as db:
            db.execute("UPDATE session_openings SET failure=? WHERE session=?", (message, str(session_id)))
        raise ApprovedRunClosed(message)

    def _close(self, session_id: UUID, reason: str) -> Never:
        with self.driver.connection() as db:
            db.execute("UPDATE session_openings SET close_reason=? WHERE session=?", (reason, str(session_id)))
        self.driver.close(session_id)
        self._reject(session_id, reason)

    def _complete(
        self, session: SessionJobs, session_id: UUID, submission: StudioJobSubmission,
    ) -> StudioJobSubmission:
        if all(item.disposition == "rejected" for item in submission.items):
            reason = next((item.reason for item in submission.items if item.reason), "No approved run was admitted.")
            identity = str(session_id)
            with self.driver.connection() as db:
                session._write_submission(db, submission)
                db.execute("UPDATE session_openings SET close_reason=? WHERE session=?", (reason, identity))
            return self._finish_refused_submission(session_id, submission)
        return session._record_submission(submission)

    def _finish_refused_submission(self, session_id: UUID, submission: StudioJobSubmission) -> StudioJobSubmission:
        self.driver.close(session_id)
        with self.driver.connection() as db:
            db.execute("UPDATE session_openings SET close_reason=NULL WHERE session=?", (str(session_id),))
        return submission

    def resume(
        self, folders: ProjectFolder, session_id: UUID, *, project: str | None = None,
    ) -> StudioJobSubmission | None:
        identity = str(session_id)
        with self.driver.connection() as db:
            owned = db.execute(
                """SELECT 1 FROM session_openings a
                   JOIN session_delegations d ON d.session=a.session AND d.token=a.session
                   WHERE a.session=? AND (? IS NULL OR json_extract(d.request, '$.project')=?)""",
                (identity, project, project),
            ).fetchone()
        if owned is None:
            return None
        with self.driver._exclusive(session_id, approval=True):
            with self.driver.connection() as db:
                staged = db.execute(
                    """SELECT a.approval, a.adoption_token, a.failure, a.close_reason, d.request, d.submission
                       FROM session_openings a
                       JOIN session_delegations d ON d.session=a.session AND d.token=a.session
                       WHERE a.session=?""",
                    (identity,),
                ).fetchone()
            assert staged is not None
            approval_json, adoption_token, failure, close_reason, delegation_json, submission_json = staged
            submission = StudioJobSubmission.model_validate_json(submission_json) if submission_json is not None else None
            if submission is not None and close_reason is None:
                return submission
            if failure is not None:
                raise ApprovedRunClosed(failure)
            approval = _OpeningApproval.model_validate_json(approval_json)
            delegated = Delegation.model_validate_json(delegation_json)
            record = self.driver.find_record(session_id)
            if record is not None:
                launch = record.launch
                if (launch.request, launch.authorization, launch.hourly_microusd) != (
                    approval.request, approval.authorization, approval.hourly_microusd,
                ):
                    raise SessionError("The approved delegation does not own this machine.")
                self.driver._check_owner(launch, adoption_token)
                if submission is not None:
                    return self._finish_refused_submission(session_id, submission)
                if close_reason or record.phase in ("closing", "closed"):
                    self._close(session_id, close_reason or "The machine closed before the approved runs could be confirmed.")
                if launch.ends_at_ms <= self.driver._clock_ms():
                    if record.phase == "open":
                        try:
                            session = self._attach(self.driver, session_id)
                            recovered = session._recover_submission(identity, delegated)
                        except (HostError, OSError, SessionError):
                            if any(resource.kind == "venue" and (
                                resource == record.venue or resource.intent == launch.intent
                            ) for resource in self.driver.inventory()):
                                raise
                        else:
                            if recovered is not None:
                                return self._complete(session, session_id, recovered)
                    self._close(session_id, "The machine reached its deadline before the approved runs could be confirmed.")
            try:
                opened = self.driver.open(
                    approval.request, approval.authorization, adoption_token=adoption_token,
                    expected_hourly_microusd=approval.hourly_microusd,
                )
            except SessionOpeningRejected as error:
                self._reject(session_id, str(error))
            except SessionOpeningFailed as error:
                closed = self.driver.find_record(session_id)
                if closed is not None and closed.phase == "closed":
                    self._reject(session_id, str(error))
                raise
            if opened.phase != "open":
                raise SessionError("The approved machine is closing.")
            session = self._attach(self.driver, session_id)
            recovered = session._recover_submission(identity, delegated) if record is not None else None
            submission = recovered if recovered is not None else session._deliver(folders, identity, delegated)
            return self._complete(session, session_id, submission)


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
        _create_journal(driver)
        self.history = VenueHistory(driver)

    def destination_stores(self) -> dict[str, Path]:
        """Return the exact destination Store captured by each saved delegation."""
        return self.history.destination_stores(self.session_id)

    def delegate(
        self, folders: ProjectFolder, reviewed: SettledProjectPlan, *,
        run_keys: Sequence[RunKey], token: str,
    ) -> StudioJobSubmission:
        if not token:
            raise SessionError("Delegation requires saved runs and a submission token.")
        request = _saved_delegation(reviewed, run_keys, destination=self._venue_jobs is not None)
        with self.driver.connection() as db:
            select = "SELECT request, submission FROM session_delegations WHERE session=? AND token=?"
            key = (self.session_id, token)
            previous = db.execute(select, key).fetchone()
            if previous is None:
                pending = _freeze_delegation(folders, request) if self._venue_jobs is not None else request
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
        return self._record_submission(self._deliver(folders, token, pending))

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
        if submission.target != self.placement or submission.token != token:
            raise SessionError("The venue admission returned a different realized target.")
        return submission

    def _recover_submission(self, token: str, delegated: Delegation) -> StudioJobSubmission | None:
        assert delegated.request_ids is not None and delegated.local_store is not None
        store = str(self._venue_jobs.store(delegated.project).root) if self._venue_jobs is not None else delegated.local_store
        try:
            return JobService.recover_saved_submission(
                self.host, token=token, project=delegated.project, revision=delegated.revision,
                semantic_root=delegated.semantic_root, store=store, run_keys=delegated.run_keys,
                request_ids=delegated.request_ids, placement=self.placement,
            )
        except JobConflict as error:
            raise SessionError(str(error)) from error

    def _record_submission(self, submission: StudioJobSubmission) -> StudioJobSubmission:
        with self.driver.connection() as db:
            self._write_submission(db, submission)
        return submission

    def _write_submission(self, db, submission: StudioJobSubmission) -> None:
        db.execute(
            "UPDATE session_delegations SET submission=? WHERE session=? AND token=?",
            (submission.model_dump_json(), self.session_id, submission.token),
        )

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
                """SELECT token, request FROM session_delegations d
                   WHERE session=? AND submission IS NULL AND NOT EXISTS (
                       SELECT 1 FROM session_openings a WHERE a.session=d.session AND a.session=d.token
                   ) ORDER BY rowid""",
                (self.session_id,),
            ).fetchall()
            stops = db.execute(
                """SELECT job, mode FROM session_stops
                   WHERE session=? AND reply IS NULL ORDER BY rowid""",
                (self.session_id,),
            ).fetchall()
        for token, request in submissions:
            self._record_submission(self._deliver(folders, token, Delegation.model_validate_json(request)))
        for job_id, mode in stops:
            reply = self.host.stop((job_id,), mode)
            with self.driver.connection() as db:
                db.execute(
                    "UPDATE session_stops SET reply=? WHERE session=? AND job=? AND mode=?",
                    (json.dumps(reply), self.session_id, job_id, mode),
                )

    def _observation(self) -> tuple[str, int, JobsSnapshot] | None:
        return self.history.observation(self.session_id)

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
        self.history.record(self.session_id, observed, stream, cursor, snapshot, events)

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
        return self._collect_to_store(local_jobs, project, store, observed_only=observed_only)

    def collect_delegated(
        self, local_jobs: JobService, *, project: str, store: Path,
        progress: Callable[[str, StoreTransferProgress], None] | None = None,
        pending: Mapping[str, str] | None = None,
    ) -> tuple[CollectedAttempt, ...]:
        if self.destination_stores().get(project) != store:
            raise SessionError("The local Store differs from the saved delegation.")
        return self._collect_to_store(local_jobs, project, store, progress=progress, pending=pending)

    def _collect_to_store(
        self, local_jobs: JobService, project: str, store: Path, *, observed_only: bool = False,
        progress: Callable[[str, StoreTransferProgress], None] | None = None,
        pending: Mapping[str, str] | None = None,
    ) -> tuple[CollectedAttempt, ...]:
        if self._venue_jobs is None:
            raise SessionError("Project collection requires a venue Store transport.")
        source = self._venue_jobs.store(project)

        def collect(attempt: JsonObject) -> CollectedAttempt:
            def transferred(update: StoreTransferProgress) -> None:
                assert progress is not None
                progress(str(attempt["id"]), update)

            result = local_jobs.collect_session_attempt(
                source, attempt, project=project, store=store, progress=transferred if progress is not None else None,
            )
            return CollectedAttempt(result.activity.job_id, result.evidence_sha256)

        if pending is not None:
            _, snapshot = self.observation()
            if snapshot is None:
                raise SessionError("This controller has no recorded venue Jobs observation.")
            return self._collect(snapshot, collect, acknowledge=True, project=project, pending=pending)
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
        acknowledge: bool, project: str | None, pending: Mapping[str, str] | None = None,
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
            if pending is not None and pending.get(str(attempt["id"])) != observed_ref.sha256:
                continue
            result = collector(attempt)
            if (result.attempt_id, result.sha256) != (attempt["id"], observed_ref.sha256):
                raise SessionError("Local custody differs from the observed terminal attempt.")
            if acknowledge:
                self.host.acknowledge_collection(result.attempt_id, result.sha256)
            collected.append(result)
        return tuple(collected)
