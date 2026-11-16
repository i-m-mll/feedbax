"""Review a saved Run against the same session request Jobs will open."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
from secrets import token_urlsafe
from threading import Lock
from time import time_ns
from uuid import UUID, uuid4

from feedbax.contracts.execution_target import (
    LOCAL_TARGET_ID, CloseWhenIdle, ExecutionTargets, SessionTarget, SessionTargetProfile,
    TargetRunKind,
)
from feedbax.contracts.jobs import (
    JobStatus, StopMode, StudioActivityJob, StudioJobSnapshot, StudioJobStopItem, StudioJobSubmission,
    StudioMachineQuote, StudioMachineQuoteRow, StudioRemoteSession, StudioSavedMachineQuoteRequest,
)
from feedbax.contracts.project_plan import RunKey
from feedbax.host import HostError
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.orchestration.session import SessionAuthorization, SessionDriver, SessionError, SessionRecord, SessionRequest
from feedbax.orchestration.session_jobs import SessionJobs
from feedbax.orchestration.targets import TargetRegistry
from feedbax.project_folder import ProjectFolder, SettledProjectPlan


log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class StudioVenue:
    driver: SessionDriver
    environment: str
    jobs: Callable[[SessionDriver, UUID], SessionJobs]
    authorization_seconds: int


@dataclass
class _PendingRun:
    quote: StudioMachineQuote
    project: str
    revision: str
    keys: tuple[RunKey, ...]
    request: SessionRequest
    venue: StudioVenue
    adoption: str = field(default_factory=token_urlsafe)
    lock: Lock = field(default_factory=Lock)
    authorization: SessionAuthorization | None = None
    result: StudioJobSubmission | None = None


@dataclass(frozen=True)
class _VenueObservation:
    record: SessionRecord
    jobs: tuple[StudioActivityJob, ...] = ()
    stores: dict[str, Path] = field(default_factory=dict)
    reachable: bool = False
    can_close: bool = False
    idle_since_ms: int | None = None


class StudioRemoteRuns:
    def __init__(
        self, profiles: Iterable[SessionTargetProfile], venues: Iterable[StudioVenue],
        *, attest: Callable[[StudioMachineQuote, SessionRequest], SessionAuthorization],
    ) -> None:
        venues = tuple(venues)
        if any(venue.authorization_seconds <= 0 for venue in venues):
            raise ValueError("Machine approval requires a positive lifetime.")
        profiles = tuple(profiles)
        self.registry = TargetRegistry(profiles, (venue.driver.provider for venue in venues))
        self._labels = {profile.id: profile.label for profile in profiles}
        self._venues = {venue.driver.provider.target: venue for venue in venues}
        self._attest = attest
        self._quotes: dict[UUID, _PendingRun] = {}
        self._lock = Lock()
        self._observations: dict[UUID, _VenueObservation] = {}
        self._version = 0

    def catalog(self, kinds: Sequence[TargetRunKind]) -> ExecutionTargets:
        catalog = self.registry.catalog(kinds=kinds, session_submission=True)
        return catalog.model_copy(update={
            "targets": tuple(option for option in catalog.targets if set(kinds).issubset(option.run_kinds)),
        })

    def quote(self, reviewed: SettledProjectPlan, body: StudioSavedMachineQuoteRequest) -> StudioMachineQuote:
        if body.target == LOCAL_TARGET_ID:
            raise JobConflict("This Mac does not need approval.")
        keys = tuple(RunKey(key.variant, key.step, key.row) for key in body.run_keys)
        runs = {run.key: run for run in reviewed.plan.runs}
        if len(set(keys)) != len(keys) or any(key not in runs or runs[key].admission_problem for key in keys):
            raise JobConflict("Machine approval requires distinct runnable saved rows.")
        selected = [runs[key] for key in keys]
        kinds = tuple(run.kind for run in selected)
        if any(kind not in {"train", "evaluate"} for kind in kinds):
            raise JobConflict("This machine cannot run the selection.")
        profile = self.registry.require_session(body.target)
        venue = self._venues[profile.driver]
        request = self.registry.session_request(profile.id, session_id=uuid4(), environment=venue.environment)
        rate = venue.driver.provider.hourly_microusd(request)
        if rate > 0 and request.maximum_microusd == 0:
            raise JobConflict("This machine has no budget.")
        now = time_ns() // 1_000_000
        values = [{axis: item.value for axis, item in run.coordinates.items()} for run in selected]
        varied = {
            axis for data in values for axis in data
            if any(axis not in other or data[axis] != other[axis] for other in values)
        }
        quote = StudioMachineQuote(
            id=request.id, machine=profile.label, resources=request.resources,
            hourly_microusd=rate, ceiling_microusd=request.maximum_microusd,
            expires_at_ms=now + venue.authorization_seconds * 1_000,
            idle_timeout_seconds=request.lifecycle.seconds if isinstance(request.lifecycle, CloseWhenIdle) else None,
            idle_with_results="on", status="awaiting_authorization",
            rows=[
                StudioMachineQuoteRow(key=key, label=run.label or run.key.step,
                                      axes={axis: value for axis, value in data.items() if axis in varied})
                for key, run, data in zip(body.run_keys, selected, values, strict=True)
            ],
        )
        pending = _PendingRun(quote, reviewed.project_id, reviewed.revision, keys, request, venue)
        with self._lock:
            self._quotes = {key: item for key, item in self._quotes.items() if item.quote.expires_at_ms > now}
            self._quotes[quote.id] = pending
        return quote

    def approve(self, folders: ProjectFolder, project: str, quote_id: UUID) -> StudioJobSubmission:
        with self._lock:
            pending = self._quotes.get(quote_id)
        if pending is None or pending.project != project:
            raise JobConflict("Review the machine quote again.")
        with pending.lock:
            if pending.result is not None:
                return pending.result
            quote = pending.quote
            if quote.expires_at_ms <= time_ns() // 1_000_000:
                raise JobConflict("The machine quote has expired.")
            folders.read_plan(project, revision=pending.revision)
            request = pending.request
            if pending.authorization is None:
                authorization = self._attest(quote.model_copy(deep=True), request.model_copy(deep=True))
                if (authorization.request_digest, authorization.expires_at_ms, authorization.ceiling_microusd) != (
                    request.digest, quote.expires_at_ms, quote.ceiling_microusd,
                ):
                    raise SessionError("Approval differs from the reviewed machine quote.")
                pending.authorization = authorization
            reviewed = folders.read_plan(project, revision=pending.revision)
            driver = pending.venue.driver
            opened = driver.open(
                request, pending.authorization, adoption_token=pending.adoption,
                expected_hourly_microusd=quote.hourly_microusd,
            )
            if opened.phase != "open":
                raise SessionError("The approved machine is closed.")
            jobs = pending.venue.jobs(driver, quote_id)
            result = jobs.delegate(folders, reviewed, run_keys=pending.keys, token=str(quote_id))
            if not isinstance(result.target, SessionTarget) or result.target != opened.placement:
                raise SessionError("The runs returned a different machine from the approval.")
            pending.result = result
            return result

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _publish(self, observation: _VenueObservation) -> None:
        session_id = observation.record.launch.request.id
        with self._lock:
            if self._observations.get(session_id) != observation:
                self._observations[session_id] = observation
                self._version += 1

    def observe(self, folders: ProjectFolder, local_jobs: JobService) -> None:
        """Advance the durable Jobs observations even with no Studio window open."""
        for venue in self._venues.values():
            for record in venue.driver.records():
                if (
                    record.launch.request.target != venue.driver.provider.target
                    or record.launch.request.profile not in self._labels
                ):
                    continue
                session_id = record.launch.request.id
                with self._lock:
                    previous = self._observations.get(session_id)
                if record.phase != "open":
                    if previous is not None or record.phase != "closed":
                        self._publish(_VenueObservation(record, jobs=previous.jobs if previous else (),
                                                        stores=previous.stores if previous else {}))
                    continue
                try:
                    session = venue.jobs(venue.driver, session_id)
                    session.replay(folders)
                    page = session.poll()
                    _, snapshot = session.observation()
                    assert snapshot is not None
                    state = session.host.session_status()
                    stores = session.destination_stores()
                    tails, last_terminal = session.activity_facts()
                except (HostError, OSError, SessionError) as error:
                    if previous is None or previous.reachable:
                        log.info("Venue %s is unreachable: %s", session_id, error)
                    self._publish(_VenueObservation(
                        record, jobs=previous.jobs if previous else (), stores=previous.stores if previous else {},
                    ))
                    continue
                pending = set(state["pending"])
                # The snapshot can be newer than the replayed terminal event page.
                pending.update(
                    str(row["id"]) for row in snapshot["jobs"]
                    if row["state"] == "succeeded" and row["id"] not in state["collections"]
                    and isinstance(outcome := row["outcome"], dict) and "store_success" in outcome
                )
                activity = local_jobs.project_venue_activity(
                    snapshot["jobs"], pending=pending, collected=set(state["collections"]), logs=tails,
                )
                self._publish(_VenueObservation(
                    record, activity, stores, True, not (state["live_jobs"] or pending or state["expired"]),
                    (last_terminal or record.launch.acquired_at_ms) if not state["live_jobs"] else None,
                ))
                if page["cursor"] < page["head"] or not pending:
                    continue
                for project, store in stores.items():
                    if any(job.job_id in pending and job.project_id == project for job in activity):
                        try:
                            session.collect_delegated(local_jobs, project=project, store=store)
                        except (HostError, OSError, JobConflict, SessionError):
                            log.exception("Result collection is unresolved for %s on %s", project, session_id)

    def snapshot(self, local: StudioJobSnapshot, *, project: str | None = None) -> StudioJobSnapshot:
        with self._lock:
            observations = tuple(self._observations.values())
            version = self._version
        now = time_ns() // 1_000_000
        jobs = {job.job_id: job for job in local.jobs}
        machines = list(local.sessions)
        for observation in observations:
            record = observation.record
            launch = record.launch
            for job in observation.jobs:
                if project is not None and job.project_id != project:
                    continue
                current = jobs.get(job.job_id)
                if current is None or job.status == JobStatus.STOPPED and record.phase == "open":
                    jobs[job.job_id] = job
            if record.phase == "closed":
                continue
            active = record.phase if observation.reachable or record.phase != "open" else "unreachable"
            elapsed_ms = max(0, min(now, launch.ends_at_ms) - launch.acquired_at_ms)
            machines.append(StudioRemoteSession(
                id=launch.request.id, name=self._labels[launch.request.placement.profile],
                state=active, hourly_microusd=launch.hourly_microusd,
                estimated_cost_microusd=elapsed_ms * launch.hourly_microusd // 3_600_000,
                closes_at_ms=launch.ends_at_ms, ceiling_microusd=launch.request.maximum_microusd,
                idle_since_ms=observation.idle_since_ms, job_ids=[job.job_id for job in observation.jobs],
                can_close=observation.can_close,
            ))
        return local.model_copy(update={"jobs": list(jobs.values()), "sessions": machines, "remote_version": version})

    def project_attempts(self, reviewed: SettledProjectPlan) -> tuple[StudioActivityJob, ...]:
        with self._lock:
            return tuple(
                job for observation in self._observations.values()
                if observation.stores.get(reviewed.project_id) == reviewed.store_path
                for job in observation.jobs if job.project_id == reviewed.project_id
            )

    def _find_job(self, job_id: str) -> tuple[_VenueObservation, StudioActivityJob] | None:
        with self._lock:
            for observation in self._observations.values():
                for job in observation.jobs:
                    if job.job_id == job_id:
                        return observation, job
        return None

    def _session(self, observation: _VenueObservation) -> SessionJobs:
        request = observation.record.launch.request
        venue = self._venues[request.target]
        return venue.jobs(venue.driver, request.id)

    def stop(
        self, local: JobService, job_ids: Sequence[str], mode: StopMode,
    ) -> list[StudioJobStopItem]:
        results: dict[str, StudioJobStopItem] = {}
        local_ids = []
        for job_id in dict.fromkeys(job_ids):
            found = self._find_job(job_id)
            if found is None or found[0].record.phase != "open":
                local_ids.append(job_id)
                continue
            observation, job = found
            session = self._session(observation)
            if not observation.reachable and mode in job.allowed_actions:
                session.queue_stop(job_id, mode)
                results[job_id] = StudioJobStopItem(job_id=job_id, job=job)
            else:
                service = JobService(session.host, local.registries)
                (results[job_id],) = service.stop((job_id,), mode)
        if local_ids:
            results.update((item.job_id, item) for item in local.stop(local_ids, mode))
        return [results[job] for job in dict.fromkeys(job_ids)]

    def resume(self, local: JobService, job_id: str, token: str) -> StudioActivityJob:
        found = self._find_job(job_id)
        if found is None or found[0].record.phase != "open":
            return local.resume(job_id, token)
        session = self._session(found[0])
        return JobService(session.host, local.registries).resume(job_id, token)

    def acknowledge(self, local: JobService, job_id: str) -> StudioActivityJob:
        if (found := self._find_job(job_id)) is None:
            return local.acknowledge(job_id)
        return local.mark_observed_job_seen(found[1])

    def close(self, session_id: UUID) -> None:
        with self._lock:
            observation = self._observations.get(session_id)
        if observation is None:
            raise JobNotFound(str(session_id))
        self._venues[observation.record.launch.request.target].driver.close(session_id)
