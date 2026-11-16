"""Review a saved Run against the same session request Jobs will open."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from threading import Lock
from time import monotonic, time_ns
from uuid import UUID, uuid4

from feedbax.contracts.artifact_custody import StoreTransferProgress
from feedbax.contracts.execution_target import (
    LOCAL_TARGET_ID, CloseWhenIdle, ExecutionTargets, SessionTargetProfile,
    TargetRunKind,
)
from feedbax.contracts.jobs import (
    JobStatus, StopMode, StudioActivityJob, StudioJobEventHistory, StudioJobSnapshot, StudioJobStopItem,
    StudioMachineQuote, StudioMachineQuoteRow, StudioRemoteSession, StudioSavedMachineQuoteRequest, StudioVenueCursor,
    StudioVenueStream,
)
from feedbax.contracts.project_plan import RunKey
from feedbax.host import HostError
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.orchestration.session import (
    ProviderMachineObservation, ProviderUnreachable, SessionAuthorization, SessionDriver, SessionError, SessionRecord, SessionRequest,
)
from feedbax.orchestration.session_jobs import ApprovedRowStatus, ApprovedRunClosed, ApprovedRunVisibility, ApprovedSessionRuns, SessionJobs
from feedbax.orchestration.targets import TargetRegistry
from feedbax.orchestration.venue_history import VenueHistory
from feedbax.project_folder import ProjectFolder, SettledProjectPlan


log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class StudioVenue:
    driver: SessionDriver
    environment: str
    jobs: Callable[[SessionDriver, UUID], SessionJobs]
    authorization_seconds: int
    quote_rate: Callable[[SessionRequest], int]


@dataclass(frozen=True)
class _PendingRun:
    quote: StudioMachineQuote
    project: str
    reviewed: SettledProjectPlan
    keys: tuple[RunKey, ...]
    request: SessionRequest
    lock: Lock = field(default_factory=Lock)


@dataclass(frozen=True)
class _VenueObservation:
    record: SessionRecord
    jobs: tuple[StudioActivityJob, ...] = ()
    stores: dict[str, Path] = field(default_factory=dict)
    reachable: bool = False
    can_close: bool = False
    idle_since_ms: int | None = None
    provider: ProviderMachineObservation | None = None
    verified_cutoff_ms: int | None = None


@dataclass(frozen=True)
class _ProviderOutage:
    request: SessionRequest
    reason: str
    retry_at: float


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
        self._openings = {target: ApprovedSessionRuns(venue.driver, venue.jobs) for target, venue in self._venues.items()}
        self._history = {target: VenueHistory(venue.driver) for target, venue in self._venues.items()}
        self._attest = attest
        self._quotes: dict[UUID, _PendingRun] = {}
        self._lock = Lock()
        self._observations: dict[UUID, _VenueObservation] = {}
        self._opening_errors: dict[UUID, str] = {}
        self._awaiting_ack: set[UUID] = set()
        self._visible_openings: tuple[ApprovedRunVisibility, ...] = ()
        self._unreachable: dict[str, _ProviderOutage] = {}
        self._catalog_version = 0
        self._version = 0
        self._transfer_progress: dict[tuple[UUID, str], StoreTransferProgress] = {}
        self._collecting: set[tuple[UUID, str]] = set()
        self._collector: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=2, thread_name_prefix="venue-collection")

    def catalog(self, kinds: Sequence[TargetRunKind]) -> ExecutionTargets:
        with self._lock:
            refused = {target: outage.reason for target, outage in self._unreachable.items()}
        return self.registry.catalog(kinds=kinds, session_submission=True, unreachable=refused)

    def _provider_response(self, request: SessionRequest, error: ProviderUnreachable | None = None) -> None:
        with self._lock:
            previous = self._unreachable.get(request.target)
            reason = str(error) if error is not None else None
            if reason is not None:
                self._unreachable[request.target] = _ProviderOutage(request, reason, monotonic() + 5)
            elif previous is not None:
                del self._unreachable[request.target]
            if (previous.reason if previous is not None else None) != reason:
                self._catalog_version += 1
                self._version += 1

    def _retry_provider(self, target: str, venue: StudioVenue) -> None:
        with self._lock:
            outage = self._unreachable.get(target)
        if outage is None or outage.retry_at > monotonic():
            return
        try:
            venue.quote_rate(outage.request)
        except ProviderUnreachable as error:
            self._provider_response(outage.request, error)
        except SessionError:
            # The provider answered, but could not confirm this quote.
            self._provider_response(outage.request)
        else:
            self._provider_response(outage.request)

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
        try:
            rate = venue.quote_rate(request)
        except ProviderUnreachable as error:
            self._provider_response(request, error)
            raise
        self._provider_response(request)
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
        pending = _PendingRun(quote, reviewed.project_id, reviewed, keys, request)
        with self._lock:
            self._quotes = {key: item for key, item in self._quotes.items() if item.quote.expires_at_ms > now}
            self._quotes[quote.id] = pending
        return quote

    def approve(self, folders: ProjectFolder, project: str, quote_id: UUID) -> bool:
        """Stage once and hold fresh decisions until their response is sent."""
        with self._lock:
            pending = self._quotes.get(quote_id)
        if pending is None:
            if any(journal.staged(quote_id, project=project) for journal in self._openings.values()):
                return False
            raise JobConflict("Review the machine quote again.")
        if pending.project != project:
            raise JobConflict("Review the machine quote again.")
        with pending.lock:
            openings = self._openings[pending.request.target]
            if openings.staged(quote_id, project=project):
                return False
            quote = pending.quote
            if quote.expires_at_ms <= time_ns() // 1_000_000:
                raise JobConflict("The machine quote has expired.")
            folders.read_authored_snapshot(project, revision=pending.reviewed.revision)
            request = pending.request
            authorization = self._attest(quote.model_copy(deep=True), request.model_copy(deep=True))
            if (authorization.request_digest, authorization.expires_at_ms, authorization.ceiling_microusd) != (
                request.digest, quote.expires_at_ms, quote.ceiling_microusd,
            ):
                raise SessionError("Approval differs from the reviewed machine quote.")
            with self._lock:
                self._awaiting_ack.add(quote_id)
            try:
                openings.stage(
                    folders, pending.reviewed, run_keys=pending.keys, request=request,
                    authorization=authorization, hourly_microusd=quote.hourly_microusd,
                )
            except BaseException:
                self.acknowledge_opening(quote_id)
                raise
            with self._lock:
                self._quotes.pop(quote_id, None)
            return True

    def acknowledge_opening(self, quote_id: UUID) -> None:
        with self._lock:
            self._awaiting_ack.discard(quote_id)
            self._version += 1

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _publish(self, observation: _VenueObservation) -> None:
        session_id = observation.record.launch.request.id
        with self._lock:
            if self._observations.get(session_id) != observation:
                self._observations[session_id] = observation
                for job in observation.jobs:
                    if job.status != JobStatus.COLLECTING:
                        self._transfer_progress.pop((session_id, job.job_id), None)
                self._version += 1

    def advance_openings(self, folders: ProjectFolder) -> None:
        for openings in self._openings.values():
            for session_id in openings.pending():
                with self._lock:
                    if session_id in self._awaiting_ack:
                        continue
                try:
                    openings.resume(folders, session_id)
                except ApprovedRunClosed as error:
                    self._opening_errors.pop(session_id, None)
                    log.warning("Approved runs %s were not confirmed: %s", session_id, error)
                    self.acknowledge_opening(session_id)
                except (HostError, OSError, SessionError) as error:
                    message = str(error)
                    if self._opening_errors.get(session_id) != message:
                        self._opening_errors[session_id] = message
                        log.warning("Approved runs %s await recovery: %s", session_id, error)
                else:
                    self._opening_errors.pop(session_id, None)
                    self.acknowledge_opening(session_id)

    def stop_collection(self) -> None:
        with self._lock:
            collector, self._collector = self._collector, None
        if collector is not None:
            collector.shutdown(cancel_futures=True)

    def _queue_collection(
        self, session: SessionJobs, local: JobService, project: str, store: Path,
        pending: dict[str, str], session_id: UUID,
    ) -> None:
        key = (session_id, project)

        def report(job: str, progress: StoreTransferProgress) -> None:
            with self._lock:
                transfer = (session_id, job)
                if self._transfer_progress.get(transfer) != progress:
                    self._transfer_progress[transfer] = progress
                    self._version += 1

        def collect() -> None:
            try:
                session.collect_delegated(local, project=project, store=store, progress=report, pending=pending)
            except (HostError, OSError, JobConflict, SessionError):
                log.exception("Result collection is unresolved for %s on %s", project, session_id)
            finally:
                with self._lock:
                    self._collecting.remove(key)

        with self._lock:
            if self._collector is not None and key not in self._collecting:
                self._collecting.add(key)
                self._collector.submit(collect)

    def _provider_state(
        self, driver: SessionDriver, record: SessionRecord, previous: _VenueObservation | None,
    ) -> ProviderMachineObservation | None:
        if record.venue is None:
            return None
        last = previous.provider if previous else None
        if last is not None and time_ns() // 1_000_000 - last.observed_at_ms < 5_000:
            return last
        try:
            result = driver.provider.machine_observation(record.venue)
        except ProviderUnreachable as error:
            self._provider_response(record.launch.request, error)
            return None
        except (OSError, SessionError) as error:
            if last is not None:
                log.info("Machine status is unavailable for %s: %s", record.launch.request.id, error)
            return None
        else:
            self._provider_response(record.launch.request)
            return result

    def _archived(
        self, local: JobService, record: SessionRecord, provider: ProviderMachineObservation | None,
    ) -> _VenueObservation:
        history = self._history[record.launch.request.target]
        session_id = record.launch.request.id
        observed = history.observation(session_id)
        if observed is None:
            return _VenueObservation(record, provider=provider)
        snapshot = observed[2]
        pending = {
            str(row["id"]) for row in snapshot["jobs"]
            if row["state"] == "succeeded" and isinstance(outcome := row["outcome"], dict) and "store_success" in outcome
        }
        activity = local.project_venue_activity(
            snapshot["jobs"], pending=pending, collected=set(), logs=history.log_tails(session_id),
        )
        off = record.phase == "closed" or provider is not None and provider.state == "off"
        for job in activity:
            job.can_resume = False
            job.allowed_actions = [] if off else [action for action in job.allowed_actions if action != "resume"]
            if off and job.status == JobStatus.COLLECTING:
                job.status = JobStatus.FAILED
                job.reason = "The machine went off before the result reached this project."
        return _VenueObservation(
            record, activity,
            history.destination_stores(session_id), provider=provider,
        )

    def observe(self, folders: ProjectFolder, local_jobs: JobService) -> None:
        """Advance the durable Jobs observations even with no Studio window open."""
        visible_openings = []
        for target, venue in self._venues.items():
            openings = self._openings[target]
            records = {entry.launch.request.id: entry for entry in venue.driver.records()}
            visible_openings.extend(openings.visibility(records))
            for record in records.values():
                if (
                    record.launch.request.target != venue.driver.provider.target
                    or record.launch.request.profile not in self._labels
                ):
                    continue
                session_id = record.launch.request.id
                with self._lock:
                    previous = self._observations.get(session_id)
                if record.phase != "open":
                    if previous is None or record != previous.record:
                        provider = self._provider_state(venue.driver, record, previous) if record.phase == "opening" else None
                        self._publish(self._archived(local_jobs, record, provider))
                    continue
                provider = self._provider_state(venue.driver, record, previous)
                if provider is not None and provider.state == "off":
                    self._publish(self._archived(local_jobs, record, provider))
                    continue
                try:
                    session = venue.jobs(venue.driver, session_id)
                    session.replay(folders)
                    page = session.poll()
                    _, snapshot = session.observation()
                    assert snapshot is not None
                    state = session.host.session_status()
                    stores = session.destination_stores()
                    tails = session.history.log_tails(session_id)
                    cutoff = round(state["expires_at"] * 1000)
                    if state["session_id"] != str(session_id) or cutoff != record.launch.ends_at_ms:
                        raise SessionError("The machine did not confirm the authorized cutoff.")
                except (HostError, OSError, SessionError) as error:
                    if previous is None or previous.reachable:
                        log.info("Venue %s is unreachable: %s", session_id, error)
                    self._publish(self._archived(local_jobs, record, provider))
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
                    record, activity, stores, True, not (state["live_jobs"] or pending or state["expired"] or state["closing"]),
                    state.get("idle_since_ms"), provider, cutoff,
                ))
                if page["cursor"] < page["head"] or not state["pending"]:
                    continue
                for project, store in stores.items():
                    collect = {job.job_id: state["pending"][job.job_id] for job in activity
                               if job.project_id == project and job.job_id in state["pending"]}
                    if collect:
                        self._queue_collection(session, local_jobs, project, store, collect, session_id)
            self._retry_provider(target, venue)
        with self._lock:
            staged = tuple(visible_openings)
            if self._visible_openings != staged:
                self._visible_openings = staged
                self._version += 1

    def opening_rows(self, reviewed: SettledProjectPlan) -> dict[RunKey, ApprovedRowStatus]:
        current = {run.key: run.request_id for run in reviewed.plan.runs}
        rows = {}
        with self._lock:
            observed_ids = {job.job_id for machine in self._observations.values() for job in machine.jobs}
            openings = self._visible_openings
        for entry in openings:
            if (entry.project, entry.store) != (reviewed.project_id, reviewed.store_path):
                continue
            for row in entry.rows:
                if current.get(row.key) != row.request_id:
                    continue
                if row.job_id in observed_ids:
                    rows.pop(row.key, None)
                else:
                    rows[row.key] = row
        return rows

    def venue_cursors(self) -> tuple[StudioVenueCursor, ...]:
        result = []
        for target, venue in self._venues.items():
            cursors = self._history[target].cursors()
            for record in venue.driver.records():
                session_id = record.launch.request.id
                if record.launch.request.target == target and record.launch.request.profile in self._labels and session_id in cursors:
                    stream, cursor = cursors[session_id]
                    result.append(StudioVenueCursor(session_id=session_id, stream_id=stream, cursor=cursor))
        return tuple(result)

    def venue_history(
        self, stream: StudioVenueStream, *, after: int, through: int, projects: Sequence[str],
    ) -> StudioJobEventHistory:
        for target, history in self._history.items():
            observed = history.observation(stream.session_id)
            if observed is None:
                continue
            record = self._venues[target].driver.record(stream.session_id)
            if record.launch.request.target != target or record.launch.request.profile not in self._labels:
                continue
            cursor, events = history.events(
                stream.session_id, stream.stream_id, after=after, through=through, projects=projects,
            )
            owned_jobs = {
                str(job["id"]) for job in observed[2]["jobs"] if isinstance(data := job["data"], dict) and "run_key" in data
            }
            return StudioJobEventHistory(
                cursor=cursor, oldest_cursor=0,
                events=JobService.venue_events(tuple(event for event in events if event["job_id"] in owned_jobs), stream),
            )
        raise JobNotFound(str(stream.session_id))

    def _project_jobs(self, observation: _VenueObservation) -> tuple[StudioActivityJob, ...]:
        session = observation.record.launch.request.id
        return tuple(
            job.model_copy(update={"collection_progress": progress})
            if job.status == JobStatus.COLLECTING and (progress := self._transfer_progress.get((session, job.job_id))) is not None
            else job
            for job in observation.jobs
        )

    def snapshot(self, local: StudioJobSnapshot, *, project: str | None = None) -> StudioJobSnapshot:
        with self._lock:
            observations = tuple(replace(value, jobs=self._project_jobs(value)) for value in self._observations.values())
            version = self._version
            catalog_version = self._catalog_version
            openings = self._visible_openings
        now = time_ns() // 1_000_000
        jobs = {job.job_id: job for job in local.jobs}
        machines = list(local.sessions)
        observed_ids = {entry.record.launch.request.id for entry in observations}
        opening_failures = {
            entry.request.id: next((row.reason for row in entry.rows if row.reason is not None), None)
            for entry in openings if entry.rows and all(row.status == JobStatus.FAILED for row in entry.rows)
        }
        for opening in openings:
            request = opening.request
            if request.id in observed_ids or request.profile not in self._labels:
                continue
            failure = opening_failures.get(request.id)
            machines.append(StudioRemoteSession(
                id=request.id, name=self._labels[request.profile],
                state="off" if failure else "queued",
                hourly_microusd=opening.hourly_microusd, estimated_cost_microusd=0,
                closes_at_ms=opening.expires_at_ms, ceiling_microusd=request.maximum_microusd,
                idle_since_ms=None, job_ids=[], can_close=False, opening_failure=failure,
            ))
        for observation in observations:
            record = observation.record
            launch = record.launch
            opening_failure = record.opening_failure or opening_failures.get(launch.request.id)
            for job in observation.jobs:
                if project is not None and job.project_id != project:
                    continue
                current = jobs.get(job.job_id)
                if current is None or job.status == JobStatus.STOPPED and observation.reachable:
                    jobs[job.job_id] = job
            if record.phase == "closed" and opening_failure is None:
                continue
            provider = observation.provider
            if record.phase == "closed" or provider is not None and provider.state == "off":
                active = "off"
            else:
                active = record.phase if observation.reachable or record.phase != "open" else "unreachable"
            end = provider.observed_at_ms if provider is not None and provider.state == "off" else now
            start = provider.billing_started_at_ms if provider is not None else None
            elapsed_ms = max(0, min(end, launch.ends_at_ms) - (start or launch.acquired_at_ms))
            machines.append(StudioRemoteSession(
                id=launch.request.id, name=self._labels[launch.request.placement.profile],
                state=active, hourly_microusd=launch.hourly_microusd,
                estimated_cost_microusd=elapsed_ms * launch.hourly_microusd // 3_600_000,
                closes_at_ms=observation.verified_cutoff_ms or launch.ends_at_ms, ceiling_microusd=launch.request.maximum_microusd,
                idle_since_ms=observation.idle_since_ms, job_ids=[job.job_id for job in observation.jobs],
                can_close=observation.can_close,
                opening_observations=record.opening_observations, opening_failure=opening_failure,
                billed_cost_microusd=provider.billed_microusd if provider is not None else None,
                provider_observed_at_ms=provider.observed_at_ms if provider is not None else None,
                power_state=provider.state if provider is not None else None,
            ))
        return local.model_copy(update={
            "jobs": list(jobs.values()), "sessions": machines, "remote_version": version,
            "machine_catalog_version": catalog_version,
            "venue_cursors": list(self.venue_cursors()),
        })

    def project_attempts(self, reviewed: SettledProjectPlan) -> tuple[StudioActivityJob, ...]:
        with self._lock:
            return tuple(
                job for observation in self._observations.values()
                if observation.stores.get(reviewed.project_id) == reviewed.store_path
                for job in self._project_jobs(observation) if job.project_id == reviewed.project_id
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
        if found is None or not found[0].reachable:
            return local.resume(job_id, token)
        session = self._session(found[0])
        return JobService(session.host, local.registries).resume(job_id, token)

    def acknowledge(self, local: JobService, job_id: str) -> StudioActivityJob:
        if (found := self._find_job(job_id)) is None:
            return local.acknowledge(job_id)
        seen = local.mark_observed_job_seen(found[1])
        session_id = found[0].record.launch.request.id
        with self._lock:
            current = self._observations[session_id]
            jobs = tuple(job.model_copy(update={"seen": True}) if job.job_id == job_id else job for job in current.jobs)
            self._observations[session_id] = replace(current, jobs=jobs)
            self._version += 1
        return seen

    def close(self, session_id: UUID) -> None:
        with self._lock:
            observation = self._observations.get(session_id)
        if observation is None:
            raise JobNotFound(str(session_id))
        self._venues[observation.record.launch.request.target].driver.close(session_id)
