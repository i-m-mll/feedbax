"""Admission and activity for local Studio training requests."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from feedbax.contracts.base import default_manifest_root
from feedbax.contracts.jobs import (
    JobOutcome,
    JobProgress,
    JobStatus,
    StopMode,
    StudioActivityJob,
    StudioJobEvent,
    StudioJobEventHistory,
    StudioJobSnapshot,
    StudioJobStopItem,
    StudioJobSubmission,
    StudioJobSubmissionRequest,
    StudioLocalSession,
    StudioSubmissionItem,
)
from feedbax.contracts.manifest import canonical_manifest_path
from feedbax.host import Admission, AdmissionItem, HostClient, HostError, Job, Launch
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.studio.training_requests import (
    StudioTrainingLoweringError,
    lower_studio_training_runs,
    studio_training_run_key,
)


_LIVE_STATES = frozenset({"queued", "assigned", "running", "stopping"})
_TERMINAL_RESUMABLE = frozenset({"failed", "interrupted", "stopped"})
_ACTIVITY_EVENTS = frozenset(
    {
        "queued",
        "assigned",
        "accepted",
        "phase",
        "progress",
        "publication",
        "stopping",
        "blocked",
        "terminal",
    }
)


class JobNotFound(LookupError):
    pass


class JobConflict(ValueError):
    pass


class JobService:
    """Bind authored requests to immutable worker launches over the durable host."""

    def __init__(
        self,
        host: HostClient,
        registries: ApplicationRegistryBundle,
        *,
        bootstrap_modules: tuple[str, ...] = (),
    ) -> None:
        self.host = host
        self.registries = registries
        self.bootstrap_modules = bootstrap_modules
        self.manifest_root = default_manifest_root().resolve()
        self._training_worker = Launch(
            argv=(sys.executable,),
            env={
                key: os.environ[key]
                for key in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS")
                if key in os.environ
            },
            cwd=str(Path.cwd()),
            warm_entrypoint="feedbax.training_host_worker:main",
            preload=("jax", "feedbax.training.invocation", "feedbax.plugins.composition"),
        )
        self._launch_lock = Lock()
        self._accepting = True
        self._warmed = False

    def admit(self, request: StudioJobSubmissionRequest) -> StudioJobSubmission:
        project = request.project_id
        if previous := self._retry(request.token, project=project, continues=None):
            return previous

        items: dict[str, AdmissionItem] = {}
        runs: dict[str, dict[str, Any]] = {}
        for selection in request.selections:
            keys = selection.run_keys
            try:
                lowered = lower_studio_training_runs(
                    request.workspace,
                    request.graph,
                    stage_id=selection.stage_id,
                    run_keys=keys,
                    registry_bundle=self.registries,
                )
            except StudioTrainingLoweringError as exc:
                for key in keys or [studio_training_run_key(selection.stage_id)]:
                    items.setdefault(key, _rejected(key, str(exc)))
                continue

            selected = set(keys or ())
            for run in lowered:
                key = run.run_key
                selected.discard(key)
                if key in items:
                    continue
                data = {
                    "run_key": key,
                    "stage_id": run.stage_id,
                    "scenario_id": run.scenario_id,
                    "label": run.label,
                    "coordinates": dict(run.coordinates),
                }
                runs[key] = data | {"request_id": run.request_id}
                launch = replace(
                    self._training_worker,
                    payload={
                        "spec": run.spec.model_dump(mode="json", exclude_none=True),
                        "project_id": project,
                        "request_id": run.request_id,
                        "manifest_root": str(self.manifest_root),
                        "bootstrap_modules": list(self.bootstrap_modules),
                    },
                )
                items[key] = AdmissionItem(
                    run_key=key,
                    job=Job(
                        id=uuid4().hex,
                        project=project,
                        request=run.request_id,
                        launch=launch,
                        data=data
                        | {"manifest_root": str(self.manifest_root), "runtime": sys.executable},
                    ),
                )
            for key in keys or ():
                if key in selected:
                    items.setdefault(
                        key, _rejected(key, "The selected run is absent from this step.")
                    )

        record = {"project": project, "continues": None, "runs": runs}
        candidates = list(items.values())
        if request.policy == "all":
            admitted = self._admit(request.token, candidates, record)
        else:
            admitted = self._admit_with_reuse(request.token, candidates, record)
        return self._submission_for_action(admitted, project=project, continues=None)

    def _admit_with_reuse(
        self,
        token: str,
        candidates: list[AdmissionItem],
        record: dict[str, Any],
    ) -> Admission:
        for attempt in range(3):
            snapshot = self.host.jobs_snapshot(project=record["project"])
            history: dict[str, list[dict[str, Any]]] = {}
            for row in snapshot["jobs"]:
                history.setdefault(row["request"], []).append(row)
            chosen = []
            runs = {key: dict(run) for key, run in record["runs"].items()}
            for item in candidates:
                if item.job is None:
                    chosen.append(item)
                elif reusable := self._reusable(history.get(item.job.request, ())):
                    chosen.append(AdmissionItem(run_key=item.run_key, disposition="reused"))
                    runs[item.run_key]["reused_job_id"] = reusable
                else:
                    chosen.append(replace(item, checked_through=snapshot["cursor"]))
            try:
                return self._admit(token, chosen, record | {"runs": runs})
            except HostError as exc:
                if exc.code != "admission_changed" or attempt == 2:
                    raise
        raise AssertionError("Admission retry exhausted without an outcome")

    def _admit(self, token: str, items: list[AdmissionItem], record: dict[str, Any]) -> Admission:
        with self._launch_lock:
            if not self._accepting:
                raise JobConflict("Studio is stopping and cannot start a run.")
            return self.host.admit(token, items, record=record)

    def warm(self) -> None:
        with self._launch_lock:
            if self._accepting and not self._warmed:
                self.host.warm(self._training_worker)
                self._warmed = True

    def begin_runtime_shutdown(self, disposition: str) -> bool:
        """Quiesce jobs using this interpreter before replacing its runtime."""
        with self._launch_lock:
            jobs = self.live_runtime_jobs()
            if disposition == "defer" and jobs:
                return False
            self._accepting = False
            if jobs:
                self.host.stop(
                    [row["id"] for row in jobs], "force" if disposition == "force" else "checkpoint"
                )
            return True

    def live_runtime_jobs(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.host.jobs()
            if row["state"] in _LIVE_STATES and row["data"].get("runtime") == sys.executable
        ]

    def submissions(self, *, project: str | None = None) -> list[StudioJobSubmission]:
        return [_project_submission(record) for record in self.host.submissions(project=project)]

    def submission(self, submission_id: str) -> StudioJobSubmission:
        records = self.host.submissions(id=submission_id)
        if not records:
            raise JobNotFound(submission_id)
        return _project_submission(records[0])

    def jobs(self, *, project: str | None = None) -> StudioJobSnapshot:
        snapshot = self.host.jobs_snapshot(project=project)
        return StudioJobSnapshot(
            jobs=list(self._activity_rows(snapshot["jobs"], project=project).values()),
            cursor=snapshot["cursor"],
            oldest_cursor=snapshot["oldest_cursor"],
            sessions=[StudioLocalSession()],
        )

    def stop(self, job_ids: Sequence[str], mode: StopMode) -> list[StudioJobStopItem]:
        ids = tuple(dict.fromkeys(job_ids))
        outcomes = self.host.stop(ids, mode)
        jobs = self._activity_rows(self.host.jobs_snapshot(job_ids=ids)["jobs"])
        return [
            StudioJobStopItem(
                job_id=result["job_id"], job=jobs.get(result["job_id"]), error=result.get("error")
            )
            for result in outcomes
        ]

    def resume(self, job_id: str, token: str) -> StudioActivityJob:
        previous = self._retry(token, continues=job_id)
        if previous is None:
            rows = self.host.jobs(job_id=job_id, include_launch=True)
            if not rows:
                raise JobNotFound(job_id)
            row = rows[0]
            if not _can_resume(row):
                raise JobConflict("This run has no checkpoint to resume.")
            launch_data = row["launch"]
            payload = launch_data["payload"] | {"resume": row.get("checkpoint")}
            launch = Launch(**(launch_data | {"payload": payload}))
            key = row["data"]["run_key"]
            job = Job(
                id=uuid4().hex,
                project=row["project"],
                request=row["request"],
                launch=launch,
                continues=job_id,
                data=row["data"],
            )
            result = self._admit(
                token,
                [AdmissionItem(run_key=key, job=job)],
                {"project": row["project"], "continues": job_id, "runs": {}},
            )
            previous = self._submission_for_action(result, project=row["project"], continues=job_id)

        item = previous.items[0]
        if item.job_id is None:
            raise JobConflict(item.reason or "The run could not resume.")
        return self._activity_rows(self.host.jobs(job_id=item.job_id))[item.job_id]

    def acknowledge(self, job_id: str) -> StudioActivityJob:
        rows = self.host.jobs(job_id=job_id)
        if not rows:
            raise JobNotFound(job_id)
        if rows[0]["state"] in _LIVE_STATES:
            raise JobConflict("Only a finished run can be acknowledged.")
        self.host.record_put(_seen_key(job_id), {"seen": True})
        return self._activity_rows(rows)[job_id]

    def events(
        self,
        *,
        after: int,
        projects: tuple[str, ...],
        wait: float,
        machine: bool,
    ) -> tuple[int, StudioJobEventHistory]:
        page = self.host.events(after=after, projects=() if machine else projects, wait=wait)
        activity_events = {
            event["job_id"]: event["cursor"]
            for event in page["events"]
            if event["kind"] in _ACTIVITY_EVENTS
        }
        activity = (
            self._activity_rows(self.host.jobs_snapshot(job_ids=tuple(activity_events))["jobs"])
            if activity_events
            else {}
        )
        events = []
        for event in page["events"]:
            job_id = event["job_id"]
            detail = not machine or event["project"] in projects
            snapshot = (
                activity.get(job_id) if activity_events.get(job_id) == event["cursor"] else None
            )
            if detail or snapshot is not None:
                events.append(_project_event(event, activity=snapshot, detail=detail))
        return page["head"], StudioJobEventHistory(
            cursor=page["cursor"],
            oldest_cursor=page["oldest_cursor"],
            events=events,
        )

    def event_history(self, *, after: int, project: str, through: int) -> StudioJobEventHistory:
        page = self.host.events(after=after, projects=(project,), wait=0)
        return StudioJobEventHistory(
            cursor=min(through, page["cursor"]),
            oldest_cursor=page["oldest_cursor"],
            events=[
                _project_event(event) for event in page["events"] if event["cursor"] <= through
            ],
        )

    def _retry(
        self,
        token: str,
        *,
        continues: str | None,
        project: str | None = None,
    ) -> StudioJobSubmission | None:
        rows = self.host.submissions(token=token)
        return (
            self._submission_for_action(rows[0], project=project, continues=continues)
            if rows
            else None
        )

    @staticmethod
    def _submission_for_action(
        response: dict[str, Any],
        *,
        project: str | None,
        continues: str | None,
    ) -> StudioJobSubmission:
        record = response["record"]
        if record["continues"] != continues or (
            project is not None and project != record["project"]
        ):
            raise JobConflict("This submission token belongs to another action.")
        return _project_submission(response)

    def _reusable(self, jobs: Sequence[dict[str, Any]]) -> str | None:
        if any(job["state"] in _LIVE_STATES for job in jobs):
            return None
        return next(
            (
                job["id"]
                for job in reversed(jobs)
                if job["state"] == "succeeded" and self._receipt_matches(job)
            ),
            None,
        )

    def _receipt_matches(self, job: dict[str, Any]) -> bool:
        receipt = (job.get("outcome") or {}).get("receipt")
        if not receipt:
            return False
        path = canonical_manifest_path(
            "TrainingRunManifest", receipt["manifest_id"], root=job["data"]["manifest_root"]
        )
        try:
            return sha256(path.read_bytes()).hexdigest() == receipt["manifest_sha256"]
        except FileNotFoundError:
            return False

    def _activity_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        project: str | None = None,
    ) -> dict[str, StudioActivityJob]:
        rows = [row for row in rows if "run_key" in row["data"]]
        if not rows:
            return {}
        seen = self.host.record_get_many(
            [_seen_key(row["id"]) for row in rows if row["state"] not in _LIVE_STATES]
        )
        current: dict[tuple[str, str], str] = {}
        for submission in reversed(self.host.submissions(project=project)):
            record = submission["record"]
            for run_key, run in record.get("runs", {}).items():
                current.setdefault((record["project"], run_key), run["request_id"])
        result = {}
        for row in rows:
            key = (row["project"], row["data"]["run_key"])
            request_id = row["request"]
            result[row["id"]] = _project_activity(
                row,
                receipt_intact=row["state"] != "succeeded" or self._receipt_matches(row),
                seen=_seen_key(row["id"]) in seen,
                earlier_definition=current.get(key, request_id) != request_id,
            )
        return result


def _rejected(run_key: str, reason: str) -> AdmissionItem:
    return AdmissionItem(run_key=run_key, disposition="rejected", reason=reason)


def _can_resume(job: dict[str, Any]) -> bool:
    return job["state"] in _TERMINAL_RESUMABLE and (
        job.get("checkpoint") is not None or (job["state"] == "stopped" and job["delivery"] == 0)
    )


def _project_submission(response: dict[str, Any]) -> StudioJobSubmission:
    record = response["record"]
    return StudioJobSubmission(
        id=response["id"],
        token=response["token"],
        project_id=record["project"],
        items=[
            StudioSubmissionItem(
                run_key=item["run_key"],
                job_id=item.get("job_id")
                or record["runs"].get(item["run_key"], {}).get("reused_job_id"),
                disposition=item["disposition"],
                reason=item.get("reason"),
            )
            for item in response["items"]
        ],
    )


def _project_event(
    event: dict[str, Any],
    *,
    activity: StudioActivityJob | None = None,
    detail: bool = True,
) -> StudioJobEvent:
    return StudioJobEvent(
        cursor=event["cursor"],
        job_id=event["job_id"],
        project_id=event["project"],
        request_id=event["request"],
        kind=event["kind"] if detail else "activity",
        data=event["data"] if detail else {},
        activity=activity,
    )


def _seen_key(job_id: str) -> str:
    return f"activity/seen/{job_id}"


def _project_activity(
    job: dict[str, Any],
    *,
    receipt_intact: bool,
    seen: bool,
    earlier_definition: bool,
) -> StudioActivityJob:
    state = job["state"]
    phase = (job.get("phase") or {}).get("phase")
    outcome = job.get("outcome")
    progress = job.get("progress")
    completed = progress["completed_updates"] if progress is not None else 0
    reason = (outcome.get("error") or {}).get("message") if outcome is not None else None
    if state == "succeeded" and not receipt_intact:
        status = JobStatus.CORRUPT
        reason = "The training receipt is missing or has changed."
    elif state == "assigned" or (state == "running" and not completed):
        status = JobStatus.STARTING
    else:
        status = JobStatus(state)
    if state == "queued":
        reason = "Waiting for a training slot."
    resumable = _can_resume(job)
    if state == "stopping":
        actions = ["force"]
    elif state in _LIVE_STATES:
        actions = ["checkpoint", "force"]
    else:
        actions = ["resume"] if resumable else []
    return StudioActivityJob(
        job_id=job["id"],
        project_id=job["project"],
        request_id=job["request"],
        continues=job.get("continues"),
        status=status,
        phase=phase,
        reason=reason,
        can_resume=resumable,
        allowed_actions=actions,
        seen=seen,
        earlier_definition=earlier_definition,
        progress=JobProgress(completed_updates=completed, total_updates=progress["total_updates"])
        if progress is not None
        else None,
        outcome=JobOutcome.model_validate(
            {key: value for key, value in outcome.items() if key in JobOutcome.model_fields}
        )
        if outcome is not None
        else None,
        **{
            key: value
            for key, value in job["data"].items()
            if key in StudioActivityJob.model_fields
        },
    )
