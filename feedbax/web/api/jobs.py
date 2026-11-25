"""HTTP projection of the shared job service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from time import monotonic
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from feedbax.contracts.execution_target import ExecutionTargets, LOCAL_TARGET_ID, TargetRunKind
from feedbax.contracts.jobs import (
    StudioActivityJob,
    StudioEvaluationObjectiveSeries,
    StudioEventReset,
    StudioEventWatermark,
    StudioJobEventHistory,
    StudioJobResumeRequest,
    StudioJobSnapshot,
    StudioJobStopItem,
    StudioJobStopRequest,
    StudioJobSubmission,
    StudioLocalSession,
    StudioRemoteSession,
    StudioVenueStream,
    StudioMachineApproval,
    StudioMachineApprovalReceipt,
    StudioMachineQuote,
    StudioSavedEvaluationRead,
    StudioSavedEvaluationReplayRequest,
    StudioSavedMachineQuoteRequest,
    StudioSavedTrainRunRequest,
    StudioSavedTrainRuns,
)
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.planner.project_plan import RunKey
from feedbax.contracts.receipt_validity import ReceiptValidity, ValidityAcceptance, ValidityAction
from feedbax.contracts.trial_clock import StudioSavedFigureMeanRequest
from feedbax.contracts.trial_record import StudioTrialReplayResponse
from feedbax.host import HostError
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.orchestration.session import SessionError
from feedbax.web.api.projects import ProjectRevisionWatch, _project_response
from feedbax.web.services.job_activity import FollowerStopped, JobActivity
from feedbax.web.services.remote_runs import StudioRemoteRuns


router = APIRouter(prefix="/api", tags=["jobs"])
_FOLDER_OBSERVATION_INTERVAL = 0.5


def _jobs(request: Request) -> JobService:
    return request.app.state.job_service


def _activity(request: Request) -> JobActivity:
    return request.app.state.job_activity


def _activity_error(exc: FollowerStopped) -> HTTPException:
    return HTTPException(503, detail={"code": "job_activity_stopped", "message": str(exc)})


def _host_error(exc: HostError) -> HTTPException:
    status = 503 if exc.code in {"unavailable", "disconnected", "protocol_version", "runtime_unavailable"} else 409
    return HTTPException(status, detail={"code": exc.code, "message": str(exc)})


@router.get("/projects/{project_id}/train-runs", response_model_exclude_unset=True)
def saved_train_runs(
    project_id: ProjectName, revision: Sha256, request: Request, response: Response,
    window_map: bool = False, stopped_checkpoint_job: str | None = None,
) -> StudioSavedTrainRuns:
    def read() -> StudioSavedTrainRuns:
        service = request.app.state.project_plan
        view = service.read(
            project_id, revision=revision,
            stopped_checkpoint_jobs=(stopped_checkpoint_job,) if stopped_checkpoint_job is not None else (),
            maintain_validity=True,
        )
        return service.response(view, include_window_map=window_map)

    try:
        return _project_response(response, read)
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/projects/{project_id}/train-runs", status_code=202)
def run_saved_train(
    project_id: ProjectName, body: StudioSavedTrainRunRequest, request: Request, response: Response,
    background: BackgroundTasks,
) -> StudioJobSubmission:
    if body.target != LOCAL_TARGET_ID:
        raise HTTPException(409, detail="Review and approve the machine quote first.")
    folders = request.app.state.project_folders

    def admit() -> StudioJobSubmission:
        reviewed = folders.read_plan(project_id, revision=body.revision)
        return _jobs(request).submit_plan(
            folders, reviewed,
            run_keys=[RunKey(key.variant, key.step, key.row) for key in body.run_keys],
            token=body.token,
            reuse_completed=body.reuse_completed,
        )

    try:
        submission = _project_response(response, admit)
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    _activity(request).advance()
    background.add_task(_jobs(request).warm)
    return submission


def _result_validity_action(
    project_id: ProjectName, body: ValidityAction, request: Request, response: Response, *,
    judgement: ValidityAcceptance | None = None,
) -> ReceiptValidity:
    def perform() -> ReceiptValidity:
        plan = request.app.state.project_folders.read_plan(project_id, revision=body.revision)
        return _jobs(request).result_validity_action(
            plan, body.request_id, body.receipt,
            judgement=(judgement.differences, judgement.who, judgement.why) if judgement is not None else None,
        )
    try:
        return _project_response(response, perform)
    except JobNotFound as exc:
        raise HTTPException(404, detail="No successful receipt for this request in the saved Plan.") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/projects/{project_id}/result-validity/check", status_code=202)
def check_saved_result(
    project_id: ProjectName, body: ValidityAction, request: Request, response: Response,
) -> ReceiptValidity:
    return _result_validity_action(project_id, body, request, response)


@router.post("/projects/{project_id}/result-validity/accept")
def accept_saved_result(
    project_id: ProjectName, body: ValidityAcceptance, request: Request, response: Response,
) -> ReceiptValidity:
    return _result_validity_action(project_id, body, request, response, judgement=body)


def _remote_runs(request: Request) -> StudioRemoteRuns:
    remote = request.app.state.remote_runs
    if remote is None:
        raise HTTPException(503, detail="Machine approval is unavailable.")
    return remote


@router.post("/projects/{project_id}/train-runs/quote")
def quote_saved_run(
    project_id: ProjectName, body: StudioSavedMachineQuoteRequest, request: Request, response: Response,
) -> StudioMachineQuote:
    remote = _remote_runs(request)

    def quote() -> StudioMachineQuote:
        reviewed = request.app.state.project_folders.read_plan(project_id, revision=body.revision)
        return remote.quote(reviewed, body)

    try:
        return _project_response(response, quote)
    except (JobConflict, SessionError) as exc:
        raise HTTPException(409, detail=str(exc)) from exc


class _ApprovalResponse(Response):
    def __init__(self, result: StudioMachineApprovalReceipt, headers, release: Callable[[], None]) -> None:
        super().__init__(result.model_dump_json(), status_code=202, media_type="application/json", headers=headers)
        self.release_opening = release

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.release_opening()


@router.post("/projects/{project_id}/train-runs/approve", status_code=202, response_model=StudioMachineApprovalReceipt)
def approve_saved_run(
    project_id: ProjectName, body: StudioMachineApproval, request: Request, response: Response,
) -> StudioMachineApprovalReceipt | Response:
    remote = _remote_runs(request)
    try:
        fresh = _project_response(
            response, lambda: remote.approve(request.app.state.project_folders, project_id, body.quote),
        )
    except (JobConflict, SessionError) as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    result = StudioMachineApprovalReceipt(session_id=body.quote)
    if fresh:
        return _ApprovalResponse(result, response.headers, lambda: remote.acknowledge_opening(body.quote))
    return result


@router.post("/projects/{project_id}/evaluate-replay")
def saved_evaluation_replay(
    project_id: ProjectName, body: StudioSavedEvaluationReplayRequest, request: Request, response: Response,
) -> StudioTrialReplayResponse:
    return _evaluation_response(
        response, lambda: request.app.state.project_plan.evaluation_replay(project_id, body),
    )


@router.get("/projects/{project_id}/light-results/{revision}/{job_id}/{role}/{member:path}")
def saved_light_product(
    project_id: ProjectName, revision: Sha256, job_id: str, role: str, member: str, request: Request,
) -> Response:
    response = Response()
    try:
        product = _project_response(response, lambda: request.app.state.project_plan.light_product(
            project_id, revision, job_id, role, member,
        ))
    except JobNotFound as exc:
        raise HTTPException(404, detail="Product is absent from the saved Plan.") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    headers = {"Cache-Control": response.headers["cache-control"], "X-Content-Type-Options": "nosniff"}
    if product.content_security_policy is not None:
        headers["Content-Security-Policy"] = product.content_security_policy
    return Response(product.content, media_type=product.media_type, headers=headers)


@router.post("/projects/{project_id}/figures/mean")
def saved_figure_mean(
    project_id: ProjectName, body: StudioSavedFigureMeanRequest, request: Request, response: Response,
) -> Response:
    try:
        product = _project_response(response, lambda: request.app.state.project_plan.figure_mean(project_id, body))
    except JobNotFound as exc:
        raise HTTPException(404, detail="Figure is absent from the saved Plan.") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    return Response(
        product.content, media_type=product.media_type,
        headers={"Cache-Control": response.headers["cache-control"], "X-Content-Type-Options": "nosniff"},
    )


@router.post("/projects/{project_id}/evaluate-objective")
def saved_evaluation_objective(
    project_id: ProjectName, body: StudioSavedEvaluationRead, request: Request, response: Response,
) -> StudioEvaluationObjectiveSeries:
    return _evaluation_response(
        response, lambda: request.app.state.project_plan.evaluation_objective(project_id, body),
    )


def _evaluation_response[T](response: Response, read: Callable[[], T]) -> T:
    try:
        return _project_response(response, read)
    except JobNotFound as exc:
        raise HTTPException(404, detail="Evaluate receipt is absent from the saved Plan.") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.get("/submissions")
def list_submissions(request: Request, project: str | None = None) -> list[StudioJobSubmission]:
    try:
        return _jobs(request).submissions(project=project)
    except HostError as exc:
        raise _host_error(exc) from exc


@router.get("/submissions/{submission_id}")
def get_submission(submission_id: str, request: Request) -> StudioJobSubmission:
    try:
        return _jobs(request).submission(submission_id)
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown submission") from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.get("/jobs")
def list_jobs(request: Request, project: str | None = None) -> StudioJobSnapshot:
    try:
        snapshot = _activity(request).snapshot(project=project)
        remote = request.app.state.remote_runs
        return remote.snapshot(snapshot, project=project) if remote is not None else snapshot
    except FollowerStopped as exc:
        raise _activity_error(exc) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/jobs/warm", status_code=202)
def warm_workers(request: Request, background: BackgroundTasks) -> None:
    background.add_task(_jobs(request).warm)


@router.post("/jobs/stop", status_code=202)
def stop_jobs(body: StudioJobStopRequest, request: Request) -> list[StudioJobStopItem]:
    try:
        remote = request.app.state.remote_runs
        stopped = (
            remote.stop(_jobs(request), body.job_ids, body.mode) if remote is not None
            else _jobs(request).stop(body.job_ids, body.mode)
        )
    except SessionError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    _activity(request).advance()
    return stopped


@router.post("/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str, body: StudioJobResumeRequest, request: Request) -> StudioActivityJob:
    try:
        remote = request.app.state.remote_runs
        resumed = (
            remote.resume(_jobs(request), job_id, body.token) if remote is not None
            else _jobs(request).resume(job_id, body.token)
        )
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown job") from exc
    except (JobConflict, SessionError) as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    _activity(request).advance()
    return resumed


@router.post("/jobs/{job_id}/seen")
def acknowledge_job(job_id: str, request: Request) -> StudioActivityJob:
    try:
        remote = request.app.state.remote_runs
        seen = (
            remote.acknowledge(_jobs(request), job_id) if remote is not None
            else _jobs(request).acknowledge(job_id)
        )
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown job") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc
    _activity(request).put(seen)
    return seen


@router.get("/sessions")
def list_sessions(request: Request) -> list[StudioLocalSession | StudioRemoteSession]:
    remote = request.app.state.remote_runs
    local = StudioJobSnapshot(jobs=[], unreadable=[], cursor=0, oldest_cursor=0, sessions=[StudioLocalSession()])
    return remote.snapshot(local).sessions if remote is not None else local.sessions


@router.post("/sessions/{session_id}/close", status_code=204)
def close_session(session_id: UUID, request: Request) -> None:
    try:
        _remote_runs(request).close(session_id)
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown machine") from exc
    except SessionError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.get("/orchestration/targets")
def list_targets(
    request: Request, kind: Annotated[list[TargetRunKind] | None, Query()] = None,
) -> ExecutionTargets:
    jobs = _jobs(request)
    if request.app.state.remote_runs is not None:
        catalog = _remote_runs(request).catalog(kind or ())
    else:
        catalog = jobs.targets.catalog(kinds=kind or (), session_submission=False)
    try:
        runtime = jobs.host.runtime_inventory()
    except HostError as exc:
        raise _host_error(exc) from exc
    if runtime is None:
        return catalog
    return catalog.model_copy(update={
        "targets": tuple(
            target.model_copy(update={"profile": target.profile.model_copy(update={"runtime": runtime})})
            if target.profile.kind == "local" else target
            for target in catalog.targets
        ),
    })


@router.get("/events/history")
def job_event_history(
    request: Request,
    project: Annotated[str, Query(min_length=1)],
    through: Annotated[int, Query(ge=0)],
    after: Annotated[int, Query(ge=0)] = 0,
    venue: UUID | None = None,
    stream: Annotated[str | None, Query(min_length=1)] = None,
) -> StudioJobEventHistory:
    if after > through:
        raise HTTPException(422, detail="Replay start follows its boundary.")
    try:
        if venue is not None and stream is not None:
            return _remote_runs(request).venue_history(
                StudioVenueStream(session_id=venue, stream_id=stream), after=after, through=through, projects=(project,),
            )
        if venue is not None or stream is not None:
            raise HTTPException(422, detail="Machine history requires both source identifiers.")
        return _activity(request).history(after=after, project=project, through=through)
    except FollowerStopped as exc:
        raise _activity_error(exc) from exc
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown machine history.") from exc
    except SessionError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.get(
    "/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def job_events(
    request: Request,
    after: Annotated[int, Query(ge=0)] = 0,
    project: Annotated[list[str] | None, Query()] = None,
    last_event_id: Annotated[int | None, Header(ge=0)] = None,
    machine: bool = False,
    folder: Annotated[list[ProjectName] | None, Query()] = None,
) -> StreamingResponse:
    service = _activity(request)
    remote = request.app.state.remote_runs if machine else None
    projects = tuple(dict.fromkeys(project or ()))
    folders = tuple(dict.fromkeys(folder or ()))
    watch = ProjectRevisionWatch(request.app.state.project_folders, folders) if folders else None
    start = last_event_id if last_event_id is not None else after
    try:
        boundary, initial_page = await run_in_threadpool(
            service.events,
            after=start,
            projects=projects,
            wait=0,
            machine=machine,
        )
    except FollowerStopped as exc:
        raise _activity_error(exc) from exc
    except HostError as exc:
        raise _host_error(exc) from exc

    venue_positions = {
        (cursor.session_id, cursor.stream_id): cursor.cursor
        for cursor in await run_in_threadpool(remote.venue_cursors)
    } if remote is not None and projects else {}

    async def stream() -> AsyncIterator[str]:
        head = boundary
        page = initial_page
        cursor = start
        replay_complete = False
        next_observation = monotonic()
        last_remote_version = 0
        while True:
            if cursor < page.oldest_cursor or cursor > head:
                reset = StudioEventReset(oldest_cursor=page.oldest_cursor, cursor=head)
                yield f"event: reset\ndata: {reset.model_dump_json()}\n\n"
                return
            for event in page.events:
                if not replay_complete and event.cursor > boundary:
                    replay_complete = True
                    yield _watermark_event("replay_complete", boundary)
                yield f"id: {event.cursor}\ndata: {event.model_dump_json()}\n\n"
            cursor = page.cursor
            if not replay_complete and cursor >= boundary:
                replay_complete = True
                yield _watermark_event("replay_complete", boundary)
            if not page.events:
                yield _watermark_event("heartbeat", cursor)
            if remote is not None and replay_complete and remote.version != last_remote_version:
                last_remote_version = remote.version
                yield f'event: machines_changed\ndata: {{"version":{last_remote_version}}}\n\n'
            if remote is not None and replay_complete and projects:
                for source in await run_in_threadpool(remote.venue_cursors):
                    key = (source.session_id, source.stream_id)
                    position = venue_positions.get(key, 0)
                    if position == source.cursor:
                        continue
                    events = await run_in_threadpool(
                        remote.venue_history, StudioVenueStream(session_id=source.session_id, stream_id=source.stream_id),
                        after=position, through=source.cursor, projects=projects,
                    )
                    venue_positions[key] = events.cursor
                    for event in events.events:
                        yield f"event: venue_event\ndata: {event.model_dump_json()}\n\n"
            if watch is not None and replay_complete and monotonic() >= next_observation:
                revisions = await run_in_threadpool(watch.changed)
                next_observation = monotonic() + _FOLDER_OBSERVATION_INTERVAL
                for revision in revisions:
                    yield f"event: project_revision\ndata: {revision.model_dump_json()}\n\n"
            wait = 5 if remote is None else 0.75
            if watch is not None and replay_complete:
                wait = min(wait, max(0, next_observation - monotonic()))
            try:
                head, page = await run_in_threadpool(
                    service.events,
                    after=cursor,
                    projects=projects,
                    wait=wait,
                    machine=machine,
                )
            except (HostError, FollowerStopped):
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _watermark_event(kind: str, cursor: int) -> str:
    return f"id: {cursor}\nevent: {kind}\ndata: {StudioEventWatermark(cursor=cursor).model_dump_json()}\n\n"
