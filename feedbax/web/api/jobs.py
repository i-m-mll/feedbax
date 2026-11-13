"""HTTP projection of the shared job service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from time import monotonic
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

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
    StudioSavedEvaluationRead,
    StudioSavedEvaluationReplayRequest,
    StudioSavedTrainRunRequest,
    StudioSavedTrainRuns,
)
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.workspace_replay import WorkspaceReplayTrial
from feedbax.host import HostError
from feedbax.jobs import JobConflict, JobNotFound, JobService
from feedbax.web.api.projects import ProjectRevisionWatch, _project_response


router = APIRouter(prefix="/api", tags=["jobs"])
_FOLDER_OBSERVATION_INTERVAL = 0.5


def _jobs(request: Request) -> JobService:
    return request.app.state.job_service


def _host_error(exc: HostError) -> HTTPException:
    status = 503 if exc.code in {"unavailable", "disconnected", "protocol_version"} else 409
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
        )
        return service.response(view, include_window_map=window_map)

    try:
        return _project_response(response, read)
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/projects/{project_id}/train-runs", status_code=202)
def run_saved_train(
    project_id: ProjectName, body: StudioSavedTrainRunRequest, request: Request, response: Response,
) -> StudioJobSubmission:
    folders = request.app.state.project_folders

    def admit() -> StudioJobSubmission:
        reviewed = folders.read_plan(project_id, revision=body.revision)
        return _jobs(request).admit_plan(
            folders, reviewed,
            run_keys=[RunKey(key.variant, key.step, key.row) for key in body.run_keys],
            token=body.token,
        )

    try:
        return _project_response(response, admit)
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/projects/{project_id}/evaluate-replay")
def saved_evaluation_replay(
    project_id: ProjectName, body: StudioSavedEvaluationReplayRequest, request: Request, response: Response,
) -> WorkspaceReplayTrial:
    return _evaluation_response(
        response, lambda: request.app.state.project_plan.evaluation_replay(project_id, body),
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
        service = _jobs(request)
        service.warm()
        return service.jobs(project=project)
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/jobs/stop", status_code=202)
def stop_jobs(body: StudioJobStopRequest, request: Request) -> list[StudioJobStopItem]:
    try:
        return _jobs(request).stop(body.job_ids, body.mode)
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str, body: StudioJobResumeRequest, request: Request) -> StudioActivityJob:
    try:
        return _jobs(request).resume(job_id, body.token)
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown job") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.post("/jobs/{job_id}/seen")
def acknowledge_job(job_id: str, request: Request) -> StudioActivityJob:
    try:
        return _jobs(request).acknowledge(job_id)
    except JobNotFound as exc:
        raise HTTPException(404, detail="Unknown job") from exc
    except JobConflict as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except HostError as exc:
        raise _host_error(exc) from exc


@router.get("/sessions")
def list_sessions() -> list[StudioLocalSession]:
    return [StudioLocalSession()]


@router.get("/events/history")
def job_event_history(
    request: Request,
    project: Annotated[str, Query(min_length=1)],
    through: Annotated[int, Query(ge=0)],
    after: Annotated[int, Query(ge=0)] = 0,
) -> StudioJobEventHistory:
    if after > through:
        raise HTTPException(422, detail="Replay start follows its boundary.")
    try:
        return _jobs(request).event_history(after=after, project=project, through=through)
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
    service = _jobs(request)
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
    except HostError as exc:
        raise _host_error(exc) from exc

    async def stream() -> AsyncIterator[str]:
        head = boundary
        page = initial_page
        cursor = start
        replay_complete = False
        next_observation = monotonic()
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
            if watch is not None and replay_complete and monotonic() >= next_observation:
                revisions = await run_in_threadpool(watch.changed)
                next_observation = monotonic() + _FOLDER_OBSERVATION_INTERVAL
                for revision in revisions:
                    yield f"event: project_revision\ndata: {revision.model_dump_json()}\n\n"
            wait = 5
            if watch is not None and replay_complete:
                wait = max(0, next_observation - monotonic())
            try:
                head, page = await run_in_threadpool(
                    service.events,
                    after=cursor,
                    projects=projects,
                    wait=wait,
                    machine=machine,
                )
            except HostError:
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _watermark_event(kind: str, cursor: int) -> str:
    return f"id: {cursor}\nevent: {kind}\ndata: {StudioEventWatermark(cursor=cursor).model_dump_json()}\n\n"
