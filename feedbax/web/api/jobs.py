"""HTTP projection of the shared job service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from feedbax.contracts.jobs import (
    StudioActivityJob,
    StudioJobEventHistory,
    StudioJobResumeRequest,
    StudioJobSnapshot,
    StudioJobStopItem,
    StudioJobStopRequest,
    StudioJobSubmission,
    StudioJobSubmissionRequest,
    StudioLocalSession,
)
from feedbax.host import HostError
from feedbax.jobs import JobConflict, JobNotFound, JobService


router = APIRouter(prefix="/api", tags=["jobs"])


def _jobs(request: Request) -> JobService:
    return request.app.state.job_service


def _host_error(exc: HostError) -> HTTPException:
    status = 503 if exc.code in {"unavailable", "disconnected", "protocol_version"} else 409
    return HTTPException(status, detail={"code": exc.code, "message": str(exc)})


@router.post("/submissions", status_code=202)
def admit_submission(body: StudioJobSubmissionRequest, request: Request) -> StudioJobSubmission:
    try:
        return _jobs(request).admit(body)
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
) -> StreamingResponse:
    service = _jobs(request)
    projects = tuple(dict.fromkeys(project or ()))
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
        while True:
            if cursor < page.oldest_cursor or cursor > head:
                yield (
                    "event: reset\n"
                    f'data: {{"oldest_cursor":{page.oldest_cursor},"cursor":{head}}}\n\n'
                )
                return
            for event in page.events:
                if not replay_complete and event.cursor > boundary:
                    replay_complete = True
                    yield f'id: {boundary}\nevent: replay_complete\ndata: {{"cursor":{boundary}}}\n\n'
                yield f"id: {event.cursor}\ndata: {event.model_dump_json()}\n\n"
            cursor = page.cursor
            if not replay_complete and cursor >= boundary:
                replay_complete = True
                yield f'id: {boundary}\nevent: replay_complete\ndata: {{"cursor":{boundary}}}\n\n'
            if not page.events:
                yield f'id: {cursor}\nevent: heartbeat\ndata: {{"cursor":{cursor}}}\n\n'
            try:
                head, page = await run_in_threadpool(
                    service.events,
                    after=cursor,
                    projects=projects,
                    wait=5,
                    machine=machine,
                )
            except HostError:
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
