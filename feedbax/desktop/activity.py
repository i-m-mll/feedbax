"""Private background Activity commands and live window presence."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import PositiveInt

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import ProjectName
from feedbax.desktop.control import CONTROL_ROUTE
from feedbax.host import HostError
from feedbax.host.protocol import StopMode


class _WindowPresence(StrictModel):
    pid: PositiveInt
    project: ProjectName | None
    ready: bool


class _StopAll(StrictModel):
    mode: StopMode


def register_activity(app: FastAPI, release: str) -> None:
    prefix = f"{CONTROL_ROUTE}/activity"
    gui: _WindowPresence | None = None
    last_report = 0.0
    open_until = 0.0
    intent = 0

    def check_release(x_feedbax_release: str | None = Header(None)) -> None:
        if x_feedbax_release != release:
            raise HTTPException(409, "Studio background Activity belongs to a different release")

    router = APIRouter(prefix=prefix, dependencies=[Depends(check_release)], include_in_schema=False)

    def current_gui() -> _WindowPresence | None:
        return gui if time.monotonic() - last_report < 3 else None

    @router.get("")
    async def read(after: int | None = None):
        try:
            cursor, active, terminal = await asyncio.to_thread(app.state.job_service.background_activity, after)
        except HostError as error:
            raise HTTPException(503, str(error)) from error
        window = current_gui()
        notice = None
        if terminal is not None:
            job, project, name, status = terminal
            notice = {"id": job, "project": project, "name": name, "status": status}
        return {
            "after": after,
            "cursor": cursor,
            "count": len(active),
            "terminal": notice,
            "window": window.model_dump(include={"pid", "project"}) if window is not None else None,
        }

    @router.post("/stop")
    async def stop(body: _StopAll):
        jobs = app.state.job_service
        try:
            _, active, _ = await asyncio.to_thread(jobs.background_activity, None)
            return await asyncio.to_thread(jobs.stop, active, body.mode) if active else []
        except HostError as error:
            raise HTTPException(503, str(error)) from error

    @router.post("/open")
    async def open_activity():
        nonlocal open_until, intent
        intent += 1
        open_until = time.monotonic() + 20
        return {"pid": gui.pid if gui is not None else None}

    @router.post("/window")
    async def window_report(body: _WindowPresence):
        nonlocal gui, last_report
        gui, last_report = body, time.monotonic()
        return {"open": intent if body.ready and last_report < open_until else None}

    @router.post("/ack/{opened}")
    async def acknowledge(opened: int):
        nonlocal open_until
        if opened == intent:
            open_until = 0.0
        return {}

    app.include_router(router)
