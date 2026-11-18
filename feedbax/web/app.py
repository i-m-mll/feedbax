from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from feedbax.contracts.studio_api import StudioRenderFailure
from feedbax.host import HostClient
from feedbax.jobs import JobService, SessionSubmission
from feedbax.orchestration.targets import TargetRegistry
from feedbax.plugins.composition import compose_application
from feedbax.validity.admission import QUICK_RECEIPT_DURATION

from feedbax.web.api import (
    components,
    domains,
    execution,
    figures,
    graphs,
    inspection,
    jobs,
    penzai,
    projects,
    statistics,
    trajectories,
)
from feedbax.web.config import BASE_DIR
from feedbax.web.services.desktop_jobs import StudioJobLifecycle
from feedbax.web.services.job_activity import JobActivity
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.services.remote_runs import StudioRemoteRuns

STUDIO_DEV_ORIGINS = ["http://localhost:3008"]
service_log = logging.getLogger("uvicorn.error")


async def _maintain_remote_runs(advance: Callable[[], None], failure: str, interval: float) -> None:
    while True:
        try:
            await asyncio.to_thread(advance)
        except Exception:
            service_log.exception(failure)
        await asyncio.sleep(interval)


def create_app(
    *,
    desktop_release: Path | None = None,
    projects_dir: Path | None = None,
    targets: TargetRegistry | None = None,
    submit_session: SessionSubmission | None = None,
    remote_runs: StudioRemoteRuns | None = None,
    automatic_run: Literal["all", "no_train", "off"] = "all",
    quick_duration: timedelta = QUICK_RECEIPT_DURATION,
) -> FastAPI:
    project_root = projects_dir if projects_dir is not None else BASE_DIR / "projects"

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.bootstrap_state = await compose_application()
        app.state.project_folders = ProjectFolderService(
            project_root, operations=app.state.bootstrap_state.bundle.operations,
            components=app.state.bootstrap_state.bundle.components,
            functions=app.state.bootstrap_state.bundle.expression_functions,
            python_tasks=app.state.bootstrap_state.bundle.python_tasks,
            training_methods=app.state.bootstrap_state.bundle.training_methods,
        )
        app.state.job_service = JobService(
            HostClient(), app.state.bootstrap_state.bundle,
            targets=targets, submit_session=submit_session,
            automatic_run=automatic_run, quick_duration=quick_duration,
        )
        app.state.job_activity = JobActivity(app.state.job_service)
        app.state.remote_runs = remote_runs
        app.state.project_plan = ProjectPlanService(
            app.state.project_folders, app.state.job_service,
            remote_runs.project_attempts if remote_runs is not None else None,
            remote_runs.opening_rows if remote_runs is not None else None,
        )
        if desktop_release is not None:
            app.state.desktop_jobs = StudioJobLifecycle(app.state.job_service, desktop_release)
        maintenance = () if remote_runs is None else (
            asyncio.create_task(_maintain_remote_runs(
                partial(remote_runs.advance_openings, app.state.project_folders), "Could not advance approved machines.",
                remote_runs.refresh_seconds,
            )),
            asyncio.create_task(_maintain_remote_runs(
                partial(remote_runs.observe, app.state.project_folders, app.state.job_service),
                "Could not refresh the machine activity.", remote_runs.refresh_seconds,
            )),
        )
        try:
            yield
        finally:
            app.state.job_activity.close()
            for task in maintenance:
                task.cancel()
            for task in maintenance:
                with suppress(asyncio.CancelledError):
                    await task
            if remote_runs is not None:
                await asyncio.to_thread(remote_runs.stop_collection)

    app = FastAPI(
        title="Feedbax Web API",
        version="0.1.0",
        description="API for Feedbax model construction and training",
        lifespan=_lifespan,
    )
    app.state.project_folders = ProjectFolderService(project_root)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=STUDIO_DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(graphs.router, prefix="/api/graphs", tags=["graphs"])
    app.include_router(components.router, prefix="/api/components", tags=["components"])
    app.include_router(domains.router, prefix="/api/domains", tags=["domains"])
    app.include_router(penzai.router, prefix="/api/penzai", tags=["penzai"])
    @app.get("/api/health")
    async def health():
        return {"status": "ready"}

    @app.post("/api/studio/render-failure", status_code=204)
    async def render_failure(report: StudioRenderFailure) -> None:
        service_log.error(
            "Studio render failure region=%r type=%r detail=%r components=%r",
            report.region, report.error_type, report.detail, report.component_stack,
        )

    app.include_router(jobs.router)
    app.include_router(projects.router)
    app.include_router(execution.router, prefix="/api/execution", tags=["execution"])
    app.include_router(inspection.router, prefix="/api/inspection", tags=["inspection"])
    app.include_router(
        trajectories.router,
        prefix="/api/trajectories",
        tags=["trajectories"],
    )
    app.include_router(
        statistics.router,
        prefix="/api/trajectories",
        tags=["statistics"],
    )

    app.include_router(figures.router, prefix="/api/figures", tags=["figures"])

    return app


app = create_app()
