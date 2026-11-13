from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from feedbax.host import HostClient
from feedbax.jobs import JobService, evaluation_candidate_configured
from feedbax.plugins.composition import compose_application

from feedbax.web.api import (
    analysis,
    analyses,
    components,
    domains,
    execution,
    figures,
    graphs,
    inspection,
    jobs,
    orchestration,
    penzai,
    projects,
    provider,
    runs,
    statistics,
    trajectories,
)
from feedbax.web.config import BASE_DIR
from feedbax.web.orchestration.controller import StudioController
from feedbax.web.services.analysis_service import AnalysisJobService
from feedbax.web.services.desktop_jobs import StudioJobLifecycle
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.services.training_service import training_service

STUDIO_DEV_ORIGINS = ["http://localhost:3008"]
logger = logging.getLogger(__name__)


def create_app(
    *,
    bootstrap_modules: tuple[str, ...] = (),
    studio_controller: StudioController | None = None,
    desktop_job_root: Path | None = None,
    projects_dir: Path | None = None,
    enable_evaluation_candidate: bool = False,
) -> FastAPI:
    project_root = projects_dir if projects_dir is not None else BASE_DIR / "projects"

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.bootstrap_modules = bootstrap_modules
        app.state.bootstrap_state = await compose_application(modules=bootstrap_modules)
        app.state.project_folders = ProjectFolderService(
            project_root, operations=app.state.bootstrap_state.bundle.operations,
            components=app.state.bootstrap_state.bundle.components,
            functions=app.state.bootstrap_state.bundle.expression_functions,
            python_tasks=app.state.bootstrap_state.bundle.python_tasks,
        )
        app.state.job_service = JobService(
            HostClient(), app.state.bootstrap_state.bundle, bootstrap_modules=bootstrap_modules,
            enable_evaluation_candidate=app.state.enable_evaluation_candidate,
        )
        app.state.project_plan = ProjectPlanService(app.state.project_folders, app.state.job_service)
        if desktop_job_root is None:
            try:
                training_service.rebuild_cache_from_state_docs()
            except Exception:
                logger.exception("Failed to rebuild the Studio run-state projection")
        analysis_jobs = AnalysisJobService(
            job_root=desktop_job_root / "analysis" if desktop_job_root is not None else None
        )
        app.state.analysis_jobs = analysis_jobs
        desktop_jobs = None
        if desktop_job_root is not None:
            await asyncio.to_thread(training_service.reconcile_desktop_worker, desktop_job_root)
            desktop_jobs = StudioJobLifecycle(
                analysis_jobs, training_service, desktop_job_root, app.state.job_service,
            )
            app.state.desktop_jobs = desktop_jobs
        controller = studio_controller or StudioController(training_service=training_service)
        app.state.studio_controller = controller
        try:
            async with controller.startup_recovery():
                yield
        finally:
            try:
                if desktop_jobs is not None:
                    analysis_jobs.begin_stop(force=True)
                await analysis_jobs.aclose()
            finally:
                training_service._terminate_worker()

    app = FastAPI(
        title="Feedbax Web API",
        version="0.1.0",
        description="API for Feedbax model construction and training",
        lifespan=_lifespan,
    )
    app.state.enable_evaluation_candidate = enable_evaluation_candidate
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
    app.include_router(provider.router, prefix="/api/provider", tags=["provider"])
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

    app.include_router(
        orchestration.router,
        prefix="/api/orchestration",
        tags=["orchestration"],
    )
    app.include_router(figures.router, prefix="/api/figures", tags=["figures"])
    app.include_router(analysis.router, prefix="/api/analyses/jobs", tags=["analysis-jobs"])

    app.include_router(analyses.router, prefix="/api/analyses", tags=["analyses"])
    app.include_router(runs.router, prefix="/api/runs", tags=["runs"])

    return app


app = create_app(enable_evaluation_candidate=evaluation_candidate_configured())
