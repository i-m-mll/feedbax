from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from feedbax.host import HostClient
from feedbax.jobs import JobService
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
    penzai,
    projects,
    statistics,
    trajectories,
)
from feedbax.web.config import BASE_DIR
from feedbax.web.services.analysis_service import AnalysisJobService
from feedbax.web.services.desktop_jobs import StudioJobLifecycle
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService

STUDIO_DEV_ORIGINS = ["http://localhost:3008"]


def create_app(
    *,
    bootstrap_modules: tuple[str, ...] = (),
    desktop_job_root: Path | None = None,
    desktop_release: Path | None = None,
    projects_dir: Path | None = None,
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
        )
        app.state.project_plan = ProjectPlanService(app.state.project_folders, app.state.job_service)
        analysis_jobs = AnalysisJobService(
            job_root=desktop_job_root / "analysis" if desktop_job_root is not None else None
        )
        app.state.analysis_jobs = analysis_jobs
        if desktop_job_root is not None:
            app.state.desktop_jobs = StudioJobLifecycle(analysis_jobs, app.state.job_service, desktop_release)
        try:
            yield
        finally:
            if desktop_job_root is not None:
                analysis_jobs.begin_stop(force=True)
            await analysis_jobs.aclose()

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
    app.include_router(analysis.router, prefix="/api/analyses/jobs", tags=["analysis-jobs"])

    app.include_router(analyses.router, prefix="/api/analyses", tags=["analyses"])

    return app


app = create_app()
