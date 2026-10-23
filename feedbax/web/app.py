from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    orchestration,
    penzai,
    provider,
    runs,
    statistics,
    training,
    trajectories,
)
from feedbax.web.orchestration.controller import get_studio_controller
from feedbax.web.services.training_service import training_service
from feedbax.web.ws import training as ws_training
from feedbax.web.ws import simulation as ws_simulation

STUDIO_DEV_ORIGINS = ["http://localhost:3008"]
logger = logging.getLogger(__name__)


def create_app(*, bootstrap_modules: tuple[str, ...] = ()) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.bootstrap_modules = bootstrap_modules
        app.state.bootstrap_state = await compose_application(modules=bootstrap_modules)
        try:
            await get_studio_controller(training_service).reconcile_on_startup()
        except Exception:
            logger.exception("Failed to reconcile durable controller state during startup")
        try:
            training_service.rebuild_cache_from_state_docs()
        except Exception:
            logger.exception("Failed to rebuild the Studio run-state projection")
        try:
            yield
        finally:
            training_service._terminate_worker()

    app = FastAPI(
        title="Feedbax Web API",
        version="0.1.0",
        description="API for Feedbax model construction and training",
        lifespan=_lifespan,
    )

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
    app.include_router(training.router, prefix="/api/training", tags=["training"])
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

    app.include_router(ws_training.router, prefix="/ws", tags=["websocket"])
    app.include_router(ws_simulation.router, prefix="/ws", tags=["websocket"])

    return app


app = create_app()
