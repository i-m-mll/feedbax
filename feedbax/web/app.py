from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from feedbax.web.api import (
    analysis,
    analyses,
    components,
    execution,
    figures,
    graphs,
    inspection,
    orchestration,
    provider,
    runs,
    statistics,
    training,
    trajectories,
)
from feedbax.web.ws import training as ws_training
from feedbax.web.ws import simulation as ws_simulation

STUDIO_DEV_ORIGINS = ["http://localhost:3008"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Feedbax Web API",
        version="0.1.0",
        description="API for Feedbax model construction and training",
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
