from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from feedbax.web.api import trajectories, statistics


def create_app() -> FastAPI:
    app = FastAPI(
        title='Feedbax Web API',
        version='0.1.0',
        description='API for Feedbax model construction and training',
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:5173'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.include_router(
        trajectories.router, prefix='/api/trajectories', tags=['trajectories'],
    )
    app.include_router(
        statistics.router, prefix='/api/trajectories', tags=['statistics'],
    )

    return app


app = create_app()
