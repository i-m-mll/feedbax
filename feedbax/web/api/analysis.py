"""API router for demand-driven analysis generation and job status polling.

Endpoints:
    POST /generate  -- trigger background figure generation for an analysis node
    GET  /status/{request_id} -- poll the status of a generation job
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from feedbax.web.services.analysis_service import JobStatus, job_tracker

logger = logging.getLogger(__name__)
router = APIRouter()

# Dedicated thread pool for CPU-bound JAX work so we don't starve the
# asyncio event loop.  A single worker prevents concurrent JAX compilations
# from fighting over device memory.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Body for ``POST /generate``."""

    node_id: str
    force_rerun: bool = False


class GenerateResponse(BaseModel):
    """Returned immediately by ``POST /generate``."""

    request_id: str
    status: str


class StatusResponse(BaseModel):
    """Returned by ``GET /status/{request_id}``."""

    request_id: str
    status: str
    figure_hashes: Optional[list[str]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


def _run_analysis_sync(node_id: str, force_rerun: bool) -> list[str]:
    """Run the analysis pipeline synchronously (called inside the executor).

    Returns a list of figure hashes produced during the run.
    """
    import jax.random as jr

    from feedbax.analysis.execution import run_analysis_module
    from feedbax.config import PATHS, load_config
    from feedbax.database import FigureRecord, db_session
    from feedbax.plugins import EXPERIMENT_REGISTRY

    # The node_id from the frontend corresponds to the analysis module key
    # (e.g. "part2.plant_perts").  Load its YAML config via the registry.
    module_key = node_id
    module_config = load_config(
        module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY,
    )

    # Use a deterministic PRNG key -- reproducible unless the user
    # explicitly varies the seed via config.
    key = jr.PRNGKey(0)

    states_pkl_dir = PATHS.cache / "states"
    states_pkl_dir.mkdir(parents=True, exist_ok=True)

    data, common_inputs, all_analyses, all_results, all_figs = run_analysis_module(
        module_key=module_key,
        module_config=module_config,
        no_pickle=force_rerun,
        states_pkl_dir=states_pkl_dir,
        requested_outputs={node_id},
        key=key,
    )

    # Collect figure hashes that were saved during this run by querying the
    # database for figures belonging to the evaluation.
    figure_hashes: list[str] = []
    with db_session(autocommit=False) as session:
        records = (
            session.query(FigureRecord.hash)
            .filter(FigureRecord.archived == False)  # noqa: E712
            .order_by(FigureRecord.created_at.desc())
            .limit(100)
            .all()
        )
        figure_hashes = [r.hash for r in records]

    return figure_hashes


async def _run_analysis_background(request_id: str, node_id: str, force_rerun: bool) -> None:
    """Wrapper that updates the job tracker around the synchronous pipeline."""
    await job_tracker.update_status(request_id, JobStatus.RUNNING)
    try:
        loop = asyncio.get_running_loop()
        figure_hashes = await loop.run_in_executor(
            _executor,
            _run_analysis_sync,
            node_id,
            force_rerun,
        )
        await job_tracker.update_status(
            request_id, JobStatus.COMPLETE, figure_hashes=figure_hashes,
        )
    except Exception:
        tb = traceback.format_exc()
        logger.error("Analysis job %s failed:\n%s", request_id, tb)
        await job_tracker.update_status(
            request_id, JobStatus.ERROR, error=str(tb),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=GenerateResponse)
async def generate_figure(payload: GenerateRequest) -> GenerateResponse:
    """Trigger demand-driven figure generation for an analysis node.

    The computation runs in a background thread; this endpoint returns
    immediately with a ``request_id`` that can be polled via
    ``GET /status/{request_id}``.
    """
    request_id = await job_tracker.create_job(payload.node_id)
    asyncio.create_task(
        _run_analysis_background(request_id, payload.node_id, payload.force_rerun),
    )
    return GenerateResponse(request_id=request_id, status=JobStatus.PENDING.value)


@router.get("/status/{request_id}", response_model=StatusResponse)
async def get_status(request_id: str) -> StatusResponse:
    """Poll the status of a figure generation job."""
    entry = await job_tracker.get_status(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id '{request_id}'")
    return StatusResponse(
        request_id=entry.request_id,
        status=entry.status.value,
        figure_hashes=entry.figure_hashes,
        error=entry.error,
    )
