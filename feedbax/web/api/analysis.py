"""API router for demand-driven analysis generation and job status polling.

Endpoints:
    POST /api/analyses/jobs -- trigger background figure generation
    GET  /api/analyses/jobs/status/{request_id} -- poll a generation job
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException

from feedbax.contracts.studio_api import (
    AnalysisJobStatusResponse,
    GenerateAnalysisRequest,
    GenerateAnalysisResponse,
)
from feedbax.contracts.manifest import AnalysisRunSpec, ParentRef, analysis_run_manifest_id
from feedbax.web.services.analysis_service import JobStatus, job_tracker

logger = logging.getLogger(__name__)
router = APIRouter()

# Dedicated thread pool for CPU-bound JAX work so we don't starve the
# asyncio event loop.  A single worker prevents concurrent JAX compilations
# from fighting over device memory.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisJobResult:
    """Durable result identifiers produced by one Studio analysis job."""

    manifest_id: str
    manifest_path: str
    figure_hashes: list[str]
    artifact_ids: list[str]
    artifact_paths: list[str]


def _spec_for_analysis_request(payload: GenerateAnalysisRequest) -> AnalysisRunSpec:
    """Build the executable analysis spec demanded by the Studio request."""
    if not payload.eval_run_id:
        raise HTTPException(
            status_code=400,
            detail="eval_run_id is required for Studio analysis execution",
        )
    return AnalysisRunSpec(
        analysis_type=payload.node_id,
        inputs=[
            ParentRef(
                kind="EvaluationRunManifest",
                id=payload.eval_run_id,
                role="evaluation_run",
            )
        ],
        params={
            "requested_outputs": [payload.node_id],
            "studio": {
                "node_id": payload.node_id,
                "force_rerun": payload.force_rerun,
            },
        },
    )


def _run_analysis_sync(spec: AnalysisRunSpec) -> AnalysisJobResult:
    """Run the executable analysis spec synchronously inside the executor."""
    from feedbax.analysis.specs import execute_analysis_run_spec

    manifest, path = execute_analysis_run_spec(
        spec,
        fig_dump_formats=("json",),
    )
    return AnalysisJobResult(
        manifest_id=manifest.id,
        manifest_path=str(path),
        figure_hashes=[
            artifact.sha256
            for artifact in manifest.artifacts
            if artifact.role == "figure" and artifact.sha256 is not None
        ],
        artifact_ids=[artifact.artifact_id for artifact in manifest.artifacts],
        artifact_paths=[
            str(Path(artifact.uri))
            for artifact in manifest.artifacts
            if artifact.uri is not None
        ],
    )


async def _run_analysis_background(request_id: str, spec: AnalysisRunSpec) -> None:
    """Wrapper that updates the job tracker around the synchronous pipeline."""
    await job_tracker.update_status(request_id, JobStatus.RUNNING)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_analysis_sync,
            spec,
        )
        await job_tracker.update_status(
            request_id,
            JobStatus.COMPLETE,
            figure_hashes=result.figure_hashes,
            manifest_id=result.manifest_id,
            manifest_path=result.manifest_path,
            artifact_ids=result.artifact_ids,
            artifact_paths=result.artifact_paths,
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


@router.post("", response_model=GenerateAnalysisResponse)
async def generate_figure(payload: GenerateAnalysisRequest) -> GenerateAnalysisResponse:
    """Trigger demand-driven figure generation for an analysis node.

    The computation runs in a background thread; this endpoint returns
    immediately with a ``request_id`` that can be polled via
    ``GET /status/{request_id}``.

    ``eval_run_id`` identifies the evaluation manifest consumed by the analysis
    spec. The in-memory tracker is UX-only; the returned manifest ID is the
    durable result identity.
    """
    spec = _spec_for_analysis_request(payload)
    manifest_id = analysis_run_manifest_id(spec)
    logger.info(
        "Generate request for node_id=%s with eval_run_id=%s manifest_id=%s",
        payload.node_id,
        payload.eval_run_id,
        manifest_id,
    )
    request_id = await job_tracker.create_job(payload.node_id, manifest_id=manifest_id)
    asyncio.create_task(
        _run_analysis_background(request_id, spec),
    )
    return GenerateAnalysisResponse(
        data={
            "request_id": request_id,
            "status": JobStatus.PENDING.value,
            "manifest_id": manifest_id,
        }
    )


@router.get("/status/{request_id}", response_model=AnalysisJobStatusResponse)
async def get_status(request_id: str) -> AnalysisJobStatusResponse:
    """Poll the status of a figure generation job."""
    entry = await job_tracker.get_status(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id '{request_id}'")
    return AnalysisJobStatusResponse(
        data={
            "request_id": entry.request_id,
            "status": entry.status.value,
            "figure_hashes": entry.figure_hashes,
            "manifest_id": entry.manifest_id,
            "manifest_path": entry.manifest_path,
            "artifact_ids": entry.artifact_ids,
            "artifact_paths": entry.artifact_paths,
            "error": entry.error,
        }
    )
