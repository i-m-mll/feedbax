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

from fastapi import APIRouter, HTTPException, Request

from feedbax.plugins.application import ApplicationRegistryBundle

from feedbax.contracts.studio_api import (
    AnalysisJobStatusResponse,
    GenerateAnalysisRequest,
    GenerateAnalysisResponse,
)
from feedbax.contracts.figures import FIGURE_COMPOSITION_SPEC_SCHEMA_ID, FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    analysis_run_manifest_id,
    figure_manifest_id,
)
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


def _job_artifact_location(artifact: ArtifactRef) -> str:
    """Return a stable job-status location without path-normalizing artifact URIs."""
    if artifact.uri is None:
        raise ValueError("analysis job artifacts require a URI")
    if artifact.uri.startswith("artifact://sha256/"):
        return artifact.uri
    return str(Path(artifact.uri))


def _spec_for_analysis_request(
    payload: GenerateAnalysisRequest,
    *,
    root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Build the executable analysis spec demanded by the Studio request."""
    if not payload.eval_run_id:
        raise HTTPException(
            status_code=400,
            detail="eval_run_id is required for Studio analysis execution",
        )
    evaluation_parent = ParentRef(
        kind="EvaluationRunManifest",
        id=payload.eval_run_id,
        role="evaluation_run",
    )
    if payload.evaluation_states_policy == "require_durable":
        from feedbax.analysis.manifest_inputs import authenticated_manifest_ref
        from feedbax.analysis.specs import find_manifest_by_id
        from feedbax.contracts.manifest import EvaluationRunManifest

        manifest, path = find_manifest_by_id(payload.eval_run_id, root=root)
        if not isinstance(manifest, EvaluationRunManifest):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"eval_run_id {payload.eval_run_id!r} does not resolve to an "
                    "EvaluationRunManifest"
                ),
            )
        evaluation_parent = authenticated_manifest_ref(manifest, path, "evaluation_run")

    return AnalysisRunSpec(
        analysis_type=payload.node_id,
        inputs=[evaluation_parent],
        evaluation_states_policy=payload.evaluation_states_policy,
        params={
            "requested_outputs": [payload.node_id],
            "studio": {
                "node_id": payload.node_id,
                "force_rerun": payload.force_rerun,
            },
        },
    )


def _figure_spec_for_request(payload: GenerateAnalysisRequest) -> FigureSpec:
    """Build a direct figure spec without accepting client-controlled source roots."""
    if payload.figure_spec is None:
        raise HTTPException(
            status_code=400,
            detail="figure_spec is required when job_kind='figure'",
        )
    if payload.figure_spec.get("schema_id") == FIGURE_COMPOSITION_SPEC_SCHEMA_ID:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "figure_composition_not_supported_in_studio",
                "message": (
                    "Studio accepts resolved FigureSpec v2 only; server-owned composition "
                    "source roots are not configured for this endpoint"
                ),
            },
        )
    spec = FigureSpec.model_validate(payload.figure_spec)
    if payload.eval_run_id and not spec.inputs:
        spec = spec.model_copy(
            update={
                "inputs": [
                    ParentRef(
                        kind="EvaluationRunManifest",
                        id=payload.eval_run_id,
                        role="evaluation_run",
                    )
                ]
            }
        )
    return spec


def _run_analysis_sync(
    spec: AnalysisRunSpec, registries: ApplicationRegistryBundle
) -> AnalysisJobResult:
    """Run the executable analysis spec synchronously inside the executor."""
    from feedbax.analysis.specs import execute_analysis_run_spec

    manifest, path = execute_analysis_run_spec(
        spec,
        registry=registries.analysis_recipes,
        evaluation_registry=registries.evaluation_recipes,
        experiment_registry=registries.experiment_packages,
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
            _job_artifact_location(artifact)
            for artifact in manifest.artifacts
            if artifact.uri is not None
        ],
    )


async def _run_analysis_background(
    request_id: str, spec: AnalysisRunSpec, registries: ApplicationRegistryBundle
) -> None:
    """Wrapper that updates the job tracker around the synchronous pipeline."""
    await job_tracker.update_status(request_id, JobStatus.RUNNING)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_analysis_sync,
            spec,
            registries,
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
            request_id,
            JobStatus.ERROR,
            error=str(tb),
        )


def _run_figure_sync(spec: FigureSpec, registries: ApplicationRegistryBundle) -> AnalysisJobResult:
    """Run a declarative figure spec synchronously inside the executor."""
    from feedbax.analysis.figures import FIGURE_RENDER_ROLE, execute_figure_spec

    manifest, path = execute_figure_spec(spec, registry=registries.figures)
    return AnalysisJobResult(
        manifest_id=manifest.id,
        manifest_path=str(path),
        figure_hashes=[manifest.id],
        artifact_ids=[
            artifact.artifact_id
            for artifact in manifest.artifacts
            if artifact.role == FIGURE_RENDER_ROLE and artifact.artifact_id is not None
        ],
        artifact_paths=[
            _job_artifact_location(artifact)
            for artifact in manifest.artifacts
            if artifact.role == FIGURE_RENDER_ROLE and artifact.uri is not None
        ],
    )


async def _run_figure_background(
    request_id: str, spec: FigureSpec, registries: ApplicationRegistryBundle
) -> None:
    """Wrapper that updates the job tracker around declarative figure execution."""
    await job_tracker.update_status(request_id, JobStatus.RUNNING)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, _run_figure_sync, spec, registries)
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
        logger.error("Figure job %s failed:\n%s", request_id, tb)
        await job_tracker.update_status(request_id, JobStatus.ERROR, error=str(tb))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=GenerateAnalysisResponse)
async def generate_figure(
    payload: GenerateAnalysisRequest, request: Request
) -> GenerateAnalysisResponse:
    """Trigger demand-driven figure generation for an analysis node.

    The computation runs in a background thread; this endpoint returns
    immediately with a ``request_id`` that can be polled via
    ``GET /status/{request_id}``.

    ``eval_run_id`` identifies the evaluation manifest consumed by the analysis
    spec. The in-memory tracker is UX-only; the returned manifest ID is the
    durable result identity.
    """
    if payload.job_kind == "figure":
        figure_spec = _figure_spec_for_request(payload)
        manifest_id = figure_manifest_id(figure_spec)
        logger.info(
            "Generate figure request for node_id=%s manifest_id=%s",
            payload.node_id,
            manifest_id,
        )
        request_id = await job_tracker.create_job(payload.node_id, manifest_id=manifest_id)
        asyncio.create_task(
            _run_figure_background(
                request_id, figure_spec, request.app.state.bootstrap_state.bundle
            )
        )
        return GenerateAnalysisResponse(
            data={
                "request_id": request_id,
                "status": JobStatus.PENDING.value,
                "manifest_id": manifest_id,
            }
        )

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
        _run_analysis_background(request_id, spec, request.app.state.bootstrap_state.bundle),
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
