"""API router for trajectory statistics (summary, timeseries, histogram, scatter, diagnostics)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from feedbax.web.models.statistics import (
    DiagnosticsResponse,
    HistogramResponse,
    ScatterResponse,
    StatisticsResponse,
    TimeseriesResponse,
)
from feedbax.web.services.statistics_service import (
    SCALAR_METRICS,
    TIMESERIES_METRICS,
    StatisticsService,
)
from feedbax.web.services.trajectory_service import TrajectoryService

router = APIRouter()

# Shared service instances -- trajectory service is reused from the trajectories router.
_trajectory_service = TrajectoryService()
_statistics_service = StatisticsService(_trajectory_service)

_VALID_GROUP_BY = {"none", "task_type", "body_idx", "body_x_task"}


@router.get('/{dataset}/stats/summary')
async def get_summary(
    dataset: str,
    group_by: str = Query(default="none"),
) -> StatisticsResponse:
    """Return grouped summary statistics for all scalar metrics."""
    _validate_group_by(group_by)
    try:
        return _statistics_service.summary(dataset, group_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/{dataset}/stats/timeseries')
async def get_timeseries(
    dataset: str,
    metric: str = Query(default="distance_to_target"),
    group_by: str = Query(default="none"),
) -> TimeseriesResponse:
    """Return percentile bands for a time-varying metric."""
    _validate_group_by(group_by)
    if metric not in TIMESERIES_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown timeseries metric '{metric}'. "
                   f"Available: {sorted(TIMESERIES_METRICS)}",
        )
    try:
        return _statistics_service.timeseries(dataset, metric, group_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/{dataset}/stats/histogram')
async def get_histogram(
    dataset: str,
    metric: str = Query(default="final_distance"),
    group_by: str = Query(default="none"),
    bins: int = Query(default=30, ge=2, le=500),
) -> HistogramResponse:
    """Return histogram bins for a scalar metric."""
    _validate_group_by(group_by)
    if metric not in SCALAR_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scalar metric '{metric}'. "
                   f"Available: {sorted(SCALAR_METRICS)}",
        )
    try:
        return _statistics_service.histogram(dataset, metric, group_by, bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/{dataset}/stats/scatter')
async def get_scatter(
    dataset: str,
    x_metric: str = Query(default="effort"),
    y_metric: str = Query(default="final_distance"),
) -> ScatterResponse:
    """Return per-trajectory scatter data for two scalar metrics."""
    for m in (x_metric, y_metric):
        if m not in SCALAR_METRICS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown scalar metric '{m}'. "
                       f"Available: {sorted(SCALAR_METRICS)}",
            )
    try:
        return _statistics_service.scatter(dataset, x_metric, y_metric)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/{dataset}/stats/diagnostics')
async def get_diagnostics(dataset: str) -> DiagnosticsResponse:
    """Run diagnostic checks on a dataset."""
    try:
        return _statistics_service.diagnostics(dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_group_by(group_by: str) -> None:
    if group_by not in _VALID_GROUP_BY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by '{group_by}'. "
                   f"Available: {sorted(_VALID_GROUP_BY)}",
        )
