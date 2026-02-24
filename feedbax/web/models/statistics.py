"""Pydantic models for trajectory statistics endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class MetricSummary(BaseModel):
    """Descriptive statistics for a single scalar metric within a group."""

    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    count: int


class GroupStatistics(BaseModel):
    """Aggregated metric summaries for a single group of trajectories."""

    group_key: str          # e.g., "task_type=0", "body_idx=42", "all"
    group_label: str        # e.g., "Reach", "Body 42", "All"
    metrics: dict[str, MetricSummary]


class StatisticsResponse(BaseModel):
    """Response for the summary statistics endpoint."""

    dataset: str
    group_by: str
    groups: list[GroupStatistics]


class TimeseriesPercentiles(BaseModel):
    """Percentile bands for a time-varying metric within a single group."""

    group_key: str
    group_label: str
    timesteps: list[int]
    p50: list[float]
    p25: list[float]
    p75: list[float]
    p05: list[float]
    p95: list[float]


class TimeseriesResponse(BaseModel):
    """Response for the timeseries percentiles endpoint."""

    dataset: str
    metric: str
    group_by: str
    series: list[TimeseriesPercentiles]


class HistogramBin(BaseModel):
    """A single bin in a histogram."""

    lo: float
    hi: float
    count: int


class HistogramGroup(BaseModel):
    """Histogram of a scalar metric for a single group."""

    group_key: str
    group_label: str
    bins: list[HistogramBin]


class HistogramResponse(BaseModel):
    """Response for the histogram endpoint."""

    dataset: str
    metric: str
    group_by: str
    groups: list[HistogramGroup]


class ScatterPoint(BaseModel):
    """A single point in a scatter plot of two scalar metrics."""

    x: float
    y: float
    body_idx: int
    task_type: int


class ScatterResponse(BaseModel):
    """Response for the scatter plot endpoint."""

    dataset: str
    x_metric: str
    y_metric: str
    points: list[ScatterPoint]


class DiagnosticCheck(BaseModel):
    """Result of a single diagnostic check on a dataset."""

    name: str
    status: str           # "pass", "warn", "fail"
    reason: str
    evidence: dict
    hint: str | None = None


class DiagnosticsResponse(BaseModel):
    """Response for the diagnostics endpoint."""

    dataset: str
    checks: list[DiagnosticCheck]
