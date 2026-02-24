"""Pydantic models for training specifications."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from feedbax.web.models.graph import ParamValue


class OptimizerSpec(BaseModel):
    """Specification for an optimizer."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)


class TimeAggregationSpec(BaseModel):
    """Specification for time aggregation in loss computation."""

    mode: Literal["all", "final", "range", "segment", "custom"] = "all"
    start: Optional[int] = None
    end: Optional[int] = None
    segment_name: Optional[str] = None
    time_idxs: Optional[List[int]] = None
    discount: Optional[Literal["none", "power", "linear"]] = None
    discount_exp: Optional[float] = None


class LossTermSpec(BaseModel):
    """Specification for a loss term."""

    type: str
    label: str
    weight: float = 1.0
    selector: Optional[str] = None
    norm: Optional[Literal["squared_l2", "l2", "l1", "huber"]] = None
    time_agg: Optional[TimeAggregationSpec] = None
    children: Optional[Dict[str, "LossTermSpec"]] = None


class EarlyStoppingSpec(BaseModel):
    """Specification for early stopping."""

    metric: str
    patience: int
    min_delta: float


class TrainingSpec(BaseModel):
    """Complete specification for a training run."""

    optimizer: OptimizerSpec
    loss: LossTermSpec
    n_batches: int
    batch_size: int
    n_epochs: Optional[int] = None
    checkpoint_interval: Optional[int] = None
    early_stopping: Optional[EarlyStoppingSpec] = None


class TaskSpec(BaseModel):
    """Specification for a task."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    timeline: Optional[Dict[str, ParamValue]] = None


# Enable forward references
LossTermSpec.model_rebuild()
