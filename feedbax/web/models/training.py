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


class TrainingConfig(BaseModel):
    """Structured configuration for the real JAX training backend (Phase 6).

    Passed verbatim to the worker via the ``/start`` request body under the
    ``training_config`` key.  The worker converts this into a ``_TrainingCfg``
    dataclass via ``_extract_training_cfg``.  All fields have sensible defaults
    so callers can override only the parameters they care about.

    Attributes:
        n_batches: Number of training steps.
        batch_size: Trials per gradient update.
        learning_rate: AdamW learning rate.
        grad_clip: Global gradient clipping norm.
        hidden_dim: GRU / CDE hidden state dimension.
        network_type: Controller architecture — ``"gru"`` or ``"cde"``.
        n_reach_steps: Number of control steps per episode.
        effort_weight: Relative weight for the muscle-effort penalty.
        snapshot_interval: Emit a ``training_trajectory`` event every N batches.
    """

    n_batches: int = 2000
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    hidden_dim: int = 128
    network_type: str = "gru"
    n_reach_steps: int = 80
    effort_weight: float = 2.5
    snapshot_interval: int = 100


# Enable forward references
LossTermSpec.model_rebuild()
