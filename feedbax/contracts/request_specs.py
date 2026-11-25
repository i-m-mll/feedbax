"""Typed request parameters. Inputs and attempt policy live outside these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from feedbax.contracts.figure_document import FigureDocument
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_refs import Name
from feedbax.contracts.training import TrialDocuments as TrialDocuments
from feedbax.plugins.declarations import validate_declaration_key


class NumericalSettings(BaseModel):
    """Runtime settings that change the computation."""

    model_config = ConfigDict(extra="forbid")
    floating_point_bits: Literal[32, 64] | None = None
    matmul_precision: (
        Literal["default", "high", "highest", "bfloat16", "tensorfloat32", "float32"] | None
    ) = None
    deterministic: bool | None = Field(default=None, strict=True)


class RequestSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numerical_settings: NumericalSettings = Field(default_factory=NumericalSettings)


class DocumentTrainingConfig(BaseModel):
    """Run controls authored alongside a Model, Task and Objective."""

    model_config = ConfigDict(extra="forbid")

    n_batches: int = Field(default=2000, gt=0, lt=2**31, title="Batches")
    batch_size: int = Field(default=128, gt=0)
    snapshot_interval: int = Field(default=100, gt=0, title="Checkpoint Interval")
    replicates: int = Field(default=1, gt=0)


class TrainContinuation(BaseModel):
    """Continue the completed-update clock while selecting which runtime state to carry."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["continue_from"] = "continue_from"
    source_completed_updates: int = Field(gt=0, lt=2**31)
    state: Literal["preserve", "reset"]
    rng: Literal["carry", "reset"]
    schedule: Literal["continue", "restart"]


class DocumentTrainingRequestSpec(RequestSpec):
    """Managed Train parameters with method settings separate from run controls."""

    trial_documents: TrialDocuments
    method: str
    settings: dict[str, JsonValue]
    training_config: DocumentTrainingConfig
    seed: int | None = Field(default=None, strict=True, ge=0, lt=2**32)
    checkpoint_initialization: Literal["initialize_from"] | TrainContinuation | None = None

    @field_validator("method")
    @classmethod
    def _method_key(cls, key: str) -> str:
        return validate_declaration_key(key)

    @property
    def continuation(self) -> TrainContinuation | None:
        policy = self.checkpoint_initialization
        return policy if isinstance(policy, TrainContinuation) else None

    @property
    def target_total(self) -> int:
        start = self.continuation.source_completed_updates if self.continuation else 0
        return start + self.training_config.n_batches

    @model_validator(mode="after")
    def _supported_run_controls(self) -> DocumentTrainingRequestSpec:
        if self.target_total >= 2**31:
            raise ValueError("The completed-update clock exceeds its supported range.")
        if self.numerical_settings.model_dump(exclude_none=True):
            raise ValueError("Train does not support overriding numerical settings.")
        return self


class EvaluationRequestSpec(RequestSpec):
    evaluation_type: Name
    params: dict[str, JsonValue] = Field(default_factory=dict)


class AnalysisRequestSpec(RequestSpec):
    analysis_type: Name
    params: dict[str, JsonValue] = Field(default_factory=dict)


class RenderRequestSpec(RequestSpec):
    document: FigureDocument
    names: dict[Name, dict[Name, str]] = Field(default_factory=dict)
    """The printed name of each value ID of the step-valued labels the figure splits or selects by."""


class ReportRequestSpec(RequestSpec):
    document: ReportDocument
