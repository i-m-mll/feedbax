"""Typed request parameters. Inputs and attempt policy live outside these models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from feedbax.contracts.figure_document import FigureDocument, ResolvedFigureTemplate
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.operation import OperationComputation
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity_guard import IdentityWideInteger
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

    n_batches: int = Field(default=2000, gt=0, lt=2**31)
    batch_size: int = Field(default=128, gt=0)
    snapshot_interval: int = Field(default=100, gt=0)
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
    method_revision: int = Field(default=1, gt=0, strict=True)
    settings: dict[str, JsonValue]
    training_config: DocumentTrainingConfig
    seed: Annotated[int | None, IdentityWideInteger()] = Field(default=None, strict=True, ge=0, lt=2**32)
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
    computation: OperationComputation | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)
    evaluation_states_policy: Literal["recompute", "require_durable"] = "recompute"


class RenderRequestSpec(RequestSpec):
    figure: FigureSpec | None
    document: FigureDocument | None = None
    template: ResolvedFigureTemplate | None = None

    @model_validator(mode="after")
    def _require_lowered_inputs(self) -> RenderRequestSpec:
        if (self.figure is None) == (self.document is None):
            raise ValueError("Declare one render figure or dataset figure document.")
        if self.template is not None and self.document is None:
            raise ValueError("A resolved template requires a dataset figure document.")
        if self.figure is not None and (self.figure.inputs or self.figure.input_authorities):
            raise ValueError("Lower figure input authorities to kernel inputs before encoding")
        return self


class ReportRequestSpec(RequestSpec):
    report_type: Name | None
    params: dict[str, JsonValue] = Field(default_factory=dict)
    narrative: str | None = None
    document: ReportDocument | None = None

    @model_validator(mode="after")
    def _require_one_report(self) -> ReportRequestSpec:
        if (self.report_type is None) == (self.document is None):
            raise ValueError("Declare one report recipe type or Report document.")
        if self.document is not None and (self.params or self.narrative is not None):
            raise ValueError("Report documents cannot carry recipe parameters or narrative.")
        return self
