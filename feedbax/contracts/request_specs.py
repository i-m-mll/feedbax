"""Typed request parameters. Inputs and attempt policy live outside these models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.request_identity_guard import IdentityWideInteger
from feedbax.contracts.request_refs import Name
from feedbax.contracts.training import (
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    RiskAggregationSpec,
    TaskSpec,
    TrainingConfig,
)


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


class TrainingRequestSpec(RequestSpec):
    graph: GraphSpec
    task: TaskSpec
    objective: ObjectiveSlotSpec
    method_payload: MethodPayloadEnvelope
    method_ref: MethodRefSpec
    training_config: TrainingConfig
    task_binding_spec: StudioTaskBindingSpec | None = None
    risk_aggregation: RiskAggregationSpec = Field(default_factory=RiskAggregationSpec)
    seed: Annotated[int | None, IdentityWideInteger()] = Field(default=None, strict=True)
    on_nan: Literal["raise", "halt_restore_checkpoint"] = "raise"


class EvaluationRequestSpec(RequestSpec):
    evaluation_type: Name
    params: dict[str, JsonValue] = Field(default_factory=dict)


class AnalysisRequestSpec(RequestSpec):
    analysis_type: Name
    params: dict[str, JsonValue] = Field(default_factory=dict)
    evaluation_states_policy: Literal["recompute", "require_durable"] = "recompute"


class RenderRequestSpec(RequestSpec):
    figure: FigureSpec

    @model_validator(mode="after")
    def _require_lowered_inputs(self) -> RenderRequestSpec:
        if self.figure.inputs or self.figure.input_authorities:
            raise ValueError("Lower figure input authorities to kernel inputs before encoding")
        return self


class ReportRequestSpec(RequestSpec):
    report_type: Name
    params: dict[str, JsonValue] = Field(default_factory=dict)
    narrative: str | None = None
