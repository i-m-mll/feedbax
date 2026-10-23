"""Governed native-execution context contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.worker import AxisCoordinateSpec
from feedbax.orchestration.bundle import ExecutionIdentityEnvelope


NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_ID = "feedbax.spec.native_execution_context"
NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_VERSION = (
    "feedbax.spec.native_execution_context.v1"
)


class LearningRateDiagnostic(StrictModel):
    """One realized learning-rate sample at a training-batch coordinate."""

    step: int = Field(ge=0)
    learning_rate: float
    axis_coordinates: tuple[AxisCoordinateSpec, ...] | None = None


class ScheduleContextDiagnostic(StrictModel):
    """Concrete schedule clock context used by the native executor."""

    schedule_origin_step: int = Field(ge=0)
    current_step: int = Field(ge=0)
    optimizer_count_at_current_step: int = Field(ge=0)


class NativeTrainingDiagnosticsInput(StrictModel):
    """Runtime observations supplied to native diagnostics production.

    Learning-rate samples are observations from the live trainer. They are not
    reconstructed from the declared schedule because that would turn
    conformance into a self-comparison.
    """

    seeds: list[int] = Field(default_factory=list)
    lr_trace: list[LearningRateDiagnostic] = Field(default_factory=list)
    resume_context: ScheduleContextDiagnostic | None = None
    optimizer_build_context: ScheduleContextDiagnostic | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NativeExecutionProducerContext(StrictModel):
    """Assembly identity and row provenance consumed before native execution.

    ``execution.row_provenance.planned_run_id`` is the exact manifest identity. It is used
    verbatim so planned identities that already carry the Feedbax prefix are
    never double-prefixed by the executor.
    """

    schema_id: Literal["feedbax.spec.native_execution_context"] = (
        NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.native_execution_context.v1"] = (
        NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_VERSION
    )
    execution: ExecutionIdentityEnvelope
    environment_fingerprint: str = Field(min_length=1)
    collection_root: str | None = None
    diagnostics: NativeTrainingDiagnosticsInput = Field(
        default_factory=NativeTrainingDiagnosticsInput
    )
