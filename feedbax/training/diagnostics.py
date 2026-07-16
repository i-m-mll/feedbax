"""Typed native training diagnostics emitted beside a training-run manifest."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.worker import AxisCoordinateSpec, ProgressCoordinate
from feedbax.orchestration.bundle import ExecutionIdentityEnvelope


TRAINING_DIAGNOSTICS_SCHEMA_ID = "feedbax.manifest.training_diagnostics"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V1 = "feedbax.manifest.training_diagnostics.v1"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2 = "feedbax.manifest.training_diagnostics.v2"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3 = "feedbax.manifest.training_diagnostics.v3"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION = TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2
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


class CheckpointTransactionDiagnostic(StrictModel):
    """Checkpoint transaction coordinate relevant to cadence verification."""

    transaction_id: str = Field(min_length=1)
    completed_batches: int = Field(ge=0)
    cumulative_completed_batches: int = Field(ge=0)
    coordinate: ProgressCoordinate


class MethodTrainingTraceRecord(StrictModel):
    """One method-authored observation at an explicit batch and replica coordinate."""

    completed_batch: int = Field(ge=0)
    replica_index: int = Field(ge=0)
    value: Any


class MethodTrainingTrace(StrictModel):
    """Durable method trace with authored scientific and coordinate provenance."""

    method_ref: str = Field(min_length=1)
    trace_schema_id: str = Field(min_length=1)
    trace_schema_version: str = Field(min_length=1)
    measurement_basis: str = Field(min_length=1)
    metric_payload_slot: str = Field(min_length=1)
    replica_axis: str = Field(min_length=1)
    records: list[MethodTrainingTraceRecord]


class TrainingDiagnostics(StrictModel):
    """Durable diagnostics emitted by the native training executor."""

    kind: Literal["TrainingDiagnostics"] = "TrainingDiagnostics"
    schema_id: Literal["feedbax.manifest.training_diagnostics"] = (
        TRAINING_DIAGNOSTICS_SCHEMA_ID
    )
    schema_version: Literal[
        "feedbax.manifest.training_diagnostics.v2",
        "feedbax.manifest.training_diagnostics.v3",
    ] = (
        TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2
    )
    manifest_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    terminal_status: Literal["completed", "cancelled"]
    completed_batches: int = Field(ge=0)
    segment_completed_batches: int = Field(ge=0)
    cumulative_completed_batches: int = Field(ge=0)
    seeds: list[int] = Field(default_factory=list)
    lr_trace: list[LearningRateDiagnostic] = Field(default_factory=list)
    checkpoint_coordinates: list[int] = Field(default_factory=list)
    checkpoint_transactions: list[CheckpointTransactionDiagnostic] = Field(
        default_factory=list
    )
    resume_context: ScheduleContextDiagnostic | None = None
    optimizer_build_context: ScheduleContextDiagnostic | None = None
    method_trace: MethodTrainingTrace | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_checkpoint_coordinates(self) -> "TrainingDiagnostics":
        transaction_coordinates = [
            transaction.completed_batches for transaction in self.checkpoint_transactions
        ]
        if self.checkpoint_coordinates != transaction_coordinates:
            raise ValueError(
                "checkpoint_coordinates must exactly match checkpoint transaction batch "
                "coordinates in emission order"
            )
        if self.cumulative_completed_batches != self.completed_batches:
            raise ValueError(
                "completed_batches is the cumulative native batch count and must match "
                "cumulative_completed_batches"
            )
        if self.segment_completed_batches > self.completed_batches:
            raise ValueError(
                "segment_completed_batches cannot exceed the cumulative batch count"
            )
        if (
            self.method_trace is not None
            and self.schema_version != TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3
        ):
            raise ValueError("method_trace requires TrainingDiagnostics schema v3")
        return self
