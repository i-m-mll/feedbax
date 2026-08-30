"""Typed native training diagnostics emitted beside a training-run manifest."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from feedbax.contracts.execution_context import (
    NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_ID,
    NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_VERSION,
    LearningRateDiagnostic,
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
)
from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError, ArtifactBlobProvider
from feedbax.contracts.manifest import StrictModel, TrainingRunManifest
from feedbax.contracts.metric_values import NumericBooleanJsonValue
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import ProgressCoordinate


__all__ = [
    "METHOD_TRAINING_TRACE_ARTIFACT_ROLE",
    "NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_ID",
    "NATIVE_EXECUTION_PRODUCER_CONTEXT_SCHEMA_VERSION",
    "TRAINING_DIAGNOSTICS_SCHEMA_ID",
    "TRAINING_DIAGNOSTICS_SCHEMA_VERSION",
    "TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V1",
    "TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2",
    "TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3",
    "TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4",
    "CheckpointTransactionDiagnostic",
    "MethodTrainingTrace",
    "MethodTrainingTraceLoadError",
    "MethodTrainingTraceRecord",
    "NativeExecutionProducerContext",
    "NativeTrainingDiagnosticsInput",
    "TrainingDiagnostics",
    "load_method_training_trace",
]


TRAINING_DIAGNOSTICS_SCHEMA_ID = "feedbax.manifest.training_diagnostics"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V1 = "feedbax.manifest.training_diagnostics.v1"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2 = "feedbax.manifest.training_diagnostics.v2"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3 = "feedbax.manifest.training_diagnostics.v3"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4 = "feedbax.manifest.training_diagnostics.v4"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION = TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4
METHOD_TRAINING_TRACE_ARTIFACT_ROLE = "training_diagnostics"


class MethodTrainingTraceLoadError(ValueError):
    """Typed fail-closed diagnostic for one governed trace contract violation."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


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
    value: NumericBooleanJsonValue


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
        "feedbax.manifest.training_diagnostics.v4",
    ] = (
        TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2
    )
    manifest_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    terminal_status: Literal["completed", "cancelled", "failed"]
    failure_kind: Literal["nan_guard"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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
        if self.method_trace is not None and self.schema_version not in {
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
        }:
            raise ValueError("method_trace requires TrainingDiagnostics schema v3 or v4")
        if self.terminal_status == "failed":
            if self.schema_version != TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4:
                raise ValueError("failed terminal status requires TrainingDiagnostics schema v4")
            if self.failure_kind != "nan_guard":
                raise ValueError("failed terminal status requires failure_kind='nan_guard'")
        elif self.failure_kind is not None:
            raise ValueError("failure_kind is only valid for failed terminal diagnostics")
        if (
            self.schema_version != TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4
            and self.failure_kind is not None
        ):
            raise ValueError("failure_kind requires TrainingDiagnostics schema v4")
        return self


def load_method_training_trace(
    manifest: TrainingRunManifest,
    provider: ArtifactBlobProvider,
) -> MethodTrainingTrace:
    """Load one method trace through manifest-declared immutable provider custody."""

    artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == METHOD_TRAINING_TRACE_ARTIFACT_ROLE
    ]
    if len(artifacts) != 1:
        raise MethodTrainingTraceLoadError(
            "missing", "TrainingRunManifest must supply exactly one training_diagnostics artifact"
        )
    artifact = artifacts[0]
    metadata = artifact.metadata
    if artifact.sha256 is None or artifact.size_bytes is None:
        raise MethodTrainingTraceLoadError("missing", "method trace lacks digest or size authority")
    if (
        artifact.media_type != "application/json"
        or metadata.get("schema_id") != TRAINING_DIAGNOSTICS_SCHEMA_ID
        or metadata.get("schema_version")
        not in {
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
            TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
        }
    ):
        raise MethodTrainingTraceLoadError("schema", "method trace artifact schema is unsupported")
    try:
        data = provider.get_bytes(
            f"artifact://sha256/{artifact.sha256}", size_bytes=artifact.size_bytes
        )
    except (ArtifactBlobCustodyError, FileNotFoundError) as exc:
        kind = "truncated" if "size" in str(exc).lower() else "missing"
        raise MethodTrainingTraceLoadError(kind, f"method trace custody failed: {exc}") from exc
    try:
        diagnostics = TrainingDiagnostics.model_validate_json(data)
    except ValidationError as exc:
        kind = "truncated" if any(error["type"] == "json_invalid" for error in exc.errors()) else "schema"
        raise MethodTrainingTraceLoadError(kind, f"method trace payload is invalid: {exc}") from exc
    if diagnostics.schema_version not in {
        TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
        TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
    }:
        raise MethodTrainingTraceLoadError(
            "schema", "method trace requires diagnostics schema v3 or v4"
        )
    if diagnostics.manifest_id != manifest.id or diagnostics.run_id != manifest.job_id:
        raise MethodTrainingTraceLoadError("provenance", "method trace names a different manifest or run")
    try:
        if manifest.training_spec is None:
            raise ValueError("missing embedded TrainingRunSpec")
        run_spec = TrainingRunSpec.model_validate(manifest.training_spec.inline)
    except (ValueError, ValidationError) as exc:
        raise MethodTrainingTraceLoadError("provenance", f"method authority is invalid: {exc}") from exc
    declaration = run_spec.worker_execution.method_contract.training_diagnostics
    trace = diagnostics.method_trace
    if declaration is None or trace is None:
        raise MethodTrainingTraceLoadError("missing", "declared method trace is absent")
    contract = run_spec.worker_execution.method_contract
    expected = (
        contract.method_ref,
        declaration.trace_schema_id,
        declaration.trace_schema_version,
        declaration.measurement_basis,
        declaration.metric_payload_slot,
        declaration.replica_axis,
    )
    observed = (
        trace.method_ref,
        trace.trace_schema_id,
        trace.trace_schema_version,
        trace.measurement_basis,
        trace.metric_payload_slot,
        trace.replica_axis,
    )
    if observed[1:3] != expected[1:3]:
        raise MethodTrainingTraceLoadError("schema", "method trace schema disagrees")
    if observed[:1] + observed[3:] != expected[:1] + expected[3:]:
        raise MethodTrainingTraceLoadError("provenance", "method trace declaration disagrees")
    replica_count = next(axis.size for axis in contract.axes if axis.name == trace.replica_axis)
    origin = diagnostics.completed_batches - diagnostics.segment_completed_batches
    expected_coordinates = {
        (batch, replica)
        for batch in range(origin + 1, diagnostics.completed_batches + 1)
        for replica in range(replica_count)
    }
    coordinates = [(record.completed_batch, record.replica_index) for record in trace.records]
    if len(coordinates) != len(set(coordinates)):
        raise MethodTrainingTraceLoadError("duplicate", "method trace duplicates coordinates")
    outside = set(coordinates) - expected_coordinates
    if outside:
        raise MethodTrainingTraceLoadError("out_of_range", f"method trace has out-of-range coordinates: {sorted(outside)!r}")
    missing = expected_coordinates - set(coordinates)
    if missing:
        raise MethodTrainingTraceLoadError("gap", f"method trace has coordinate gaps: {sorted(missing)!r}")
    return trace
