"""Typed, bounded attribution records for executor NaN guard failures."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.training import ScheduleProjection
from feedbax.contracts.worker import ProgressCoordinate


NAN_ATTRIBUTION_DETECTION_SCHEMA_ID = "feedbax.manifest.nan_attribution.detection"
NAN_ATTRIBUTION_DETECTION_SCHEMA_VERSION = f"{NAN_ATTRIBUTION_DETECTION_SCHEMA_ID}.v1"
NAN_ATTRIBUTION_RESTORATION_SCHEMA_ID = "feedbax.manifest.nan_attribution.restoration"
NAN_ATTRIBUTION_RESTORATION_SCHEMA_VERSION = f"{NAN_ATTRIBUTION_RESTORATION_SCHEMA_ID}.v1"
NAN_ATTRIBUTION_ARTIFACT_ROLE = "nan_attribution_detection"
NAN_RESTORATION_ARTIFACT_ROLE = "nan_attribution_restoration"

MAX_ATTRIBUTION_AXIS_SIZE = 1024
MAX_ATTRIBUTION_METRICS = 64
MAX_ATTRIBUTION_SLOTS = 128
MAX_SLOT_LEAF_EXEMPLARS = 16
MAX_SCHEDULES = 64
MAX_SCHEDULE_SAMPLES = 8


class AxisNonFiniteSummary(StrictModel):
    """NaN and Inf counts for either a shared value or one declared mapped axis."""

    mode: Literal["shared", "mapped"]
    axis_role: str | None = None
    array_axis: int | None = Field(default=None, ge=0)
    axis_size: int | None = Field(default=None, gt=0)
    axis_indices: tuple[int, ...] | None = Field(
        default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE
    )
    axis_truncated: bool = False
    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)
    nan_mask: tuple[bool, ...] | None = Field(default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE)
    inf_mask: tuple[bool, ...] | None = Field(default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE)
    nan_counts_by_axis: tuple[int, ...] | None = Field(
        default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE
    )
    inf_counts_by_axis: tuple[int, ...] | None = Field(
        default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE
    )

    @model_validator(mode="after")
    def _validate_axis_resolution(self) -> "AxisNonFiniteSummary":
        mapped_fields = (
            self.nan_mask,
            self.inf_mask,
            self.nan_counts_by_axis,
            self.inf_counts_by_axis,
            self.axis_indices,
        )
        if self.mode == "shared":
            if (
                self.axis_role is not None
                or self.array_axis is not None
                or self.axis_size is not None
            ):
                raise ValueError("shared non-finite summaries forbid mapped-axis identity")
            if any(value is not None for value in mapped_fields):
                raise ValueError("shared non-finite summaries forbid axis-resolved arrays")
            return self

        if not self.axis_role:
            raise ValueError("mapped non-finite summaries require axis_role")
        if self.array_axis is None or self.axis_size is None:
            raise ValueError("mapped non-finite summaries require array_axis and axis_size")
        if any(value is None for value in mapped_fields):
            raise ValueError("mapped non-finite summaries require masks and per-axis counts")
        assert self.axis_indices is not None
        assert self.nan_mask is not None
        assert self.inf_mask is not None
        assert self.nan_counts_by_axis is not None
        assert self.inf_counts_by_axis is not None
        if any(count < 0 for count in (*self.nan_counts_by_axis, *self.inf_counts_by_axis)):
            raise ValueError("axis-resolved non-finite counts must be non-negative")
        if tuple(sorted(set(self.axis_indices))) != self.axis_indices:
            raise ValueError("mapped axis_indices must be unique and sorted")
        if any(index >= self.axis_size for index in self.axis_indices):
            raise ValueError("mapped axis_indices must fall within axis_size")
        if any(len(value) != len(self.axis_indices) for value in mapped_fields if value is not None):
            raise ValueError("mapped masks and counts must exactly match axis_indices")
        if self.axis_truncated != (len(self.axis_indices) < self.axis_size):
            raise ValueError("axis_truncated must report omitted axis positions")
        if self.nan_mask != tuple(count > 0 for count in self.nan_counts_by_axis):
            raise ValueError("nan_mask must exactly identify positive per-axis NaN counts")
        if self.inf_mask != tuple(count > 0 for count in self.inf_counts_by_axis):
            raise ValueError("inf_mask must exactly identify positive per-axis Inf counts")
        if (not self.axis_truncated and self.nan_count != sum(self.nan_counts_by_axis)) or (
            self.axis_truncated and self.nan_count < sum(self.nan_counts_by_axis)
        ):
            raise ValueError("nan_count disagrees with reported per-axis NaN counts")
        if (not self.axis_truncated and self.inf_count != sum(self.inf_counts_by_axis)) or (
            self.axis_truncated and self.inf_count < sum(self.inf_counts_by_axis)
        ):
            raise ValueError("inf_count disagrees with reported per-axis Inf counts")
        return self


class AxisPredicateSummary(StrictModel):
    """False predicate-component counts without re-evaluating predicate logic."""

    mode: Literal["shared", "mapped"]
    axis_role: str | None = None
    array_axis: int | None = Field(default=None, ge=0)
    axis_size: int | None = Field(default=None, gt=0)
    axis_indices: tuple[int, ...] | None = Field(
        default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE
    )
    axis_truncated: bool = False
    false_count: int = Field(ge=1)
    false_mask: tuple[bool, ...] | None = Field(
        default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE
    )
    false_counts_by_axis: tuple[int, ...] | None = Field(
        default=None, max_length=MAX_ATTRIBUTION_AXIS_SIZE
    )

    @model_validator(mode="after")
    def _validate_predicate_axis(self) -> "AxisPredicateSummary":
        if self.mode == "shared":
            if any(
                value is not None
                for value in (
                    self.axis_role,
                    self.array_axis,
                    self.axis_size,
                    self.axis_indices,
                    self.false_mask,
                    self.false_counts_by_axis,
                )
            ):
                raise ValueError("shared predicate summaries forbid mapped-axis evidence")
            if self.axis_truncated:
                raise ValueError("shared predicate summaries cannot be axis-truncated")
            return self
        if (
            not self.axis_role
            or self.array_axis is None
            or self.axis_size is None
            or self.axis_indices is None
            or self.false_mask is None
            or self.false_counts_by_axis is None
        ):
            raise ValueError("mapped predicate summaries require complete axis identity")
        if tuple(sorted(set(self.axis_indices))) != self.axis_indices:
            raise ValueError("predicate axis_indices must be unique and sorted")
        if any(index >= self.axis_size for index in self.axis_indices):
            raise ValueError("predicate axis_indices must fall within axis_size")
        if len(self.false_mask) != len(self.axis_indices) or len(
            self.false_counts_by_axis
        ) != len(self.axis_indices):
            raise ValueError("predicate masks and counts must match axis_indices")
        if self.false_mask != tuple(count > 0 for count in self.false_counts_by_axis):
            raise ValueError("false_mask must identify positive false counts")
        if (
            not self.axis_truncated
            and self.false_count != sum(self.false_counts_by_axis)
        ) or (
            self.axis_truncated
            and self.false_count < sum(self.false_counts_by_axis)
        ):
            raise ValueError("false_count disagrees with reported per-axis false counts")
        if self.axis_truncated != (len(self.axis_indices) < self.axis_size):
            raise ValueError("axis_truncated must report omitted predicate positions")
        return self


class MetricNonFiniteSummary(StrictModel):
    """One metric's shared or declared-axis-resolved non-finiteness."""

    metric_name: str = Field(min_length=1, max_length=256)
    non_finite: AxisNonFiniteSummary


class SlotLeafNonFiniteExemplar(StrictModel):
    """One bounded offending slot-leaf example without raw tensor values."""

    leaf_path: str = Field(min_length=1, max_length=512)
    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)
    non_finite: AxisNonFiniteSummary

    @model_validator(mode="after")
    def _validate_counts(self) -> "SlotLeafNonFiniteExemplar":
        if self.nan_count != self.non_finite.nan_count:
            raise ValueError("leaf nan_count must match its axis summary")
        if self.inf_count != self.non_finite.inf_count:
            raise ValueError("leaf inf_count must match its axis summary")
        if self.nan_count == 0 and self.inf_count == 0:
            raise ValueError("slot-leaf exemplars must be non-finite")
        return self


class SlotLeafPredicateExemplar(StrictModel):
    """One bounded false boolean component already carried in a runtime slot."""

    leaf_path: str = Field(min_length=1, max_length=512)
    false_count: int = Field(ge=1)
    predicate: AxisPredicateSummary

    @model_validator(mode="after")
    def _validate_count(self) -> "SlotLeafPredicateExemplar":
        if self.false_count != self.predicate.false_count:
            raise ValueError("leaf false_count must match its predicate summary")
        return self


class SlotNonFiniteSummary(StrictModel):
    """Bounded aggregate and exemplars for one executor state or predicate slot."""

    slot_name: str = Field(min_length=1, max_length=256)
    total_leaf_count: int = Field(ge=0)
    offending_leaf_count: int = Field(ge=0)
    nonfinite_leaf_count: int = Field(ge=0)
    predicate_false_leaf_count: int = Field(ge=0)
    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)
    exemplars: tuple[SlotLeafNonFiniteExemplar, ...] = Field(
        default_factory=tuple,
        max_length=MAX_SLOT_LEAF_EXEMPLARS,
    )
    predicate_exemplars: tuple[SlotLeafPredicateExemplar, ...] = Field(
        default_factory=tuple,
        max_length=MAX_SLOT_LEAF_EXEMPLARS,
    )
    truncated: bool = False

    @model_validator(mode="after")
    def _validate_exemplar_bounds(self) -> "SlotNonFiniteSummary":
        if self.offending_leaf_count > self.total_leaf_count:
            raise ValueError("offending_leaf_count cannot exceed total_leaf_count")
        if self.offending_leaf_count > self.nonfinite_leaf_count + self.predicate_false_leaf_count:
            raise ValueError("offending_leaf_count exceeds the union inputs")
        if len(self.exemplars) > self.nonfinite_leaf_count:
            raise ValueError("non-finite exemplars exceed nonfinite_leaf_count")
        if len(self.predicate_exemplars) > self.predicate_false_leaf_count:
            raise ValueError("predicate exemplars exceed predicate_false_leaf_count")
        omitted = self.nonfinite_leaf_count > len(self.exemplars) or (
            self.predicate_false_leaf_count > len(self.predicate_exemplars)
        )
        if self.truncated != omitted:
            raise ValueError("truncated must exactly report omitted offending leaves")
        if sum(item.nan_count for item in self.exemplars) > self.nan_count:
            raise ValueError("exemplar NaN counts cannot exceed the slot aggregate")
        if sum(item.inf_count for item in self.exemplars) > self.inf_count:
            raise ValueError("exemplar Inf counts cannot exceed the slot aggregate")
        return self


class NanGuardCoordinate(StrictModel):
    """Training-batch count observed before authoritative post-guard extraction."""

    completed_batches_as_observed: int | None = Field(default=None, ge=0)
    completed_batches_observation: Literal["declared_authority", "unavailable"]
    completed_batches_observation_error: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
    )
    program_step: int = Field(ge=0)
    outer_step: int | None = Field(default=None, ge=0)
    inner_step: int | None = Field(default=None, ge=0)
    provenance: Literal["guard_pre_authoritative_extraction"] = "guard_pre_authoritative_extraction"

    @model_validator(mode="after")
    def _validate_observation(self) -> "NanGuardCoordinate":
        if self.completed_batches_observation == "declared_authority":
            if (
                self.completed_batches_as_observed is None
                or self.completed_batches_observation_error is not None
            ):
                raise ValueError("declared batch authority requires only an observed value")
        elif (
            self.completed_batches_as_observed is not None
            or self.completed_batches_observation_error is None
        ):
            raise ValueError("unavailable batch authority requires only an observation error")
        return self


class NanAttributionDetection(StrictModel):
    """Immutable guard-point evidence persisted before checkpoint restoration."""

    kind: Literal["NanAttributionDetection"] = "NanAttributionDetection"
    schema_id: Literal["feedbax.manifest.nan_attribution.detection"] = (
        NAN_ATTRIBUTION_DETECTION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.manifest.nan_attribution.detection.v1"] = (
        NAN_ATTRIBUTION_DETECTION_SCHEMA_VERSION
    )
    run_id: str = Field(min_length=1)
    on_nan: Literal["raise", "halt_restore_checkpoint"]
    trip_condition: Literal["nan_only"] = "nan_only"
    coordinate: NanGuardCoordinate
    metrics: tuple[MetricNonFiniteSummary, ...] = Field(
        min_length=1,
        max_length=MAX_ATTRIBUTION_METRICS,
    )
    total_offending_metrics: int = Field(gt=0)
    metrics_truncated: bool = False
    slots: tuple[SlotNonFiniteSummary, ...] = Field(
        default_factory=tuple,
        max_length=MAX_ATTRIBUTION_SLOTS,
    )
    total_offending_slots: int = Field(ge=0)
    slots_truncated: bool = False
    schedule_state: ScheduleProjection
    schedule_coordinate_source: Literal["observed_batch_authority", "segment_start_fallback"]
    total_schedules: int = Field(ge=0)
    schedule_state_truncated: bool = False

    @model_validator(mode="after")
    def _validate_detection(self) -> "NanAttributionDetection":
        if not any(metric.non_finite.nan_count > 0 for metric in self.metrics):
            raise ValueError("NaN attribution detection requires at least one NaN metric")
        if len(self.metrics) > self.total_offending_metrics:
            raise ValueError("metric summaries cannot exceed total_offending_metrics")
        if self.metrics_truncated != (self.total_offending_metrics > len(self.metrics)):
            raise ValueError("metrics_truncated must exactly report omitted offending metrics")
        if len(self.slots) > self.total_offending_slots:
            raise ValueError("slot summaries cannot exceed total_offending_slots")
        if self.slots_truncated != (self.total_offending_slots > len(self.slots)):
            raise ValueError("slots_truncated must exactly report omitted offending slots")
        if len(self.schedule_state.schedules) > self.total_schedules:
            raise ValueError("schedule_state cannot exceed total_schedules")
        if self.schedule_state_truncated != (
            self.total_schedules > len(self.schedule_state.schedules)
        ):
            raise ValueError("schedule_state_truncated must report omitted schedules")
        if any(
            len(schedule.samples) > MAX_SCHEDULE_SAMPLES
            for schedule in self.schedule_state.schedules.values()
        ):
            raise ValueError(f"schedule_state exceeds {MAX_SCHEDULE_SAMPLES} samples per schedule")
        return self


class RestoredCheckpointIdentity(StrictModel):
    """Exact checkpoint transaction selected by a successful restoration."""

    transaction_id: str = Field(min_length=1)
    completed_batches: int = Field(ge=0)
    coordinate: ProgressCoordinate


class NanAttributionRestorationOutcome(StrictModel):
    """Immutable outcome linked to the already-persisted guard detection."""

    kind: Literal["NanAttributionRestorationOutcome"] = "NanAttributionRestorationOutcome"
    schema_id: Literal["feedbax.manifest.nan_attribution.restoration"] = (
        NAN_ATTRIBUTION_RESTORATION_SCHEMA_ID
    )
    schema_version: Literal["feedbax.manifest.nan_attribution.restoration.v1"] = (
        NAN_ATTRIBUTION_RESTORATION_SCHEMA_VERSION
    )
    run_id: str = Field(min_length=1)
    detection_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["restored", "failed", "not_attempted"]
    restored_transaction: RestoredCheckpointIdentity | None = None
    restore_failure: str | None = Field(default=None, min_length=1, max_length=2048)
    not_attempted_reason: str | None = Field(default=None, min_length=1, max_length=2048)

    @model_validator(mode="after")
    def _validate_outcome(self) -> "NanAttributionRestorationOutcome":
        if self.status == "restored":
            if (
                self.restored_transaction is None
                or self.restore_failure is not None
                or self.not_attempted_reason is not None
            ):
                raise ValueError("restored outcome requires only restored_transaction")
        elif self.status == "failed" and (
            self.restored_transaction is not None
            or self.restore_failure is None
            or self.not_attempted_reason is not None
        ):
            raise ValueError("failed outcome requires only restore_failure")
        elif self.status == "not_attempted" and (
            self.restored_transaction is not None
            or self.restore_failure is not None
            or self.not_attempted_reason is None
        ):
            raise ValueError("not_attempted outcome requires only not_attempted_reason")
        return self
