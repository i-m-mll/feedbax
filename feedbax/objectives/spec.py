"""Durable declarative objective specifications.

These models describe objective/loss terms for saved artifacts and provider
contracts. They intentionally do not execute losses; lowering into
``feedbax.objectives.loss`` or a future streaming objective backend is a separate step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


OBJECTIVE_SCHEMA_VERSION = "feedbax.spec.objective.v1"

MetricKind = Literal["squared_l2", "l2", "l1", "squared", "absolute", "huber"]
ReductionKind = Literal["mean", "sum", "none"]
TimeReductionKind = Literal["mean", "sum", "none", "final"]
MatrixPayloadKind = Literal["dense", "diagonal"]
SelectorKind = Literal[
    "state",
    "port",
    "wire",
    "graph_output",
    "recurrent_carry",
    "task_data",
    "objective",
    "probe",
]


class ObjectiveSpecModel(BaseModel):
    """Base model for durable objective specs."""

    model_config = ConfigDict(extra="forbid")


class SelectorAddressSpec(ObjectiveSpecModel):
    """Address of a selectable value in a graph, rollout, task, or retained set."""

    selector: str = Field(min_length=1)
    kind: SelectorKind = "state"
    value_dtype: Optional[str] = None
    value_shape: Optional[list[Any]] = None
    value_units: Optional[str] = None
    temporal_axis: Optional[Literal["time", "step", "sample"]] = "time"
    feature_axis: Optional[Literal["feature", "coordinate", "unit", "channel"]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetValueSpec(ObjectiveSpecModel):
    """Static or selector-addressed target for a target-state objective term."""

    kind: Literal["constant", "selector", "task_target"] = "task_target"
    value: Any = None
    selector: Optional[SelectorAddressSpec] = None
    target_key: Optional[str] = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "TargetValueSpec":
        if self.kind == "selector" and self.selector is None:
            raise ValueError("selector target requires selector")
        if self.kind == "task_target" and not self.target_key:
            raise ValueError("task_target target requires target_key")
        return self


class EpochMaskSpec(ObjectiveSpecModel):
    """Mask selecting named timeline epochs before objective reduction."""

    type: Literal["epoch_mask"] = "epoch_mask"
    epochs: list[str] = Field(min_length=1)
    mode: Literal["include", "exclude"] = "include"
    alignment: Literal["sample", "right_edge"] = "sample"

    @model_validator(mode="after")
    def _validate_epochs(self) -> "EpochMaskSpec":
        if len(set(self.epochs)) != len(self.epochs):
            raise ValueError("epoch mask epochs must be unique")
        return self


class ConstantScheduleSpec(ObjectiveSpecModel):
    """A schedule with a fixed scalar multiplier."""

    type: Literal["constant"] = "constant"
    value: float = 1.0


class PowerLawScheduleSpec(ObjectiveSpecModel):
    """Power-law schedule over a named timeline window."""

    type: Literal["power_law"] = "power_law"
    exponent: float = Field(gt=0)
    window: Literal["full_trial", "post_go", "epoch"] = "full_trial"
    epoch: Optional[str] = None
    start_value: float = 0.0
    end_value: float = 1.0
    normalize: bool = True

    @model_validator(mode="after")
    def _validate_window(self) -> "PowerLawScheduleSpec":
        if self.window == "epoch" and not self.epoch:
            raise ValueError("epoch power-law schedule requires epoch")
        if self.window != "epoch" and self.epoch is not None:
            raise ValueError("epoch may only be set when window='epoch'")
        return self


class MovementEpochRampScheduleSpec(ObjectiveSpecModel):
    """Ramp schedule anchored to the movement epoch of a task timeline."""

    type: Literal["movement_epoch_ramp"] = "movement_epoch_ramp"
    anchor_epoch: str = "movement"
    duration_steps: int = Field(gt=0)
    shape: Literal["linear", "cosine", "power"] = "linear"
    power: Optional[float] = None
    start_value: float = 0.0
    end_value: float = 1.0
    hold_after: bool = True

    @model_validator(mode="after")
    def _validate_shape(self) -> "MovementEpochRampScheduleSpec":
        if self.shape == "power":
            if self.power is None:
                raise ValueError("power ramp schedule requires power")
            if self.power <= 0:
                raise ValueError("power ramp schedule requires power > 0")
        elif self.power is not None:
            raise ValueError("power may only be set when shape='power'")
        return self


ScheduleSpec = Annotated[
    ConstantScheduleSpec | PowerLawScheduleSpec | MovementEpochRampScheduleSpec,
    Field(discriminator="type"),
]


class MetricSpec(ObjectiveSpecModel):
    """Metric applied before time/trial/objective reduction."""

    kind: MetricKind = "squared_l2"
    axis: Literal["feature", "coordinate", "unit", "channel", "all"] = "feature"
    huber_delta: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_metric(self) -> "MetricSpec":
        if self.kind == "huber" and self.huber_delta is None:
            raise ValueError("huber metric requires huber_delta")
        if self.kind != "huber" and self.huber_delta is not None:
            raise ValueError("huber_delta may only be set for huber metric")
        return self


class MatrixPayloadSpec(ObjectiveSpecModel):
    """Matrix payload for selector-addressed quadratic objective terms."""

    kind: MatrixPayloadKind = "dense"
    value: Any
    dtype: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_matrix_payload(self) -> "MatrixPayloadSpec":
        shape = _json_array_shape(self.value)
        if self.kind == "dense":
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("dense matrix payload must be a square rank-2 array")
        elif self.kind == "diagonal":
            if len(shape) != 1:
                raise ValueError("diagonal matrix payload must be a rank-1 array")
        return self


class ReductionSpec(ObjectiveSpecModel):
    """Reduction applied after metric, mask, and schedule weighting."""

    time: TimeReductionKind = "mean"
    trial: ReductionKind = "mean"
    feature: ReductionKind = "sum"
    empty_mask: Literal["zero", "error"] = "zero"


class ObjectiveTermBase(ObjectiveSpecModel):
    """Shared durable fields for objective terms."""

    label: str = Field(min_length=1)
    weight: float = 1.0
    selector: SelectorAddressSpec
    metric: MetricSpec = Field(default_factory=MetricSpec)
    reduction: ReductionSpec = Field(default_factory=ReductionSpec)
    mask: Optional[EpochMaskSpec] = None
    schedule: ScheduleSpec = Field(default_factory=ConstantScheduleSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schedule_mask_pair(self) -> "ObjectiveTermBase":
        if (
            isinstance(self.schedule, MovementEpochRampScheduleSpec)
            and self.mask is not None
            and self.schedule.anchor_epoch not in self.mask.epochs
        ):
            raise ValueError(
                "movement_epoch_ramp anchor_epoch must be included in the term epoch mask"
            )
        return self


class TargetStateLossSpec(ObjectiveTermBase):
    """Penalize a selected value against a static or task-provided target."""

    type: Literal["target_state"] = "target_state"
    target: TargetValueSpec = Field(default_factory=TargetValueSpec)


class FiniteDifferenceLossSpec(ObjectiveTermBase):
    """Penalize an Nth finite difference of a selected temporal value."""

    type: Literal["finite_difference"] = "finite_difference"
    order: int = Field(ge=1)
    time_axis: Literal["time", "step", "sample"] = "time"
    alignment: Literal["right_edge", "left_edge", "center"] = "right_edge"
    quantity: Optional[
        Literal["hidden_derivative", "output_derivative", "output_jerk", "velocity_jerk"]
    ] = None

    @model_validator(mode="after")
    def _validate_temporal_selector(self) -> "FiniteDifferenceLossSpec":
        if self.selector.temporal_axis is None:
            raise ValueError("finite_difference selector must declare a temporal_axis")
        if self.selector.kind in {"objective", "probe"}:
            raise ValueError("finite_difference selector must address a temporal value")
        if self.mask is not None and self.mask.alignment == "sample":
            raise ValueError(
                "finite_difference epoch masks must use right_edge alignment "
                "until lowering defines another convention"
            )
        return self


class MatrixQuadraticLossSpec(ObjectiveTermBase):
    """Penalize a selected vector with ``(value - target)^T M (value - target)``."""

    type: Literal["matrix_quadratic"] = "matrix_quadratic"
    matrix: MatrixPayloadSpec
    target: Optional[TargetValueSpec] = None

    @model_validator(mode="after")
    def _validate_matrix_quadratic(self) -> "MatrixQuadraticLossSpec":
        if self.selector.feature_axis is None:
            raise ValueError("matrix_quadratic selector must declare a feature_axis")
        if self.metric.kind != "squared_l2":
            raise ValueError("matrix_quadratic terms use matrix payloads, not metric kinds")
        return self


ObjectiveTermSpec = Annotated[
    TargetStateLossSpec | FiniteDifferenceLossSpec | MatrixQuadraticLossSpec,
    Field(discriminator="type"),
]


class TimelineEpochSpec(ObjectiveSpecModel):
    """Named epoch in a task timeline."""

    name: str = Field(min_length=1)
    index: int = Field(ge=0)
    length_range: Optional[tuple[int, int]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_length_range(self) -> "TimelineEpochSpec":
        if self.length_range is not None:
            lo, hi = self.length_range
            if lo < 0 or hi < lo:
                raise ValueError("length_range must satisfy 0 <= min <= max")
        return self


class TimelineEventSpec(ObjectiveSpecModel):
    """Named event in a task timeline."""

    name: str = Field(min_length=1)
    epoch: Optional[str] = None
    offset_steps: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskTimelineSpec(ObjectiveSpecModel):
    """Durable timeline provenance needed to interpret objective schedules."""

    type: Literal["task_timeline"] = "task_timeline"
    n_steps: Optional[int] = Field(default=None, gt=0)
    epochs: list[TimelineEpochSpec] = Field(default_factory=list)
    events: list[TimelineEventSpec] = Field(default_factory=list)
    movement_epoch: Optional[str] = None
    go_event: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_names(self) -> "TaskTimelineSpec":
        epoch_names = [epoch.name for epoch in self.epochs]
        if len(epoch_names) != len(set(epoch_names)):
            raise ValueError("timeline epoch names must be unique")
        event_names = [event.name for event in self.events]
        if len(event_names) != len(set(event_names)):
            raise ValueError("timeline event names must be unique")
        epoch_name_set = set(epoch_names)
        if self.movement_epoch is not None and self.movement_epoch not in epoch_name_set:
            raise ValueError("movement_epoch must name a declared epoch")
        if self.go_event is not None and self.go_event not in set(event_names):
            raise ValueError("go_event must name a declared event")
        for event in self.events:
            if event.epoch is not None and event.epoch not in epoch_name_set:
                raise ValueError(f"event {event.name!r} references unknown epoch")
        return self


class ObjectiveSpec(ObjectiveSpecModel):
    """A durable objective made of selector-addressed terms."""

    kind: Literal["objective_spec"] = "objective_spec"
    schema_version: str = OBJECTIVE_SCHEMA_VERSION
    terms: list[ObjectiveTermSpec] = Field(default_factory=list)
    timeline: Optional[TaskTimelineSpec] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_term_references(self) -> "ObjectiveSpec":
        labels = [term.label for term in self.terms]
        if len(labels) != len(set(labels)):
            raise ValueError("objective term labels must be unique")

        epoch_names = {epoch.name for epoch in self.timeline.epochs} if self.timeline else set()
        for term in self.terms:
            if term.mask is not None and epoch_names:
                unknown = sorted(set(term.mask.epochs) - epoch_names)
                if unknown:
                    raise ValueError(
                        f"term {term.label!r} references unknown mask epochs {unknown!r}"
                    )
            if isinstance(term.schedule, PowerLawScheduleSpec) and term.schedule.epoch:
                if epoch_names and term.schedule.epoch not in epoch_names:
                    raise ValueError(
                        f"term {term.label!r} references unknown schedule epoch "
                        f"{term.schedule.epoch!r}"
                    )
            if isinstance(term.schedule, MovementEpochRampScheduleSpec):
                if self.timeline is None:
                    raise ValueError(
                        f"term {term.label!r} uses movement_epoch_ramp without timeline"
                    )
                if term.schedule.anchor_epoch not in epoch_names:
                    raise ValueError(
                        f"term {term.label!r} references unknown movement epoch "
                        f"{term.schedule.anchor_epoch!r}"
                    )
        return self


def objective_schema_models() -> dict[str, type[BaseModel]]:
    """Return Pydantic models that should be exposed by provider schemas."""

    return {
        "ObjectiveSpec": ObjectiveSpec,
        "TargetStateLossSpec": TargetStateLossSpec,
        "FiniteDifferenceLossSpec": FiniteDifferenceLossSpec,
        "MatrixQuadraticLossSpec": MatrixQuadraticLossSpec,
        "SelectorAddressSpec": SelectorAddressSpec,
        "TargetValueSpec": TargetValueSpec,
        "MatrixPayloadSpec": MatrixPayloadSpec,
        "EpochMaskSpec": EpochMaskSpec,
        "ConstantScheduleSpec": ConstantScheduleSpec,
        "PowerLawScheduleSpec": PowerLawScheduleSpec,
        "MovementEpochRampScheduleSpec": MovementEpochRampScheduleSpec,
        "MetricSpec": MetricSpec,
        "ReductionSpec": ReductionSpec,
        "TaskTimelineSpec": TaskTimelineSpec,
        "TimelineEpochSpec": TimelineEpochSpec,
        "TimelineEventSpec": TimelineEventSpec,
    }


def canonical_objective_payload(spec: ObjectiveSpec | dict[str, Any]) -> dict[str, Any]:
    """Validate and return a deterministic JSON-compatible payload."""

    objective = validate_objective_spec(spec)
    return objective.model_dump(mode="json", exclude_none=True)


def migrate_objective_spec(
    spec: ObjectiveSpec | dict[str, Any],
    *,
    source_version: str | None = None,
    target_version: str | None = None,
) -> Any:
    """Migrate or explicitly reject a durable objective payload by schema version."""
    from feedbax.contracts.migrations import migrate_structured_spec_payload

    payload = spec.model_dump(mode="json") if isinstance(spec, ObjectiveSpec) else spec
    return migrate_structured_spec_payload(
        "ObjectiveSpec",
        payload,
        source_version=source_version,
        target_version=target_version,
        path="objective_spec",
    )


def validate_objective_spec(spec: ObjectiveSpec | dict[str, Any]) -> ObjectiveSpec:
    """Migrate a durable objective payload, then validate it as ``ObjectiveSpec``."""
    if isinstance(spec, ObjectiveSpec):
        return spec
    migrated = migrate_objective_spec(spec).payload
    return ObjectiveSpec.model_validate(migrated)


def _json_array_shape(value: Any) -> tuple[int, ...]:
    """Return the rectangular shape of a JSON-like array payload."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        length = len(value)
        if length == 0:
            return (0,)
        child_shapes = [_json_array_shape(item) for item in value]
        first = child_shapes[0]
        if any(shape != first for shape in child_shapes):
            raise ValueError("matrix payload must be rectangular")
        return (length, *first)
    return ()
