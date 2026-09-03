"""Method-neutral training worker contracts.

This module defines the durable vocabulary shared by Feedbax training workers
and downstream method implementations. A method reference resolves to these
declarations plus update kernels; it does not resolve to a method-owned runner.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.canonical_json import (
    CANONICAL_JSON_V1,
    CANONICAL_JSON_V2,
    canonical_json_v2_bytes,
)
from feedbax.contracts.base import StrictModel


WORKER_CONTRACT_SCHEMA_ID = "feedbax.spec.worker.execution_program"
WORKER_CONTRACT_SCHEMA_VERSION_V1 = "feedbax.spec.worker.execution_program.v1"
WORKER_CONTRACT_SCHEMA_VERSION = "feedbax.spec.worker.execution_program.v2"
CONSISTENCY_PREDICATE_SCHEMA_ID = "feedbax.manifest.worker.consistency_predicate"
CONSISTENCY_PREDICATE_SCHEMA_VERSION_V2 = "feedbax.manifest.worker.consistency_predicate.v2"
CONSISTENCY_PREDICATE_SCHEMA_VERSION = "feedbax.manifest.worker.consistency_predicate.v3"
FIXED_UPDATE_KERNEL_SIGNATURE = ("slots", "coordinate", "context")
NATIVE_TRAINING_COLLECTION_OUTPUTS = (
    "manifest.json",
    "training-diagnostics.json",
    "checkpoints",
    "manifests",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/@+-]*$")
_GENERATOR_SOURCE = (
    "feedbax.training.worker_contract.v2:"
    "slot-init-read-write+lifetime+optimizer-bindings+objective-reads+"
    "measurement-control+metric-guards"
)
CONSISTENCY_PREDICATE_GENERATOR_HASH = hashlib.sha256(_GENERATOR_SOURCE.encode("utf-8")).hexdigest()

AxisRole = Literal[
    "authored_sweep",
    "batch",
    "environment",
    "replicate",
    "member",
    "realization",
    "phase",
    "rollout",
    "epoch",
    "minibatch",
]
StateSlotRole = Literal[
    "model",
    "optimizer",
    "prng",
    "auxiliary",
    "population",
    "environment",
    "objective",
    "checkpoint",
    "metric",
]
SlotLifetime = Literal["persistent", "per-phase-init", "per-outer-step-init"]
PhaseKind = Literal[
    "warmup",
    "collect",
    "advantage",
    "inner_loop",
    "outer_loop",
    "adversarial",
    "evaluation",
    "checkpoint",
    "custom",
]
UpdateStepKind = Literal[
    "collect",
    "gradient",
    "projection",
    "reduce",
    "checkpoint",
    "measurement",
    "control",
    "custom",
]
OptimizationDirection = Literal["minimize", "maximize"]
ReducerOwner = Literal["objective", "worker"]


class WorkerContractError(ValueError):
    """Raised when a worker contract value is structurally invalid."""


def validate_worker_identifier(value: str, *, path: str = "identifier") -> str:
    """Validate one stable worker-contract identifier."""
    if not isinstance(value, str) or not value:
        raise WorkerContractError(f"{path} must be a non-empty string")
    if not _IDENTIFIER_RE.match(value):
        raise WorkerContractError(f"{path} {value!r} contains unsupported characters")
    return value


def _migrate_legacy_global_step(value: Any, *, model_name: str) -> Any:
    """Map an old coordinate field to the explicit cumulative program field.

    This intentionally only migrates coordinate naming. It never supplies a
    training-batch total from a legacy coordinate.
    """
    if not isinstance(value, dict) or "global_step" not in value:
        return value
    migrated = dict(value)
    legacy = migrated.pop("global_step")
    current = migrated.get("program_step")
    if current is not None and current != legacy:
        raise ValueError(
            f"{model_name} has conflicting legacy global_step={legacy!r} and "
            f"program_step={current!r}"
        )
    migrated["program_step"] = legacy
    return migrated


class AxisReducerSpec(StrictModel):
    """Declares that the worker owns reduction over an intra-run axis."""

    owner: ReducerOwner
    reduction: Literal["none", "sum", "mean", "min", "max", "custom"] = "mean"
    path: str


class AxisSpec(StrictModel):
    """One run-set or intra-run axis in a method declaration."""

    name: str
    role: AxisRole
    size: int | None = None
    reducer: AxisReducerSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_worker_identifier(value, path="axis.name")

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("axis size must be positive when provided")
        return value


class AxisCoordinateSpec(StrictModel):
    """One coordinate in ordered declared-axis space."""

    axis: str
    index: int = Field(ge=0)

    @field_validator("axis")
    @classmethod
    def _validate_axis(cls, value: str) -> str:
        return validate_worker_identifier(value, path="axis_coordinate.axis")


class SlotAxisBindingSpec(StrictModel):
    """Authored binding between one state slot and one declared axis."""

    axis: str
    mode: Literal["mapped", "shared"]
    array_axis: int | None = None
    leaf_policy: Literal["all_array_leaves"] = "all_array_leaves"

    @field_validator("axis")
    @classmethod
    def _validate_axis(cls, value: str) -> str:
        return validate_worker_identifier(value, path="axis_binding.axis")

    @model_validator(mode="after")
    def _validate_mode(self) -> "SlotAxisBindingSpec":
        if self.mode == "mapped" and self.array_axis is None:
            raise ValueError("mapped slot-axis binding requires array_axis")
        if self.mode == "shared" and self.array_axis is not None:
            raise ValueError("shared slot-axis binding forbids array_axis")
        return self


class MappingLevelSpec(StrictModel):
    """One authored execution mapping level, ordered outermost first."""

    axis: str

    @field_validator("axis")
    @classmethod
    def _validate_axis(cls, value: str) -> str:
        return validate_worker_identifier(value, path="mapping_level.axis")


class MaterializedMappingLevelSpec(StrictModel):
    """One mapping level resolved against a sized declared axis."""

    axis: str
    role: AxisRole
    size: int = Field(gt=0)
    level: int = Field(ge=0)


class MaterializedSlotAxisBinding(StrictModel):
    """One slot binding resolved against a materialized mapping level."""

    axis: str
    role: AxisRole
    size: int = Field(gt=0)
    level: int = Field(ge=0)
    mode: Literal["mapped", "shared"]
    array_axis: int | None = None
    leaf_policy: Literal["all_array_leaves"] = "all_array_leaves"

    @model_validator(mode="after")
    def _validate_mode(self) -> "MaterializedSlotAxisBinding":
        if self.mode == "mapped" and self.array_axis is None:
            raise ValueError("mapped materialized binding requires array_axis")
        if self.mode == "shared" and self.array_axis is not None:
            raise ValueError("shared materialized binding forbids array_axis")
        return self


class StateSlotSpec(StrictModel):
    """Opaque state slot declared by a training method."""

    name: str
    role: StateSlotRole
    required: bool = True
    axis: str | None = None
    axis_bindings: list[SlotAxisBindingSpec] | None = None
    shape: tuple[int | str, ...] | None = None
    dtype: str | None = None
    lifetime: SlotLifetime = "persistent"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_worker_identifier(value, path="state_slot.name")

    @model_validator(mode="after")
    def _validate_axis_bindings(self) -> "StateSlotSpec":
        bindings = self.axis_bindings or ()
        seen_axes: set[str] = set()
        seen_array_axes: set[int] = set()
        for index, binding in enumerate(bindings):
            if binding.axis in seen_axes:
                raise ValueError(
                    f"axis_bindings/{index}/axis duplicates axis {binding.axis!r}"
                )
            seen_axes.add(binding.axis)
            if binding.mode == "mapped":
                assert binding.array_axis is not None
                if binding.array_axis in seen_array_axes:
                    raise ValueError(
                        f"axis_bindings/{index}/array_axis overlaps array position "
                        f"{binding.array_axis}"
                    )
                seen_array_axes.add(binding.array_axis)
        return self


class OptimizerTargetBinding(StrictModel):
    """Binds one optimizer state slot to the state slot it updates."""

    name: str
    optimizer_slot: str
    target_slot: str
    target_selector: str | None = None
    direction: OptimizationDirection = "minimize"
    projection: Literal["none", "after_step", "phase_end"] = "none"
    phase_scope: list[str] = Field(default_factory=list)
    objective_reads: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "optimizer_slot", "target_slot")
    @classmethod
    def _validate_identifiers(cls, value: str) -> str:
        return validate_worker_identifier(value)


class UpdateKernelSpec(StrictModel):
    """Named callable update kernel with the fixed worker signature."""

    kernel_ref: str
    signature: tuple[str, str, str] = FIXED_UPDATE_KERNEL_SIGNATURE
    jit_static_payload_fields: list[str] = Field(default_factory=list)
    resource_significant_payload_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_kernel_ref(self) -> "UpdateKernelSpec":
        validate_worker_identifier(self.kernel_ref, path="kernel_ref")
        leaf = self.kernel_ref.rsplit(".", 1)[-1]
        if leaf.startswith("run_") or leaf in {"runner", "train", "fit"}:
            raise ValueError(
                "kernel_ref must name an update kernel, not a method-owned training runner"
            )
        if self.signature != FIXED_UPDATE_KERNEL_SIGNATURE:
            raise ValueError(
                f"update kernel signature must be exactly {FIXED_UPDATE_KERNEL_SIGNATURE!r}"
            )
        return self


class UpdateStepSpec(StrictModel):
    """One typed update step inside a phase."""

    name: str
    kind: UpdateStepKind
    kernel: UpdateKernelSpec
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    axes: list[str] = Field(default_factory=list)
    optimizer_binding: str | None = None
    data_member: str | None = None
    schedule_coordinate: str | None = None
    emits_progress: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_worker_identifier(value, path="update_step.name")


class PhaseSpec(StrictModel):
    """One phase node in a formal phase execution program."""

    name: str
    kind: PhaseKind
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    initializes: list[str] = Field(default_factory=list)
    update_steps: list[str] = Field(default_factory=list)
    legal_next: list[str] = Field(default_factory=list)
    checkpoint_barrier: str | None = None
    graph_binding: str | None = None
    loop_axis: str | None = None
    schedule_origin: "ScheduleOriginSpec | None" = None
    max_steps: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_worker_identifier(value, path="phase.name")

    @field_validator("max_steps")
    @classmethod
    def _validate_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("phase max_steps must be positive")
        return value


class PhaseTransitionSpec(StrictModel):
    """Legal transition between phase nodes."""

    source: str
    target: str
    barrier: str | None = None
    condition: str | None = None
    guard: "MetricGuardSpec | None" = None


class ScheduleOriginSpec(StrictModel):
    """Phase-program step-clock origin recorded when entering a phase.

    This origin applies to phase execution and resume coordinates. For
    training-batch-parameterized schedules, use
    :class:`feedbax.contracts.training.BatchScheduleOriginSpec` instead.
    """

    mode: Literal["run_start", "phase_entry", "resume_barrier"] = "phase_entry"
    step_offset: int = 0


class MetricGuardSpec(StrictModel):
    """Metric-slot transition guard evaluated at a checkpoint barrier."""

    predicate_ref: str
    metric_slots: list[str]
    bookkeeping_slots: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("predicate_ref")
    @classmethod
    def _validate_predicate_ref(cls, value: str) -> str:
        validate_worker_identifier(value, path="guard.predicate_ref")
        leaf = value.rsplit(".", 1)[-1]
        if leaf.startswith("run_") or leaf in {"runner", "train", "fit"}:
            raise ValueError(
                "guard predicate must name a predicate kernel, not a method-owned runner"
            )
        return value


class ResumeCoordinateSpec(StrictModel):
    """Cumulative program coordinate at which a phase program can resume."""

    phase: str
    completed_barrier: str | None = None
    program_step: int = 0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_global_step(cls, value: Any) -> Any:
        return _migrate_legacy_global_step(value, model_name=cls.__name__)


class TrainingBatchProgressSpec(StrictModel):
    """Declared authority for completed training batches at a custody barrier.

    ``slot`` names a declared persistent state slot. ``field_path`` is traversed
    through mapping keys (or sequence indices) to obtain a non-negative integer
    total. This is deliberately separate from ``ProgressCoordinate.program_step``:
    a program may take one orchestration step for many training batches.
    """

    slot: str
    field_path: tuple[str | int, ...] = ()

    @field_validator("slot")
    @classmethod
    def _validate_slot(cls, value: str) -> str:
        return validate_worker_identifier(value, path="batch_progress.slot")

    @field_validator("field_path")
    @classmethod
    def _validate_field_path(cls, value: tuple[str | int, ...]) -> tuple[str | int, ...]:
        for index, segment in enumerate(value):
            if isinstance(segment, int) and segment >= 0:
                continue
            if isinstance(segment, str) and segment:
                continue
            raise ValueError(f"batch_progress.field_path[{index}] must be a non-empty key or index")
        return value


class CheckpointSlotSpec(StrictModel):
    """One state slot captured in a checkpoint transaction."""

    slot: str
    required: bool = True
    axis: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BarrierArtifactSinkSpec(StrictModel):
    """One slot materialized as a local artifact each time a barrier fires."""

    slot: str
    role: str = "barrier_artifact"
    logical_name: str | None = None
    media_type: str = "application/octet-stream"
    encoding: Literal["raw", "json", "pickle"] = "raw"
    suffix: str | None = None
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("slot")
    @classmethod
    def _validate_slot(cls, value: str) -> str:
        return validate_worker_identifier(value, path="artifact_sink.slot")

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        return validate_worker_identifier(value, path="artifact_sink.role")

    @field_validator("logical_name")
    @classmethod
    def _validate_logical_name(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("artifact sink logical_name must be non-empty when provided")
        return value


class CheckpointBarrierSpec(StrictModel):
    """Checkpoint barrier and the slots it captures."""

    name: str
    phase: str
    slots: list[CheckpointSlotSpec]
    artifact_sinks: list[BarrierArtifactSinkSpec] = Field(default_factory=list)
    resume_coordinate: ResumeCoordinateSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_worker_identifier(value, path="checkpoint_barrier.name")


class PhaseProgramSpec(StrictModel):
    """Formal method-neutral phase execution program."""

    schema_id: str = WORKER_CONTRACT_SCHEMA_ID
    schema_version: str = WORKER_CONTRACT_SCHEMA_VERSION
    phases: list[PhaseSpec]
    initial_phase: str
    transitions: list[PhaseTransitionSpec] = Field(default_factory=list)
    update_steps: list[UpdateStepSpec] = Field(default_factory=list)
    optimizer_bindings: list[OptimizerTargetBinding] = Field(default_factory=list)
    checkpoint_barriers: list[CheckpointBarrierSpec] = Field(default_factory=list)
    batch_progress: TrainingBatchProgressSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "PhaseProgramSpec":
        if self.schema_id != WORKER_CONTRACT_SCHEMA_ID:
            raise ValueError(
                f"unsupported PhaseProgramSpec schema_id: {self.schema_id!r}, "
                f"expected {WORKER_CONTRACT_SCHEMA_ID!r}"
            )
        if self.schema_version != WORKER_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported PhaseProgramSpec schema_version: "
                f"{self.schema_version!r}, expected {WORKER_CONTRACT_SCHEMA_VERSION!r}"
            )
        return self


class ReducerRequirement(StrictModel):
    """Reducer ownership declaration from either objective or worker layer."""

    axis: str
    owner: ReducerOwner
    path: str


class MethodTrainingDiagnosticsSpec(StrictModel):
    """Authored declaration for one method's per-update training trace."""

    trace_schema_id: str
    trace_schema_version: str
    measurement_basis: str
    metric_payload_slot: str
    replica_axis: str

    @field_validator(
        "trace_schema_id",
        "trace_schema_version",
        "measurement_basis",
        "metric_payload_slot",
        "replica_axis",
    )
    @classmethod
    def _validate_authored_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("method training diagnostics values must be non-empty")
        return value


class MethodContractSpec(StrictModel):
    """Resolved method declaration consumed by the generic worker executor."""

    schema_id: str = WORKER_CONTRACT_SCHEMA_ID
    schema_version: str = WORKER_CONTRACT_SCHEMA_VERSION
    method_ref: str
    method_payload_schema_version: str
    axes: list[AxisSpec]
    state_slots: list[StateSlotSpec]
    phase_program: PhaseProgramSpec
    objective_reducers: list[ReducerRequirement] = Field(default_factory=list)
    worker_reducers: list[ReducerRequirement] = Field(default_factory=list)
    training_diagnostics: MethodTrainingDiagnosticsSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "MethodContractSpec":
        if self.schema_id != WORKER_CONTRACT_SCHEMA_ID:
            raise ValueError(
                f"unsupported MethodContractSpec schema_id: {self.schema_id!r}, "
                f"expected {WORKER_CONTRACT_SCHEMA_ID!r}"
            )
        if self.schema_version != WORKER_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported MethodContractSpec schema_version: "
                f"{self.schema_version!r}, expected {WORKER_CONTRACT_SCHEMA_VERSION!r}"
            )
        validate_worker_identifier(self.method_ref, path="method_ref")
        leaf = self.method_ref.rsplit(".", 1)[-1]
        if leaf.startswith("run_") or leaf in {"runner", "train", "fit"}:
            raise ValueError(
                "method_ref must resolve to a governed method payload, phase program, "
                "and update kernels; it must not name a method-owned runner"
            )
        diagnostics = self.training_diagnostics
        if diagnostics is not None:
            axis = next((item for item in self.axes if item.name == diagnostics.replica_axis), None)
            if axis is None or axis.role != "replicate" or axis.size is None:
                raise ValueError(
                    "training_diagnostics.replica_axis must name a sized replicate axis"
                )
            slot = next(
                (item for item in self.state_slots if item.name == diagnostics.metric_payload_slot),
                None,
            )
            if slot is None or slot.role != "metric":
                raise ValueError(
                    "training_diagnostics.metric_payload_slot must name a metric state slot"
                )
            bindings = slot.axis_bindings or ()
            if (
                len(bindings) != 1
                or bindings[0].axis != diagnostics.replica_axis
                or bindings[0].mode != "mapped"
            ):
                raise ValueError(
                    "training_diagnostics metric slot must map over the declared replica axis"
                )
        return self


class ProgressCoordinate(StrictModel):
    """Cumulative program coordinate for worker progress streams.

    ``program_step`` counts executed phase-program steps across phase
    transitions and resumes. It is neither a training-batch total nor a
    checkpoint count; custody records those values separately.
    """

    run_id: str
    phase: str
    program_step: int = 0
    outer_step: int | None = None
    inner_step: int | None = None
    adversary_member: int | None = None
    replicate: int | None = None
    risk_realization: int | None = None
    member_id: str | None = None
    replicate_id: str | None = None
    completed_barrier: str | None = None
    schedule_origin_step: int | None = None
    completed_batches: int | None = Field(default=None, ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_global_step(cls, value: Any) -> Any:
        return _migrate_legacy_global_step(value, model_name=cls.__name__)


class ConsistencyPredicateRule(StrictModel):
    """One machine-readable consistency rule derived from a phase program."""

    path: str
    rule: str
    expected: dict[str, Any] = Field(default_factory=dict)


class ConsistencyPredicateSpec(StrictModel):
    """Governed predicate consumed by checkpoint and executor integrations."""

    schema_id: str = CONSISTENCY_PREDICATE_SCHEMA_ID
    schema_version: str = CONSISTENCY_PREDICATE_SCHEMA_VERSION
    generator_hash: str = CONSISTENCY_PREDICATE_GENERATOR_HASH
    rules: list[ConsistencyPredicateRule]
    phase_program_digest: str
    pin_algorithm: Literal["canonical_json_v1", "canonical_json_v2"]

    @model_validator(mode="before")
    @classmethod
    def _migrate_v2(cls, value: Any) -> Any:
        return migrate_consistency_predicate_payload(value)

    @model_validator(mode="after")
    def _validate_schema_identity(self) -> "ConsistencyPredicateSpec":
        if self.schema_id != CONSISTENCY_PREDICATE_SCHEMA_ID:
            raise ValueError(
                f"unsupported ConsistencyPredicateSpec schema_id: {self.schema_id!r}, "
                f"expected {CONSISTENCY_PREDICATE_SCHEMA_ID!r}"
            )
        if self.schema_version != CONSISTENCY_PREDICATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ConsistencyPredicateSpec schema_version: "
                f"{self.schema_version!r}, expected {CONSISTENCY_PREDICATE_SCHEMA_VERSION!r}"
            )
        return self


class EffectivePhaseSpec(StrictModel):
    """Validated phase-program bundle handed to executors."""

    schema_id: str = WORKER_CONTRACT_SCHEMA_ID
    schema_version: str = WORKER_CONTRACT_SCHEMA_VERSION
    method_ref: str
    axes: list[AxisSpec]
    state_slots: list[StateSlotSpec]
    phase_program: PhaseProgramSpec
    consistency_predicate: ConsistencyPredicateSpec
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointSlotRecord(StrictModel):
    """Loaded checkpoint coordinate for one state slot."""

    slot: str
    barrier: str
    program_step: int
    axis_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_global_step(cls, value: Any) -> Any:
        return _migrate_legacy_global_step(value, model_name=cls.__name__)


class CheckpointSlotManifest(StrictModel):
    """Manifest describing checkpoint slot coordinates for resume validation."""

    schema_id: str = "feedbax.manifest.worker.checkpoint_slots"
    schema_version: str = "feedbax.manifest.worker.checkpoint_slots.v1"
    slots: list[CheckpointSlotRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerMappingRow(StrictModel):
    """Doc-testable mapping row from an existing trainer to worker vocabulary."""

    source: str
    phase: str
    axes: list[str] = Field(default_factory=list)
    state_slots: list[str] = Field(default_factory=list)
    optimizer_bindings: list[str] = Field(default_factory=list)
    progress_coordinate: str
    checkpoint_transaction: str | None = None
    notes: str = ""


def migrate_consistency_predicate_payload(value: Any) -> Any:
    """Migrate v2 by pinning its existing digest bytes to ``canonical_json_v1``."""

    if not isinstance(value, dict):
        return value
    if value.get("schema_version") != CONSISTENCY_PREDICATE_SCHEMA_VERSION_V2:
        return value
    if "pin_algorithm" in value:
        raise ValueError("ConsistencyPredicateSpec v2 does not admit pin_algorithm")
    migrated = dict(value)
    migrated["schema_version"] = CONSISTENCY_PREDICATE_SCHEMA_VERSION
    migrated["pin_algorithm"] = CANONICAL_JSON_V1
    return migrated


def derive_consistency_predicate(program: PhaseProgramSpec) -> ConsistencyPredicateSpec:
    """Derive a governed consistency predicate from a phase program."""
    rules: list[ConsistencyPredicateRule] = []
    for phase_index, phase in enumerate(program.phases):
        prefix = f"/phase_program/phases/{phase_index}"
        rules.append(
            ConsistencyPredicateRule(
                path=prefix,
                rule="phase-slot-access",
                expected={
                    "phase": phase.name,
                    "initializes": sorted(phase.initializes),
                    "reads": sorted(phase.reads),
                    "writes": sorted(phase.writes),
                    "schedule_origin": (
                        None
                        if phase.schedule_origin is None
                        else phase.schedule_origin.model_dump(mode="json")
                    ),
                },
            )
        )
    for binding_index, binding in enumerate(program.optimizer_bindings):
        rules.append(
            ConsistencyPredicateRule(
                path=f"/phase_program/optimizer_bindings/{binding_index}",
                rule="optimizer-target-binding",
                expected={
                    "name": binding.name,
                    "optimizer_slot": binding.optimizer_slot,
                    "target_slot": binding.target_slot,
                    "direction": binding.direction,
                    "projection": binding.projection,
                    "phase_scope": sorted(binding.phase_scope),
                    "objective_reads": sorted(binding.objective_reads),
                },
            )
        )
    for transition_index, transition in enumerate(program.transitions):
        if transition.guard is None:
            continue
        rules.append(
            ConsistencyPredicateRule(
                path=f"/phase_program/transitions/{transition_index}/guard",
                rule="metric-guard",
                expected={
                    "predicate_ref": transition.guard.predicate_ref,
                    "metric_slots": sorted(transition.guard.metric_slots),
                    "bookkeeping_slots": sorted(transition.guard.bookkeeping_slots),
                    "barrier": transition.barrier,
                },
            )
        )
    payload = program.model_dump(mode="json", exclude_none=True)
    return ConsistencyPredicateSpec(
        rules=rules,
        phase_program_digest=hashlib.sha256(canonical_json_v2_bytes(payload)).hexdigest(),
        pin_algorithm=CANONICAL_JSON_V2,
    )


def supervised_executor_mapping() -> tuple[WorkerMappingRow, ...]:
    """Map the standard supervised executor loop to worker vocabulary."""
    return (
        WorkerMappingRow(
            source="feedbax.training.executor.execute_training_run_spec",
            phase="train_batch",
            axes=["batch"],
            state_slots=["model", "optimizer", "prng", "trial_specs", "auxiliary_losses"],
            optimizer_bindings=["task_optimizer_to_model"],
            progress_coordinate="run_id/phase/program_step",
            checkpoint_transaction="manifested multi-slot checkpoint transaction",
            notes="Executor-owned phase step using declared kernels and checkpoint custody.",
        ),
    )


PPO_MAPPING_TABLE: tuple[WorkerMappingRow, ...] = (
    WorkerMappingRow(
        source="feedbax.training.rl.ppo.train_ppo_batched",
        phase="collect_rollout",
        axes=["environment", "rollout"],
        state_slots=["policy", "environment_state", "prng", "rollout", "observation_norm"],
        progress_coordinate="run_id/phase/program_step/outer_step",
        checkpoint_transaction="policy+optimizer+environment_state+prng",
        notes="Rollout collection scans over time and vmaps over environments.",
    ),
    WorkerMappingRow(
        source="feedbax.training.rl.ppo.compute_gae_scan",
        phase="compute_gae",
        axes=["environment", "rollout"],
        state_slots=["rollout", "gae_state", "returns"],
        progress_coordinate="run_id/phase/outer_step",
        notes="GAE is an auxiliary/intermediate state slot, not a method runner.",
    ),
    WorkerMappingRow(
        source="feedbax.training.rl.ppo.train_ppo_batched",
        phase="ppo_epoch_minibatch",
        axes=["epoch", "minibatch", "environment"],
        state_slots=["policy", "policy_optimizer", "value_optimizer", "gae_state", "prng"],
        optimizer_bindings=["policy_loss_to_policy", "value_loss_to_policy"],
        progress_coordinate="run_id/phase/outer_step/inner_step",
        checkpoint_transaction="policy+optimizer+prng after update",
        notes="Epoch x minibatch loop is represented as nested worker axes.",
    ),
    WorkerMappingRow(
        source="feedbax.training.rl.ppo.train_ppo_batched",
        phase="batched_body_collect_and_update",
        axes=["replicate", "environment", "rollout", "minibatch"],
        state_slots=["batched_policy", "batched_optimizer", "environment_state", "prng"],
        optimizer_bindings=["body_policy_loss_to_body_policy"],
        progress_coordinate="run_id/phase/outer_step/replicate",
        checkpoint_transaction="body-indexed policy+optimizer+environment_state+prng",
        notes="Body/population axis covers independent batched policies.",
    ),
)


def toy_minimax_method_contract() -> MethodContractSpec:
    """Return a feedbax-core toy minimax-shaped method contract for tests."""
    axes = [
        AxisSpec(name="batch", role="batch"),
        AxisSpec(name="adversary_member", role="member", size=2),
    ]
    slots = [
        StateSlotSpec(name="controller", role="model"),
        StateSlotSpec(name="controller_optimizer", role="optimizer"),
        StateSlotSpec(name="adversary_population", role="population", axis="adversary_member"),
        StateSlotSpec(name="adversary_optimizer", role="optimizer", axis="adversary_member"),
        StateSlotSpec(name="rng", role="prng"),
        StateSlotSpec(name="loss", role="auxiliary", required=False),
    ]
    update_steps = [
        UpdateStepSpec(
            name="warmup_update",
            kind="gradient",
            kernel=UpdateKernelSpec(kernel_ref="toy_minimax.warmup_update"),
            reads=["controller", "controller_optimizer", "rng"],
            writes=["controller", "controller_optimizer", "rng", "loss"],
            optimizer_binding="controller_optimizer_to_controller",
        ),
        UpdateStepSpec(
            name="adversary_update",
            kind="gradient",
            kernel=UpdateKernelSpec(kernel_ref="toy_minimax.adversary_update"),
            reads=["controller", "adversary_population", "adversary_optimizer", "rng"],
            writes=["adversary_population", "adversary_optimizer", "rng", "loss"],
            optimizer_binding="adversary_optimizer_to_population",
            axes=["adversary_member"],
        ),
    ]
    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="warmup",
                kind="warmup",
                reads=["controller", "controller_optimizer", "rng"],
                writes=["controller", "controller_optimizer", "rng", "loss"],
                update_steps=["warmup_update"],
                legal_next=["adversarial"],
                checkpoint_barrier="after_warmup",
            ),
            PhaseSpec(
                name="adversarial",
                kind="adversarial",
                reads=["controller", "adversary_population", "adversary_optimizer", "rng"],
                writes=["adversary_population", "adversary_optimizer", "rng", "loss"],
                update_steps=["adversary_update"],
                checkpoint_barrier="after_adversarial",
            ),
        ],
        initial_phase="warmup",
        transitions=[
            PhaseTransitionSpec(
                source="warmup",
                target="adversarial",
                barrier="after_warmup",
            )
        ],
        update_steps=update_steps,
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="controller_optimizer_to_controller",
                optimizer_slot="controller_optimizer",
                target_slot="controller",
                direction="minimize",
                projection="after_step",
                phase_scope=["warmup"],
            ),
            OptimizerTargetBinding(
                name="adversary_optimizer_to_population",
                optimizer_slot="adversary_optimizer",
                target_slot="adversary_population",
                direction="maximize",
                phase_scope=["adversarial"],
            ),
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name="after_warmup",
                phase="warmup",
                slots=[
                    CheckpointSlotSpec(slot="controller"),
                    CheckpointSlotSpec(slot="controller_optimizer"),
                    CheckpointSlotSpec(slot="adversary_population", axis="adversary_member"),
                    CheckpointSlotSpec(slot="adversary_optimizer", axis="adversary_member"),
                    CheckpointSlotSpec(slot="rng"),
                ],
                resume_coordinate=ResumeCoordinateSpec(
                    phase="adversarial",
                    completed_barrier="after_warmup",
                    program_step=1,
                ),
            ),
            CheckpointBarrierSpec(
                name="after_adversarial",
                phase="adversarial",
                slots=[
                    CheckpointSlotSpec(slot="controller"),
                    CheckpointSlotSpec(slot="adversary_population", axis="adversary_member"),
                    CheckpointSlotSpec(slot="adversary_optimizer", axis="adversary_member"),
                    CheckpointSlotSpec(slot="rng"),
                ],
            ),
        ],
    )
    return MethodContractSpec(
        method_ref="feedbax.toy_minimax",
        method_payload_schema_version="feedbax.spec.worker.toy_minimax_payload.v1",
        axes=axes,
        state_slots=slots,
        phase_program=program,
    )


def toy_adaptive_curriculum_method_contract() -> MethodContractSpec:
    """Return a toy adaptive-curriculum contract for guard/control tests."""
    axes = [
        AxisSpec(name="realization", role="realization", size=1),
    ]
    slots = [
        StateSlotSpec(name="controller", role="model"),
        StateSlotSpec(name="heldout_metric", role="metric"),
        StateSlotSpec(name="adaptive_lambda", role="auxiliary"),
        StateSlotSpec(name="guard_counter", role="auxiliary"),
    ]
    update_steps = [
        UpdateStepSpec(
            name="heldout_measurement",
            kind="measurement",
            kernel=UpdateKernelSpec(kernel_ref="toy_adaptive.measure_heldout"),
            reads=["controller"],
            writes=["heldout_metric"],
            data_member="heldout_realization",
        ),
        UpdateStepSpec(
            name="lambda_control",
            kind="control",
            kernel=UpdateKernelSpec(kernel_ref="toy_adaptive.update_lambda"),
            reads=["heldout_metric"],
            writes=["adaptive_lambda", "guard_counter"],
            schedule_coordinate="schedule_origin_step",
        ),
    ]
    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="adaptive",
                kind="custom",
                reads=["controller", "heldout_metric", "adaptive_lambda", "guard_counter"],
                writes=["heldout_metric", "adaptive_lambda", "guard_counter"],
                update_steps=["heldout_measurement", "lambda_control"],
                legal_next=["done", "adaptive"],
                checkpoint_barrier="after_adaptive_measurement",
                schedule_origin=ScheduleOriginSpec(mode="phase_entry"),
            ),
            PhaseSpec(
                name="done",
                kind="evaluation",
                reads=["heldout_metric", "adaptive_lambda", "guard_counter"],
            ),
        ],
        initial_phase="adaptive",
        transitions=[
            PhaseTransitionSpec(
                source="adaptive",
                target="done",
                barrier="after_adaptive_measurement",
                guard=MetricGuardSpec(
                    predicate_ref="toy_adaptive.stop_when_counter_satisfied",
                    metric_slots=["heldout_metric"],
                    bookkeeping_slots=["guard_counter"],
                ),
            ),
            PhaseTransitionSpec(
                source="adaptive",
                target="adaptive",
                barrier="after_adaptive_measurement",
            ),
        ],
        update_steps=update_steps,
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name="after_adaptive_measurement",
                phase="adaptive",
                slots=[
                    CheckpointSlotSpec(slot="controller"),
                    CheckpointSlotSpec(slot="heldout_metric"),
                    CheckpointSlotSpec(slot="adaptive_lambda"),
                    CheckpointSlotSpec(slot="guard_counter"),
                ],
                resume_coordinate=ResumeCoordinateSpec(
                    phase="adaptive",
                    completed_barrier="after_adaptive_measurement",
                    program_step=1,
                ),
            ),
        ],
    )
    return MethodContractSpec(
        method_ref="feedbax.toy_adaptive_curriculum",
        method_payload_schema_version="feedbax.spec.worker.toy_adaptive_payload.v1",
        axes=axes,
        state_slots=slots,
        phase_program=program,
    )


PhaseSpec.model_rebuild()
PhaseTransitionSpec.model_rebuild()
