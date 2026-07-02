"""Method-neutral training worker contracts.

This module defines the durable vocabulary shared by Feedbax training workers
and downstream method implementations. A method reference resolves to these
declarations plus update kernels; it does not resolve to a method-owned runner.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.manifest import StrictModel


WORKER_CONTRACT_SCHEMA_ID = "feedbax.spec.worker.execution_program"
WORKER_CONTRACT_SCHEMA_VERSION = "feedbax.spec.worker.execution_program.v1"
CONSISTENCY_PREDICATE_SCHEMA_ID = "feedbax.manifest.worker.consistency_predicate"
CONSISTENCY_PREDICATE_SCHEMA_VERSION = "feedbax.manifest.worker.consistency_predicate.v2"
FIXED_UPDATE_KERNEL_SIGNATURE = ("slots", "coordinate", "context")

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/@+-]*$")
_GENERATOR_SOURCE = (
    "feedbax.training.worker_contract.v2:"
    "slot-init-read-write+lifetime+optimizer-bindings+objective-reads+"
    "measurement-control+metric-guards"
)
CONSISTENCY_PREDICATE_GENERATOR_HASH = hashlib.sha256(
    _GENERATOR_SOURCE.encode("utf-8")
).hexdigest()

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


class StateSlotSpec(StrictModel):
    """Opaque state slot declared by a training method."""

    name: str
    role: StateSlotRole
    required: bool = True
    axis: str | None = None
    shape: tuple[int | str, ...] | None = None
    dtype: str | None = None
    lifetime: SlotLifetime = "persistent"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_worker_identifier(value, path="state_slot.name")


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
                "update kernel signature must be exactly "
                f"{FIXED_UPDATE_KERNEL_SIGNATURE!r}"
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
    """Schedule-origin coordinate recorded when entering a phase."""

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
    """Coordinate at which a phase program can resume."""

    phase: str
    completed_barrier: str | None = None
    global_step: int = 0


class CheckpointSlotSpec(StrictModel):
    """One state slot captured in a checkpoint transaction."""

    slot: str
    required: bool = True
    axis: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointBarrierSpec(StrictModel):
    """Checkpoint barrier and the slots it captures."""

    name: str
    phase: str
    slots: list[CheckpointSlotSpec]
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
        return self


class ProgressCoordinate(StrictModel):
    """Generic nested-loop coordinate for worker progress streams."""

    run_id: str
    phase: str
    global_step: int = 0
    outer_step: int | None = None
    inner_step: int | None = None
    adversary_member: int | None = None
    replicate: int | None = None
    risk_realization: int | None = None
    member_id: str | None = None
    replicate_id: str | None = None
    completed_barrier: str | None = None
    schedule_origin_step: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


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
    global_step: int
    axis_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


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
        phase_program_digest=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    )


def supervised_task_trainer_mapping() -> tuple[WorkerMappingRow, ...]:
    """Map the existing supervised ``TaskTrainer`` loop to worker vocabulary."""
    return (
        WorkerMappingRow(
            source="feedbax.training.trainer.TaskTrainer.__call__",
            phase="train_batch",
            axes=["batch"],
            state_slots=["model", "optimizer", "prng", "trial_specs", "auxiliary_losses"],
            optimizer_bindings=["task_optimizer_to_model"],
            progress_coordinate="run_id/phase/global_step",
            checkpoint_transaction="model+optimizer+prng at checkpoint interval",
            notes="One optimizer update per batch followed by component-parameter projection.",
        ),
    )


PPO_MAPPING_TABLE: tuple[WorkerMappingRow, ...] = (
    WorkerMappingRow(
        source="feedbax.training.rl.ppo.train_ppo",
        phase="collect_rollout",
        axes=["environment", "rollout"],
        state_slots=["policy", "environment_state", "prng", "rollout", "observation_norm"],
        progress_coordinate="run_id/phase/global_step/outer_step",
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
        source="feedbax.training.rl.ppo.train_ppo",
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
                    global_step=1,
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
                    global_step=1,
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
