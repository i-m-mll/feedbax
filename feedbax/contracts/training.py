"""Pydantic models for training specifications."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.graph import GraphSpec, ParamValue, RetentionPolicySpec
from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    CheckpointSlotManifest,
    CheckpointSlotSpec,
    EffectivePhaseSpec,
    MethodContractSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    ProgressCoordinate,
    ResumeCoordinateSpec,
    StateSlotSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)


TRAINING_RUN_SPEC_SCHEMA_ID = "feedbax.spec.training_run"
TRAINING_RUN_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training_run.v1"
TRAINING_RUN_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run.v2"
LR_SCHEDULE_SPEC_SCHEMA_ID = "feedbax.spec.training.lr_schedule"
LR_SCHEDULE_SPEC_SCHEMA_VERSION = "feedbax.spec.training.lr_schedule.v1"
LOSS_TERM_SPEC_SCHEMA_ID = "feedbax.spec.training.loss_term"
LOSS_TERM_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training.loss_term.v1"
LOSS_TERM_SPEC_SCHEMA_VERSION = "feedbax.spec.training.loss_term.v2"
STANDARD_SUPERVISED_METHOD_REF = "feedbax/standard_supervised/v1"
STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID = (
    "feedbax.spec.training_method.standard_supervised_payload"
)
STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION = (
    "feedbax.spec.training_method.standard_supervised_payload.v1"
)


class LrScheduleSpec(BaseModel):
    """Schema-versioned learning-rate schedule specification.

    Attributes:
        schema_id: Stable schema identity for Feedbax learning-rate schedules.
        schema_version: Version of this schedule payload schema.
        kind: Schedule family. ``"constant"`` holds ``learning_rate_0``;
            ``"warmup_cosine"`` linearly warms from
            ``warmup_init_fraction * learning_rate_0`` over
            ``constant_lr_iterations`` steps, then cosine-anneals to
            ``cosine_annealing_alpha * learning_rate_0`` at ``total_steps``;
            ``"delayed_cosine"`` holds ``learning_rate_0`` for
            ``constant_lr_iterations`` steps, then cosine-anneals to the same
            terminal fraction at ``total_steps``.
        learning_rate_0: Peak learning rate. This maps from rlrmp
            ``controller_lr``.
        total_steps: Origin-relative step at which cosine schedules reach
            their terminal value. This maps from rlrmp ``n_batches_condition``.
        constant_lr_iterations: Warmup length for ``"warmup_cosine"`` and
            constant-prefix length for ``"delayed_cosine"``. This maps from
            rlrmp ``lr_warmup_batches``.
        warmup_init_fraction: Initial learning-rate fraction for
            ``"warmup_cosine"``. This maps from rlrmp
            ``lr_warmup_init_fraction``.
        cosine_annealing_alpha: Terminal learning-rate fraction for cosine
            schedules. This maps from rlrmp ``lr_cosine_alpha``.
    """

    schema_id: Literal["feedbax.spec.training.lr_schedule"] = LR_SCHEDULE_SPEC_SCHEMA_ID
    schema_version: str = LR_SCHEDULE_SPEC_SCHEMA_VERSION
    kind: Literal["constant", "warmup_cosine", "delayed_cosine"] = "constant"
    learning_rate_0: float = Field(gt=0.0)
    total_steps: int | None = Field(default=None, gt=0)
    constant_lr_iterations: int = Field(default=0, ge=0)
    warmup_init_fraction: float = Field(default=0.0, ge=0.0)
    cosine_annealing_alpha: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_schedule_shape(self) -> "LrScheduleSpec":
        if self.schema_id != LR_SCHEDULE_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported LrScheduleSpec schema_id {self.schema_id!r}; "
                f"expected {LR_SCHEDULE_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != LR_SCHEDULE_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported LrScheduleSpec schema_version "
                f"{self.schema_version!r}; expected {LR_SCHEDULE_SPEC_SCHEMA_VERSION!r}"
            )
        if self.kind == "constant":
            return self
        if self.total_steps is None:
            raise ValueError(f"/total_steps is required when lr_schedule.kind={self.kind!r}")
        if self.kind == "warmup_cosine":
            if self.constant_lr_iterations < 1:
                raise ValueError(
                    "/constant_lr_iterations must be >= 1 for warmup_cosine schedules"
                )
            if self.constant_lr_iterations >= self.total_steps:
                raise ValueError(
                    "/constant_lr_iterations must be < /total_steps for warmup_cosine"
                )
        if self.kind == "delayed_cosine" and self.constant_lr_iterations >= self.total_steps:
            raise ValueError(
                "/constant_lr_iterations must be < /total_steps for delayed_cosine"
            )
        return self


class OptimizerSpec(BaseModel):
    """Specification for an optimizer.

    Attributes:
        type: Optimizer factory name. The public builder supports ``"adamw"``,
            ``"adam"``, ``"sgd"``, and ``"rmsprop"``.
        params: Static optimizer parameters other than scheduled learning rate.
        lr_schedule: Optional declarative learning-rate schedule. ``None``
            preserves legacy method-owned optimizer construction.
    """

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    lr_schedule: LrScheduleSpec | None = None


class TimeAggregationSpec(BaseModel):
    """Specification for time aggregation in loss computation."""

    mode: Literal["all", "mean", "sum", "final", "range", "segment", "custom"] = "all"
    start: Optional[int] = None
    end: Optional[int] = None
    segment_name: Optional[str] = None
    time_idxs: Optional[List[int]] = None
    discount: Optional[Literal["none", "power", "linear"]] = None
    discount_exp: Optional[float] = None


class LossTermSpec(BaseModel):
    """Specification for a loss term."""

    schema_id: Literal["feedbax.spec.training.loss_term"] = LOSS_TERM_SPEC_SCHEMA_ID
    schema_version: str = LOSS_TERM_SPEC_SCHEMA_VERSION
    type: str
    label: str
    weight: float = 1.0
    selector: Optional[str] = None
    target_selector: Optional[str] = None
    target_value: Optional[Any] = None
    retention: Optional[RetentionPolicySpec] = None
    norm: Optional[Literal["squared_l2", "l2", "l1", "huber"]] = None
    matrix: Optional[Any] = None
    matrix_kind: Optional[Literal["dense", "diagonal"]] = None
    time_agg: Optional[TimeAggregationSpec] = None
    children: Optional[Dict[str, "LossTermSpec"]] = None


class EarlyStoppingSpec(BaseModel):
    """Specification for early stopping."""

    metric: str
    patience: int
    min_delta: float


class TrainingSpec(BaseModel):
    """Complete specification for a training run."""

    optimizer: OptimizerSpec
    loss: LossTermSpec
    n_batches: int
    batch_size: int
    n_epochs: Optional[int] = None
    checkpoint_interval: Optional[int] = None
    early_stopping: Optional[EarlyStoppingSpec] = None


class TaskSpec(BaseModel):
    """Specification for a task."""

    type: str
    params: Dict[str, ParamValue] = Field(default_factory=dict)
    timeline: Optional[Dict[str, ParamValue]] = None


class TrainingConfig(BaseModel):
    """Structured configuration for the real JAX training backend (Phase 6).

    Passed verbatim to the worker via the ``/start`` request body under the
    ``training_config`` key.  The worker converts this into a ``_TrainingCfg``
    dataclass via ``_extract_training_cfg``.  All fields have sensible defaults
    so callers can override only the parameters they care about.

    Attributes:
        n_batches: Number of training steps.
        batch_size: Trials per gradient update. The current generic graph worker
            supports only ``batch_size=1`` and rejects larger values at compile
            time instead of silently ignoring them.
        learning_rate: AdamW learning rate.
        grad_clip: Global gradient clipping norm. The schema default remains 1.0;
            set explicitly to None to disable gradient clipping.
        hidden_dim: GRU / CDE hidden state dimension.
        network_type: Controller architecture — ``"gru"`` or ``"cde"``.
        n_reach_steps: Number of control steps per episode.
        effort_weight: Relative weight for the muscle-effort penalty.
        snapshot_interval: Emit a ``training_trajectory`` event every N batches.
    """

    n_batches: int = 2000
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float | None = 1.0
    hidden_dim: int = 128
    network_type: str = "gru"
    n_reach_steps: int = 80
    effort_weight: float = 2.5
    snapshot_interval: int = 100


class TrainingRunContractModel(StrictModel):
    """Base model for durable training-run request contracts."""

    model_config = ConfigDict(extra="forbid")


class MethodRefSpec(TrainingRunContractModel):
    """Durable namespaced/versioned method identity."""

    package: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @property
    def key(self) -> str:
        """Return the registry key for this method reference."""
        return f"{self.package}/{self.name}/{self.version}"

    @model_validator(mode="after")
    def _validate_segments(self) -> "MethodRefSpec":
        for path, value in (
            ("/method_ref/package", self.package),
            ("/method_ref/name", self.name),
            ("/method_ref/version", self.version),
        ):
            if "/" in value:
                raise ValueError(f"{path} must not contain '/'")
        return self


class GraphTopologySourceSpec(TrainingRunContractModel):
    """GraphSpec source for the run topology."""

    kind: Literal["GraphSpec"] = "GraphSpec"
    ref: str | None = None
    inline: dict[str, Any] | None = None
    schema_id: str | None = None
    schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> "GraphTopologySourceSpec":
        if self.ref is None and self.inline is None:
            raise ValueError("/graph requires ref or inline GraphSpec payload")
        if self.inline is not None:
            GraphSpec.model_validate(self.inline)
        return self


class ObjectiveSlotSpec(TrainingRunContractModel):
    """Objective/loss payload slot consumed by future objective-governance work."""

    kind: Literal["loss_term", "objective_spec", "external"] = "loss_term"
    loss: LossTermSpec | None = None
    payload: dict[str, Any] | None = None
    schema_id: str | None = None
    schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "ObjectiveSlotSpec":
        if self.kind == "loss_term" and self.loss is None:
            raise ValueError("/objective/loss is required when objective.kind='loss_term'")
        if self.kind != "loss_term" and self.payload is None:
            raise ValueError(f"/objective/payload is required when objective.kind={self.kind!r}")
        return self


class RiskAggregationSpec(TrainingRunContractModel):
    """Objective/risk aggregation policy, separate from method identity."""

    realization: Literal["none", "mean", "sum", "min", "max", "custom"] = "none"
    replicate: Literal["none", "mean", "sum", "min", "max", "custom"] = "none"
    time: TimeAggregationSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodPayloadEnvelope(TrainingRunContractModel):
    """Governed method-owned payload plus schema identity."""

    schema_id: str
    schema_version: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodExtensionsSpec(TrainingRunContractModel):
    """Explicitly non-semantic method extension metadata."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardSupervisedMethodPayload(TrainingRunContractModel):
    """Payload owned by Feedbax's standard supervised training method."""

    optimizer: OptimizerSpec = Field(default_factory=lambda: OptimizerSpec(type="adamw"))
    gradient_clip: float | None = None
    model_slot: str = "model"
    optimizer_slot: str = "optimizer"
    objective_slot: str = "objective"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerExecutionSpec(TrainingRunContractModel):
    """Worker declarations required before launch."""

    method_contract: MethodContractSpec
    effective_phase: EffectivePhaseSpec
    checkpoint_slots: CheckpointSlotManifest | None = None
    resume: ResumeCoordinateSpec | None = None
    progress: ProgressCoordinate | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionPolicySpec(TrainingRunContractModel):
    """Execution policy for a request, without performing launch side effects."""

    mode: Literal["dry_run", "local", "remote"] = "local"
    require_review: bool = True
    allow_cloud: bool = False
    max_wallclock_seconds: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactPolicySpec(TrainingRunContractModel):
    """Artifact custody and manifest emission policy for a training request."""

    manifest_root: str | None = None
    artifact_root: str | None = None
    custody: Literal["local", "mandible", "external"] = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointProgressPolicySpec(TrainingRunContractModel):
    """Checkpoint and progress policy for worker-visible run records."""

    checkpoint_interval: int | None = None
    progress_interval: int | None = None
    resume_from: ResumeCoordinateSpec | None = None
    checkpoint_slots: CheckpointSlotManifest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TrainingMethodRegistration:
    """One method-ref keyed payload governance registration."""

    method_ref: str
    payload_schema_id: str
    payload_schema_version: str
    payload_model: type[BaseModel]
    contract_factory: Callable[[], MethodContractSpec]
    update_kernels_factory: Callable[[BaseModel], Mapping[str, Callable[..., Mapping[str, Any]]]]
    guard_predicates_factory: Callable[[BaseModel], Mapping[str, Callable[..., Mapping[str, Any]]]] = (
        lambda _payload: {}
    )
    rejected_payload_versions: tuple[str, ...] = ()
    owner: str = "feedbax"
    package: str | None = None


class TrainingMethodRegistry:
    """Registry for method-ref keyed payload validation and migration dispatch."""

    def __init__(self) -> None:
        self._registrations: dict[str, TrainingMethodRegistration] = {}

    def register(self, registration: TrainingMethodRegistration) -> None:
        """Register one method payload governance row."""
        if registration.method_ref in self._registrations:
            raise ValueError(f"training method already registered: {registration.method_ref!r}")
        self._registrations[registration.method_ref] = registration

    def available_keys(self) -> tuple[str, ...]:
        """Return method refs known to this registry."""
        return tuple(sorted(self._registrations))

    def resolve(self, method_ref: MethodRefSpec | str, *, path: str) -> TrainingMethodRegistration:
        """Resolve a method ref or raise with available registry keys."""
        key = method_ref.key if isinstance(method_ref, MethodRefSpec) else method_ref
        try:
            return self._registrations[key]
        except KeyError as exc:
            raise ValueError(
                f"{path}: unknown method_ref {key!r}; "
                f"available registry keys={list(self.available_keys())!r}. "
                "Install a package exposing a feedbax.plugins training-method hook, "
                "or pass --plugin <module> to CLI commands that validate "
                "TrainingRunSpec payloads."
            ) from exc

    def validate_payload(
        self,
        method_ref: MethodRefSpec,
        envelope: MethodPayloadEnvelope,
        *,
        path: str,
    ) -> BaseModel:
        """Validate a method-owned payload through its registered schema row."""
        registration = self.resolve(method_ref, path="/method_ref")
        if envelope.schema_id != registration.payload_schema_id:
            raise ValueError(
                f"{path}/schema_id: unsupported method payload schema_id "
                f"{envelope.schema_id!r}; expected {registration.payload_schema_id!r}; "
                f"available registry keys={list(self.available_keys())!r}"
            )
        if envelope.schema_version != registration.payload_schema_version:
            if envelope.schema_version in registration.rejected_payload_versions:
                raise ValueError(
                    f"{path}/schema_version: unsupported method payload schema version "
                    f"{envelope.schema_version!r}; current_version="
                    f"{registration.payload_schema_version!r}; "
                    "migration_intentionally_absent=yes; "
                    f"available registry keys={list(self.available_keys())!r}"
                )
            raise ValueError(
                f"{path}/schema_version: no method payload migration path registered from "
                f"{envelope.schema_version!r} to {registration.payload_schema_version!r}; "
                f"available registry keys={list(self.available_keys())!r}"
            )
        return registration.payload_model.model_validate(envelope.payload)


def standard_supervised_method_ref() -> MethodRefSpec:
    """Return the public method ref for Feedbax's standard supervised method."""
    return MethodRefSpec(package="feedbax", name="standard_supervised", version="v1")


def standard_supervised_method_payload() -> MethodPayloadEnvelope:
    """Return a default governed payload envelope for the standard supervised method."""
    return MethodPayloadEnvelope(
        schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=StandardSupervisedMethodPayload().model_dump(mode="json"),
    )


def standard_supervised_method_contract() -> MethodContractSpec:
    """Return the worker declaration for Feedbax's standard supervised method."""
    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="train_batch",
                kind="outer_loop",
                reads=["model", "optimizer", "prng", "objective"],
                writes=["model", "optimizer", "prng", "train_loss"],
                update_steps=["supervised_gradient_update"],
                checkpoint_barrier="after_train_batch",
            )
        ],
        initial_phase="train_batch",
        update_steps=[
            UpdateStepSpec(
                name="supervised_gradient_update",
                kind="gradient",
                kernel=UpdateKernelSpec(
                    kernel_ref="feedbax.training.standard_supervised.gradient_update"
                ),
                reads=["model", "optimizer", "prng", "objective"],
                writes=["model", "optimizer", "prng", "train_loss"],
                axes=["batch"],
                optimizer_binding="optimizer_to_model",
            )
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="optimizer_to_model",
                optimizer_slot="optimizer",
                target_slot="model",
                projection="after_step",
                phase_scope=["train_batch"],
                objective_reads=["objective"],
            )
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name="after_train_batch",
                phase="train_batch",
                slots=[
                    CheckpointSlotSpec(slot="model"),
                    CheckpointSlotSpec(slot="optimizer"),
                    CheckpointSlotSpec(slot="prng"),
                ],
            )
        ],
    )
    return MethodContractSpec(
        method_ref=STANDARD_SUPERVISED_METHOD_REF,
        method_payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        axes=[AxisSpec(name="batch", role="batch")],
        state_slots=[
            StateSlotSpec(name="model", role="model"),
            StateSlotSpec(name="optimizer", role="optimizer"),
            StateSlotSpec(name="prng", role="prng"),
            StateSlotSpec(name="objective", role="objective"),
            StateSlotSpec(name="train_loss", role="metric", required=False),
        ],
        phase_program=program,
    )


def standard_supervised_effective_phase_spec() -> EffectivePhaseSpec:
    """Return an effective phase bundle for the standard supervised method."""
    contract = standard_supervised_method_contract()
    return EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=derive_consistency_predicate(contract.phase_program),
    )


def standard_supervised_update_kernels(
    _payload: BaseModel | None = None,
) -> Mapping[str, Callable[..., Mapping[str, Any]]]:
    """Return generic kernels for the standard supervised worker declaration.

    The kernel is intentionally method-neutral: it updates declared slots and lets
    real trainers provide richer kernels by overriding the registry entry.
    """

    def gradient_update(
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        updates: dict[str, Any] = {
            "model": slots["model"],
            "optimizer": slots["optimizer"],
            "prng": slots["prng"],
            "train_loss": float(coordinate.global_step + 1),
        }
        model = slots["model"]
        optimizer = slots["optimizer"]
        if isinstance(optimizer, Mapping) and "count" in optimizer:
            updates["optimizer"] = {**dict(optimizer), "count": optimizer["count"] + 1}
            try:
                updates["model"] = model + optimizer["count"]
            except TypeError:
                updates["model"] = model
        return updates

    return {"feedbax.training.standard_supervised.gradient_update": gradient_update}


def default_training_method_registry() -> TrainingMethodRegistry:
    """Return the default method-ref keyed payload registry."""
    registry = TrainingMethodRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref=STANDARD_SUPERVISED_METHOD_REF,
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=standard_supervised_method_contract,
            update_kernels_factory=standard_supervised_update_kernels,
            rejected_payload_versions=(
                "feedbax.spec.training_method.standard_supervised_payload.v0",
            ),
            owner="feedbax.contracts.training",
            package="feedbax",
        )
    )
    return registry


DEFAULT_TRAINING_METHOD_REGISTRY = default_training_method_registry()


class TrainingRunSpec(TrainingRunContractModel):
    """Public durable request contract for one Feedbax training run."""

    schema_id: str = TRAINING_RUN_SPEC_SCHEMA_ID
    schema_version: str = TRAINING_RUN_SPEC_SCHEMA_VERSION
    on_nan: Literal["raise", "halt_restore_checkpoint"] = "raise"
    graph: GraphTopologySourceSpec
    task: TaskSpec
    training_config: TrainingConfig
    objective: ObjectiveSlotSpec
    risk_aggregation: RiskAggregationSpec = Field(default_factory=RiskAggregationSpec)
    method_ref: MethodRefSpec
    method_payload: MethodPayloadEnvelope
    method_extensions: MethodExtensionsSpec = Field(default_factory=MethodExtensionsSpec)
    worker_execution: WorkerExecutionSpec
    execution: ExecutionPolicySpec = Field(default_factory=ExecutionPolicySpec)
    artifacts: ArtifactPolicySpec = Field(default_factory=ArtifactPolicySpec)
    checkpoint_progress: CheckpointProgressPolicySpec = Field(
        default_factory=CheckpointProgressPolicySpec
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> "TrainingRunSpec":
        if self.schema_id != TRAINING_RUN_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported TrainingRunSpec schema_id {self.schema_id!r}; "
                f"expected {TRAINING_RUN_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != TRAINING_RUN_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported TrainingRunSpec schema_version "
                f"{self.schema_version!r}; expected {TRAINING_RUN_SPEC_SCHEMA_VERSION!r}"
            )

        registration = DEFAULT_TRAINING_METHOD_REGISTRY.resolve(
            self.method_ref,
            path="/method_ref",
        )
        DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
            self.method_ref,
            self.method_payload,
            path="/method_payload",
        )
        method_key = self.method_ref.key
        method_contract = self.worker_execution.method_contract
        effective_phase = self.worker_execution.effective_phase
        if method_contract.method_ref != method_key:
            raise ValueError(
                "/worker_execution/method_contract/method_ref must match /method_ref; "
                f"found {method_contract.method_ref!r}, expected {method_key!r}"
            )
        if effective_phase.method_ref != method_key:
            raise ValueError(
                "/worker_execution/effective_phase/method_ref must match /method_ref; "
                f"found {effective_phase.method_ref!r}, expected {method_key!r}"
            )
        if method_contract.method_payload_schema_version != self.method_payload.schema_version:
            raise ValueError(
                "/worker_execution/method_contract/method_payload_schema_version must match "
                f"/method_payload/schema_version; found "
                f"{method_contract.method_payload_schema_version!r}, expected "
                f"{self.method_payload.schema_version!r}"
            )
        if not method_contract.axes:
            raise ValueError("/worker_execution/method_contract/axes must declare worker axes")
        if not method_contract.state_slots:
            raise ValueError(
                "/worker_execution/method_contract/state_slots must declare worker state slots"
            )
        expected_contract = registration.contract_factory()
        if not expected_contract.axes or not expected_contract.state_slots:
            raise ValueError(
                f"/method_ref {method_key!r} registry row has incomplete worker declaration"
            )
        return self


# Enable forward references
LossTermSpec.model_rebuild()
