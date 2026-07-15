"""Pydantic models for training specifications."""

from __future__ import annotations

import json
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from feedbax.contracts.graph import GraphSpec, ParamValue, RetentionPolicySpec
from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    CheckpointSlotManifest,
    CheckpointSlotSpec,
    EffectivePhaseSpec,
    MethodContractSpec,
    MappingLevelSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    ProgressCoordinate,
    ResumeCoordinateSpec,
    StateSlotSpec,
    TrainingBatchProgressSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)

if TYPE_CHECKING:
    from feedbax.training.preparation import ExecutionPreparationProvider
    from feedbax.training.run_matrix import TrainingRowLowerer


TRAINING_RUN_SPEC_SCHEMA_ID = "feedbax.spec.training_run"
TRAINING_RUN_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training_run.v1"
TRAINING_RUN_SPEC_SCHEMA_VERSION_V2 = "feedbax.spec.training_run.v2"
TRAINING_RUN_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run.v3"
RUN_CONTROL_SPEC_SCHEMA_ID = "feedbax.spec.training.run_control"
RUN_CONTROL_SPEC_SCHEMA_VERSION = "feedbax.spec.training.run_control.v1"
LR_SCHEDULE_SPEC_SCHEMA_ID = "feedbax.spec.training.lr_schedule"
LR_SCHEDULE_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training.lr_schedule.v1"
LR_SCHEDULE_SPEC_SCHEMA_VERSION = "feedbax.spec.training.lr_schedule.v2"
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
TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID = (
    "feedbax.spec.training_manifest_metadata_projection"
)
TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION = (
    "feedbax.spec.training_manifest_metadata_projection.v1"
)


class BatchScheduleOriginSpec(BaseModel):
    """Typed clock origin for a batch-parameterized schedule.

    This origin selects the training-batch clock for learning-rate and other
    batch schedules. For phase-program step clocks, use
    :class:`feedbax.contracts.worker.ScheduleOriginSpec` instead.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["segment_start", "run_start", "absolute"]
    batch: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_origin(self) -> "BatchScheduleOriginSpec":
        if self.kind == "absolute" and self.batch is None:
            raise ValueError("/batch is required when schedule origin kind='absolute'")
        if self.kind != "absolute" and self.batch is not None:
            raise ValueError("/batch is only allowed when schedule origin kind='absolute'")
        return self


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
    origin: BatchScheduleOriginSpec
    allow_inert: bool = False
    kind: Literal["constant", "warmup_cosine", "delayed_cosine"] = "constant"
    learning_rate_0: float = Field(gt=0.0)
    total_steps: int | None = Field(default=None, gt=0)
    constant_lr_iterations: int = Field(default=0, ge=0)
    warmup_init_fraction: float = Field(default=0.0, ge=0.0)
    cosine_annealing_alpha: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_v1_origin(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        if payload.get("schema_version") == LR_SCHEDULE_SPEC_SCHEMA_VERSION_V1:
            payload["schema_version"] = LR_SCHEDULE_SPEC_SCHEMA_VERSION
            payload.setdefault("origin", {"kind": "run_start"})
        elif "schema_version" not in payload and "origin" not in payload:
            # Python construction predates durable serialization. Preserve its
            # historical run-global clock while ensuring emitted v2 payloads
            # always contain the mandatory origin field.
            payload["origin"] = {"kind": "run_start"}
        return payload

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
                raise ValueError("/constant_lr_iterations must be >= 1 for warmup_cosine schedules")
            if self.constant_lr_iterations >= self.total_steps:
                raise ValueError("/constant_lr_iterations must be < /total_steps for warmup_cosine")
        if self.kind == "delayed_cosine" and self.constant_lr_iterations >= self.total_steps:
            raise ValueError("/constant_lr_iterations must be < /total_steps for delayed_cosine")
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


class RunControlSpec(TrainingRunContractModel):
    """Small, method-agnostic control surface for one training segment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_id: Literal["feedbax.spec.training.run_control"] = RUN_CONTROL_SPEC_SCHEMA_ID
    schema_version: str = RUN_CONTROL_SPEC_SCHEMA_VERSION
    n_batches: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    checkpoint_interval: int | None = Field(default=None, gt=0)
    progress_interval: int | None = Field(default=None, gt=0)
    continuation: CheckpointContinuationRequest | None = None

    @model_validator(mode="after")
    def _validate_control(self) -> "RunControlSpec":
        if self.schema_version != RUN_CONTROL_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported RunControlSpec schema_version "
                f"{self.schema_version!r}; expected {RUN_CONTROL_SPEC_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        if self.continuation is not None and self.continuation.additional_batches != self.n_batches:
            raise ValueError(
                "/n_batches must equal /continuation/additional_batches for a continuation"
            )
        return self


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


class TrainingMethodAuthoringContribution(BaseModel):
    """Runtime-only method values consumed by the typed authoring compiler.

    This hook result is not a durable spec. The authoring compiler embeds its
    values in the existing ``TrainingRunSpec`` contract.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    training_config: TrainingConfig
    method_extensions: MethodExtensionsSpec = Field(default_factory=MethodExtensionsSpec)
    mapping_levels: list[MappingLevelSpec] | None = None


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
    mapping_levels: list[MappingLevelSpec] | None = None
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
    continuation: CheckpointContinuationRequest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TrainingMethodRegistration:
    """Low-level method registration retained for direct runtime adapters."""

    method_ref: str
    payload_schema_id: str
    payload_schema_version: str
    payload_model: type[BaseModel]
    contract_factory: Callable[[], MethodContractSpec] | None
    update_kernels_factory: Callable[[BaseModel], Mapping[str, Callable[..., Mapping[str, Any]]]]
    guard_predicates_factory: Callable[
        [BaseModel], Mapping[str, Callable[..., Mapping[str, Any]]]
    ] = lambda _payload: {}
    rejected_payload_versions: tuple[str, ...] = ()
    owner: str = "feedbax"
    package: str | None = None
    requires_execution_preparation: bool = False
    contract_compiler: Callable[[BaseModel], MethodContractSpec] | None = None

    def compile_contract(self, payload: BaseModel) -> MethodContractSpec:
        """Compile the worker contract, preserving legacy zero-argument factories."""
        if self.contract_compiler is not None:
            return self.contract_compiler(payload)
        if self.contract_factory is None:
            raise ValueError(
                f"training method {self.method_ref!r} has no payload-aware contract compiler"
            )
        return self.contract_factory()


PayloadT = TypeVar("PayloadT", bound=BaseModel)

TrainingMethodOptimizerSpecProjector = Callable[[PayloadT], OptimizerSpec | Mapping[str, Any]]
TrainingMethodOptimizerStepExtractor = Callable[[PayloadT, Mapping[str, Any]], int]


@dataclass(frozen=True)
class TrainingMethodAuthoringHook(Generic[PayloadT]):
    """Stable runtime hook for method-owned authoring contributions."""

    lowerer_id: str
    lowerer_version: str
    compile: Callable[[PayloadT], TrainingMethodAuthoringContribution]

    def validate_structure(self) -> None:
        """Validate the stable lowerer identity and callable boundary."""
        identities = (self.lowerer_id, self.lowerer_version)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("training method authoring hook identity must not be empty")
        if not callable(self.compile):
            raise TypeError("training method authoring hook compile must be callable")


@dataclass(frozen=True)
class TrainingMethodMetadataProjector(Generic[PayloadT]):
    """Typed runtime hook for projecting method-owned manifest metadata."""

    schema_id: str
    schema_version: str
    output_model: type[BaseModel]
    projector: Callable[[PayloadT], object]

    def validate_structure(self) -> None:
        """Validate the stable projection identity and callable boundary."""
        identities = (self.schema_id, self.schema_version)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("training method metadata projector identity must not be empty")
        if not isinstance(self.output_model, type) or not issubclass(self.output_model, BaseModel):
            raise TypeError("training method metadata projector output_model must extend BaseModel")
        if self.output_model.model_config.get("extra") != "forbid":
            raise ValueError("training method metadata projector output_model must set extra='forbid'")
        if self.output_model.model_config.get("strict") is not True:
            raise ValueError("training method metadata projector output_model must set strict=True")
        if not callable(self.projector):
            raise TypeError("training method metadata projector projector must be callable")

    def project(self, payload: PayloadT) -> BaseModel:
        """Project and validate metadata through the declared output model."""
        self.validate_structure()
        return self.output_model.model_validate(self.projector(payload))


@dataclass(frozen=True)
class TrainingMethodDescriptor(Generic[PayloadT]):
    """Atomic runtime extension surface for one typed training method.

    Descriptor hooks are runtime-only. They do not alter ``TrainingRunSpec`` or
    any other durable schema.
    """

    method_ref: str
    payload_schema_id: str
    payload_schema_version: str
    payload_model: type[PayloadT]
    contract_compiler: Callable[[PayloadT], MethodContractSpec]
    update_kernels_factory: Callable[[PayloadT], Mapping[str, Callable[..., Mapping[str, Any]]]]
    guard_predicates_factory: Callable[
        [PayloadT], Mapping[str, Callable[..., Mapping[str, Any]]]
    ] = lambda _payload: {}
    preparation_provider: ExecutionPreparationProvider | None = None
    row_compiler: TrainingRowLowerer | None = None
    authoring_hook: TrainingMethodAuthoringHook[PayloadT] | None = None
    metadata_projector: TrainingMethodMetadataProjector[PayloadT] | None = None
    optimizer_spec_projector: TrainingMethodOptimizerSpecProjector[PayloadT] | None = None
    optimizer_step_extractor: TrainingMethodOptimizerStepExtractor[PayloadT] | None = None
    rejected_payload_versions: tuple[str, ...] = ()
    owner: str = "feedbax"
    package: str | None = None

    def registration(self) -> TrainingMethodRegistration:
        """Derive the single low-level registry row for this descriptor."""
        return TrainingMethodRegistration(
            method_ref=self.method_ref,
            payload_schema_id=self.payload_schema_id,
            payload_schema_version=self.payload_schema_version,
            payload_model=self.payload_model,
            contract_factory=None,
            contract_compiler=self.contract_compiler,
            update_kernels_factory=self.update_kernels_factory,
            guard_predicates_factory=self.guard_predicates_factory,
            rejected_payload_versions=self.rejected_payload_versions,
            owner=self.owner,
            package=self.package,
            requires_execution_preparation=self.preparation_provider is not None,
        )


@dataclass(frozen=True)
class ResolvedTrainingMethod(Generic[PayloadT]):
    """Typed, identity-checked runtime projection of one method request."""

    descriptor: TrainingMethodDescriptor[PayloadT] | None
    registration: TrainingMethodRegistration
    payload: PayloadT
    contract: MethodContractSpec
    effective_phase: EffectivePhaseSpec
    update_kernels: Mapping[str, Callable[..., Mapping[str, Any]]] = field(compare=False)
    guard_predicates: Mapping[str, Callable[..., Mapping[str, Any]]] = field(compare=False)


class TrainingManifestMetadataProjection(TrainingRunContractModel):
    """Governed request to project selected external payload metadata."""

    schema_id: Literal["feedbax.spec.training_manifest_metadata_projection"] = (
        TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID
    )
    schema_version: str = TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION
    source_payload_kind: str = Field(min_length=1)
    source_payload_schema_id: str = Field(min_length=1)
    source_payload_schema_version: str = Field(min_length=1)
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_schema_id: str = Field(min_length=1)
    projection_schema_version: str = Field(min_length=1)
    values: dict[str, Any]

    @model_validator(mode="after")
    def _validate_schema(self) -> "TrainingManifestMetadataProjection":
        if self.schema_version != TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION:
            raise ValueError(
                "/manifest_metadata_projection/schema_version unsupported version "
                f"{self.schema_version!r}; expected "
                f"{TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION!r}; "
                "migration_intentionally_absent=yes"
            )
        if not self.values:
            raise ValueError("/manifest_metadata_projection/values must not be empty")
        if any(not key for key in self.values):
            raise ValueError("/manifest_metadata_projection/values keys must be non-empty strings")
        return self


@dataclass(frozen=True)
class TrainingManifestMetadataProjectionRegistration:
    """Governance row for one external payload metadata projection schema."""

    source_payload_kind: str
    source_payload_schema_id: str
    source_payload_schema_version: str
    projection_schema_id: str
    projection_schema_version: str
    values_model: type[BaseModel]
    owner: str
    package: str

    @property
    def source_key(self) -> tuple[str, str, str]:
        """Return the external training payload identity governed by this row."""
        return (
            self.source_payload_kind,
            self.source_payload_schema_id,
            self.source_payload_schema_version,
        )


class TrainingMethodRegistry:
    """Registry for method payloads and independent manifest projection governance."""

    def __init__(self) -> None:
        self._registrations: dict[str, TrainingMethodRegistration] = {}
        self._descriptors: dict[str, TrainingMethodDescriptor[Any]] = {}
        self._metadata_projection_registrations: dict[
            tuple[str, str, str], TrainingManifestMetadataProjectionRegistration
        ] = {}

    def register(self, registration: TrainingMethodRegistration) -> None:
        """Register one method payload governance row."""
        if registration.method_ref in self._registrations:
            raise ValueError(f"training method already registered: {registration.method_ref!r}")
        producer_count = sum(
            producer is not None
            for producer in (registration.contract_factory, registration.contract_compiler)
        )
        if producer_count != 1:
            raise ValueError(
                f"training method {registration.method_ref!r} must define exactly one contract "
                "producer: contract_factory or contract_compiler"
            )
        self._registrations[registration.method_ref] = registration

    def register_descriptor(self, descriptor: TrainingMethodDescriptor[Any]) -> None:
        """Atomically register one descriptor and its derived low-level row."""
        required = {
            "method_ref": descriptor.method_ref,
            "payload_schema_id": descriptor.payload_schema_id,
            "payload_schema_version": descriptor.payload_schema_version,
            "owner": descriptor.owner,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"training method descriptor has empty fields={missing!r}")
        if descriptor.method_ref in self._descriptors:
            raise ValueError(
                f"training method descriptor already registered: {descriptor.method_ref!r}"
            )
        if not issubclass(descriptor.payload_model, BaseModel):
            raise TypeError("training method descriptor payload_model must extend BaseModel")
        hooks = {
            "contract_compiler": descriptor.contract_compiler,
            "update_kernels_factory": descriptor.update_kernels_factory,
            "guard_predicates_factory": descriptor.guard_predicates_factory,
            "preparation_provider": descriptor.preparation_provider,
            "row_compiler": descriptor.row_compiler,
            "optimizer_spec_projector": descriptor.optimizer_spec_projector,
            "optimizer_step_extractor": descriptor.optimizer_step_extractor,
        }
        invalid_hooks = [
            name for name, hook in hooks.items() if hook is not None and not callable(hook)
        ]
        if invalid_hooks:
            raise TypeError(f"training method descriptor has non-callable hooks={invalid_hooks!r}")
        if descriptor.authoring_hook is not None:
            if not isinstance(descriptor.authoring_hook, TrainingMethodAuthoringHook):
                raise TypeError(
                    "training method descriptor authoring_hook must be a "
                    "TrainingMethodAuthoringHook"
                )
            descriptor.authoring_hook.validate_structure()
        if descriptor.metadata_projector is not None:
            if not isinstance(descriptor.metadata_projector, TrainingMethodMetadataProjector):
                raise TypeError(
                    "training method descriptor metadata_projector must be a "
                    "TrainingMethodMetadataProjector"
                )
            descriptor.metadata_projector.validate_structure()
        self.register(descriptor.registration())
        self._descriptors[descriptor.method_ref] = descriptor

    def descriptor_keys(self) -> tuple[str, ...]:
        """Return method refs backed by atomic descriptors."""
        return tuple(sorted(self._descriptors))

    def descriptor(self, method_ref: MethodRefSpec | str) -> TrainingMethodDescriptor[Any] | None:
        """Return the descriptor for a method ref, if its row is descriptor-backed."""
        key = method_ref.key if isinstance(method_ref, MethodRefSpec) else method_ref
        return self._descriptors.get(key)

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

    def resolve_execution(
        self,
        method_ref: MethodRefSpec,
        envelope: MethodPayloadEnvelope,
        *,
        worker_execution: WorkerExecutionSpec | None = None,
    ) -> ResolvedTrainingMethod[Any]:
        """Resolve and construct one method exactly once for runtime execution."""
        registration = self.resolve(method_ref, path="/method_ref")
        payload = self.validate_payload(method_ref, envelope, path="/method_payload")
        method_key = method_ref.key
        if worker_execution is not None:
            if worker_execution.method_contract.method_ref != method_key:
                raise ValueError(
                    "/worker_execution/method_contract/method_ref must match /method_ref; "
                    f"found {worker_execution.method_contract.method_ref!r}, "
                    f"expected {method_key!r}"
                )
            if worker_execution.effective_phase.method_ref != method_key:
                raise ValueError(
                    "/worker_execution/effective_phase/method_ref must match /method_ref; "
                    f"found {worker_execution.effective_phase.method_ref!r}, "
                    f"expected {method_key!r}"
                )
            if (
                worker_execution.method_contract.method_payload_schema_version
                != envelope.schema_version
            ):
                raise ValueError(
                    "/worker_execution/method_contract/method_payload_schema_version must "
                    "match /method_payload/schema_version; found "
                    f"{worker_execution.method_contract.method_payload_schema_version!r}, "
                    f"expected {envelope.schema_version!r}"
                )
        contract = registration.compile_contract(payload)
        if contract.method_ref != method_key:
            raise ValueError(
                "/worker_execution/method_contract/method_ref compiled by registry must "
                f"match /method_ref; found {contract.method_ref!r}, expected {method_key!r}"
            )
        if contract.method_payload_schema_version != envelope.schema_version:
            raise ValueError(
                "/worker_execution/method_contract/method_payload_schema_version compiled by "
                "registry must match /method_payload/schema_version; found "
                f"{contract.method_payload_schema_version!r}, expected {envelope.schema_version!r}"
            )
        descriptor = self.descriptor(method_key)
        if worker_execution is not None and descriptor is not None:
            if worker_execution.method_contract != contract:
                raise ValueError(
                    "/worker_execution/method_contract must exactly match the payload-compiled "
                    f"contract for method_ref {method_key!r}"
                )
        update_kernels = registration.update_kernels_factory(payload)
        guard_predicates = registration.guard_predicates_factory(payload)
        if not isinstance(update_kernels, Mapping):
            raise TypeError(f"training method {method_key!r} returned non-mapping update kernels")
        if not isinstance(guard_predicates, Mapping):
            raise TypeError(f"training method {method_key!r} returned non-mapping guard predicates")
        from feedbax.training.worker_validation import validate_worker_contract

        effective_phase = validate_worker_contract(
            contract,
            update_kernels=update_kernels,
            guard_predicates=guard_predicates,
        )
        if (
            worker_execution is not None
            and descriptor is not None
            and worker_execution.effective_phase != effective_phase
        ):
            raise ValueError(
                "/worker_execution/effective_phase must exactly match the validated "
                f"effective phase for method_ref {method_key!r}"
            )
        return ResolvedTrainingMethod(
            descriptor=descriptor,
            registration=registration,
            payload=payload,
            contract=contract,
            effective_phase=effective_phase,
            update_kernels=dict(update_kernels),
            guard_predicates=dict(guard_predicates),
        )

    def register_manifest_metadata_projection(
        self,
        registration: TrainingManifestMetadataProjectionRegistration,
    ) -> None:
        """Register projection governance independently of training methods."""
        required = {
            "source_payload_kind": registration.source_payload_kind,
            "source_payload_schema_id": registration.source_payload_schema_id,
            "source_payload_schema_version": registration.source_payload_schema_version,
            "projection_schema_id": registration.projection_schema_id,
            "projection_schema_version": registration.projection_schema_version,
            "owner": registration.owner,
            "package": registration.package,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"manifest metadata projection registration is incomplete: empty fields={missing!r}"
            )
        if registration.values_model.model_config.get("extra") != "forbid":
            raise ValueError("manifest metadata projection values_model must set extra='forbid'")
        if registration.values_model.model_config.get("strict") is not True:
            raise ValueError("manifest metadata projection values_model must set strict=True")
        key = registration.source_key
        if key in self._metadata_projection_registrations:
            raise ValueError(
                "manifest metadata projection already registered for source payload "
                f"identity={key!r}"
            )
        self._metadata_projection_registrations[key] = registration

    def resolve_manifest_metadata_projection(
        self,
        source_key: tuple[str, str, str],
        *,
        path: str,
    ) -> TrainingManifestMetadataProjectionRegistration:
        """Resolve projection governance for an exact source payload identity."""
        try:
            return self._metadata_projection_registrations[source_key]
        except KeyError as exc:
            raise ValueError(
                f"{path}: no manifest metadata projection registered for source payload "
                f"identity={source_key!r}"
            ) from exc

    def validate_manifest_metadata_projection(
        self,
        projection: TrainingManifestMetadataProjection,
        *,
        path: str,
    ) -> tuple[TrainingManifestMetadataProjectionRegistration, dict[str, Any]]:
        """Validate one projection envelope and return JSON-mode canonical values."""
        registration = self.resolve_manifest_metadata_projection(
            (
                projection.source_payload_kind,
                projection.source_payload_schema_id,
                projection.source_payload_schema_version,
            ),
            path=path,
        )
        if projection.projection_schema_id != registration.projection_schema_id:
            raise ValueError(
                f"{path}/projection_schema_id unsupported schema_id "
                f"{projection.projection_schema_id!r}; expected "
                f"{registration.projection_schema_id!r}"
            )
        if projection.projection_schema_version != registration.projection_schema_version:
            raise ValueError(
                f"{path}/projection_schema_version unsupported schema version "
                f"{projection.projection_schema_version!r}; expected "
                f"{registration.projection_schema_version!r}; "
                "migration_intentionally_absent=yes"
            )
        validated = registration.values_model.model_validate(projection.values)
        return registration, validated.model_dump(mode="json", exclude_none=True)


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
                reads=["model", "optimizer", "prng", "objective", "batch_counter"],
                writes=["model", "optimizer", "prng", "train_loss", "batch_counter"],
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
                reads=["model", "optimizer", "prng", "objective", "batch_counter"],
                writes=["model", "optimizer", "prng", "train_loss", "batch_counter"],
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
                    CheckpointSlotSpec(slot="batch_counter"),
                ],
            )
        ],
        batch_progress=TrainingBatchProgressSpec(slot="batch_counter"),
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
            StateSlotSpec(name="batch_counter", role="auxiliary"),
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
            "train_loss": float(coordinate.program_step + 1),
            "batch_counter": slots["batch_counter"] + 1,
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


def standard_supervised_method_descriptor() -> TrainingMethodDescriptor[
    StandardSupervisedMethodPayload
]:
    """Return the atomic descriptor for Feedbax's standard supervised method."""
    return TrainingMethodDescriptor(
        method_ref=STANDARD_SUPERVISED_METHOD_REF,
        payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload_model=StandardSupervisedMethodPayload,
        contract_compiler=lambda _payload: standard_supervised_method_contract(),
        update_kernels_factory=standard_supervised_update_kernels,
        optimizer_spec_projector=lambda payload: payload.optimizer,
        optimizer_step_extractor=_standard_supervised_optimizer_step,
        rejected_payload_versions=("feedbax.spec.training_method.standard_supervised_payload.v0",),
        owner="feedbax.contracts.training",
        package="feedbax",
    )


def _standard_supervised_optimizer_step(
    payload: StandardSupervisedMethodPayload,
    runtime: Mapping[str, Any],
) -> int:
    """Extract the standard method's declared optimizer count without tree search."""
    value: Any
    metadata = runtime.get("metadata")
    if isinstance(metadata, Mapping) and "optimizer_step" in metadata:
        value = metadata["optimizer_step"]
    elif "optimizer_step" in runtime:
        value = runtime["optimizer_step"]
    else:
        optimizer = runtime.get(payload.optimizer_slot)
        if not isinstance(optimizer, Mapping) or "count" not in optimizer:
            raise ValueError(
                "standard supervised optimizer step requires either checkpoint "
                "/metadata/optimizer_step or the declared optimizer slot count"
            )
        value = optimizer["count"]
    item = getattr(value, "item", None)
    if callable(item):
        size = getattr(value, "size", 1)
        if size != 1:
            raise ValueError("standard supervised optimizer count must contain one scalar")
        value = item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("standard supervised optimizer count must be numeric")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("standard supervised optimizer count must be integer-valued")
    step = int(value)
    if step < 0:
        raise ValueError("standard supervised optimizer count must be non-negative")
    return step


def default_training_method_registry() -> TrainingMethodRegistry:
    """Return the default method-ref keyed payload registry."""
    registry = TrainingMethodRegistry()
    registry.register_descriptor(standard_supervised_method_descriptor())
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
    _resolved_method: ResolvedTrainingMethod[Any] | None = PrivateAttr(default=None)
    _resolved_method_cache_key: tuple[str, str, str] | None = PrivateAttr(default=None)

    def _method_resolution_cache_key(self) -> tuple[str, str, str]:
        """Return a deterministic key for all method-resolution inputs."""

        def canonical(value: BaseModel) -> str:
            return json.dumps(
                value.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )

        return (
            self.method_ref.key,
            canonical(self.method_payload),
            canonical(self.worker_execution),
        )

    @property
    def resolved_method(self) -> ResolvedTrainingMethod[Any]:
        """Lazily return the strict, validated runtime-only method projection."""
        cache_key = self._method_resolution_cache_key()
        if self._resolved_method is None or self._resolved_method_cache_key != cache_key:
            self._resolved_method = DEFAULT_TRAINING_METHOD_REGISTRY.resolve_execution(
                self.method_ref,
                self.method_payload,
                worker_execution=self.worker_execution,
            )
            self._resolved_method_cache_key = cache_key
        return self._resolved_method

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

        DEFAULT_TRAINING_METHOD_REGISTRY.resolve(
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
        return self


# Enable forward references
LossTermSpec.model_rebuild()
